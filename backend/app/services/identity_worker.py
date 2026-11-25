"""Persistent, leased publication identity work backed by paper records."""
import asyncio
import logging
import random
from datetime import datetime, timedelta, timezone

from starlette.concurrency import run_in_threadpool
from .publication_identity import extract_identity_input, extracted_paper_metadata, verify_identity_input

logger = logging.getLogger(__name__)


async def process_one(repository, discovery, read_original=None):
    paper = await repository.claim_identity_job()
    if not paper:
        return False
    job = paper["publication_identity"]
    identity_input = paper["identity_input"] or {}
    refresh = "not_requested"
    if job.get("refresh_extraction") and read_original is not None:
        try:
            content = await run_in_threadpool(read_original, paper)
            updated = await run_in_threadpool(extract_identity_input, content)
            updated.pop("tail", None)
            previous_fields = extracted_paper_metadata(identity_input)
            new_fields = extracted_paper_metadata(updated)
            refresh = "unchanged"
            identifiers_agree = all(not previous_fields.get(field) or previous_fields[field] == new_fields.get(field)
                                    for field in ("doi", "arxiv_id", "pmid", "pmcid"))
            if (updated != identity_input and len(updated["text"].strip()) >= 40
                    and previous_fields.keys() <= new_fields.keys() and identifiers_agree):
                identity_input = updated
                refresh = "updated"
        except Exception as exc:
            refresh = "original_unavailable"
            logger.warning("publication extraction refresh failed error_type=%s", type(exc).__name__)
    try:
        result = await run_in_threadpool(verify_identity_input, identity_input, discovery)
    except Exception as exc:
        logger.warning("publication identity check failed error_type=%s", type(exc).__name__)
        result = {"status": "verification_unavailable", "reason_code": "provider_unavailable", "reason": "The academic source could not be checked."}
    if result.get("reason_code") == "provider_unavailable" or result.get("reason") == "The academic source could not be checked.":
        client = getattr(discovery, "client", None)
        delay = client.retry_delay() if client is not None and callable(getattr(client, "retry_delay", None)) else 0
        attempts = paper["publication_identity"].get("attempts", 1)
        retry_at = datetime.now(timezone.utc) + timedelta(seconds=max(60 * (2 ** min(attempts - 1, 3)), delay) + random.uniform(0, 5))
        if attempts < 3 and delay < 300:
            result = {"status": "pending", "reason_code": "provider_unavailable", "reason": "The academic source is temporarily unavailable; retry scheduled.",
                      "next_retry_at": retry_at, "attempts": attempts}
            if refresh == "original_unavailable":
                result["refresh_extraction"] = True
        else:
            result["retry_after"] = retry_at
            result["attempts"] = attempts
    if refresh == "original_unavailable" and result["status"] in {"not_verified", "conflicting", "registry_record", "verification_unavailable"}:
        result["reason"] += " The original PDF could not be reopened to refresh its metadata."
        if result.get("reason_code") == "insufficient_metadata":
            result["reason_code"] = "original_pdf_unavailable"
    if job.get("refresh_extraction") and result["status"] in {"not_verified", "conflicting", "registry_record"}:
        # An unchanged search can reuse fresh responses. Wait for the shared
        # five-minute cache to expire before offering the same check again.
        result["retry_after"] = datetime.now(timezone.utc) + timedelta(minutes=5)
    fields = {"identity_input": identity_input} if refresh == "updated" else {}
    saved = await repository.finish_identity_job(paper, result, **fields)
    logger.info("publication job extraction=%s fields=%s status=%s reason=%s persisted=%s retry_scheduled=%s",
                refresh, sorted(extracted_paper_metadata(identity_input)), result["status"],
                result.get("reason_code"), saved, "next_retry_at" in result or "retry_after" in result)
    return True


async def run_worker(repository, discovery, read_original=None):
    while True:
        try:
            if not await process_one(repository, discovery, read_original):
                await asyncio.sleep(3)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("publication identity worker failed error_type=%s", type(exc).__name__)
            await asyncio.sleep(5)
