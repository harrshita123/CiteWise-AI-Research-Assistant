import unittest
from unittest.mock import AsyncMock, Mock, patch

from app.services.identity_worker import process_one


class IdentityWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_transient_errors_retry_at_most_three_times(self):
        repository = Mock()
        repository.finish_identity_job = AsyncMock(return_value=True)
        discovery = Mock()
        discovery.client.retry_delay.return_value = 120
        for attempt in (1, 2, 3):
            repository.claim_identity_job = AsyncMock(return_value={"identity_input": {}, "publication_identity": {"attempts": attempt}})
            with patch("app.services.identity_worker.verify_identity_input", return_value={
                    "status": "verification_unavailable", "reason": "The academic source could not be checked."}):
                await process_one(repository, discovery)
            result = repository.finish_identity_job.call_args.args[1]
            self.assertEqual(result["status"], "pending" if attempt < 3 else "verification_unavailable")
            self.assertIn("next_retry_at" if attempt < 3 else "retry_after", result)

    async def test_mismatch_does_not_schedule_automatic_retry(self):
        repository = Mock()
        repository.claim_identity_job = AsyncMock(return_value={"identity_input": {}, "publication_identity": {"attempts": 1}})
        repository.finish_identity_job = AsyncMock(return_value=True)
        with patch("app.services.identity_worker.verify_identity_input", return_value={
                "status": "verification_unavailable", "reason": "No reliable match."}):
            await process_one(repository, Mock())
        result = repository.finish_identity_job.call_args.args[1]
        self.assertNotIn("next_retry_at", result)

    async def test_long_provider_cooldown_is_retryable_without_a_pending_job(self):
        repository = Mock()
        repository.claim_identity_job = AsyncMock(return_value={"identity_input": {}, "publication_identity": {"attempts": 1}})
        repository.finish_identity_job = AsyncMock(return_value=True)
        discovery = Mock()
        discovery.client.retry_delay.return_value = 3600
        with patch("app.services.identity_worker.verify_identity_input", return_value={
                "status": "verification_unavailable", "reason": "The academic source could not be checked."}):
            await process_one(repository, discovery)
        result = repository.finish_identity_job.call_args.args[1]
        self.assertEqual(result["status"], "verification_unavailable")
        self.assertIn("retry_after", result)
        self.assertNotIn("next_retry_at", result)
