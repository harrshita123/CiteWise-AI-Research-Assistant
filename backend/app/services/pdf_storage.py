"""Store original PDFs separately from paper metadata and searchable text."""

from io import BytesIO
from pathlib import Path
from time import time

import cloudinary
import cloudinary.uploader
import cloudinary.utils

from ..config import Settings


class LocalPdfStorage:
    provider = "local"

    def __init__(self, directory: Path):
        self.directory = directory

    def path(self, paper_id: str) -> Path:
        return self.directory / f"{paper_id}.pdf"

    def delete(self, paper_id: str) -> None:
        self.path(paper_id).unlink(missing_ok=True)


class CloudinaryPdfStorage:
    provider = "cloudinary"

    def __init__(self, settings: Settings):
        cloudinary.config(
            cloud_name=settings.cloudinary_cloud_name,
            api_key=settings.cloudinary_api_key,
            api_secret=settings.cloudinary_api_secret.get_secret_value(),
            secure=True,
        )

    @staticmethod
    def public_id(paper_id: str) -> str:
        return f"citewise/papers/{paper_id}"

    def save(self, paper_id: str, content: bytes) -> None:
        source = BytesIO(content)
        source.name = "paper.pdf"
        result = cloudinary.uploader.upload(
            source,
            public_id=self.public_id(paper_id),
            resource_type="image",
            type="authenticated",
            overwrite=False,
        )
        if result.get("public_id") != self.public_id(paper_id):
            raise RuntimeError("Cloudinary returned an unexpected paper identifier.")

    def delete(self, paper_id: str) -> None:
        result = cloudinary.uploader.destroy(
            self.public_id(paper_id), resource_type="image", type="authenticated", invalidate=True,
        )
        if result.get("result") not in {"ok", "not found"}:
            raise RuntimeError("Cloudinary could not remove the paper.")

    def view_url(self, paper_id: str) -> str:
        return cloudinary.utils.private_download_url(
            self.public_id(paper_id), "pdf", resource_type="image",
            type="authenticated", expires_at=int(time()) + 60,
        )
