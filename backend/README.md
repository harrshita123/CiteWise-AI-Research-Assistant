# CiteWise API

This service supports email/password and Google authentication, stores users in MongoDB, and sets an HttpOnly session cookie.

It also supports the first paper ingestion milestone: PDF upload, page-aware text extraction, overlapping chunks, local embeddings, and persistent ChromaDB storage.

Google login uses Google Identity Services in popup mode. The React credential callback sends the Google-signed ID token to `POST /api/v1/auth/google`; the API verifies it with Google's Python authentication library before creating a CiteWise session. CiteWise does not currently request Google API access or refresh tokens.

## Local setup

1. Copy `.env.example` to `.env` and replace every placeholder.
2. In Google Cloud, create an OAuth 2.0 web client and add `http://localhost:5173` to **Authorized JavaScript origins**.
3. Install dependencies and start the API:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --reload
```

The frontend development server proxies `/api` requests to `http://127.0.0.1:8000`.

Passwords are hashed with Argon2. Plaintext passwords and Google ID tokens are never stored.

## Roles

Every new account receives the `user` role. An administrator must be assigned directly in MongoDB by a trusted operator. The frontend never chooses or changes roles.

Authenticated users can access `GET /api/v1/auth/me`. Administrators can also access `GET /api/v1/admin/users`. The admin endpoint supports `limit` and `offset` query parameters and never returns password hashes or Google account identifiers.

## Current local compatibility note

The current Python 3.14 environment has shown an intermittent Atlas TLS failure with PyMongo's asynchronous client even when a direct TLS handshake and synchronous PyMongo ping succeed. Do not disable TLS verification. Prefer a tested Python 3.12 or 3.13 environment while this compatibility issue is unresolved.

## PDF to embeddings milestone

Sign in through the frontend and open `/papers`. Upload one text-based PDF, then inspect its page count, chunk count, and first ten chunks. Each chunk preserves the original PDF page number and a SHA-256 content hash. The filename is display metadata only; stored files use server-generated identifiers.

The flow is `papers router -> IngestionService -> PyMuPDF -> token-aware chunks -> LocalEmbeddings -> ChromaDB`. MongoDB stores paper metadata and ownership. PDFs, vectors, and the model cache live under the ignored `backend/storage/` directory.

- Local model: `all-MiniLM-L6-v2`, CPU inference, 384 dimensions, no API key.
- First use downloads approximately 80 MB of model files. Later inference runs locally. For a smooth demo, run the real embedding check below before uploading.
- Each chunk has at most 220 model tokens, with 30 overlapping tokens within the same page. Chunking uses a non-truncating copy of the model's tokenizer.
- Embeddings are written in batches of 32. Settings can reduce the batch size and upload limits.
- Default limits: 25 MB, 300 pages, and two million extracted characters. Encrypted PDFs and PDFs without readable text are rejected. Blank pages are skipped without changing later page numbers. Mixed scanned/text PDFs index only their extractable text; OCR and advanced multi-column layout analysis are future work.
- Upload is a single request that returns `201` after embedding completes. CPU and storage work run outside the async event loop. Run **one API worker** for this milestone; concurrent uploads receive `429` with `Retry-After`.
- Duplicate PDFs are checked per owner. Reads and vector previews enforce ownership. Failed processing removes partial vectors, files, and the metadata reservation so the upload can be retried. A cleanup failure leaves a failed record for operator attention.

| Endpoint | Result |
|---|---|
| `POST /api/v1/papers` | Multipart `file`; completed paper and up to ten chunk previews |
| `GET /api/v1/papers?limit=50&offset=0` | Paginated papers belonging to the signed-in user |
| `GET /api/v1/papers/summary` | Total and per-status counts for the signed-in user's dashboard |
| `GET /api/v1/papers/{id}` | Owned paper and up to ten chunk previews |
| `DELETE /api/v1/papers/{id}` | Remove an owned, non-processing paper, its PDF, and its vectors |

This is a learning milestone, not the complete ingestion system. Durable jobs, restart recovery, multi-file upload, retries, retrieval, LLM answers, and WebSockets are not implemented here. A process crash can leave a `processing` record requiring operator cleanup; keep the API running during the demo. Production needs request-body limits at the proxy and a durable worker queue. Model quality on research papers has not yet been evaluated.

## Verify locally

From `backend`:

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m scripts.check_embeddings
```

Unit/API tests use in-memory MongoDB substitutes and deterministic test vectors, alongside real PyMuPDF and ChromaDB. `check_embeddings` separately runs the real local model, checks persisted vectors, and verifies token limits and ownership filtering without reading MongoDB credentials or writing to Atlas.
