# CiteWise API

This service supports email/password and Google authentication, stores users in MongoDB, and sets an HttpOnly session cookie.

It also supports PDF upload, page-aware text extraction, overlapping chunks, local embeddings, persistent ChromaDB storage, Cloudinary storage for new original PDFs, and owner-scoped semantic passage search.

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
If Vite uses port 5174 because 5173 is busy, include `http://localhost:5174` in `ALLOWED_ORIGINS` and restart the API. The upload route checks the browser's exact origin as well as CORS.

New PDFs are stored in Cloudinary. Fill in `CLOUDINARY_CLOUD_NAME`, `CLOUDINARY_API_KEY`, and `CLOUDINARY_API_SECRET` in `backend/.env`; the API will refuse to start without them. Keep the API secret on the backend; never put it in a `VITE_` variable. PDFs uploaded before this change remain readable and deletable from the server's legacy `storage/uploads/` folder; they are not migrated automatically.

The API key must be allowed to create and delete assets. A live upload check with a key assigned only the Cloudinary **Media Library User** role returned `Request forbidden due to missing permissions (actions=["create"])`; that role alone is insufficient. On this account, the **Assign Roles** menu offers only **Master Admin** and **Media Library User**. With Master Admin on the dedicated local-development key, a temporary PDF upload, signed download, byte comparison, and deletion all passed. Master Admin grants broad product-environment access; a narrower production setup needs appropriate folder-scoped permissions configured for the key through Cloudinary's API.

Cloudinary uploads use authenticated PDF assets. After checking paper ownership, the API redirects the browser to a signed download URL that expires in 60 seconds. Cloudinary's Free-plan security settings may block PDF delivery until **Allow delivery of PDF and ZIP files** is enabled. This setting is separate from making assets public.

Passwords are hashed with Argon2. Plaintext passwords and Google ID tokens are never stored.

## Roles

Every new account receives the `user` role. An administrator must be assigned directly in MongoDB by a trusted operator. The frontend never chooses or changes roles.

Authenticated users can access `GET /api/v1/auth/me`. Administrators can also access `GET /api/v1/admin/users`. The admin endpoint supports `limit` and `offset` query parameters and never returns password hashes or Google account identifiers.

## Current local compatibility note

Atlas TLS handshakes failed intermittently in the local Python 3.14 environment, including with Uvicorn's `--loop asyncio`. After confirming the Atlas IP access list, a separate Python 3.13 environment passed five MongoDB pings, started the API, returned `{"status":"ok"}` from `/health`, and passed 24 backend tests. On this machine, use the installed `.venv313/bin/uvicorn app.main:app --reload` after stopping the old server on port 8000. For a fresh Python 3.13 setup, create `.venv313` with `python3.13 -m venv .venv313` and install `requirements.txt`. Keep TLS certificate verification enabled.

## PDF to embeddings milestone

Sign in through the frontend and open `/papers`. Upload one text-based PDF, then inspect its page count, chunk count, and first ten chunks. Each chunk preserves the original PDF page number and a SHA-256 content hash. The filename is display metadata only; stored files use server-generated identifiers.

The flow is `papers router -> IngestionService -> PyMuPDF -> token-aware chunks -> LocalEmbeddings -> ChromaDB -> Cloudinary`. MongoDB stores paper metadata and ownership. New PDFs are uploaded as authenticated Cloudinary assets; vectors and the model cache remain under the ignored `backend/storage/` directory. Existing local PDFs stay under `backend/storage/uploads/` until removed or migrated.

- Local model: `all-MiniLM-L6-v2`, CPU inference, 384 dimensions, no API key.
- First use downloads approximately 80 MB of model files. Later inference runs locally. For a smooth demo, run the real embedding check below before uploading.
- Each chunk has at most 220 model tokens, with 30 overlapping tokens within the same page. Chunking uses a non-truncating copy of the model's tokenizer.
- Repeated text lines near the top or bottom of at least 60% of PDF pages are removed before chunking, including running headers and footers with changing page numbers. Unique margin text is kept. This does not change the original PDF or previously stored chunks; existing papers need to be reprocessed to use the cleaner.
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
| `GET /api/v1/papers/{id}/file` | Owned original PDF, locally served or redirected to a short-lived Cloudinary URL |
| `DELETE /api/v1/papers/{id}` | Remove an owned, non-processing paper, its PDF, and its vectors |
| `POST /api/v1/research/search` | Search one owned ready paper using `paper_id`, `question`, and optional `limit` (1–10); return ranked passages with page numbers |
| `GET /api/v1/research/capabilities` | Report whether the configured answer provider is available |
| `POST /api/v1/research/ask` | Assess retrieved PDF chunks, answer from them when sufficient, and conditionally search external sources for missing information |
| `GET /api/v1/research/conversations` | List the signed-in user's saved single-paper Q&A sessions |
| `GET /api/v1/research/conversations/{id}` | Read a saved session with owner enforcement |
| `DELETE /api/v1/research/conversations/{id}` | Delete an owned saved session |
| `GET /api/v1/evaluation/runs` | List an admin's saved retrieval evaluation runs |
| `POST /api/v1/evaluation/runs` | Admin only: evaluate 1–50 labeled questions against owned ready papers; save Hit@5 and per-question pages |

## Question answering

Answer generation uses [Groq](https://console.groq.com/) with `openai/gpt-oss-120b`. Create a Groq API key and set `GROQ_API_KEY` in the ignored `backend/.env`; never put it in the frontend or commit it. Groq receives the question and selected PDF excerpts, not the full PDF. Restart the backend after changing `.env`. The Research page shows Generate answer when Groq is configured. Free API usage has account-specific rate limits, so check them in Groq Console. `GROQ_MODEL` can select another compatible model that supports strict JSON schema output.

Before answering, Groq returns `sufficient`, `partial`, or `none`, supporting chunk IDs and verbatim quote fragments, and the missing information for a partial result. The backend checks IDs and quotes against retrieved chunks, requires a question-term match for supported assessments, and rejects an unsupported `none` when a retrieved chunk has a strong lexical match. An explicit sample size can also be read directly from a retrieved sentence. Invalid assessments do not authorize external search. These checks validate source linkage, not full semantic entailment or every page of the PDF. A request for a missing statistic from this particular paper, such as its p-value, never triggers web search; another source cannot provide it. For `sufficient`, answer generation uses only assessed PDF evidence. For `partial` or `none`, a configured Tavily search supplies external excerpts; Groq returns separate PDF and external sections with IDs that the backend checks against supplied sources. Indian legal searches accept only official `.gov.in` and `.nic.in` results. External links and excerpts and PDF pages are displayed separately and saved with the conversation. Set `TAVILY_API_KEY` in the ignored `backend/.env` to enable fallback. Without it, CiteWise reports the PDF evidence limit and does not guess. Search snippets and generated claims may still be incomplete or mistaken; verify important claims against the linked sources. Q&A sessions and individual answer feedback are saved in MongoDB with owner checks. Answer quality still needs evaluation on a labeled set of real-paper questions.

This is a learning milestone, not the complete ingestion and Q&A system. Durable jobs, restart recovery, multi-file upload, retries, and WebSockets are not implemented here. A process crash can leave a `processing` record requiring operator cleanup; keep the API running during the demo. Production needs request-body limits at the proxy and a durable worker queue. Retrieval and answer quality on research papers have not yet been evaluated.

## Verify locally

From `backend`:

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m scripts.check_embeddings
```

Unit/API tests use in-memory MongoDB substitutes and deterministic test vectors, alongside real PyMuPDF and ChromaDB. `check_embeddings` separately runs the real local model, checks persisted vectors, and verifies token limits and ownership filtering without reading MongoDB credentials or writing to Atlas.

## Retrieval evaluation

Create a JSON Lines file with one labelled question per line. Each case needs an indexed paper's `owner_id` and `paper_id`, the `question`, and one or more correct original PDF page numbers in `expected_pages`:

```json
{"owner_id":"example-owner","paper_id":"example-paper","question":"Which libraries worked with OCLC?","expected_pages":[2]}
```

Run `.venv313/bin/python -m scripts.evaluate_retrieval cases.jsonl` from `backend/`. The script searches ChromaDB with the same embedding and ranking path as `/research/search` and reports page-level Hit@5. It does not call Groq or print passage text. Use a fixed set of real paper questions before claiming retrieval or answer reliability; the example above is a format illustration, not an evaluation result.

The admin Retrieval evaluation dashboard offers the same page-level check through the API. Label each question with an expected PDF viewer page, run retrieval, and inspect saved Hit@5 scores plus returned pages. Runs are owner-scoped in MongoDB. This metric does not assess generated answer correctness.
