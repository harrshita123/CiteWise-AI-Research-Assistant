# CiteWise

CiteWise is an academic PDF research assistant in development. The current repository includes authentication, an owner-scoped paper library, PDF processing, and locally stored embeddings. Its goal is to answer questions with verifiable links to the supporting PDF pages.

## Repository layout

```text
backend/
  app/{providers,repositories,routers,services}/  API and processing code
  scripts/                                Maintenance and evaluation tools
  tests/                                  Backend tests
frontend/
  src/{components,pages,services}/       React UI and API clients
  src/styles/                             Shared and page-specific CSS
  public/                                 Static assets
```

`frontend/src/styles/index.css` defines the CSS load order. Local `work/` documents, virtual environments, build output, and storage data are ignored by Git.

## Current features

- Email/password and Google sign-in with an HttpOnly CiteWise session cookie
- Owner-scoped paper library, PDF upload, preview, view, and deletion
- Page-aware text extraction with PyMuPDF, overlapping chunks, and local MiniLM embeddings
- Single-paper question answering with page sources and saved conversations
- Answer checks with source passages and per-answer feedback; admin retrieval evaluation with saved Hit@5 runs

The original PDFs are stored as authenticated Cloudinary assets. MongoDB stores users and paper metadata; ChromaDB stores searchable chunks and vectors. The backend checks paper ownership before searching, answering, or opening an original PDF.

## Processing flow

```text
PDF upload → validate and check ownership → extract text by page → chunk
           → embed with MiniLM → store chunks and vectors in ChromaDB
           → store original PDF in Cloudinary
```

The app can preview extracted chunks, open owned PDFs, answer questions from retrieved passages, and measure whether labeled source pages appear among retrieved results.

## Run locally

You need Python 3.13, Node.js with npm, MongoDB, a Cloudinary account with PDF upload access, and a Google OAuth web client ID.

1. Copy `backend/.env.example` to `backend/.env` and fill in the MongoDB, Google, JWT, Cloudinary, and allowed-origin settings. Copy `frontend/.env.example` to `frontend/.env` and set the same Google web client ID. Keep secrets in `backend/.env` only.
2. Start the API from `backend/`:

   ```bash
   python3.13 -m venv .venv313
   .venv313/bin/pip install -r requirements.txt
   .venv313/bin/uvicorn app.main:app --reload
   ```

3. Start the website from `frontend/` in another terminal

   ```bash
   npm install
   npm run dev
   ```

Open the URL printed by Vite, normally `http://localhost:5173`. If Vite selects another port, add that exact origin to `ALLOWED_ORIGINS` in `backend/.env` and restart the API.

For detailed configuration and troubleshooting, see [backend/README.md](backend/README.md) and [frontend/README.md](frontend/README.md).

## Check the project

```bash
cd backend
.venv313/bin/python -m unittest discover -s tests -v
```

```bash
cd frontend
npm run build
```

## Current limits

Upload processing runs within the request and uses an in-process lock; it is not a durable job queue. Scanned-only PDFs need OCR, which is not implemented. Multi-paper comparison, labeled answer-quality evaluation, and deployment are still in progress. Retrieval Hit@5 does not measure whether generated answers are correct.
