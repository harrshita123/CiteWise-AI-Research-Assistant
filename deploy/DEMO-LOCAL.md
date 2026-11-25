# Run CiteWise Demo Mode locally

Start local MongoDB on port 27017. From the repository root, create a private demo environment file once. The normal backend continues to load `backend/.env`.

```bash
mkdir -p backend/storage
(
  umask 077
  set -C
  {
    printf '%s\n' 'CITEWISE_DEMO_MODE=true' \
      'CITEWISE_DEMO_MONGODB_URI=mongodb://127.0.0.1:27017' \
      'CITEWISE_DEMO_DATABASE=citewise_demo_local' \
      'ALLOWED_ORIGINS=http://localhost:5174' 'SESSION_COOKIE_SECURE=false'
    printf 'CITEWISE_DEMO_JWT_SECRET=%s\n' "$(openssl rand -hex 32)"
  } > backend/storage/.env.demo.local
)
```

Start FastAPI and Vite in separate terminals after installing their normal dependencies. Substitute your backend virtual environment name if it differs.

```bash
cd backend
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8004 --env-file storage/.env.demo.local
```

```bash
cd frontend
VITE_API_BASE_URL= npm run dev -- --mode demo --port 5174 --strictPort
```

Vite in demo mode sends relative API requests to port 8004. The normal Vite mode still targets port 8000. Open `http://localhost:5174/` and choose **Open demo workspace**. Check `http://localhost:5174/api/v1/auth/demo` for `{"demo_mode":true}`.

## Verify the demo

Start a fresh guest workspace to receive the bundled PLOS ONE article. If you opened the earlier demo, sign out before starting the new workspace. Select **How did the authors simulate each experiment?** and open its page 2 citation. Then select the **External sample** question about the 2007 exchange to see why the PDF alone is insufficient, the disagreement between two published commentaries, and their original source links. Sample answers and claim checks are predefined. Other uploaded research papers use local passage search.

In the Library, upload a text-based research paper, an ambiguous PDF, and a certificate to check the three upload outcomes. Only the accepted paper appears in the research selector. Demo uploads indicate when an arXiv identifier was found in the PDF. Live publication verification remains a production service, while the bundled paper has a checked publisher record.

## Host the demo separately

Use the service and HTTPS proxy templates in [the deployment guide](README.md) on a separate demo host. Use [the demo environment template](demo.env.example) for its service environment, a fresh demo session secret, a MongoDB account scoped to the demo database, and a separate persistent storage directory. Leave the frontend API base URL empty so the demo hostname proxies its own API. Keep the bundled file under `backend/app/fixtures/neves-et-al-2022.pdf` in the deployed checkout. [Paper attribution and license](../backend/app/fixtures/NEVES-2022.md)

No OpenRouter or Tavily credentials are required. Warm the local MiniLM cache before the first guest session, or allow time for the initial model download during sample preparation.
