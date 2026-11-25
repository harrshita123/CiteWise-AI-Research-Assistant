# CiteWise mentor demo on Vercel and Render

Vercel serves the React build and forwards `/api/*` to Render. The browser stays on the Vercel domain, including for guest sign-in and PDF links. The FastAPI service uses a separate database on your existing Atlas cluster.

## 1. Pick the Render storage option

The Free instance is suitable for a short trial. Render can spin it down, and its ephemeral filesystem loses uploaded PDFs and Chroma indexes on restart. Atlas metadata survives, so an old workspace may list files whose local data is gone. Start a fresh guest session after a restart. The source checkout always includes the bundled paper.

For a mentor link with persistent uploads and indexes, attach a Render persistent disk on an approved paid instance. Mount it at `/var/data` and set `STORAGE_DIRECTORY=/var/data/citewise-demo`. Keep one backend instance. [Render Free instances](https://render.com/docs/free) · [Persistent disks](https://render.com/docs/disks)

## 2. Prepare Atlas and Render

1. In your existing Atlas project, create a database user with `readWrite` scoped to `citewise_demo_mentor`. Scope that user to the existing cluster if Atlas offers the option. Use its own password, and build its Atlas SRV connection string. Leave production database users and reservations alone. [Atlas database users](https://www.mongodb.com/docs/atlas/security-add-mongodb-users/)
2. Publish the reviewed demo code to a dedicated branch of your existing repository, such as `citewise-hosted-demo`. Render and Vercel must build from a commit containing the demo code and bundled PDF. Confirm the branch before linking either project.
3. Create a separate **Web Service** in Render from that branch. Use the following settings. [Render FastAPI setup](https://render.com/docs/deploy-fastapi)

| Setting | Value |
| --- | --- |
| Root Directory | `backend` |
| Runtime | Python |
| Build Command | `python -m pip install -r requirements.txt` |
| Start Command | `python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT` |
| Health Check Path | `/health` |

4. In Render's **Environment** settings, set the values from [render.demo.env.example](render.demo.env.example). Enter the scoped Atlas connection string and a fresh `CITEWISE_DEMO_JWT_SECRET` directly there. Generate the latter with `openssl rand -hex 32`. Keep `CITEWISE_DEMO_MODE=true`. The demo uses neither OpenRouter nor Tavily, so leave both provider keys unset.
5. In Atlas, allowlist the outgoing IP ranges shown on the Render service's **Connect** menu. Add those ranges alongside the existing entries. Once the service is healthy, open its `/api/v1/auth/demo` endpoint and confirm it returns `{"demo_mode":true}`. [Render outbound IP addresses](https://render.com/docs/outbound-ip-addresses) · [Atlas IP access list](https://www.mongodb.com/docs/atlas/security/ip-access-list/)

## 3. Route Vercel to Render

1. Copy [vercel.demo.json](vercel.demo.json) to `frontend/vercel.json` on the demo branch and replace both placeholder destinations with the actual Render HTTPS origin. Keep `/api/:path*` in the API destination. The SPA fallback covers direct visits to routes such as `/research`. [Vercel rewrites](https://vercel.com/docs/routing/rewrites) · [Vite on Vercel](https://vercel.com/docs/frameworks/frontend/vite)
2. Create a separate Vercel project using that repository and demo branch. Select **Vite**, set **Root Directory** to `frontend`, and use `npm ci`, `npm run build`, and `dist` for install, build, and output. Leave `VITE_API_BASE_URL` and `VITE_GOOGLE_CLIENT_ID` unset. All frontend API calls stay relative to the Vercel domain. The database URI and session secret belong only in Render.
3. Set Render's `ALLOWED_ORIGINS` to the exact Vercel HTTPS hostname, including `https://` and without a trailing slash. Keep `SESSION_COOKIE_SECURE=true`. If testing a preview URL, add that precise origin during testing. Redeploy Render to apply the settings.

## 4. Verify the URL before sharing

- Open `/health` and `/api/v1/auth/demo` through the Vercel URL.
- Use a private browser window. Open the guest workspace, choose a suggested question, open its cited PDF page, and load one sample external answer.
- Upload a test paper, search for a passage, and confirm an invalid PDF is rejected.
- If using a persistent disk, verify that the test PDF and its passage search still work after a restart before sharing the link. On Free, allow the service and embedding model to warm up before your demo. Vercel limits proxied requests to 120 seconds. [Vercel limits](https://vercel.com/docs/limits#proxied-request-timeout)

Share the verified Vercel HTTPS URL with your mentor.
