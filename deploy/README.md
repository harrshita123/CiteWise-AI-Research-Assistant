# CiteWise deployment

For the isolated mentor demo, use [the demo setup guide](DEMO-LOCAL.md) and the demo environment template.

The template assumes one Linux VM with systemd, Python 3.13, Node.js 24, Caddy 2, and a public DNS name. Caddy serves the Vite build and proxies `/api/*` and `/health` to Uvicorn on loopback. Use a persistent disk for Chroma, cached models, and any legacy PDFs. Mount or copy the existing storage directory if the demo must include previously uploaded PDFs.

## Configuration

Use a dedicated `citewise` service user and a readable checkout at `/opt/citewise`. Create `/etc/citewise/backend.env` with access restricted to the operator. Use the current MongoDB Atlas database and OpenRouter key.

| Setting | Production value |
| --- | --- |
| `MONGODB_URI`, `MONGODB_DATABASE` | Existing Atlas connection and database. Allow the VM's egress IP in Atlas. |
| `STORAGE_DIRECTORY` | Existing or restored persistent storage, such as `/var/lib/citewise`. The service user needs write access. |
| `JWT_SECRET` | A server-only secret of at least 32 characters. Preserve the existing secret to keep current sessions valid. |
| `ALLOWED_ORIGINS` | Exact public origin, including `https://` and without a trailing slash. |
| `SESSION_COOKIE_SECURE` | `true` |
| `GOOGLE_CLIENT_ID`, `VITE_GOOGLE_CLIENT_ID` | Same Google OAuth web client ID. Register the public HTTPS origin under Authorized JavaScript origins. If the consent screen is in Testing, add the mentor as a test user. |
| `CLOUDINARY_CLOUD_NAME`, `CLOUDINARY_API_KEY`, `CLOUDINARY_API_SECRET` | Existing configured Cloudinary environment and authenticated PDF access. |
| `OPENROUTER_API_KEY` | Existing server-only key. Keep the same account ledger and quota settings. |
| `TAVILY_API_KEY` and academic-source credentials | Existing values if those optional integrations are used. |
| `VITE_API_BASE_URL` | Empty for same-origin API routing. Set before building. |
| `CITEWISE_DOMAIN` | Hostname in Caddy's service environment. Point DNS to the VM and permit inbound ports 80 and 443. |

## Commands after approval

Prepare the service user and storage on the VM. Restore the existing storage before starting the service if the demo needs existing papers.

```bash
getent passwd citewise || sudo useradd --system --home-dir /var/lib/citewise --shell /usr/sbin/nologin citewise
sudo install -d -m 0750 -o citewise -g citewise /var/lib/citewise
sudo install -d -m 0750 /etc/citewise
sudo touch /etc/citewise/backend.env
sudo chmod 0600 /etc/citewise/backend.env
sudoedit /etc/citewise/backend.env
```

Run `sudo systemctl edit caddy` and add this drop-in with the real hostname.

```ini
[Service]
Environment=CITEWISE_DOMAIN=citewise.example.com
```

```bash
cd /opt/citewise/backend
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

cd /opt/citewise/frontend
npm ci
VITE_API_BASE_URL= VITE_GOOGLE_CLIENT_ID='<public-web-client-id>' npm run build

cd /opt/citewise
sudo install -m 0644 deploy/citewise.service /etc/systemd/system/citewise.service
sudo install -m 0644 deploy/Caddyfile /etc/caddy/Caddyfile
sudo systemctl daemon-reload
sudo systemctl enable citewise
sudo systemctl restart citewise
curl --fail --silent http://127.0.0.1:8000/health
sudo env CITEWISE_DOMAIN='<public-hostname>' caddy validate --config /etc/caddy/Caddyfile
sudo systemctl enable caddy
sudo systemctl restart caddy
curl --fail --silent 'https://<public-hostname>/health'
```

Caddyfile-only edits can be applied with `sudo systemctl reload caddy`. Keep the previous release and Caddyfile for rollback. Check Google sign-in, PDF upload, and passage search on the public origin. The Research composer switches to passage search while an answer cooldown is active. An answer requires a legitimately available local reservation and an eligible upstream provider.
