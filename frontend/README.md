# CiteWise frontend

This is the first frontend milestone for CiteWise. It currently includes:

- Public landing page at `/`
- Sign-up page at `/signup`
- Login page at `/login`
- Signed-in home dashboard at `/home`
- Administrator dashboard at `/admin`
- Signed-in paper library and PDF upload at `/papers`
- Research (`/research`) searches one ready paper and displays ranked passages with page numbers. When Groq is configured, it assesses PDF evidence before answering. A sufficient assessment stays PDF-only; a validated partial or no-evidence assessment can search external sources when `TAVILY_API_KEY` is configured on the backend. PDF passages and external links are displayed separately.
- Research shows answers with linked source passages and saves per-answer feedback. All users can revisit them in Answer check (`/evaluation`), linked from the Dashboard and sidebar. Admins can use `/admin/evaluation` for labeled page-retrieval tests and saved Hit@5 reports. Compare (`/compare`) remains an information page.
- Responsive desktop and mobile layouts
- Email and password account creation and login
- Google sign-in for account creation and login
- Session-aware login results and backend-enforced administrator access

Passwords are hashed by the API with Argon2id before storage. Google ID tokens are verified on the server. Both methods create a CiteWise session in an HttpOnly cookie. The paper library uploads one PDF at a time, shows processing results, and previews page-aware chunks after embeddings are saved. Research questions can retrieve passages from one ready paper. The answer flow checks citation IDs against retrieved chunks; real-paper answer quality remains to be evaluated.

The landing page remains public for first-time visitors. After sign-up or login, the browser opens `/home`. The dashboard uses the signed-in user's real paper counts, recent papers, and processing states. It links to `/papers` for upload and text previews. Research saves single-paper conversations. Evaluation measures whether expected PDF pages appear in the top one or five retrieval results; answer correctness and citation quality are not measured yet.

The workspace sidebar opens a distinct route for each tab. Dashboard, Library, Research, and Answer check show implemented work. Compare explains its planned scope.

## Run locally

1. Copy `.env.example` to `.env` and add your Google web client ID.
2. Start the authentication API from `../backend`.
3. Start the frontend:

```bash
npm install
npm run dev
```

## Create a production build

```bash
npm run build
```
