# CiteWise frontend

This is the first frontend milestone for CiteWise. It currently includes:

- Public landing page at `/`
- Sign-up page at `/signup`
- Login page at `/login`
- Signed-in home dashboard at `/home`
- Administrator dashboard at `/admin`
- Signed-in paper library and PDF upload at `/papers`
- Separate Research (`/research`), Compare (`/compare`), and Evaluation (`/evaluation`) information pages; the underlying workflows are still planned
- Responsive desktop and mobile layouts
- Email and password account creation and login
- Google sign-in for account creation and login
- Session-aware login results and backend-enforced administrator access

Passwords are hashed by the API with Argon2id before storage. Google ID tokens are verified on the server. Both methods create a CiteWise session in an HttpOnly cookie. The paper library uploads one PDF at a time, shows processing results, and previews page-aware chunks after embeddings are saved. Research questions and citation answers are still future work.

The landing page remains public for first-time visitors. After sign-up or login, the browser opens `/home`. The dashboard uses the signed-in user's real paper counts, recent papers, and processing states. It links to `/papers` for upload and text previews. Conversations, citation precision, research, comparison, and evaluation remain future work and are not shown as fabricated data.

The workspace sidebar opens a distinct route for each tab. Dashboard and Library show implemented work. Research, Compare, and Evaluation explain their planned scope and link back to the available library and dashboard.

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
