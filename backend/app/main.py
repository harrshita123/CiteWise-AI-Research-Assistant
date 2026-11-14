from contextlib import asynccontextmanager
import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pymongo import AsyncMongoClient

from .config import get_settings
from .repositories.users import UserRepository
from .routers.admin import router as admin_router
from .routers.auth import router as auth_router
from .repositories.papers import PaperRepository
from .repositories.conversations import ConversationRepository
from .repositories.evaluations import EvaluationRepository
from .routers.papers import router as papers_router
from .routers.research import router as research_router
from .routers.evaluation import router as evaluation_router
from .services.ingestion import IngestionService
from .providers.groq_answers import GroqAnswerProvider
from .providers.external_search import TavilyExternalSearch

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = AsyncMongoClient(
        settings.mongodb_uri,
    )
    await client.admin.command({"ping": 1})

    user_repository = UserRepository(client[settings.mongodb_database]["users"])
    await user_repository.ensure_indexes()
    app.state.user_repository = user_repository
    paper_repository = PaperRepository(client[settings.mongodb_database]["papers"])
    await paper_repository.ensure_indexes()
    app.state.paper_repository = paper_repository
    conversation_repository = ConversationRepository(client[settings.mongodb_database]["conversations"])
    await conversation_repository.ensure_indexes()
    app.state.conversation_repository = conversation_repository
    evaluation_repository = EvaluationRepository(client[settings.mongodb_database]["evaluation_runs"])
    await evaluation_repository.ensure_indexes()
    app.state.evaluation_repository = evaluation_repository
    app.state.ingestion_service = IngestionService(settings)
    key = settings.groq_api_key.get_secret_value() if settings.groq_api_key else ""
    app.state.answer_provider = GroqAnswerProvider(key, settings.groq_model, settings.answer_timeout_seconds)
    app.state.external_search = TavilyExternalSearch(
        settings.tavily_api_key.get_secret_value() if settings.tavily_api_key else "")
    app.state.upload_lock = asyncio.Lock()

    try:
        yield
    finally:
        await client.close()


app = FastAPI(title="CiteWise API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)

app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(papers_router)
app.include_router(research_router)
app.include_router(evaluation_router)


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}
