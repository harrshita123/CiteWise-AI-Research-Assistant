from contextlib import asynccontextmanager
import asyncio
from contextlib import suppress

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pymongo import AsyncMongoClient, MongoClient

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
from .routers.discovery import router as discovery_router
from .services.paper_discovery import AcademicClient, PaperDiscoveryService
from .services.ingestion import IngestionService
from .services.identity_worker import run_worker
from .services.provider_coordination import ProviderCoordinator
from .providers.openrouter_answers import OpenRouterAnswerProvider
from .providers.openrouter_routing import ModelRouter, SharedRequestBudget
from .providers.external_search import TavilyExternalSearch

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = AsyncMongoClient(
        settings.citewise_demo_mongodb_uri if settings.citewise_demo_mode else settings.mongodb_uri,
    )
    await client.admin.command({"ping": 1})
    database = client[settings.citewise_demo_database if settings.citewise_demo_mode else settings.mongodb_database]

    user_repository = UserRepository(database["users"])
    await user_repository.ensure_indexes()
    app.state.user_repository = user_repository
    paper_repository = PaperRepository(database["papers"])
    await paper_repository.ensure_indexes()
    app.state.paper_repository = paper_repository
    conversation_repository = ConversationRepository(database["conversations"])
    await conversation_repository.ensure_indexes()
    app.state.conversation_repository = conversation_repository
    evaluation_repository = EvaluationRepository(database["evaluation_runs"])
    await evaluation_repository.ensure_indexes()
    app.state.evaluation_repository = evaluation_repository
    app.state.ingestion_service = IngestionService(settings)
    app.state.discovery_records = database["discovery_records"]
    app.state.discovery_audit = database["discovery_audit"]
    app.state.upload_lock = asyncio.Lock()
    app.state.demo_mode = settings.citewise_demo_mode
    if settings.citewise_demo_mode:
        app.state.discovery_service = None
        app.state.answer_provider = None
        app.state.external_search = None
        try:
            yield
        finally:
            await client.close()
        return
    coordination_client = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=5000)
    app.state.discovery_service = PaperDiscoveryService(AcademicClient(
        settings.openalex_api_key.get_secret_value() if settings.openalex_api_key else "",
        settings.crossref_contact_email, ProviderCoordinator(coordination_client[settings.mongodb_database]["academic_provider_state"]),
        ncbi_key=settings.ncbi_api_key.get_secret_value() if settings.ncbi_api_key else ""))
    key = settings.openrouter_api_key.get_secret_value() if settings.openrouter_api_key else ""
    models = ModelRouter(settings.openrouter_generation_model, settings.answer_timeout_seconds,
                         verification=settings.openrouter_verification_model,
                         recommendation=settings.openrouter_recommendation_model,
                         fallback=settings.openrouter_generation_fallback,
                         structured_fallback=settings.openrouter_structured_fallback_model,
                         external_generation=settings.openrouter_external_generation_model,
                         rerank=settings.openrouter_rerank_model, vision=settings.openrouter_vision_model,
                         output_limits=settings.openrouter_output_limits)
    budget = SharedRequestBudget(coordination_client[settings.mongodb_database]["llm_request_budget"],
                                 "openrouter", settings.openrouter_daily_limit, settings.openrouter_minute_limit)
    app.state.answer_provider = OpenRouterAnswerProvider(key, models, budget=budget,
                                                        require_zdr=settings.openrouter_require_zdr,
                                                        rerank_enabled=settings.openrouter_rerank_enabled,
                                                        vision_enabled=settings.openrouter_vision_enabled)
    app.state.external_search = TavilyExternalSearch(
        settings.tavily_api_key.get_secret_value() if settings.tavily_api_key else "")
    identity_worker = asyncio.create_task(run_worker(
        paper_repository, app.state.discovery_service, app.state.ingestion_service.read_original))

    try:
        yield
    finally:
        identity_worker.cancel()
        with suppress(asyncio.CancelledError):
            await identity_worker
        coordination_client.close()
        await client.close()


app = FastAPI(title="CiteWise API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
    expose_headers=["Retry-After"],
)

app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(papers_router)
app.include_router(discovery_router)
app.include_router(research_router)
app.include_router(evaluation_router)


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}
