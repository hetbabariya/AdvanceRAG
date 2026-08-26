from __future__ import annotations

import os
import contextlib
import asyncio

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from backend.api.api import limiter, router
from backend.api.cache import cache_manager
from backend.api.database import check_db_connection, init_db
from backend.api.schemas import HealthResponse
from backend.api.utils import configure_logging, get_logger
from backend.rag.pinecone_hybrid import build_hybrid_index, init_bm25_template
from backend.rag.service import RagService
from backend.rag.settings import load_settings

# ---------------------------------------------------------------------------
# Bootstrap logging first — before any other module logs
# ---------------------------------------------------------------------------
configure_logging(level=os.getenv("LOG_LEVEL", "INFO"))
logger = get_logger(__name__)

load_dotenv()

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AskMyDocs API...")

    # Create required directories
    required_dirs = ["backend_storage/bm25", "backend_storage/uploads"]
    for dir_path in required_dirs:
        os.makedirs(dir_path, exist_ok=True)
        logger.info("Ensured directory exists: %s", dir_path)

    # Skip blocking BM25 pre-initialization at startup to prevent hangs.
    # It will initialize lazily on the first retrieval/ingest request.
    logger.info("Skipping blocking BM25 pre-initialization to avoid startup hang.")

    # Initialize RAG service once per process and store in app state
    try:
        settings = load_settings()
        hybrid = build_hybrid_index(
            pinecone_api_key=settings.pinecone_api_key,
            index_name=settings.pinecone_index_name,
            embedding_model=settings.embedding_model,
            cloud=settings.pinecone_cloud,
            region=settings.pinecone_region,
        )
        app.state.rag_service = RagService.create(settings, hybrid)
        logger.info("RAG service initialized")
    except Exception:
        logger.exception("Failed to initialize RAG service")
        app.state.rag_service = None

    await init_db()
    await cache_manager.connect()
    db_healthy = await check_db_connection()
    cache_healthy = await cache_manager.ping()
    if db_healthy and cache_healthy:
        logger.info("All systems ready!")
    else:
        logger.warning("Some systems failed to initialize (db=%s cache=%s)", db_healthy, cache_healthy)

    try:
        yield
    finally:
        logger.info("Shutting down AskMyDocs API...")
        await cache_manager.disconnect()


# Create FastAPI app
app = FastAPI(
    title="AskMyDocs API",
    description="Modern RAG system with session-based authentication and caching",
    version="1.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware
def _parse_cors_origins(value: str) -> list[str]:
    origins = [o.strip() for o in (value or "").split(",")]
    return [o for o in origins if o]

cors_allow_origins = os.getenv("CORS_ALLOW_ORIGINS")
if cors_allow_origins:
    origins = _parse_cors_origins(cors_allow_origins)
else:
    origins = [os.getenv("CORS_ALLOW_ORIGIN", "http://localhost:5173")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["set-cookie"],
)


# Include API routes
app.include_router(router)


# ========== Health Check ==========

@app.get("/health")
async def health():
    """Lightweight health check — no external dependencies.

    Render uses this to detect whether the port is open and the process
    is accepting HTTP requests.  It must return instantly, even if the
    database or cache is still starting up.
    """
    return {"status": "ok"}


@app.get("/health/detailed", response_model=HealthResponse)
async def health_detailed() -> HealthResponse:
    """Detailed health check — touches database and cache.

    Use this for operational monitoring, not Render's startup probe.
    """
    db_status = "ok" if await check_db_connection() else "error"
    cache_status = "ok" if await cache_manager.ping() else "error"
    return HealthResponse(
        status="ok" if db_status == "ok" and cache_status == "ok" else "degraded",
        database=db_status,
        cache=cache_status,
    )
