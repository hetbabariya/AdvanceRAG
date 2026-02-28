from __future__ import annotations

import os

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
from backend.rag.pinecone_hybrid import init_bm25_template

# ---------------------------------------------------------------------------
# Bootstrap logging first — before any other module logs
# ---------------------------------------------------------------------------
configure_logging(level=os.getenv("LOG_LEVEL", "INFO"))
logger = get_logger(__name__)

load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="AllinOneRAG API",
    description="Modern RAG system with session-based authentication and caching",
    version="1.0.0",
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
)


# Include API routes
app.include_router(router)


# ========== Startup & Shutdown Events ==========

@app.on_event("startup")
async def startup_event():
    """Initialize database, cache, and required directories on startup."""
    logger.info("Starting AllinOneRAG API...")

    # Create required directories
    required_dirs = ["backend_storage/bm25", "backend_storage/uploads"]
    for dir_path in required_dirs:
        os.makedirs(dir_path, exist_ok=True)
        logger.info("Ensured directory exists: %s", dir_path)

    # Pre-initialize BM25 template to avoid first-request delay
    init_bm25_template()

    await init_db()
    await cache_manager.connect()
    db_healthy = await check_db_connection()
    cache_healthy = await cache_manager.ping()
    if db_healthy and cache_healthy:
        logger.info("All systems ready!")
    else:
        logger.warning("Some systems failed to initialize (db=%s cache=%s)", db_healthy, cache_healthy)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down AllinOneRAG API...")
    await cache_manager.disconnect()


# ========== Health Check ==========

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    db_status = "ok" if await check_db_connection() else "error"
    cache_status = "ok" if await cache_manager.ping() else "error"
    return HealthResponse(
        status="ok" if db_status == "ok" and cache_status == "ok" else "degraded",
        database=db_status,
        cache=cache_status,
    )
