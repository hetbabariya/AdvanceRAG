import asyncio
import re
import json
import os
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, Body, Cookie, Depends, File, HTTPException, Request, Response, UploadFile, status
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy import delete, desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.auth import authenticate_user, create_session, create_user, delete_session, get_current_user
from backend.api.cache import cache_manager
from backend.api.database import get_db
from backend.api.models import ChatHistory, FileMetadata, User
from backend.api.schemas import (
    CacheStatsResponse,
    ChatHistoryItem,
    ChatHistoryResponse,
    ChatRequest,
    ChatResponse,
    FilesResponse,
    IngestResponse,
    URLIngestRequest,
    UserLogin,
    UserRegister,
    UserResponse,
)
from backend.api.utils import (
    configure_logging,
    ensure_directory,
    get_logger,
    is_safe_url,
    parse_citations,
    safe_upload_path
)
from backend.rag.agent_graph import AgenticRagGraph
from backend.rag.pinecone_hybrid import bm25_path
from backend.rag.service import RagService
from backend.api.supabase_storage import delete_remote, download_to_file, upload_bytes, upload_file

logger = get_logger(__name__)

try:
    from langsmith import traceable  # type: ignore
except Exception:  # pragma: no cover
    def traceable(*_args, **_kwargs):  # type: ignore
        def _decorator(fn):
            return fn

        return _decorator

limiter = Limiter(key_func=get_remote_address)

def get_rag_service(request: Request) -> RagService:
    service = getattr(request.app.state, "rag_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="RAG service not initialized")
    return service

def _is_smalltalk_message(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return True
    if len(t) <= 8 and re.fullmatch(r"(hi|hello|hey|yo|sup|hii+|heyy+)", t):
        return True
    if len(t) <= 24 and re.fullmatch(r"(good\s+morning|good\s+afternoon|good\s+evening)", t):
        return True
    if len(t) <= 32 and re.fullmatch(r"(how\s+are\s+you|what'?s\s+up)", t):
        return True
    return False

from fastapi import APIRouter  # noqa: E402

router = APIRouter()


def _int_env(name: str, default: int) -> int:
    raw = (os.getenv(name, "") or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _effective_k(*, requested: Optional[int], default_env: str, default_value: int, max_env: str, max_value: int) -> int:
    default_k = _int_env(default_env, default_value)
    max_k = _int_env(max_env, max_value)
    if max_k < 1:
        max_k = max_value
    k = requested if requested is not None else default_k
    if k < 1:
        k = 1
    if k > max_k:
        k = max_k
    return k


def _dump_model(obj: object) -> dict:
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        return dump()
    dump = getattr(obj, "dict", None)
    if callable(dump):
        return dump()
    raise TypeError(f"Object is not a Pydantic model: {type(obj)}")


# ========== Authentication Endpoints ==========

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit("10/minute")
async def register(
    request: Request,
    user_data: UserRegister = Body(...),
    db: AsyncSession = Depends(get_db),
) -> UserResponse:
    user = await create_user(db, user_data.username, user_data.email, user_data.password)
    await db.commit()
    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        created_at=user.created_at,
    )


@router.post("/login")
@limiter.limit("5/minute")
async def login(
    request: Request,
    response: Response,
    credentials: UserLogin = Body(...),
    db: AsyncSession = Depends(get_db),
):
    user = await authenticate_user(db, credentials.username, credentials.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    session_token = await create_session(db, user.id)
    await db.commit()

    is_production = os.getenv("ENVIRONMENT", "development").lower() == "production"
    cookie_samesite = (os.getenv("COOKIE_SAMESITE") or ("none" if is_production else "lax")).lower()
    cookie_secure = (os.getenv("COOKIE_SECURE") or ("1" if (is_production and cookie_samesite == "none") else "0"))
    use_secure_cookie = cookie_secure in {"1", "true", "yes", "on"}
    response.set_cookie(
        key="session_token",
        value=session_token,
        httponly=True,
        secure=use_secure_cookie,
        samesite=cookie_samesite,
        max_age=7 * 24 * 60 * 60,
    )

    logger.info("User '%s' logged in successfully", user.username)
    return {
        "message": "Login successful",
        "user": UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            created_at=user.created_at,
        ),
    }


@router.post("/logout")
async def logout(
    response: Response,
    session_token: Optional[str] = Cookie(None, alias="session_token"),
    db: AsyncSession = Depends(get_db),
):
    if session_token:
        await delete_session(db, session_token)
        await db.commit()
    response.delete_cookie(key="session_token")
    return {"message": "Logout successful"}


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)) -> UserResponse:
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        created_at=current_user.created_at,
    )


# ========== File Management Endpoints ==========

@router.get("/files", response_model=FilesResponse)
async def get_files(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> FilesResponse:
    result = await db.execute(
        select(FileMetadata)
        .where(FileMetadata.user_id == current_user.id)
        .order_by(desc(FileMetadata.ingested_at))
    )
    files = result.scalars().all()
    return FilesResponse(
        files=[
            {
                "file_name": f.file_name,
                "chunks_upserted": f.chunks_count,
                "ingested_at": f.ingested_at.isoformat(),
                "file_size": f.file_size,
                "file_hash": f.file_hash,
            }
            for f in files
        ]
    )


async def _process_file_ingestion(
    saved_path: str,
    original_name: str,
    file_hash: str,
    file_size: int,
    service: RagService,
) -> tuple[int, bool]:
    cached_file = await cache_manager.get_file_by_hash(file_hash)
    if cached_file:
        cached_bm25_file = cached_file.get("bm25_file")
        bm25_file = cached_bm25_file or bm25_path(service.settings.bm25_store_dir, original_name)
        if not os.path.exists(bm25_file):
            try:
                remote_bm25 = f"bm25/{file_hash}.pkl"
                download_to_file(remote_path=remote_bm25, local_path=bm25_file)
            except Exception:
                pass
        if os.path.exists(bm25_file):
            return cached_file.get("chunks_count", 0), True
        logger.info(
            "Cache hit for file_hash but BM25 model missing on disk; re-ingesting (file=%s)",
            original_name,
        )

    count, _bm25_path = await asyncio.to_thread(
        service.ingest_file, saved_path=saved_path, original_name=original_name
    )

    await cache_manager.set_file_hash(
        file_hash,
        {
            "file_name": original_name,
            "chunks_count": count,
            "file_size": file_size,
            "bm25_file": _bm25_path,
        },
    )
    return count, False


@router.post("/ingest", response_model=IngestResponse)
@traceable(name="pipeline.ingest")
async def ingest(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    service: RagService = Depends(get_rag_service),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".docx", ".txt", ".md"}:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    ensure_directory(service.settings.upload_dir)

    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read uploaded file: {e}")

    import hashlib

    file_hash = hashlib.sha256(content).hexdigest()
    file_size = len(content)

    result = await db.execute(
        select(FileMetadata).where(
            FileMetadata.user_id == current_user.id,
            FileMetadata.file_hash == file_hash,
        )
    )
    existing_file = result.scalar_one_or_none()
    if existing_file:
        return IngestResponse(
            file_name=file.filename,
            chunks_upserted=existing_file.chunks_count,
            file_hash=file_hash,
            cached=True,
        )

    saved_path = safe_upload_path(service.settings.upload_dir, current_user.id, file_hash, suffix)
    try:
        with open(saved_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    try:
        upload_file(
            local_path=saved_path,
            remote_path=f"uploads/{current_user.id}/{file_hash}{suffix}",
            content_type=file.content_type or "application/octet-stream",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file to S3 Storage: {e}")

    count, cached = await _process_file_ingestion(saved_path, file.filename, file_hash, file_size, service)

    try:
        bm25_local_path = bm25_path(service.settings.bm25_store_dir, file.filename)
        upload_file(
            local_path=bm25_local_path,
            remote_path=f"bm25/{file_hash}.pkl",
            content_type="application/octet-stream",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload BM25 artifact to S3 Storage: {e}")

    file_metadata = FileMetadata(
        user_id=current_user.id,
        file_name=file.filename,
        file_hash=file_hash,
        file_size=file_size,
        chunks_count=count,
    )
    db.add(file_metadata)
    await db.commit()

    logger.info(
        "Ingested '%s' for user %d: %d chunks (cached=%s)",
        file.filename,
        current_user.id,
        count,
        cached,
    )
    return IngestResponse(
        file_name=file.filename,
        chunks_upserted=count,
        file_hash=file_hash,
        cached=cached,
    )


@router.post("/ingest-youtube", response_model=IngestResponse)
@traceable(name="pipeline.ingest_youtube")
async def ingest_youtube(
    req: URLIngestRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    service: RagService = Depends(get_rag_service),
):
    import hashlib
    from urllib.parse import urlparse

    url = str(req.url)

    safe, reason = is_safe_url(url)
    if not safe:
        raise HTTPException(status_code=400, detail=f"URL rejected: {reason}")

    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if host not in {"youtube.com", "m.youtube.com", "youtu.be"}:
        raise HTTPException(status_code=400, detail="Only YouTube URLs are supported")

    url_hash = hashlib.md5(url.encode()).hexdigest()

    result = await db.execute(
        select(FileMetadata).where(
            FileMetadata.user_id == current_user.id,
            FileMetadata.file_hash == url_hash,
        )
    )
    existing = result.scalar_one_or_none()
    if existing:
        return IngestResponse(
            file_name=existing.file_name,
            chunks_upserted=existing.chunks_count,
            file_hash=url_hash,
            cached=True,
        )

    try:
        from backend.rag.pinecone_hybrid import upsert_documents

        display_name, docs = service.loader.load_and_split_youtube(url, languages=["en", "hi", "en-US", "hi-IN"])
        count, _ = upsert_documents(
            hybrid=service.hybrid,
            docs=docs,
            file_name=display_name,
            bm25_store_dir=service.settings.bm25_store_dir,
        )
    except Exception as e:
        logger.exception("YouTube ingestion failed for %s", url)
        raise HTTPException(status_code=500, detail=f"Failed to ingest YouTube video: {e}")

    file_metadata = FileMetadata(
        user_id=current_user.id,
        file_name=display_name,
        file_hash=url_hash,
        file_size=0,
        chunks_count=count,
    )
    db.add(file_metadata)
    await db.commit()

    return IngestResponse(
        file_name=display_name,
        chunks_upserted=count,
        file_hash=url_hash,
        cached=False,
    )


@router.delete("/files/{file_name}")
async def delete_file(
    file_name: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(FileMetadata).where(
            FileMetadata.user_id == current_user.id,
            FileMetadata.file_name == file_name,
        )
    )
    file_metadata = result.scalar_one_or_none()

    if not file_metadata:
        raise HTTPException(status_code=404, detail="File not found")

    # Best-effort cleanup of remote artifacts (S3)
    try:
        suffix = Path(file_metadata.file_name).suffix.lower()
        delete_remote(remote_path=f"uploads/{current_user.id}/{file_metadata.file_hash}{suffix}")
        delete_remote(remote_path=f"bm25/{file_metadata.file_hash}.pkl")
    except Exception:
        pass

    await db.execute(
        delete(ChatHistory).where(
            ChatHistory.user_id == current_user.id,
            ChatHistory.file_name == file_name,
        )
    )
    await db.delete(file_metadata)
    await cache_manager.invalidate_file_cache(file_metadata.file_hash, file_metadata.file_name)
    await db.commit()

    logger.info("Deleted file '%s' for user %d", file_name, current_user.id)
    return {"message": f"File '{file_name}' deleted successfully"}


# ========== Chat Endpoints ==========


async def _resolve_retrieval(
    req: ChatRequest,
    current_user: User,
    db: AsyncSession,
    service: RagService,
    retrieval_top_k: int,
) -> tuple[list, str]:
    if not req.file_name:
        raise HTTPException(
            status_code=422, detail="file_name is required"
        )

    result = await db.execute(
        select(FileMetadata).where(
            FileMetadata.user_id == current_user.id,
            FileMetadata.file_name == req.file_name,
        )
    )
    file_meta = result.scalar_one_or_none()
    if not file_meta:
        raise HTTPException(status_code=404, detail="File not found or access denied")

    try:
        local_bm25 = bm25_path(service.settings.bm25_store_dir, req.file_name)
        if not os.path.exists(local_bm25):
            download_to_file(remote_path=f"bm25/{file_meta.file_hash}.pkl", local_path=local_bm25)
    except Exception:
        pass

    rewritten_query = req.message
    try:
        rewritten_query = service.rewrite_query(question=req.message, file_name=req.file_name)
    except Exception:
        rewritten_query = req.message

    try:
        matches = await asyncio.to_thread(
            service.retrieve,
            query=rewritten_query,
            file_name=req.file_name,
            top_k=retrieval_top_k,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Local retrieval failed")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")

    return matches, rewritten_query


@router.post("/chat", response_model=ChatResponse)
@traceable(name="pipeline.chat")
async def chat(
    req: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    service: RagService = Depends(get_rag_service),
) -> ChatResponse:
    retrieval_top_k = _effective_k(
        requested=None,
        default_env="RETRIEVAL_TOP_K_DEFAULT",
        default_value=10,
        max_env="RETRIEVAL_TOP_K_MAX",
        max_value=50,
    )
    rerank_top_k = _effective_k(
        requested=None,
        default_env="RERANK_TOP_K_DEFAULT",
        default_value=retrieval_top_k,
        max_env="RERANK_TOP_K_MAX",
        max_value=retrieval_top_k,
    )
    logger.debug(
        "Effective k values (env-only): retrieval_top_k=%s rerank_top_k=%s file=%s",
        retrieval_top_k,
        rerank_top_k,
        req.file_name,
    )
    cached_result = await cache_manager.get_query_result(req.message, req.file_name)
    if cached_result:
        chat_history = ChatHistory(
            user_id=current_user.id,
            file_name=req.file_name,
            question=req.message,
            answer=cached_result["answer"],
            citations=cached_result["citations"],
        )
        db.add(chat_history)
        await db.commit()
        return ChatResponse(
            answer=cached_result["answer"],
            citations=cached_result["citations"],
            used_context_chunks=cached_result["used_context_chunks"],
            cached=True,
        )

    matches, rewritten_query = await _resolve_retrieval(req, current_user, db, service, retrieval_top_k)
    docs = service._matches_to_documents(matches)

    if req.use_reranker and docs:
        docs = await asyncio.to_thread(
            service.rerank, query=rewritten_query, docs=docs, top_k=min(len(docs), rerank_top_k)
        )

    try:
        raw_answer = await asyncio.to_thread(service.generate, question=req.message, context_docs=docs)
    except Exception as e:
        logger.exception("Generation failed")
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    answer_text, citations = parse_citations(raw_answer, docs, fetch_missing_fn=service.fetch_chunks_by_ids)

    result_data = {
        "answer": answer_text,
        "citations": [_dump_model(c) for c in citations],
        "used_context_chunks": len(docs),
    }
    await cache_manager.set_query_result(req.message, req.file_name, result_data)

    chat_history = ChatHistory(
        user_id=current_user.id,
        file_name=req.file_name,
        question=req.message,
        answer=answer_text,
        citations=[_dump_model(c) for c in citations],
    )
    db.add(chat_history)
    await db.commit()

    return ChatResponse(
        answer=answer_text,
        citations=citations,
        used_context_chunks=len(docs),
        cached=False,
    )


@router.post("/chat/agent", response_model=ChatResponse)
@traceable(name="pipeline.chat_agent")
async def chat_agent(
    req: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    service: RagService = Depends(get_rag_service),
) -> ChatResponse:
    if _is_smalltalk_message(req.message):
        prompt = (
            "You are a helpful assistant. The user is greeting you. "
            "Reply briefly and friendly. Ask one short question to understand what they want to do with the selected document.\n\n"
            f"User: {req.message}\n"
            "Assistant:"
        )
        try:
            resp = await asyncio.to_thread(service.llm.invoke, prompt)
            answer_text = (getattr(resp, "content", "") or str(resp)).strip()
        except Exception as e:
            logger.exception("Smalltalk generation failed")
            raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

        result_data = {
            "answer": answer_text,
            "citations": [],
            "used_context_chunks": 0,
        }
        cache_message = f"[agent]{req.message}"
        await cache_manager.set_query_result(cache_message, req.file_name, result_data)

        chat_history = ChatHistory(
            user_id=current_user.id,
            file_name=req.file_name,
            question=req.message,
            answer=answer_text,
            citations=[],
        )
        db.add(chat_history)
        await db.commit()

        return ChatResponse(
            answer=answer_text,
            citations=[],
            used_context_chunks=0,
            cached=False,
        )

    retrieval_top_k = _effective_k(
        requested=None,
        default_env="RETRIEVAL_TOP_K_DEFAULT",
        default_value=10,
        max_env="RETRIEVAL_TOP_K_MAX",
        max_value=50,
    )
    rerank_top_k = _effective_k(
        requested=None,
        default_env="RERANK_TOP_K_DEFAULT",
        default_value=retrieval_top_k,
        max_env="RERANK_TOP_K_MAX",
        max_value=retrieval_top_k,
    )

    cache_message = f"[agent]{req.message}"
    cached_result = await cache_manager.get_query_result(cache_message, req.file_name)
    if cached_result:
        chat_history = ChatHistory(
            user_id=current_user.id,
            file_name=req.file_name,
            question=req.message,
            answer=cached_result["answer"],
            citations=cached_result["citations"],
        )
        db.add(chat_history)
        await db.commit()
        return ChatResponse(
            answer=cached_result["answer"],
            citations=cached_result["citations"],
            used_context_chunks=cached_result["used_context_chunks"],
            cached=True,
        )

    try:
        matches, _rewritten_for_bm25 = await _resolve_retrieval(req, current_user, db, service, retrieval_top_k)
        _ = matches
    except Exception:
        # _resolve_retrieval already raises proper HTTPException
        raise

    agent = AgenticRagGraph(
        service=service,
        retrieval_top_k=retrieval_top_k,
        rerank_top_k=rerank_top_k,
    )
    agent_state = await asyncio.to_thread(agent.run, question=req.message, file_name=req.file_name)
    docs = list(agent_state.get("docs", []) or [])
    try:
        raw_answer = await asyncio.to_thread(service.generate, question=req.message, context_docs=docs)
    except Exception as e:
        logger.exception("Generation failed")
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    answer_text, citations = parse_citations(raw_answer, docs, fetch_missing_fn=service.fetch_chunks_by_ids)

    result_data = {
        "answer": answer_text,
        "citations": [_dump_model(c) for c in citations],
        "used_context_chunks": len(docs),
    }
    await cache_manager.set_query_result(cache_message, req.file_name, result_data)

    chat_history = ChatHistory(
        user_id=current_user.id,
        file_name=req.file_name,
        question=req.message,
        answer=answer_text,
        citations=[_dump_model(c) for c in citations],
    )
    db.add(chat_history)
    await db.commit()

    return ChatResponse(
        answer=answer_text,
        citations=citations,
        used_context_chunks=len(docs),
        cached=False,
    )


@router.get("/chat/agent/stream")
async def chat_agent_stream(
    message: str,
    file_name: str,
    use_reranker: bool = True,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    service: RagService = Depends(get_rag_service),
):
    from fastapi.responses import StreamingResponse as FastAPIStreamingResponse

    async def _pipeline_event_stream():
        agent_task: Optional[asyncio.Task] = None
        try:
            if _is_smalltalk_message(message):
                prompt = (
                    "You are a helpful assistant. The user is greeting you. "
                    "Reply briefly and friendly. Ask one short question to understand what they want to do with the selected document.\n\n"
                    f"User: {message}\n"
                    "Assistant:"
                )

                full_text = ""
                try:
                    async for chunk in service.llm.astream(prompt):
                        token = getattr(chunk, "content", "") or str(chunk)
                        if token:
                            full_text += token
                            yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"
                except Exception as e:
                    logger.exception("Smalltalk streaming failed")
                    yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
                    return

                answer_text = full_text.strip()
                result_data = {
                    "answer": answer_text,
                    "citations": [],
                    "used_context_chunks": 0,
                }
                cache_message = f"[agent]{message}"
                await cache_manager.set_query_result(cache_message, file_name, result_data)

                chat_history = ChatHistory(
                    user_id=current_user.id,
                    file_name=file_name,
                    question=message,
                    answer=answer_text,
                    citations=[],
                )
                db.add(chat_history)
                await db.commit()

                yield (
                    "data: "
                    + json.dumps(
                        {
                            "type": "done",
                            "answer": answer_text,
                            "citations": [],
                            "used_context_chunks": 0,
                            "cached": False,
                        }
                    )
                    + "\n\n"
                )
                return

            agent_event_queue: asyncio.Queue = asyncio.Queue()
            req = ChatRequest(
                message=message,
                file_name=file_name,
                use_reranker=use_reranker,
            )

            retrieval_top_k = _effective_k(
                requested=None,
                default_env="RETRIEVAL_TOP_K_DEFAULT",
                default_value=10,
                max_env="RETRIEVAL_TOP_K_MAX",
                max_value=50,
            )
            rerank_top_k = _effective_k(
                requested=None,
                default_env="RERANK_TOP_K_DEFAULT",
                default_value=retrieval_top_k,
                max_env="RERANK_TOP_K_MAX",
                max_value=retrieval_top_k,
            )

            cache_message = f"[agent]{message}"
            cached_result = await cache_manager.get_query_result(cache_message, file_name)
            if cached_result:
                answer = cached_result["answer"]
                chunk_size = 8
                for i in range(0, len(answer), chunk_size):
                    token = answer[i : i + chunk_size]
                    yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"
                    await asyncio.sleep(0.01)
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "type": "done",
                            "answer": cached_result["answer"],
                            "citations": cached_result["citations"],
                            "used_context_chunks": cached_result["used_context_chunks"],
                            "cached": True,
                        }
                    )
                    + "\n\n"
                )
                return

            await _resolve_retrieval(req, current_user, db, service, retrieval_top_k)

            loop = asyncio.get_running_loop()

            def _on_agent_event(payload: dict) -> None:
                loop.call_soon_threadsafe(agent_event_queue.put_nowait, payload)

            agent = AgenticRagGraph(
                service=service,
                retrieval_top_k=retrieval_top_k,
                rerank_top_k=rerank_top_k,
                on_event=_on_agent_event,
            )

            agent_task = asyncio.create_task(
                asyncio.to_thread(agent.run, question=message, file_name=file_name)
            )

            while True:
                if agent_task.done() and agent_event_queue.empty():
                    break
                try:
                    evt = await asyncio.wait_for(agent_event_queue.get(), timeout=0.25)
                    yield f"data: {json.dumps({'type': 'agent', **evt})}\n\n"
                except asyncio.TimeoutError:
                    continue

            try:
                agent_state = await agent_task
            except Exception as e:
                msg = str(e)
                if "BM25 model not found" in msg and "Ingest the file first" in msg:
                    msg = "This file is not ingested yet (BM25 index missing). Please upload/ingest the file first, then try again."
                logger.exception("Agent run failed")
                yield f"data: {json.dumps({'type': 'error', 'message': msg})}\n\n"
                return

            docs = list(agent_state.get("docs", []) or [])
            if not docs:
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "type": "error",
                            "message": "Agent could not retrieve any context chunks, so citations cannot be produced. Please verify the correct file is selected and ingested, then try again.",
                        }
                    )
                    + "\n\n"
                )
                return

            full_text = ""
            try:
                async for chunk in service.generate_stream(question=message, context_docs=docs):
                    if isinstance(chunk, dict) and chunk.get("__done__"):
                        full_text = chunk["full_text"]
                        break
                    full_text += chunk
                    yield f"data: {json.dumps({'type': 'token', 'text': chunk})}\n\n"
            except Exception as e:
                logger.exception("Streaming generation failed")
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
                return

            answer_text, citations = parse_citations(
                full_text, docs, fetch_missing_fn=service.fetch_chunks_by_ids
            )
            result_data = {
                "answer": answer_text,
                "citations": [_dump_model(c) for c in citations],
                "used_context_chunks": len(docs),
            }
            await cache_manager.set_query_result(cache_message, file_name, result_data)

            chat_history = ChatHistory(
                user_id=current_user.id,
                file_name=file_name,
                question=message,
                answer=answer_text,
                citations=[_dump_model(c) for c in citations],
            )
            db.add(chat_history)
            await db.commit()

            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "done",
                        "answer": answer_text,
                        "citations": [_dump_model(c) for c in citations],
                        "used_context_chunks": len(docs),
                        "cached": False,
                    }
                )
                + "\n\n"
            )
        finally:
            try:
                if agent_task is not None and not agent_task.done():
                    agent_task.cancel()
            except Exception:
                pass
            try:
                await db.close()
            except Exception:
                pass

    _pipeline_event_stream = traceable(name="pipeline.chat_agent_stream")(_pipeline_event_stream)

    return FastAPIStreamingResponse(
        _pipeline_event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/chat/stream")
async def chat_stream(
    message: str,
    file_name: str,
    top_k: Optional[int] = None,
    use_reranker: bool = True,
    rerank_top_k: Optional[int] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    service: RagService = Depends(get_rag_service),
):
    from fastapi.responses import StreamingResponse as FastAPIStreamingResponse

    async def _pipeline_event_stream():
        req = ChatRequest(
            message=message,
            file_name=file_name,
            top_k=top_k,
            use_reranker=use_reranker,
            rerank_top_k=rerank_top_k,
        )

        retrieval_top_k = _effective_k(
            requested=None,
            default_env="RETRIEVAL_TOP_K_DEFAULT",
            default_value=10,
            max_env="RETRIEVAL_TOP_K_MAX",
            max_value=50,
        )
        stream_rerank_top_k = _effective_k(
            requested=None,
            default_env="RERANK_TOP_K_DEFAULT",
            default_value=retrieval_top_k,
            max_env="RERANK_TOP_K_MAX",
            max_value=retrieval_top_k,
        )
        logger.debug(
            "Effective k values (env-only stream): retrieval_top_k=%s rerank_top_k=%s file=%s",
            retrieval_top_k,
            stream_rerank_top_k,
            file_name,
        )

        cached_result = await cache_manager.get_query_result(message, file_name)
        if cached_result:
            answer = cached_result["answer"]
            chunk_size = 8
            for i in range(0, len(answer), chunk_size):
                token = answer[i : i + chunk_size]
                yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"
                await asyncio.sleep(0.01)
            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "done",
                        "answer": cached_result["answer"],
                        "citations": cached_result["citations"],
                        "used_context_chunks": cached_result["used_context_chunks"],
                        "cached": True,
                    }
                )
                + "\n\n"
            )
            return

        matches, rewritten_query = await _resolve_retrieval(req, current_user, db, service, retrieval_top_k)
        docs = service._matches_to_documents(matches)

        if use_reranker and docs:
            docs = await asyncio.to_thread(
                service.rerank, query=rewritten_query, docs=docs, top_k=min(len(docs), stream_rerank_top_k)
            )

        full_text = ""
        try:
            async for chunk in service.generate_stream(question=message, context_docs=docs):
                if isinstance(chunk, dict) and chunk.get("__done__"):
                    full_text = chunk["full_text"]
                    break
                full_text += chunk
                yield f"data: {json.dumps({'type': 'token', 'text': chunk})}\n\n"
        except Exception as e:
            logger.exception("Streaming generation failed")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            return

        answer_text, citations = parse_citations(full_text, docs, fetch_missing_fn=service.fetch_chunks_by_ids)

        result_data = {
            "answer": answer_text,
            "citations": [_dump_model(c) for c in citations],
            "used_context_chunks": len(docs),
        }
        await cache_manager.set_query_result(message, file_name, result_data)

        chat_history = ChatHistory(
            user_id=current_user.id,
            file_name=file_name,
            question=message,
            answer=answer_text,
            citations=[_dump_model(c) for c in citations],
        )
        db.add(chat_history)
        await db.commit()

        yield (
            "data: "
            + json.dumps(
                {
                    "type": "done",
                    "answer": answer_text,
                    "citations": [_dump_model(c) for c in citations],
                    "used_context_chunks": len(docs),
                    "cached": False,
                }
            )
            + "\n\n"
        )

    _pipeline_event_stream = traceable(name="pipeline.chat_stream")(_pipeline_event_stream)

    return FastAPIStreamingResponse(
        _pipeline_event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/chat/history", response_model=ChatHistoryResponse)
async def get_chat_history(
    file_name: Optional[str] = None,
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ChatHistoryResponse:
    query = select(ChatHistory).where(ChatHistory.user_id == current_user.id)
    if file_name:
        query = query.where(ChatHistory.file_name == file_name)
    query = query.order_by(desc(ChatHistory.created_at)).limit(limit)

    result = await db.execute(query)
    history = result.scalars().all()

    return ChatHistoryResponse(
        history=[
            ChatHistoryItem(
                id=item.id,
                question=item.question,
                answer=item.answer,
                citations=item.citations or [],
                file_name=item.file_name,
                created_at=item.created_at,
            )
            for item in history
        ],
        total=len(history),
    )


@router.get("/cache/stats", response_model=CacheStatsResponse)
async def get_cache_stats(
    current_user: User = Depends(get_current_user),
) -> CacheStatsResponse:
    stats = await cache_manager.get_cache_stats()
    total_requests = stats["keyspace_hits"] + stats["keyspace_misses"]
    hit_rate = stats["keyspace_hits"] / total_requests if total_requests > 0 else 0.0
    return CacheStatsResponse(
        keyspace_hits=stats["keyspace_hits"],
        keyspace_misses=stats["keyspace_misses"],
        total_keys=stats["total_keys"],
        hit_rate=hit_rate,
    )
