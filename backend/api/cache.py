from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Optional

import redis.asyncio as redis
from dotenv import load_dotenv

load_dotenv()


try:
    from langsmith import traceable  # type: ignore
except Exception:  # pragma: no cover
    def traceable(*_args, **_kwargs):  # type: ignore
        def _decorator(fn):
            return fn

        return _decorator

# Redis configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
EMBEDDING_CACHE_TTL = int(os.getenv("EMBEDDING_CACHE_TTL", 2592000))  # 30 days
QUERY_CACHE_TTL = int(os.getenv("QUERY_CACHE_TTL", 3600))  # 1 hour
SESSION_CACHE_TTL = int(os.getenv("SESSION_CACHE_TTL", 604800))  # 7 days


class CacheManager:
    """Redis cache manager for embeddings, queries, and sessions"""

    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None

    async def connect(self):
        """Connect to Redis"""
        if not self.redis_client:
            self.redis_client = await redis.from_url(
                REDIS_URL,
                encoding="utf-8",
                decode_responses=False,  # We'll handle encoding ourselves
                max_connections=50,
            )

    connect = traceable(name="cache.connect")(connect)

    async def disconnect(self):
        """Disconnect from Redis"""
        if self.redis_client:
            await self.redis_client.close()

    disconnect = traceable(name="cache.disconnect")(disconnect)

    async def ping(self) -> bool:
        """Check if Redis is connected"""
        try:
            if not self.redis_client:
                await self.connect()
            return await self.redis_client.ping()
        except Exception as e:
            print(f"Redis ping failed: {e}")
            return False

    ping = traceable(name="cache.ping")(ping)

    # ========== File Hash Cache ==========

    def _file_hash_key(self, file_hash: str) -> str:
        """Generate cache key for file hash"""
        return f"file_hash:{file_hash}"

    async def get_file_by_hash(self, file_hash: str) -> Optional[dict]:
        """Get file metadata by hash"""
        if not self.redis_client:
            await self.connect()

        key = self._file_hash_key(file_hash)
        data = await self.redis_client.get(key)
        if data:
            return json.loads(data)
        return None

    get_file_by_hash = traceable(name="cache.get_file_by_hash")(get_file_by_hash)

    async def set_file_hash(self, file_hash: str, metadata: dict, ttl: int = EMBEDDING_CACHE_TTL):
        """Cache file metadata by hash"""
        if not self.redis_client:
            await self.connect()

        key = self._file_hash_key(file_hash)
        await self.redis_client.setex(key, ttl, json.dumps(metadata))

    set_file_hash = traceable(name="cache.set_file_hash")(set_file_hash)

    # ========== Query Result Cache ==========

    def _query_cache_key(self, query: str, file_name: str) -> str:
        """Generate cache key for query results.

        The key format is ``query:{file_hash}:{query_hash}`` so that all
        cached queries for a given file share a scannable prefix.
        """
        file_name_hash = hashlib.sha256(file_name.encode()).hexdigest()[:16]
        query_hash = hashlib.sha256(query.encode()).hexdigest()
        return f"query:{file_name_hash}:{query_hash}"

    async def get_query_result(self, query: str, file_name: str) -> Optional[dict]:
        """Get cached query result"""
        if not self.redis_client:
            await self.connect()

        key = self._query_cache_key(query, file_name)
        data = await self.redis_client.get(key)
        if data:
            return json.loads(data)
        return None

    get_query_result = traceable(name="cache.get_query_result")(get_query_result)

    async def set_query_result(self, query: str, file_name: str, result: dict, ttl: int = QUERY_CACHE_TTL):
        """Cache query result"""
        if not self.redis_client:
            await self.connect()

        key = self._query_cache_key(query, file_name)
        await self.redis_client.setex(key, ttl, json.dumps(result))

    set_query_result = traceable(name="cache.set_query_result")(set_query_result)

    # ========== Session Cache ==========

    def _session_key(self, session_token: str) -> str:
        """Generate cache key for session"""
        return f"session:{session_token}"

    async def get_session(self, session_token: str) -> Optional[dict]:
        """Get cached session data"""
        if not self.redis_client:
            await self.connect()

        key = self._session_key(session_token)
        data = await self.redis_client.get(key)
        if data:
            return json.loads(data)
        return None

    async def set_session(self, session_token: str, session_data: dict, ttl: int = SESSION_CACHE_TTL):
        """Cache session data"""
        if not self.redis_client:
            await self.connect()

        key = self._session_key(session_token)
        await self.redis_client.setex(key, ttl, json.dumps(session_data))

    set_session = traceable(name="cache.set_session")(set_session)

    async def delete_session(self, session_token: str):
        """Delete session from cache"""
        if not self.redis_client:
            await self.connect()

        key = self._session_key(session_token)
        await self.redis_client.delete(key)

    delete_session = traceable(name="cache.delete_session")(delete_session)

    # ========== Cache Invalidation ==========

    async def invalidate_file_cache(self, file_hash: str, file_name: str):
        """Invalidate all cache entries for a specific file.

        Deletes:
        * The file-hash metadata entry.
        * Only the query caches whose keys match this file’s prefix — other
          users’ cached queries are NOT affected.
        """
        if not self.redis_client:
            await self.connect()

        # Delete file hash cache
        await self.redis_client.delete(self._file_hash_key(file_hash))

        # Delete query caches scoped to this file only
        file_name_hash = hashlib.sha256(file_name.encode()).hexdigest()[:16]
        pattern = f"query:{file_name_hash}:*"
        async for key in self.redis_client.scan_iter(match=pattern):
            await self.redis_client.delete(key)

    invalidate_file_cache = traceable(name="cache.invalidate_file_cache")(invalidate_file_cache)

    async def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        if not self.redis_client:
            await self.connect()

        info = await self.redis_client.info("stats")
        return {
            "keyspace_hits": info.get("keyspace_hits", 0),
            "keyspace_misses": info.get("keyspace_misses", 0),
            "total_keys": await self.redis_client.dbsize(),
        }

    get_cache_stats = traceable(name="cache.get_cache_stats")(get_cache_stats)


# Global cache manager instance
cache_manager = CacheManager()
