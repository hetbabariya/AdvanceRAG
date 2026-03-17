from __future__ import annotations

import os
import secrets
from datetime import datetime, timedelta
from typing import Optional

import bcrypt
from dotenv import load_dotenv
from fastapi import Cookie, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.cache import cache_manager
from backend.api.database import get_db
from backend.api.models import Session, User
from backend.api.utils import get_logger

load_dotenv()

logger = get_logger(__name__)

# Configuration
SESSION_EXPIRY_DAYS = int(os.getenv("SESSION_EXPIRY_DAYS", 7))
_secret = os.getenv("SECRET_KEY")
if not _secret:
    raise ValueError(
        "SECRET_KEY environment variable is not set. "
        "Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(32))\""
    )
SECRET_KEY: str = _secret


def hash_password(password: str) -> str:
    """Hash a password using bcrypt"""
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(password.encode(), salt).decode()


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its hash"""
    return bcrypt.checkpw(password.encode(), password_hash.encode())


def generate_session_token() -> str:
    """Generate a secure session token"""
    return secrets.token_urlsafe(32)


async def create_user(db: AsyncSession, username: str, email: str, password: str) -> User:
    """Create a new user"""
    # Check if user already exists
    result = await db.execute(
        select(User).where((User.username == username) | (User.email == email))
    )
    existing_user = result.scalar_one_or_none()

    if existing_user:
        if existing_user.username == username:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already exists"
            )

    # Create new user
    password_hash = hash_password(password)
    user = User(username=username, email=email, password_hash=password_hash)
    db.add(user)
    await db.flush()
    await db.refresh(user)

    return user


async def authenticate_user(db: AsyncSession, username: str, password: str) -> Optional[User]:
    """Authenticate a user by username and password"""
    result = await db.execute(select(User).where(User.username == username))
    user = result.scalar_one_or_none()

    if not user:
        return None

    if not verify_password(password, user.password_hash):
        return None

    return user


async def create_session(db: AsyncSession, user_id: int) -> str:
    """Create a new session for a user"""
    session_token = generate_session_token()
    expires_at = datetime.utcnow() + timedelta(days=SESSION_EXPIRY_DAYS)

    # Create session in database
    session = Session(
        user_id=user_id,
        session_token=session_token,
        expires_at=expires_at
    )
    db.add(session)
    await db.flush()

    # Cache session in Redis
    session_data = {
        "user_id": user_id,
        "expires_at": expires_at.isoformat(),
    }
    await cache_manager.set_session(session_token, session_data)

    return session_token


async def get_user_from_session(
    session_token: str,
    db: AsyncSession
) -> User:
    """Internal function to get user from session token"""
    # Try to get session from cache first
    cached_session = await cache_manager.get_session(session_token)

    if cached_session:
        user_id = cached_session["user_id"]
        expires_at = datetime.fromisoformat(cached_session["expires_at"])

        # Check if session is expired
        if datetime.fromisoformat(cached_session["expires_at"]).replace(tzinfo=None) < datetime.utcnow():
            await cache_manager.delete_session(session_token)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Session expired"
            )

        # Get user from database
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )

        return user

    # If not in cache, check database
    result = await db.execute(
        select(Session).where(Session.session_token == session_token)
    )
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid session"
        )

    # Check if session is expired
    if session.expires_at.replace(tzinfo=None) < datetime.utcnow():
        await db.delete(session)
        await cache_manager.delete_session(session_token)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired"
        )

    # Update last accessed time
    session.last_accessed = datetime.utcnow()

    # Get user
    result = await db.execute(select(User).where(User.id == session.user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )

    # Re-cache session
    session_data = {
        "user_id": user.id,
        "expires_at": session.expires_at.isoformat(),
    }
    await cache_manager.set_session(session_token, session_data)

    return user


async def get_current_user(
    session_token: Optional[str] = Cookie(None, alias="session_token"),
    db: AsyncSession = Depends(get_db)
) -> User:
    """FastAPI dependency to get the current authenticated user"""
    if not session_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )

    return await get_user_from_session(session_token, db)


async def delete_session(db: AsyncSession, session_token: str):
    """Delete a session (logout)"""
    # Delete from database
    result = await db.execute(
        select(Session).where(Session.session_token == session_token)
    )
    session = result.scalar_one_or_none()

    if session:
        await db.delete(session)

    # Delete from cache
    await cache_manager.delete_session(session_token)
