from __future__ import annotations

from datetime import datetime, timedelta, timezone
import time
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel

from .. import db

router = APIRouter(prefix="/api/auth", tags=["auth"])

AUTH_RATE_LIMIT_WINDOW_SECONDS = 60
AUTH_RATE_LIMIT_MAX_ATTEMPTS = 8
_auth_attempts: dict[str, list[float]] = {}


def _client_key(request: Request, email: str) -> str:
    host = request.client.host if request.client else "unknown"
    return f"{host}:{email.strip().lower()}"


def _check_auth_rate_limit(request: Request, email: str) -> None:
    key = _client_key(request, email)
    now = time.time()
    attempts = [stamp for stamp in _auth_attempts.get(key, []) if now - stamp < AUTH_RATE_LIMIT_WINDOW_SECONDS]
    if len(attempts) >= AUTH_RATE_LIMIT_MAX_ATTEMPTS:
        raise HTTPException(status_code=429, detail="Too many login attempts. Wait a minute and try again.")
    attempts.append(now)
    _auth_attempts[key] = attempts


def _validate_password(password: str) -> None:
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    if password.lower() in {"password", "password123", "12345678"}:
        raise HTTPException(status_code=400, detail="Choose a stronger password")


class AuthRequest(BaseModel):
    email: str
    password: str
    name: str = ""


class AuthUser(BaseModel):
    id: str
    email: str
    name: str


class AuthResponse(BaseModel):
    token: str
    user: AuthUser


def _bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


async def optional_user(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    token = _bearer_token(authorization)
    if not token:
        return {"id": "local", "email": "local@shamsu", "name": "Local demo user"}
    user = await db.get_user_by_token(token)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired login token")
    return user


async def required_user(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    token = _bearer_token(authorization)
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Sign in required")
    user = await db.get_user_by_token(token)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired login token")
    return user


@router.post("/register", response_model=AuthResponse)
async def register(body: AuthRequest, request: Request) -> AuthResponse:
    email = body.email.strip().lower()
    if "@" not in email or "." not in email.rsplit("@", 1)[-1]:
        raise HTTPException(status_code=400, detail="Enter a valid email address")
    _check_auth_rate_limit(request, email)
    _validate_password(body.password)
    try:
        user = await db.create_user(email=email, password=body.password, name=body.name.strip())
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    token = await db.create_auth_token(user["id"], expires_at=datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(days=7))
    return AuthResponse(token=token, user=AuthUser(**user))


@router.post("/login", response_model=AuthResponse)
async def login(body: AuthRequest, request: Request) -> AuthResponse:
    email = body.email.strip().lower()
    _check_auth_rate_limit(request, email)
    user = await db.authenticate_user(email, body.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = await db.create_auth_token(user["id"], expires_at=datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(days=7))
    return AuthResponse(token=token, user=AuthUser(**user))


@router.get("/me", response_model=AuthUser)
async def me(user: dict[str, Any] = Depends(required_user)) -> AuthUser:
    return AuthUser(**user)


@router.post("/logout")
async def logout(authorization: str | None = Header(default=None)) -> dict[str, bool]:
    token = _bearer_token(authorization)
    if token:
        await db.revoke_auth_token(token)
    return {"ok": True}

