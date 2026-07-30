from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel

from .. import db

router = APIRouter(prefix="/api/auth", tags=["auth"])


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
async def register(body: AuthRequest) -> AuthResponse:
    email = body.email.strip().lower()
    if "@" not in email or "." not in email.rsplit("@", 1)[-1]:
        raise HTTPException(status_code=400, detail="Enter a valid email address")
    if len(body.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    try:
        user = await db.create_user(email=email, password=body.password, name=body.name.strip())
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    token = await db.create_auth_token(user["id"], expires_at=datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(days=7))
    return AuthResponse(token=token, user=AuthUser(**user))


@router.post("/login", response_model=AuthResponse)
async def login(body: AuthRequest) -> AuthResponse:
    user = await db.authenticate_user(body.email.strip().lower(), body.password)
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



