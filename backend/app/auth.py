from __future__ import annotations

import uuid

from fastapi import Request, Response

from app.config import Settings


def get_anon_owner_token(request: Request, settings: Settings) -> str | None:
    return request.cookies.get(settings.anon_session_cookie_name)


def issue_anon_owner_token() -> str:
    return str(uuid.uuid4())


def set_anon_owner_cookie(response: Response, token: str, settings: Settings) -> None:
    response.set_cookie(
        key=settings.anon_session_cookie_name,
        value=token,
        httponly=True,
        max_age=settings.anon_session_cookie_max_age_seconds,
        samesite=settings.anon_session_cookie_samesite,
        secure=settings.anon_session_cookie_secure,
        path="/",
    )
