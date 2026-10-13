"""FastAPI application factory."""

from __future__ import annotations

import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.errors import AppError
from app.core.logging import logger
from app.api.routes import health
from app.api.routes import generations


def create_app() -> FastAPI:
    """Build the FastAPI application with routes and middleware."""
    settings = get_settings()
    app = FastAPI(
        title="Floor Plan Generator API",
        version="0.1.0",
        description="Approximate 2D floor plans from a single indoor-room photograph.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    @app.middleware("http")
    async def request_id_middleware(request, call_next):
        request_id = request.headers.get("X-Request-ID") or f"req_{uuid.uuid4().hex[:12]}"
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    app.include_router(health.router, prefix="/api")
    app.include_router(generations.router, prefix="/api")

    @app.exception_handler(AppError)
    async def app_error_handler(_request, exc: AppError):
        logger.warning("app_error code=%s message=%s", exc.code.value, exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code.value, "message": exc.message}},
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(_request, exc: Exception):
        logger.exception("unhandled_error")
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "internal_error",
                               "message": "An unexpected error occurred."}},
        )

    return app


app = create_app()
