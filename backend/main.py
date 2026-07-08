"""FastAPI application entry point for InsightForge-AI.

Run with:
    uvicorn backend.main:app --reload --port 8000

This backend is an OPTIONAL companion to the primary Streamlit UI — the
Streamlit app works fully standalone using the same `app/` core library.
The API exists for programmatic/integration use cases (e.g. calling
InsightForge-AI's analytics from another service).
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.utils.logger import get_logger
from backend.api.routes import router

logger = get_logger(__name__)
settings = get_settings()

app = FastAPI(
    title="InsightForge-AI API",
    description="Enterprise-grade AI Data Analytics Agent Platform — REST API layer.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1", tags=["insightforge"])


@app.get("/")
def root() -> dict:
    return {
        "service": "InsightForge-AI API",
        "status": "running",
        "docs": "/docs",
        "environment": settings.app_env,
    }


@app.on_event("startup")
def on_startup() -> None:
    logger.info(f"InsightForge-AI backend starting in '{settings.app_env}' mode.")
