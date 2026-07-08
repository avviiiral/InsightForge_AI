"""Backend-facing re-export of the shared application settings.

Kept as a thin wrapper (rather than a duplicate settings class) so the
FastAPI layer and the Streamlit layer always agree on configuration.
"""
from app.config import Settings, get_settings  # noqa: F401
