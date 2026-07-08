"""FastAPI routes exposing InsightForge-AI's agent pipeline as a REST API.

This layer is a thin adapter: it uploads/loads a dataset, runs the shared
`Orchestrator`, caches the result in `DatasetStore`, and serializes agent
outputs to JSON. All actual analytics logic lives in `app/agents/*`.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.agents.data_loader_agent import DataLoaderAgent
from app.agents.export_agent import ExportAgent
from app.agents.forecast_agent import ForecastAgent
from app.agents.query_agent import QueryAgent
from app.agents.orchestrator import Orchestrator
from app.agents.visualization_agent import VisualizationAgent
from app.llm.llm_factory import get_llm_router
from app.models.schemas import QueryRequest, QueryResponse, UploadResponse
from backend.core.dataset_store import get_dataset_store
from backend.core.logging_config import get_logger

router = APIRouter()
logger = get_logger(__name__)

loader_agent = DataLoaderAgent()
export_agent = ExportAgent()
visualization_agent = VisualizationAgent()
forecast_agent = ForecastAgent()
query_agent = QueryAgent(llm_router=get_llm_router())


def _get_record_or_404(dataset_id: str):
    record = get_dataset_store().get(dataset_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Unknown dataset_id '{dataset_id}'.")
    return record


@router.get("/health")
def health_check() -> dict:
    router_ = get_llm_router()
    return {"status": "ok", "llm_provider": router_.provider_name, "llm_enabled": router_.is_enabled()}


@router.post("/upload", response_model=UploadResponse)
async def upload_dataset(file: UploadFile = File(...)) -> UploadResponse:
    suffix = Path(file.filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = loader_agent.run(file_path=tmp_path)
        if not result.success:
            raise HTTPException(status_code=400, detail=result.error)
        df = result.data["dataframe"]
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    record = get_dataset_store().create(filename=file.filename, dataframe=df)
    return UploadResponse(
        dataset_id=record.dataset_id, filename=file.filename,
        n_rows=len(df), n_columns=df.shape[1], columns=list(df.columns),
    )


@router.post("/datasets/{dataset_id}/analyze")
def analyze_dataset(dataset_id: str, clean_data: bool = True) -> dict:
    record = _get_record_or_404(dataset_id)
    orchestrator = Orchestrator()
    context = orchestrator.run_full_pipeline(record.dataframe, dataset_name=record.filename, clean_data=clean_data)
    get_dataset_store().update_context(dataset_id, context)

    serializable = {k: v for k, v in context.items() if k != "dataframe"}
    return serializable


@router.get("/datasets/{dataset_id}/profile")
def get_profile(dataset_id: str) -> dict:
    record = _get_record_or_404(dataset_id)
    return {"profiles": record.pipeline_context.get("profiles", {}), "column_types": record.column_types}


@router.get("/datasets/{dataset_id}/quality")
def get_quality(dataset_id: str) -> dict:
    record = _get_record_or_404(dataset_id)
    return record.pipeline_context.get("quality", {})


@router.get("/datasets/{dataset_id}/correlation")
def get_correlation(dataset_id: str) -> dict:
    record = _get_record_or_404(dataset_id)
    return record.pipeline_context.get("correlation", {})


@router.get("/datasets/{dataset_id}/outliers")
def get_outliers(dataset_id: str) -> dict:
    record = _get_record_or_404(dataset_id)
    return record.pipeline_context.get("outliers", {})


@router.get("/datasets/{dataset_id}/kpis")
def get_kpis(dataset_id: str) -> dict:
    record = _get_record_or_404(dataset_id)
    return {"kpis": record.pipeline_context.get("kpis", [])}


@router.get("/datasets/{dataset_id}/insights")
def get_insights(dataset_id: str) -> dict:
    record = _get_record_or_404(dataset_id)
    return {
        "insights": record.pipeline_context.get("insights", []),
        "recommendations": record.pipeline_context.get("recommendations", []),
        "executive_summary": record.pipeline_context.get("executive_summary", {}),
        "business_narrative": record.pipeline_context.get("business_narrative"),
    }


@router.get("/datasets/{dataset_id}/charts")
def get_charts(dataset_id: str) -> dict:
    record = _get_record_or_404(dataset_id)
    specs = record.pipeline_context.get("chart_recommendations", [])
    figures = []
    for spec in specs:
        try:
            result = visualization_agent.run(dataframe=record.dataframe, spec=spec)
            if result.success:
                figures.append({**spec, "figure_json": result.data["figure_json"]})
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Chart build failed for spec {spec}: {exc}")
    return {"charts": figures}


@router.get("/datasets/{dataset_id}/forecast")
def get_forecast(dataset_id: str, date_column: str, value_column: str, periods_ahead: int = 6) -> dict:
    record = _get_record_or_404(dataset_id)
    result = forecast_agent.run(
        dataframe=record.dataframe, date_column=date_column, value_column=value_column, periods_ahead=periods_ahead
    )
    if not result.success:
        raise HTTPException(status_code=400, detail=result.error)
    return result.data


@router.post("/datasets/{dataset_id}/query", response_model=QueryResponse)
def query_dataset(dataset_id: str, request: QueryRequest) -> QueryResponse:
    record = _get_record_or_404(dataset_id)
    result = query_agent.run(dataframe=record.dataframe, question=request.question, column_types=record.column_types)
    if not result.success:
        raise HTTPException(status_code=400, detail=result.error)
    return QueryResponse(
        answer=result.data["answer"], data_preview=result.data.get("data_preview"), used_llm=result.data.get("used_llm", False)
    )


@router.post("/datasets/{dataset_id}/export")
def export_dataset(dataset_id: str, export_format: str) -> FileResponse:
    record = _get_record_or_404(dataset_id)
    report = record.pipeline_context.get("report")
    if report is None and export_format != "csv":
        raise HTTPException(status_code=400, detail="Run /analyze before exporting a report.")

    export_dir = Path(tempfile.gettempdir()) / "insightforge_exports"
    result = export_agent.run(
        export_format=export_format, output_dir=str(export_dir),
        filename_stem=f"insightforge_{dataset_id[:8]}", report=report, dataframe=record.dataframe,
    )
    if not result.success:
        raise HTTPException(status_code=400, detail=result.error)

    file_path = result.data["file_path"]
    media_types = {
        "pdf": "application/pdf",
        "excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "markdown": "text/markdown",
        "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "csv": "text/csv",
    }
    return FileResponse(file_path, media_type=media_types.get(export_format, "application/octet-stream"),
                         filename=Path(file_path).name)


@router.get("/datasets")
def list_datasets() -> dict:
    return {"dataset_ids": get_dataset_store().list_ids()}


@router.delete("/datasets/{dataset_id}")
def delete_dataset(dataset_id: str) -> dict:
    deleted = get_dataset_store().delete(dataset_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Unknown dataset_id '{dataset_id}'.")
    return {"deleted": True}
