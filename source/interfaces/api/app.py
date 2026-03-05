from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException

try:
    import boto3
except ModuleNotFoundError:
    boto3 = None

from source.infrastructure.inference import (
    InferenceRequestLogger,
    InferenceService,
    ModelBundleLoader,
    UserHistoryProvider,
)
from source.infrastructure.inference.service import InferenceRequest
from source.infrastructure.storage.postgres import PostgresClient
from source.interfaces.api.schemas import (
    HealthResponse,
    InteractionRequest,
    RecommendationRequest,
    RecommendationResponse,
    SimilarItemsResponse,
)


@dataclass
class AppState:
    service: InferenceService | None = None
    postgres_ok: bool = False
    s3_ok: bool = False


def create_app() -> FastAPI:
    state = AppState()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        model_uri = os.getenv("BOOKRECS_API_MODEL_URI", "").strip()
        s3_region = os.getenv("BOOKRECS_S3_REGION", "us-east-1")
        s3_endpoint = os.getenv("BOOKRECS_S3_ENDPOINT", "").strip()
        pg_dsn = os.getenv("BOOKRECS_PG_DSN", "").strip()
        history_table = os.getenv("BOOKRECS_API_HISTORY_TABLE", "user_item_interactions").strip()
        log_table = os.getenv("BOOKRECS_API_INFERENCE_LOG_TABLE", "inference_requests").strip()

        pg = PostgresClient(pg_dsn) if pg_dsn else None
        runtime_pg = None
        if pg is not None:
            try:
                row = pg.fetchone("SELECT 1 AS ok")
                state.postgres_ok = bool(row and row.get("ok") == 1)
                if state.postgres_ok:
                    runtime_pg = pg
            except Exception:
                state.postgres_ok = False
                runtime_pg = None

        loader = ModelBundleLoader(
            s3_region=s3_region,
            s3_endpoint=s3_endpoint,
            local_cache_root=os.getenv("BOOKRECS_API_MODEL_CACHE_DIR", "artifacts/cache/models"),
        )
        bundle = loader.load(model_uri=model_uri)
        state.s3_ok = _check_s3_available(bundle.model_dir, model_uri, s3_region, s3_endpoint)
        state.service = InferenceService(
            bundle=bundle,
            history=UserHistoryProvider(pg=runtime_pg, table_name=history_table),
            request_logger=InferenceRequestLogger(pg=runtime_pg, table_name=log_table),
        )
        yield

    app = FastAPI(
        title="BookRecs Inference API",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/healthz", response_model=HealthResponse)
    def healthz() -> HealthResponse:
        svc = _service_or_503(state)
        return HealthResponse(
            status="ok",
            model_dir=svc.model_dir,
            postgres=state.postgres_ok,
            s3=state.s3_ok,
        )

    @app.post("/v1/recommendations", response_model=RecommendationResponse)
    def recommendations(payload: RecommendationRequest) -> RecommendationResponse:
        svc = _service_or_503(state)
        result = svc.recommend(
            InferenceRequest(
                user_id=payload.user_id,
                top_k=payload.top_k,
                candidate_pool_size=payload.candidate_pool_size,
                candidate_per_source_limit=payload.candidate_per_source_limit,
                pre_top_m=payload.pre_top_m,
                seen_items=set(payload.seen_items),
                use_history=payload.use_history,
            )
        )
        return RecommendationResponse(**result)

    @app.get("/v1/items/{item_id}/similar", response_model=SimilarItemsResponse)
    def similar_items(item_id: str, limit: int = 10) -> SimilarItemsResponse:
        svc = _service_or_503(state)
        safe_limit = min(max(1, int(limit)), 100)
        try:
            return SimilarItemsResponse(**svc.similar_items(item_id=item_id, limit=safe_limit))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/interactions")
    def add_interaction(payload: InteractionRequest) -> dict[str, Any]:
        svc = _service_or_503(state)
        try:
            svc.register_interaction(
                user_id=payload.user_id,
                item_id=payload.item_id,
                event_type=payload.event_type,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"status": "ok"}

    return app


app = create_app()


def _service_or_503(state: AppState) -> InferenceService:
    if state.service is None:
        raise HTTPException(status_code=503, detail="Inference service is not initialized")
    return state.service


def _check_s3_available(model_dir: str, model_uri: str, region: str, endpoint: str) -> bool:
    _ = model_dir
    if not model_uri or not model_uri.startswith("s3://"):
        return False
    if boto3 is None:
        return False
    parsed = urlparse(model_uri)
    bucket = parsed.netloc
    key = parsed.path.strip("/")
    if not bucket or not key:
        return False
    try:
        client = boto3.client(
            "s3",
            region_name=region,
            endpoint_url=(endpoint or None),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
        client.head_object(Bucket=bucket, Key=f"{key}/stage1.pkl")
        return True
    except Exception:
        return False
