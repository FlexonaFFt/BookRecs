from __future__ import annotations

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
from source.infrastructure.inference.demo_store import DemoStore
from source.infrastructure.inference.service import InferenceRequest
from source.infrastructure.storage.postgres import PostgresClient
from source.infrastructure.config import load_api_runtime_settings
from source.interfaces.api.schemas import (
    DemoBook,
    DemoCatalogResponse,
    DemoUsersResponse,
    HealthResponse,
    InteractionRequest,
    RecommendationRequest,
    RecommendationResponse,
    SimilarItemsResponse,
)


@dataclass
class AppState:
    service: InferenceService | None = None
    demo_store: DemoStore | None = None
    postgres_ok: bool = False
    s3_ok: bool = False


def create_app() -> FastAPI:
    state = AppState()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        settings = load_api_runtime_settings()

        pg = PostgresClient(settings.pg_dsn) if settings.pg_dsn else None
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
            s3_region=settings.s3_region,
            s3_endpoint=settings.s3_endpoint,
            local_cache_root=settings.model_cache_dir,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
        )
        bundle = loader.load(model_uri=settings.model_uri)
        state.s3_ok = _check_s3_available(
            bundle.model_dir,
            settings.model_uri,
            settings.s3_region,
            settings.s3_endpoint,
            settings.aws_access_key_id,
            settings.aws_secret_access_key,
        )
        state.service = InferenceService(
            bundle=bundle,
            history=UserHistoryProvider(pg=runtime_pg, table_name=settings.history_table),
            request_logger=InferenceRequestLogger(pg=runtime_pg, table_name=settings.inference_log_table),
        )
        state.demo_store = DemoStore(pg=runtime_pg)
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

    @app.get("/v1/demo/users", response_model=DemoUsersResponse)
    def demo_users(limit: int = 100) -> DemoUsersResponse:
        store = _demo_store_or_503(state)
        safe_limit = min(max(1, int(limit)), 5000)
        items = store.list_users(limit=safe_limit)
        return DemoUsersResponse(
            items=[{"user_id": x.user_id, "history_len": x.history_len} for x in items],
            total=len(items),
        )

    @app.get("/v1/demo/catalog", response_model=DemoCatalogResponse)
    def demo_catalog(
        limit: int = 40,
        offset: int = 0,
        q: str = "",
        genre: str = "",
    ) -> DemoCatalogResponse:
        store = _demo_store_or_503(state)
        safe_limit = min(max(1, int(limit)), 100)
        safe_offset = max(0, int(offset))
        items, total = store.list_books(
            limit=safe_limit,
            offset=safe_offset,
            q=q,
            genre=genre,
        )
        return DemoCatalogResponse(
            items=[_to_demo_book(x) for x in items],
            total=total,
            limit=safe_limit,
            offset=safe_offset,
        )

    @app.get("/v1/demo/books/{item_id}", response_model=DemoBook)
    def demo_book(item_id: int) -> DemoBook:
        store = _demo_store_or_503(state)
        book = store.get_book(item_id=item_id)
        if book is None:
            raise HTTPException(status_code=404, detail=f"Book {item_id} not found")
        return _to_demo_book(book)

    return app


app = create_app()


def _service_or_503(state: AppState) -> InferenceService:
    if state.service is None:
        raise HTTPException(status_code=503, detail="Inference service is not initialized")
    return state.service


def _demo_store_or_503(state: AppState) -> DemoStore:
    if state.demo_store is None:
        raise HTTPException(status_code=503, detail="Demo store is not initialized")
    return state.demo_store


def _to_demo_book(book: Any) -> DemoBook:
    return DemoBook(
        item_id=int(book.item_id),
        title=str(book.title),
        description=str(book.description),
        url=str(book.url),
        image_url=str(book.image_url),
        authors=list(book.authors),
        tags=list(book.tags),
        series=list(book.series),
    )


def _check_s3_available(
    model_dir: str,
    model_uri: str,
    region: str,
    endpoint: str,
    aws_access_key_id: str | None,
    aws_secret_access_key: str | None,
) -> bool:
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
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
        client.head_object(Bucket=bucket, Key=f"{key}/stage1.pkl")
        return True
    except Exception:
        return False
