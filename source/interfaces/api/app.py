from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException

try:
    import boto3
except ModuleNotFoundError:
    boto3 = None

from source.infrastructure.config import load_api_runtime_settings
from source.infrastructure.inference import (
    InferenceRequestLogger,
    InferenceService,
    ModelBundleLoader,
    UserHistoryProvider,
    resolve_model_uri,
)
from source.infrastructure.inference.demo_store import DemoStore
from source.infrastructure.inference.service import InferenceRequest
from source.infrastructure.storage.postgres import PostgresClient
from source.interfaces.api.schemas import (
    DemoBook,
    DemoCatalogResponse,
    DemoUser,
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
    loader: ModelBundleLoader | None = None
    runtime_pg: PostgresClient | None = None
    postgres_ok: bool = False
    s3_ok: bool = False
    current_model_uri: str = ""
    current_model_run_id: str | None = None
    model_reload_lock: threading.Lock = field(default_factory=threading.Lock)
    model_reload_interval_sec: int = 60
    model_reload_checked_at: float = 0.0


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
        state.runtime_pg = runtime_pg

        state.loader = ModelBundleLoader(
            s3_region=settings.s3_region,
            s3_endpoint=settings.s3_endpoint,
            local_cache_root=settings.model_cache_dir,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
        )
        state.model_reload_interval_sec = max(0, settings.auto_reload_sec)
        _reload_model(state=state, settings=settings, force=True)
        svc = _service_or_503(state)
        state.s3_ok = _check_s3_available(
            svc.model_dir,
            state.current_model_uri or settings.model_uri,
            settings.s3_region,
            settings.s3_endpoint,
            settings.aws_access_key_id,
            settings.aws_secret_access_key,
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
        _maybe_reload_model(state=state)
        svc = _service_or_503(state)
        return HealthResponse(
            status="ok",
            model_dir=svc.model_dir,
            postgres=state.postgres_ok,
            s3=state.s3_ok,
        )

    @app.post("/v1/recommendations", response_model=RecommendationResponse)
    def recommendations(payload: RecommendationRequest) -> RecommendationResponse:
        _maybe_reload_model(state=state)
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

    @app.post("/v1/demo/recommendations", response_model=RecommendationResponse)
    def demo_recommendations(payload: RecommendationRequest) -> RecommendationResponse:
        _maybe_reload_model(state=state)
        svc = _service_or_503(state)
        store = _demo_store_or_503(state)
        demo_seen = set(store.list_user_seen_items(user_id=payload.user_id, limit=1000))
        payload_seen = set(payload.seen_items)
        seen_items = demo_seen | payload_seen
        result = svc.recommend(
            InferenceRequest(
                user_id=payload.user_id,
                top_k=payload.top_k,
                candidate_pool_size=payload.candidate_pool_size,
                candidate_per_source_limit=payload.candidate_per_source_limit,
                pre_top_m=payload.pre_top_m,
                seen_items=seen_items,
                use_history=True,
            )
        )
        return RecommendationResponse(**result)

    @app.get("/v1/items/{item_id}/similar", response_model=SimilarItemsResponse)
    def similar_items(item_id: str, limit: int = 10) -> SimilarItemsResponse:
        _maybe_reload_model(state=state)
        svc = _service_or_503(state)
        safe_limit = min(max(1, int(limit)), 100)
        try:
            return SimilarItemsResponse(
                **svc.similar_items(item_id=item_id, limit=safe_limit)
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/interactions")
    def add_interaction(payload: InteractionRequest) -> dict[str, Any]:
        _maybe_reload_model(state=state)
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

    @app.post("/v1/admin/reload-model")
    def reload_model() -> dict[str, Any]:
        changed = _reload_model(
            state=state, settings=load_api_runtime_settings(), force=True
        )
        svc = _service_or_503(state)
        return {
            "status": "ok",
            "changed": changed,
            "model_dir": svc.model_dir,
            "model_uri": state.current_model_uri,
            "run_id": state.current_model_run_id,
        }

    @app.get("/v1/demo/users", response_model=DemoUsersResponse)
    def demo_users(limit: int = 100) -> DemoUsersResponse:
        store = _demo_store_or_503(state)
        safe_limit = min(max(1, int(limit)), 5000)
        items = store.list_users(limit=safe_limit)
        return DemoUsersResponse(
            items=[
                DemoUser(user_id=x.user_id, history_len=x.history_len) for x in items
            ],
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
        raise HTTPException(
            status_code=503, detail="Inference service is not initialized"
        )
    return state.service


def _maybe_reload_model(state: AppState) -> None:
    interval = max(0, int(state.model_reload_interval_sec))
    if interval <= 0:
        return
    now = time.monotonic()
    if now - state.model_reload_checked_at < interval:
        return
    _reload_model(state=state, settings=load_api_runtime_settings(), force=False)


def _reload_model(state: AppState, settings: Any, *, force: bool) -> bool:
    if state.loader is None:
        raise RuntimeError("Model loader is not initialized")

    with state.model_reload_lock:
        state.model_reload_checked_at = time.monotonic()
        resolved_uri, pointer = resolve_model_uri(
            settings.model_uri, settings.active_model_pointer
        )
        if not resolved_uri:
            if state.current_model_uri:
                resolved_uri = state.current_model_uri
            else:
                raise RuntimeError(
                    "Model URI is empty and no active model pointer found"
                )

        if (
            not force
            and state.current_model_uri == resolved_uri
            and state.service is not None
        ):
            return False

        bundle = state.loader.load(model_uri=resolved_uri)
        state.service = InferenceService(
            bundle=bundle,
            history=UserHistoryProvider(
                pg=state.runtime_pg, table_name=settings.history_table
            ),
            request_logger=InferenceRequestLogger(
                pg=state.runtime_pg, table_name=settings.inference_log_table
            ),
        )
        state.current_model_uri = resolved_uri
        state.current_model_run_id = pointer.run_id if pointer is not None else None
        return True


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
