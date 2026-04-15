from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    top_k: int = Field(10, ge=1, le=100, description="Number of recommendations.")
    candidate_pool_size: int = Field(
        1000, ge=50, le=5000, description="Global candidate pool size before ranking."
    )
    candidate_per_source_limit: int = Field(
        300, ge=10, le=2000, description="Per-source candidate cap."
    )
    pre_top_m: int = Field(
        300, ge=10, le=2000, description="Top-M candidates selected after pre-ranking."
    )
    seen_items: list[Any] = Field(
        default_factory=list, description="Items that should be excluded from results."
    )
    use_history: bool = Field(
        True, description="If true, merges runtime user history into exclusion set."
    )


class RecommendationItem(BaseModel):
    item_id: Any = Field(..., description="Recommended item identifier.")
    score: float = Field(..., description="Final ranking score.")
    source: str = Field(
        ..., description="Dominant source used for this recommendation."
    )


class RecommendationResponse(BaseModel):
    user_id: str = Field(..., description="User identifier from the request.")
    history_len: int = Field(
        ..., description="Number of known historical interactions."
    )
    model_dir: str = Field(..., description="Filesystem path to active model bundle.")
    metrics_snapshot: dict[str, Any] = Field(
        ..., description="Training/evaluation metrics snapshot bundled with the model."
    )
    items: list[RecommendationItem] = Field(
        ..., description="Sorted recommendation list."
    )


class SimilarItemsResponse(BaseModel):
    item_id: Any = Field(..., description="Input item identifier.")
    content: list[dict[str, Any]] = Field(
        ..., description="Top similar items from content-based neighborhood."
    )
    cf: list[dict[str, Any]] = Field(
        ..., description="Top similar items from collaborative filtering neighborhood."
    )


class InteractionRequest(BaseModel):
    user_id: str = Field(..., description="User identifier.")
    item_id: Any = Field(..., description="Interacted item identifier.")
    event_type: str = Field(
        "implicit",
        min_length=1,
        max_length=64,
        description="Interaction type, e.g. implicit, click, add_to_cart.",
    )


class LivenessResponse(BaseModel):
    status: str = Field(..., description="Liveness status string.")
    alive: bool = Field(..., description="True when API process is alive.")


class ReadinessResponse(BaseModel):
    status: str = Field(..., description="Readiness status string.")
    ready: bool = Field(..., description="True when service is ready for traffic.")
    model_dir: str = Field(..., description="Path to active model bundle if loaded.")
    run_id: str | None = Field(
        None, description="Active training run identifier when available."
    )
    postgres: bool = Field(..., description="PostgreSQL readiness flag.")
    s3: bool = Field(..., description="S3 readiness flag when S3 check is required.")


class ApiEndpointsInfoResponse(BaseModel):
    docs_url: str = Field(..., description="Swagger UI endpoint URL.")
    openapi_url: str = Field(..., description="OpenAPI JSON endpoint URL.")
    redoc_url: str = Field(..., description="ReDoc endpoint URL.")


class DemoUser(BaseModel):
    user_id: str = Field(..., description="Demo user identifier.")
    history_len: int = Field(..., description="Count of known user interactions.")


class DemoUsersResponse(BaseModel):
    items: list[DemoUser] = Field(..., description="Demo user list.")
    total: int = Field(..., description="Total number of returned users.")


class DemoBook(BaseModel):
    item_id: int = Field(..., description="Book identifier.")
    title: str = Field(..., description="Book title.")
    description: str = Field(..., description="Book description text.")
    url: str = Field(..., description="Source URL.")
    image_url: str = Field(..., description="Cover image URL.")
    authors: list[str] = Field(..., description="Book authors.")
    tags: list[str] = Field(..., description="Book tags/genres.")
    series: list[str] = Field(..., description="Series metadata.")


class DemoCatalogResponse(BaseModel):
    items: list[DemoBook] = Field(..., description="Paginated demo books.")
    total: int = Field(..., description="Total items matching filters.")
    limit: int = Field(..., description="Applied page size.")
    offset: int = Field(..., description="Applied page offset.")
