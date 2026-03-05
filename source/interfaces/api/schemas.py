from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    top_k: int = Field(10, ge=1, le=100)
    candidate_pool_size: int = Field(1000, ge=50, le=5000)
    candidate_per_source_limit: int = Field(300, ge=10, le=2000)
    pre_top_m: int = Field(300, ge=10, le=2000)
    seen_items: list[Any] = Field(default_factory=list)
    use_history: bool = True


class RecommendationItem(BaseModel):
    item_id: Any
    score: float
    source: str


class RecommendationResponse(BaseModel):
    user_id: str
    history_len: int
    model_dir: str
    metrics_snapshot: dict[str, Any]
    items: list[RecommendationItem]


class SimilarItemsResponse(BaseModel):
    item_id: Any
    content: list[dict[str, Any]]
    cf: list[dict[str, Any]]


class InteractionRequest(BaseModel):
    user_id: str
    item_id: Any
    event_type: str = Field("implicit", min_length=1, max_length=64)


class HealthResponse(BaseModel):
    status: str
    model_dir: str
    postgres: bool
    s3: bool
