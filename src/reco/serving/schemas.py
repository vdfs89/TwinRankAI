"""Schemas Pydantic da API de serving."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field


class TrainResponse(BaseModel):
    status: str = "ok"
    runs: list[dict[str, str | dict[str, float]]]


class RecommendResponse(BaseModel):
    user_id: int
    item_ids: list[int]


class PredictRequest(BaseModel):
    user_id: int
    candidate_item_ids: list[int] = Field(default_factory=list)
    top_k: Annotated[int, Field(ge=1, le=100)] = 10


class PredictResponse(BaseModel):
    user_id: int
    recommendations: list[int]
