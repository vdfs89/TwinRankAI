"""Schemas Pydantic da API de serving."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field


class TrainResponse(BaseModel):  # noqa: D101
    status: str = "ok"
    detail: str = ""


class RecommendResponse(BaseModel):  # noqa: D101
    user_id: int
    item_ids: list[int]
    # Qual caminho produziu a resposta: "two_tower" (personalizado),
    # "popularity_fallback" (cold-start) ou "unavailable" (sem modelo).
    strategy: str = "two_tower"


class PredictRequest(BaseModel):  # noqa: D101
    user_id: int
    candidate_item_ids: list[int] = Field(default_factory=list)
    top_k: Annotated[int, Field(ge=1, le=100)] = 10


class PredictResponse(BaseModel):  # noqa: D101
    user_id: int
    recommendations: list[int]
    strategy: str = "two_tower"
