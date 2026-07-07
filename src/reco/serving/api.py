"""API FastAPI do TwinRank AI."""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException

from reco.pipelines.feature_eng import run as run_feature_engineering
from reco.pipelines.preprocess import run as run_preprocess
from reco.serving.schemas import PredictRequest
from reco.serving.schemas import PredictResponse
from reco.serving.schemas import RecommendResponse
from reco.serving.schemas import TrainResponse
from reco.serving.service import RecommendationService
from reco.serving.service import get_recommendation_service
from reco.settings import Settings, get_settings
from reco.training.train import run_training_pipeline

app = FastAPI(title="TwinRank AI", version="0.1.0")


@lru_cache(maxsize=1)
def _settings() -> Settings:
    return get_settings()


@lru_cache(maxsize=1)
def _service() -> RecommendationService:
    return get_recommendation_service(_settings())


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model/version")
def model_version() -> dict[str, str]:
    settings = _settings()
    return {
        "model_path": str(settings.model_path),
        "registered_model_name": settings.mlflow_registered_model_name,
        "stage": settings.model_stage,
    }


@app.get("/recommend/{user_id}", response_model=RecommendResponse)
def recommend(user_id: int, top_k: int = 10) -> RecommendResponse:
    item_ids = _service().recommend(user_id, top_k)
    return RecommendResponse(user_id=user_id, item_ids=item_ids)


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    recommendations = _service().predict(
        payload.user_id,
        payload.candidate_item_ids,
        payload.top_k,
    )
    return PredictResponse(
        user_id=payload.user_id,
        recommendations=recommendations,
    )


@app.post("/train", response_model=TrainResponse)
def train() -> TrainResponse:
    try:
        results = run_training_pipeline(_settings())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _service.cache_clear()
    return TrainResponse(
        runs=[
            {
                "model_type": result.model_type.value,
                "run_id": result.run_id,
                "metrics": result.metrics,
            }
            for result in results
        ]
    )


@app.post("/preprocess")
def preprocess() -> dict[str, str]:
    artifacts = run_preprocess(_settings())
    _service.cache_clear()
    return {
        "train_path": str(artifacts.train_path),
        "test_path": str(artifacts.test_path),
    }


@app.post("/feature-eng")
def feature_eng() -> dict[str, str]:
    artifacts = run_feature_engineering(_settings())
    _service.cache_clear()
    return {
        "train_features_path": str(artifacts.train_features_path),
        "test_features_path": str(artifacts.test_features_path),
    }
