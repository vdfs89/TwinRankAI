"""API FastAPI do TwinRank AI."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Annotated

from fastapi import BackgroundTasks, FastAPI, Path, Query, Request, Response

from reco.pipelines.feature_eng import run as run_feature_engineering
from reco.pipelines.preprocess import run as run_preprocess
from reco.serving.schemas import PredictRequest, PredictResponse, RecommendResponse, TrainResponse
from reco.serving.service import (
    RecommendationService,
    get_recommendation_service,
    reset_recommendation_service,
)
from reco.settings import Settings, get_settings
from reco.training.train import run_training_pipeline
from reco.utils.logging import get_logger

logger = get_logger(__name__)

app = FastAPI(title="TwinRank AI", version="0.1.0")

_settings_instance: Settings | None = None


def _settings() -> Settings:
    global _settings_instance  # noqa: PLW0603
    if _settings_instance is None:
        _settings_instance = get_settings()
    return _settings_instance


def _service() -> RecommendationService:
    return get_recommendation_service(_settings())


@app.on_event("startup")
def warm_up_model() -> None:
    """Carrega o modelo no boot em vez de no primeiro request.

    O carregamento do checkpoint (~35 MB) mais o índice FAISS custa ~7 s; feito
    de forma preguiçosa, esse custo cairia inteiro sobre o primeiro usuário.
    """
    started = time.perf_counter()
    _service()
    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info("modelo_carregado_no_startup", elapsed_ms=round(elapsed_ms, 2))


@app.middleware("http")
async def log_request_latency(request: Request, call_next: Callable) -> Response:
    """Mede e loga a latência de cada requisição no logger estruturado."""
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "request_concluida",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        latency_ms=round(elapsed_ms, 2),
    )
    response.headers["X-Response-Time-ms"] = f"{elapsed_ms:.2f}"
    return response


@app.get("/health")
def health() -> dict[str, str]:  # noqa: D103
    return {"status": "ok"}


@app.get("/model/version")
def model_version() -> dict[str, str]:  # noqa: D103
    settings = _settings()
    return {
        "model_path": str(settings.model_path),
        "registered_model_name": settings.mlflow_registered_model_name,
        "stage": settings.model_stage,
    }


@app.get("/recommend/{user_id}", response_model=RecommendResponse)
def recommend(
    user_id: Annotated[int, Path(ge=0)],
    top_k: Annotated[int, Query(ge=1, le=100)] = 10,
) -> RecommendResponse:
    """Top-k recomendações para um visitante, com fallback de cold-start."""
    item_ids, strategy = _service().recommend(user_id, top_k)
    return RecommendResponse(user_id=user_id, item_ids=item_ids, strategy=strategy)


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:  # noqa: D103
    recommendations, strategy = _service().predict(
        payload.user_id,
        payload.candidate_item_ids,
        payload.top_k,
    )
    return PredictResponse(
        user_id=payload.user_id,
        recommendations=recommendations,
        strategy=strategy,
    )


def _run_training_in_background() -> None:
    """Executa o pipeline de treino fora do ciclo de request."""
    try:
        run_training_pipeline(_settings())
    except Exception as exc:  # noqa: BLE001
        logger.error("treino_em_background_falhou", error=str(exc))
        return
    reset_recommendation_service()
    logger.info("treino_em_background_concluido")


@app.post("/train", response_model=TrainResponse, status_code=202)
def train(background_tasks: BackgroundTasks) -> TrainResponse:
    """Dispara o treino em background e responde imediatamente.

    O pipeline completo leva ~12,5 min; executá-lo dentro do request garantiria
    timeout. Ver a limitação documentada no README: sem fila de jobs dedicada,
    esta rota não é adequada para produção.
    """
    background_tasks.add_task(_run_training_in_background)
    return TrainResponse(
        status="training_started",
        detail="Treino disparado em background; acompanhe o progresso no MLflow.",
    )


@app.post("/preprocess")
def preprocess() -> dict[str, str]:  # noqa: D103
    artifacts = run_preprocess(_settings())
    reset_recommendation_service()
    return {
        "train_path": str(artifacts.train_path),
        "test_path": str(artifacts.test_path),
    }


@app.post("/feature-eng")
def feature_eng() -> dict[str, str]:  # noqa: D103
    artifacts = run_feature_engineering(_settings())
    reset_recommendation_service()
    return {
        "train_features_path": str(artifacts.train_features_path),
        "test_features_path": str(artifacts.test_features_path),
    }
