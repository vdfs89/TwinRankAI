"""Orquestração de treino, avaliação e registro no MLflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlflow
import pandas as pd

from reco.models.factory import ModelType, create_model
from reco.pipelines.feature_eng import FeatureArtifacts
from reco.pipelines.feature_eng import run as run_feature_engineering
from reco.pipelines.preprocess import PreprocessArtifacts
from reco.pipelines.preprocess import run as run_preprocess
from reco.settings import Settings, get_settings
from reco.training.evaluate import evaluate_model
from reco.training.mlflow_utils import configure_mlflow, log_and_register_model
from reco.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class TrainingRunResult:
    model_type: ModelType
    run_id: str
    model_path: Path
    metrics: dict[str, float]


def _load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return pd.read_csv(path, parse_dates=["timestamp"])


def _build_test_lookup(
    test_events: pd.DataFrame,
) -> dict[int, dict[int, float]]:
    lookup: dict[int, dict[int, float]] = {}
    for visitor_id, group in test_events.groupby("visitorid"):
        lookup[int(visitor_id)] = {
            int(row.itemid): float(row.relevance)
            for row in group.itertuples(index=False)
        }
    return lookup


def _artifact_path(settings: Settings, model_type: ModelType) -> Path:
    model_dir = settings.models_dir / model_type.value
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / "model.joblib"


def _run_single_experiment(
    settings: Settings,
    model_type: ModelType,
    train_events: pd.DataFrame,
    test_events: pd.DataFrame,
) -> TrainingRunResult:
    configure_mlflow(settings)
    model = create_model(model_type, settings)
    test_lookup = _build_test_lookup(test_events)

    with mlflow.start_run(run_name=model_type.value) as run:
        mlflow.log_params(
            {
                "model_type": model_type.value,
                "embedding_dim": settings.embedding_dim,
                "learning_rate": settings.learning_rate,
                "batch_size": settings.batch_size,
                "negative_samples_per_positive": (
                    settings.negative_samples_per_positive
                ),
                "max_epochs": settings.max_epochs,
                "early_stopping_patience": settings.early_stopping_patience,
            }
        )
        model.fit(train_events)
        metrics = evaluate_model(model, test_lookup, settings.top_k)
        mlflow.log_metrics(metrics)

        model_path = _artifact_path(settings, model_type)
        model.save(str(model_path))
        mlflow.log_artifact(str(model_path), artifact_path="model")

        if model_type is ModelType.TWO_TOWER:
            version = log_and_register_model(
                settings=settings,
                run_id=run.info.run_id,
                model_path=model_path,
                stage=settings.model_stage,
            )
            mlflow.log_param("registered_model_version", version)

        logger.info(
            "treino_concluido",
            model_type=model_type.value,
            run_id=run.info.run_id,
            metrics=metrics,
        )
        return TrainingRunResult(
            model_type=model_type,
            run_id=run.info.run_id,
            model_path=model_path,
            metrics=metrics,
        )


def run_training_pipeline(
    settings: Settings | None = None,
) -> list[TrainingRunResult]:
    current_settings = settings or get_settings()
    preprocess_artifacts: PreprocessArtifacts = run_preprocess(
        current_settings,
    )
    feature_artifacts: FeatureArtifacts = run_feature_engineering(
        current_settings,
    )
    train_events = _load_dataframe(feature_artifacts.train_features_path)
    test_events = _load_dataframe(feature_artifacts.test_features_path)

    results = [
        _run_single_experiment(
            current_settings,
            model_type,
            train_events,
            test_events,
        )
        for model_type in (
            ModelType.POPULARITY,
            ModelType.MATRIX_FACTORIZATION,
            ModelType.TWO_TOWER,
        )
    ]
    logger.info(
        "pipeline_treino_concluido",
        train_events=str(preprocess_artifacts.train_path),
        test_events=str(preprocess_artifacts.test_path),
    )
    return results


def main() -> int:
    run_training_pipeline()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
