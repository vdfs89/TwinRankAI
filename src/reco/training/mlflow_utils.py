"""Helpers de MLflow para logging e registro de modelos."""

from __future__ import annotations

from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from reco.settings import Settings


def configure_mlflow(settings: Settings) -> None:  # noqa: D103
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)


def log_and_register_model(settings: Settings, run_id: str, model_path: Path, stage: str) -> str:  # noqa: D103
    model_uri = f"runs:/{run_id}/model"
    registered = mlflow.register_model(
        model_uri=model_uri, name=settings.mlflow_registered_model_name
    )
    client = MlflowClient()
    client.transition_model_version_stage(
        name=settings.mlflow_registered_model_name,
        version=registered.version,
        stage=stage,
        archive_existing_versions=True,
    )
    return registered.version
