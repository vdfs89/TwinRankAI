"""Helpers de MLflow para logging e registro de modelos."""

from __future__ import annotations

from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from reco.settings import Settings
from reco.utils.logging import get_logger

logger = get_logger(__name__)

STAGING = "Staging"


def configure_mlflow(settings: Settings) -> None:  # noqa: D103
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)


def _transition(client: MlflowClient, name: str, version: str, stage: str) -> None:
    """Move uma versão registrada para o stage indicado."""
    client.transition_model_version_stage(
        name=name,
        version=version,
        stage=stage,
        archive_existing_versions=True,
    )
    logger.info("model_version_promovida", name=name, version=version, stage=stage)


def log_and_register_model(settings: Settings, run_id: str, model_path: Path, stage: str) -> str:
    """Registra o modelo e o promove até o stage alvo, passando por Staging.

    A promoção é feita em dois passos (Staging e depois o stage final) mesmo
    quando o alvo é Production: o ciclo de vida pedido pela Fase 2 é
    `Staging -> Production`, e saltar direto para Production apagaria esse
    histórico no Registry. Cada transição fica registrada na versão.

    Args:
    ----
        settings: configurações com o nome do modelo registrado.
        run_id: run do MLflow que produziu o artefato.
        model_path: caminho local do checkpoint (mantido por compatibilidade).
        stage: stage final desejado, normalmente `Production`.

    Returns:
    -------
        A versão registrada no Model Registry.

    """
    model_uri = f"runs:/{run_id}/model"
    registered = mlflow.register_model(
        model_uri=model_uri, name=settings.mlflow_registered_model_name
    )
    client = MlflowClient()

    _transition(client, settings.mlflow_registered_model_name, registered.version, STAGING)
    if stage != STAGING:
        _transition(client, settings.mlflow_registered_model_name, registered.version, stage)

    return registered.version
