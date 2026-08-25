# pyright: reportMissingTypeStubs=false, reportMissingImports=false
"""Pipeline de avaliação final do modelo treinado."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow  # type: ignore[import-untyped]
import pandas as pd  # type: ignore[import-untyped]

from reco.models.factory import ModelType, create_model
from reco.settings import Settings
from reco.training.evaluate import (
    build_relevance_lookup,
    evaluate_model,
    evaluate_novel_recall_at_k,
    filter_lookup_by_history,
    train_items_by_visitor,
)
from reco.training.mlflow_utils import configure_mlflow


def _load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return pd.read_csv(path, parse_dates=["timestamp"])


def _production_run_id(settings: Settings) -> str | None:
    """Busca o run_id da versão em produção no Model Registry.

    Retorna None se o registry não estiver acessível — o pipeline de avaliação
    não deve falhar por causa de um metadado informativo.
    """
    try:
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(
            settings.mlflow_registered_model_name, stages=[settings.model_stage]
        )
    except Exception:  # noqa: BLE001 - registry indisponível não invalida a avaliação
        return None
    return versions[0].run_id if versions else None


def _evaluate_one(
    model_type: ModelType,
    settings: Settings,
    lookup: dict,
    train_items: dict,
    eval_audit: dict,
) -> dict[str, float]:
    """Avalia um modelo e registra o resultado num run do MLflow."""
    model = create_model(model_type, settings)
    model.load(str(settings.models_dir / model_type.value / "model.joblib"))

    metrics = evaluate_model(model, lookup, settings.top_k)
    # Métrica secundária: descoberta pura, exclui repetição de itens já vistos
    # no treino. Calculada para os 3 modelos para saber se a vantagem do
    # two-tower é descoberta real ou só repetir conhecidos com mais precisão.
    metrics[f"recall_at_{settings.top_k}_novel"] = evaluate_novel_recall_at_k(
        model, lookup, train_items, settings.top_k
    )

    with mlflow.start_run(run_name=f"evaluate_{model_type.value}"):
        mlflow.log_params(eval_audit)
        mlflow.log_metrics(metrics)

    return metrics


def _run_metadata(
    settings: Settings,
    eval_audit: dict,
    train_events: pd.DataFrame,
    test_events: pd.DataFrame,
) -> dict:
    """Metadados lidos pelo app Streamlit, que não mantém números hardcoded."""
    return {
        "run_id": _production_run_id(settings),
        "embedding_dim": settings.embedding_dim,
        "top_k": settings.top_k,
        "eval_visitors": eval_audit.get("eval_visitors_after_filter"),
        "catalog_items": int(pd.concat([train_events.itemid, test_events.itemid]).nunique()),
        "known_visitors": int(pd.concat([train_events.visitorid, test_events.visitorid]).nunique()),
        "train_interactions": int(len(train_events)),
        "test_interactions": int(len(test_events)),
    }


def run(settings: Settings) -> Path:  # noqa: D103
    configure_mlflow(settings)

    train_events = _load_dataframe(settings.processed_data_dir / "train_features.csv")
    test_events = _load_dataframe(settings.processed_data_dir / "test_features.csv")

    # C1: mesmo critério de população do treino (ver reco.training.evaluate).
    lookup, eval_audit = filter_lookup_by_history(build_relevance_lookup(test_events), train_events)
    train_items = train_items_by_visitor(train_events)

    all_metrics: dict = {
        model_type.value: _evaluate_one(model_type, settings, lookup, train_items, eval_audit)
        for model_type in (
            ModelType.POPULARITY,
            ModelType.MATRIX_FACTORIZATION,
            ModelType.TWO_TOWER,
        )
    }
    all_metrics["_meta"] = _run_metadata(settings, eval_audit, train_events, test_events)

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = reports_dir / "metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")

    return metrics_path


def main() -> int:  # noqa: D103
    run(Settings())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
