# pyright: reportMissingTypeStubs=false, reportMissingImports=false
"""Pipeline de avaliação final do modelo treinado."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow  # type: ignore[import-untyped]
import pandas as pd  # type: ignore[import-untyped]

from reco.models.factory import ModelType, create_model
from reco.settings import Settings
from reco.training.evaluate import evaluate_model
from reco.training.mlflow_utils import configure_mlflow


def _load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return pd.read_csv(path, parse_dates=["timestamp"])


def _build_lookup(test_events: pd.DataFrame) -> dict[int, dict[int, float]]:
    lookup: dict[int, dict[int, float]] = {}
    for visitor_id, group in test_events.groupby("visitorid"):
        lookup[int(visitor_id)] = {
            int(row.itemid): float(row.relevance)
            for row in group.itertuples(index=False)
        }
    return lookup


def run(settings: Settings) -> Path:
    configure_mlflow(settings)

    model = create_model(ModelType.TWO_TOWER, settings)
    model.load(str(settings.model_path))

    test_path = settings.processed_data_dir / "test_features.csv"
    test_events = _load_dataframe(test_path)
    metrics = evaluate_model(model, _build_lookup(test_events), settings.top_k)

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = reports_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    with mlflow.start_run(run_name="evaluate"):
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(metrics_path), artifact_path="reports")

    return metrics_path


def main() -> int:
    run(Settings())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
