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
            int(row.itemid): float(row.relevance) for row in group.itertuples(index=False)
        }
    return lookup


def run(settings: Settings) -> Path:  # noqa: D103
    configure_mlflow(settings)

    test_path = settings.processed_data_dir / "test_features.csv"
    test_events = _load_dataframe(test_path)
    lookup = _build_lookup(test_events)

    all_metrics = {}

    for model_type in (ModelType.POPULARITY, ModelType.MATRIX_FACTORIZATION, ModelType.TWO_TOWER):
        model = create_model(model_type, settings)
        model_file = settings.models_dir / model_type.value / "model.joblib"
        model.load(str(model_file))

        metrics = evaluate_model(model, lookup, settings.top_k)
        all_metrics[model_type.value] = metrics

        with mlflow.start_run(run_name=f"evaluate_{model_type.value}"):
            mlflow.log_metrics(metrics)

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
