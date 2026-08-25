# pyright: reportMissingTypeStubs=false, reportMissingImports=false
"""Pipeline de feature engineering e criação de índices estáveis."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

from reco.settings import Settings
from reco.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class FeatureArtifacts:  # noqa: D101
    train_features_path: Path
    test_features_path: Path
    visitor_mapping_path: Path
    item_mapping_path: Path


def _load_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return pd.read_csv(path, parse_dates=["timestamp"])


def build_id_mappings(  # noqa: D103
    train_events: pd.DataFrame,
) -> tuple[dict[int, int], dict[int, int]]:
    visitors = sorted(train_events["visitorid"].unique().tolist())
    items = sorted(train_events["itemid"].unique().tolist())
    visitor_mapping = {int(visitor): index for index, visitor in enumerate(visitors)}
    item_mapping = {int(item): index for index, item in enumerate(items)}
    return visitor_mapping, item_mapping


def add_feature_columns(  # noqa: D103
    events: pd.DataFrame,
    visitor_mapping: dict[int, int],
    item_mapping: dict[int, int],
) -> pd.DataFrame:
    featured = events.copy()
    featured["visitor_index"] = featured["visitorid"].map(visitor_mapping)
    featured["item_index"] = featured["itemid"].map(item_mapping)
    featured = featured.dropna(subset=["visitor_index", "item_index"])
    featured["visitor_index"] = featured["visitor_index"].astype(int)
    featured["item_index"] = featured["item_index"].astype(int)
    return featured


def _persist(
    settings: Settings,
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    visitor_mapping: dict,
    item_mapping: dict,
) -> FeatureArtifacts:
    """Grava features e mapeamentos de id, devolvendo os caminhos gerados."""
    settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
    artifacts = FeatureArtifacts(
        train_features_path=settings.processed_data_dir / "train_features.csv",
        test_features_path=settings.processed_data_dir / "test_features.csv",
        visitor_mapping_path=settings.processed_data_dir / "visitor_mapping.json",
        item_mapping_path=settings.processed_data_dir / "item_mapping.json",
    )

    train_features.to_csv(artifacts.train_features_path, index=False)
    test_features.to_csv(artifacts.test_features_path, index=False)
    artifacts.visitor_mapping_path.write_text(
        json.dumps(visitor_mapping, indent=2), encoding="utf-8"
    )
    artifacts.item_mapping_path.write_text(json.dumps(item_mapping, indent=2), encoding="utf-8")
    return artifacts


def run(settings: Settings) -> FeatureArtifacts:  # noqa: D103
    train_events = _load_events(settings.processed_data_dir / "train_events.csv")
    test_events = _load_events(settings.processed_data_dir / "test_events.csv")

    # Mapeamentos derivados só do treino: ids vistos apenas no teste não podem
    # influenciar o espaço de embeddings aprendido.
    visitor_mapping, item_mapping = build_id_mappings(train_events)

    artifacts = _persist(
        settings,
        add_feature_columns(train_events, visitor_mapping, item_mapping),
        add_feature_columns(test_events, visitor_mapping, item_mapping),
        visitor_mapping,
        item_mapping,
    )

    logger.info(
        "feature_engineering_concluido",
        train_features=str(artifacts.train_features_path),
        test_features=str(artifacts.test_features_path),
    )
    return artifacts


def main() -> int:  # noqa: D103
    run(Settings())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
