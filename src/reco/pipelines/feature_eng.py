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


def run(settings: Settings) -> FeatureArtifacts:  # noqa: D103
    train_path = settings.processed_data_dir / "train_events.csv"
    test_path = settings.processed_data_dir / "test_events.csv"

    train_events = _load_events(train_path)
    test_events = _load_events(test_path)
    visitor_mapping, item_mapping = build_id_mappings(train_events)

    train_features = add_feature_columns(
        train_events,
        visitor_mapping,
        item_mapping,
    )
    test_features = add_feature_columns(
        test_events,
        visitor_mapping,
        item_mapping,
    )

    settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
    train_features_path = settings.processed_data_dir / "train_features.csv"
    test_features_path = settings.processed_data_dir / "test_features.csv"
    visitor_mapping_path = settings.processed_data_dir / "visitor_mapping.json"
    item_mapping_path = settings.processed_data_dir / "item_mapping.json"

    train_features.to_csv(train_features_path, index=False)
    test_features.to_csv(test_features_path, index=False)
    visitor_mapping_path.write_text(
        json.dumps(visitor_mapping, indent=2),
        encoding="utf-8",
    )
    item_mapping_path.write_text(
        json.dumps(item_mapping, indent=2),
        encoding="utf-8",
    )

    logger.info(
        "feature_engineering_concluido",
        train_features=str(train_features_path),
        test_features=str(test_features_path),
    )
    return FeatureArtifacts(
        train_features_path=train_features_path,
        test_features_path=test_features_path,
        visitor_mapping_path=visitor_mapping_path,
        item_mapping_path=item_mapping_path,
    )


def main() -> int:  # noqa: D103
    run(Settings())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
