# pyright: reportMissingTypeStubs=false, reportMissingImports=false
"""Pipeline de pré-processamento dos dados do RetailRocket."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

from reco.data.preprocessing import ImplicitFeedbackWeightingStrategy
from reco.data.schema import validate_events
from reco.settings import Settings
from reco.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class PreprocessArtifacts:  # noqa: D101
    train_path: Path
    test_path: Path
    processed_events_path: Path


def _raw_events_path(settings: Settings) -> Path:
    return settings.raw_data_dir / settings.events_file


def load_events(settings: Settings) -> pd.DataFrame:  # noqa: D103
    path = _raw_events_path(settings)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo bruto não encontrado: {path}")
    return pd.read_csv(path)


def preprocess_events(  # noqa: D103
    events: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    validated = validate_events(events)
    strategy = ImplicitFeedbackWeightingStrategy(settings)
    enriched = strategy.transform(validated)
    enriched["timestamp"] = pd.to_datetime(
        enriched["timestamp"],
        unit="ms",
        errors="coerce",
    )
    enriched = enriched.dropna(subset=["timestamp"])
    enriched = enriched.sort_values(
        ["visitorid", "timestamp", "itemid"],
    ).reset_index(drop=True)
    logger.info("events_preprocessados", rows=len(enriched))
    return enriched


def split_train_test(  # noqa: D103
    events: pd.DataFrame,
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = events.sort_values(
        ["timestamp", "visitorid", "itemid"],
    ).reset_index(drop=True)
    cutoff = max(1, int(len(ordered) * (1 - test_fraction)))
    train = ordered.iloc[:cutoff].copy()
    test = ordered.iloc[cutoff:].copy()
    return train, test


def _write_splits(
    settings: Settings,
    processed: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> PreprocessArtifacts:
    """Grava o conjunto processado e os splits, devolvendo os caminhos."""
    settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
    artifacts = PreprocessArtifacts(
        processed_events_path=settings.processed_data_dir / "events_processed.csv",
        train_path=settings.processed_data_dir / "train_events.csv",
        test_path=settings.processed_data_dir / "test_events.csv",
    )
    processed.to_csv(artifacts.processed_events_path, index=False)
    train.to_csv(artifacts.train_path, index=False)
    test.to_csv(artifacts.test_path, index=False)
    return artifacts


def run(settings: Settings) -> PreprocessArtifacts:  # noqa: D103
    processed = preprocess_events(load_events(settings), settings)
    train, test = split_train_test(processed)
    artifacts = _write_splits(settings, processed, train, test)

    logger.info(
        "preprocess_concluido",
        processed_events=str(artifacts.processed_events_path),
        train_events=str(artifacts.train_path),
        test_events=str(artifacts.test_path),
    )
    return artifacts


def main() -> int:  # noqa: D103
    run(Settings())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
