"""Valida que o ambiente está pronto para rodar o pipeline.

Uso:
    poetry run python scripts/validate_env.py
"""

from __future__ import annotations

import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


REQUIRED_PACKAGES = [
    "torch",
    "scikit-learn",
    "mlflow",
    "pandera",
    "fastapi",
    "pydantic",
    "pydantic-settings",
]


def _env_path(name: str, default: str) -> Path:
    return Path(os.getenv(name, default))


def check_packages() -> list[str]:
    """Retorna lista de pacotes obrigatórios ausentes."""
    missing: list[str] = []
    for pkg in REQUIRED_PACKAGES:
        try:
            version(pkg)
        except PackageNotFoundError:
            missing.append(pkg)
    return missing


def check_environment() -> bool:
    """Verifica o diretório de dados brutos."""
    raw_data_dir = _env_path("RAW_DATA_DIR", "data/raw")
    return raw_data_dir.exists()


def main() -> int:
    missing_packages = check_packages()
    environment_ok = check_environment()

    if missing_packages:
        print("error missing_packages=" + ",".join(sorted(missing_packages)))
        return 1

    if not environment_ok:
        print(
            "warning raw_data_dir_missing path=data/raw hint="
            "kaggle datasets download -d retailrocket/ecommerce-dataset",
        )
        return 0

    print("info environment_validated_successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
