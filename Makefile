.PHONY: install validate lint test train serve dvc-repro mlflow-ui

install:
	poetry install
	poetry run pre-commit install

validate:
	poetry run python scripts/validate_env.py

lint:
	poetry run ruff check src tests scripts
	poetry run ruff format --check src tests scripts

test:
	poetry run pytest

train:
	poetry run python -m reco.training.train

serve:
	poetry run uvicorn reco.serving.api:app --reload --host 0.0.0.0 --port 8000

dvc-repro:
	poetry run dvc repro

mlflow-ui:
	poetry run mlflow ui --backend-store-uri ./mlruns

mlflow-ui-local:
	poetry run mlflow ui --backend-store-uri ./mlruns
