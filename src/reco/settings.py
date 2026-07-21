"""Configurações centralizadas do projeto, carregadas de variáveis de ambiente.

Usa Pydantic Settings para validar tipos e falhar rápido (fail-fast) se uma
variável obrigatória estiver ausente ou com tipo incorreto.
"""

from pathlib import Path

from pydantic import Field
from pydantic.v1 import (  # type: ignore[attr-defined]
    BaseSettings as PydanticBaseSettings,
)


class Settings(PydanticBaseSettings):
    """Configurações de ambiente, dados, modelo e serviço."""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    # Ambiente
    env: str = Field(default="development", alias="ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    random_seed: int = Field(default=42, alias="RANDOM_SEED")

    # Dados
    raw_data_dir: Path = Field(default=Path("data/raw"), alias="RAW_DATA_DIR")
    processed_data_dir: Path = Field(
        default=Path("data/processed"),
        alias="PROCESSED_DATA_DIR",
    )
    models_dir: Path = Field(default=Path("models"), alias="MODELS_DIR")
    events_file: str = Field(default="events.csv", alias="EVENTS_FILE")
    item_properties_file: str = Field(
        default="item_properties.csv",
        alias="ITEM_PROPERTIES_FILE",
    )
    category_tree_file: str = Field(
        default="category_tree.csv",
        alias="CATEGORY_TREE_FILE",
    )

    # Pesos de feedback implícito
    weight_view: float = Field(default=1.0, alias="WEIGHT_VIEW")
    weight_addtocart: float = Field(default=3.0, alias="WEIGHT_ADDTOCART")
    weight_transaction: float = Field(default=5.0, alias="WEIGHT_TRANSACTION")

    # Modelo
    embedding_dim: int = Field(default=64, alias="EMBEDDING_DIM")
    negative_samples_per_positive: int = Field(
        default=4,
        alias="NEGATIVE_SAMPLES_PER_POSITIVE",
    )
    learning_rate: float = Field(default=1e-3, alias="LEARNING_RATE")
    batch_size: int = Field(default=1024, alias="BATCH_SIZE")
    max_epochs: int = Field(default=50, alias="MAX_EPOCHS")
    early_stopping_patience: int = Field(
        default=5,
        alias="EARLY_STOPPING_PATIENCE",
    )

    # Avaliação
    top_k: int = Field(default=10, alias="TOP_K")

    # MLflow
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000",
        alias="MLFLOW_TRACKING_URI",
    )
    mlflow_experiment_name: str = Field(
        default="retailrocket-two-tower", alias="MLFLOW_EXPERIMENT_NAME"
    )
    mlflow_registered_model_name: str = Field(
        default="twinrank-ai-two-tower", alias="MLFLOW_REGISTERED_MODEL_NAME"
    )

    # API
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    model_stage: str = Field(default="Production", alias="MODEL_STAGE")
    model_path: Path = Field(
        default=Path("models/two_tower/model.joblib"),
        alias="MODEL_PATH",
    )

    # Cache & Retrieval
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        alias="REDIS_URL",
    )
    faiss_index_path: Path = Field(
        default=Path("models/two_tower/item_index.faiss"),
        alias="FAISS_INDEX_PATH",
    )


def get_settings() -> Settings:
    """Factory simples para permitir override em testes."""
    return Settings()
