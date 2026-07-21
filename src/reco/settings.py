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
    env: str = Field(default="development", env="ENV")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    random_seed: int = Field(default=42, env="RANDOM_SEED")

    # Dados
    raw_data_dir: Path = Field(default=Path("data/raw"), env="RAW_DATA_DIR")
    processed_data_dir: Path = Field(
        default=Path("data/processed"),
        env="PROCESSED_DATA_DIR",
    )
    models_dir: Path = Field(default=Path("models"), env="MODELS_DIR")
    events_file: str = Field(default="events.csv", env="EVENTS_FILE")
    item_properties_file: str = Field(
        default="item_properties.csv",
        env="ITEM_PROPERTIES_FILE",
    )
    category_tree_file: str = Field(
        default="category_tree.csv",
        env="CATEGORY_TREE_FILE",
    )

    # Pesos de feedback implícito
    weight_view: float = Field(default=1.0, env="WEIGHT_VIEW")
    weight_addtocart: float = Field(default=3.0, env="WEIGHT_ADDTOCART")
    weight_transaction: float = Field(default=5.0, env="WEIGHT_TRANSACTION")

    # Modelo
    embedding_dim: int = Field(default=64, env="EMBEDDING_DIM")
    negative_samples_per_positive: int = Field(
        default=4,
        env="NEGATIVE_SAMPLES_PER_POSITIVE",
    )
    learning_rate: float = Field(default=1e-3, env="LEARNING_RATE")
    batch_size: int = Field(default=1024, env="BATCH_SIZE")
    max_epochs: int = Field(default=50, env="MAX_EPOCHS")
    early_stopping_patience: int = Field(
        default=5,
        env="EARLY_STOPPING_PATIENCE",
    )

    # Avaliação
    top_k: int = Field(default=10, env="TOP_K")

    # MLflow
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000",
        env="MLFLOW_TRACKING_URI",
    )
    mlflow_experiment_name: str = Field(
        default="retailrocket-two-tower", env="MLFLOW_EXPERIMENT_NAME"
    )
    mlflow_registered_model_name: str = Field(
        default="twinrank-ai-two-tower", env="MLFLOW_REGISTERED_MODEL_NAME"
    )

    # API
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    model_stage: str = Field(default="Production", env="MODEL_STAGE")
    model_path: Path = Field(
        default=Path("models/two_tower/model.joblib"),
        env="MODEL_PATH",
    )

    # Cache & Retrieval
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        env="REDIS_URL",
    )
    faiss_index_path: Path = Field(
        default=Path("models/two_tower/item_index.faiss"),
        env="FAISS_INDEX_PATH",
    )


def get_settings() -> Settings:
    """Factory simples para permitir override em testes."""
    return Settings()
