"""Factory de modelos de recomendação (padrão Factory).

Desacopla a escolha do modelo (config/CLI) da lógica de treino,
facilitando comparação justa entre baseline e modelo neural.
"""

from enum import Enum

from reco.models.base import RecommenderModel
from reco.settings import Settings


class ModelType(str, Enum):
    """Tipos de modelo suportados pelo pipeline."""

    POPULARITY = "popularity"
    MATRIX_FACTORIZATION = "matrix_factorization"
    TWO_TOWER = "two_tower"


def create_model(model_type: ModelType, settings: Settings) -> RecommenderModel:
    """Instancia o modelo solicitado, isolando os imports pesados (torch etc.)."""
    if model_type is ModelType.POPULARITY:
        from reco.models.baseline import PopularityRecommender

        return PopularityRecommender()

    if model_type is ModelType.MATRIX_FACTORIZATION:
        from reco.models.baseline import MatrixFactorizationRecommender

        return MatrixFactorizationRecommender(embedding_dim=settings.embedding_dim)

    if model_type is ModelType.TWO_TOWER:
        from reco.models.two_tower import TwoTowerRecommender

        return TwoTowerRecommender(settings=settings)

    raise ValueError(f"Tipo de modelo desconhecido: {model_type}")
