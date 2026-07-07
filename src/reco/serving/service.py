"""Carregamento e consulta do modelo treinado."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from reco.models.factory import ModelType, create_model
from reco.settings import Settings


class RecommendationService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = create_model(ModelType.TWO_TOWER, settings)
        self._load_model_if_available()

    def _load_model_if_available(self) -> None:
        model_path = self._settings.model_path
        if Path(model_path).exists():
            self._model.load(str(model_path))

    def recommend(self, user_id: int, top_k: int) -> list[int]:
        return self._model.predict_top_k(user_id, top_k)

    def predict(
        self,
        user_id: int,
        candidate_item_ids: list[int],
        top_k: int,
    ) -> list[int]:
        """Filtra recomendações por um conjunto opcional de candidatos."""
        if not candidate_item_ids:
            return self.recommend(user_id, top_k)

        candidate_total = max(top_k, len(candidate_item_ids))
        recommended = self.recommend(user_id, candidate_total)
        candidate_set = set(candidate_item_ids)

        filtered = [
            item_id
            for item_id in recommended
            if item_id in candidate_set
        ]
        return filtered[:top_k]


@lru_cache(maxsize=1)
def get_recommendation_service(settings: Settings) -> RecommendationService:
    return RecommendationService(settings)
