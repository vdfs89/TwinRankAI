"""Carregamento e consulta do modelo treinado."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import redis

from reco.models.factory import ModelType, create_model
from reco.settings import Settings
from reco.utils.logging import get_logger

logger = get_logger(__name__)


class RecommendationService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = create_model(ModelType.TWO_TOWER, settings)
        self._load_model_if_available()

        try:
            self._redis = redis.Redis.from_url(
                self._settings.redis_url, decode_responses=True, socket_timeout=2.0
            )
            self._redis.ping()
        except (redis.ConnectionError, redis.TimeoutError):
            logger.warning("redis_unavailable", url=self._settings.redis_url)
            self._redis = None

    def _load_model_if_available(self) -> None:
        model_path = self._settings.model_path
        if Path(model_path).exists():
            self._model.load(str(model_path))

    def recommend(self, user_id: int, top_k: int) -> list[int]:
        cache_key = f"reco:user:{user_id}:k:{top_k}"

        if self._redis:
            try:
                cached = self._redis.get(cache_key)
                if cached:
                    logger.info("cache_hit", user_id=user_id, top_k=top_k)
                    return json.loads(cached)
            except redis.RedisError as e:
                logger.warning("redis_read_error", error=str(e))

        logger.info("cache_miss", user_id=user_id, top_k=top_k)
        recommendations = self._model.predict_top_k(user_id, top_k)

        if self._redis and recommendations:
            try:
                self._redis.setex(cache_key, 3600, json.dumps(recommendations))
            except redis.RedisError as e:
                logger.warning("redis_write_error", error=str(e))

        return recommendations

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

        filtered = [item_id for item_id in recommended if item_id in candidate_set]
        return filtered[:top_k]


@lru_cache(maxsize=1)
def get_recommendation_service(settings: Settings) -> RecommendationService:
    return RecommendationService(settings)
