"""Carregamento e consulta do modelo treinado."""

from __future__ import annotations

import json
from pathlib import Path

import redis

from reco.models.baseline import PopularityRecommender
from reco.models.factory import ModelType, create_model
from reco.settings import Settings
from reco.utils.logging import get_logger

logger = get_logger(__name__)


class RecommendationService:  # noqa: D101
    def __init__(self, settings: Settings) -> None:  # noqa: D107
        self._settings = settings
        self._model = create_model(ModelType.TWO_TOWER, settings)
        self._load_model_if_available()
        self._fallback = self._load_fallback_if_available()

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

    def _load_fallback_if_available(self) -> PopularityRecommender | None:
        """Carrega o baseline de popularidade usado no cold-start.

        O two-tower é embedding de ID puro: um visitante ausente do índice de
        treino não tem vetor e receberia lista vazia. O ranking global de
        popularidade é a resposta degradada aceitável nesse caso.
        """
        fallback_path = self._settings.models_dir / ModelType.POPULARITY.value / "model.joblib"
        if not fallback_path.exists():
            logger.warning("fallback_indisponivel", path=str(fallback_path))
            return None
        model = create_model(ModelType.POPULARITY, self._settings)
        model.load(str(fallback_path))
        return model

    def recommend(self, user_id: int, top_k: int) -> tuple[list[int], str]:
        """Recomenda top-k itens; devolve também a estratégia efetivamente usada."""
        cache_key = f"reco:user:{user_id}:k:{top_k}"

        if self._redis:
            try:
                cached = self._redis.get(cache_key)
                if cached:
                    payload = json.loads(cached)
                    # Entradas gravadas antes da introdução do campo `strategy`
                    # eram uma lista simples; ignoramos essas para não quebrar
                    # em deploy com cache quente.
                    if isinstance(payload, dict):
                        logger.info("cache_hit", user_id=user_id, top_k=top_k)
                        return payload["item_ids"], payload["strategy"]
                    logger.info("cache_formato_antigo_descartado", user_id=user_id)
            except redis.RedisError as e:
                logger.warning("redis_read_error", error=str(e))

        logger.info("cache_miss", user_id=user_id, top_k=top_k)
        # int() explícito: o two-tower devolve numpy.int64 (os ids vêm de
        # `unique()` do pandas), que não é serializável por `json.dumps` e
        # derrubava a gravação no cache com TypeError sempre que o Redis
        # estivesse disponível.
        recommendations = [int(item_id) for item_id in self._model.predict_top_k(user_id, top_k)]
        strategy = ModelType.TWO_TOWER.value

        if not recommendations and self._fallback is not None:
            logger.info("cold_start_fallback", user_id=user_id, top_k=top_k)
            recommendations = [
                int(item_id) for item_id in self._fallback.predict_top_k(user_id, top_k)
            ]
            strategy = "popularity_fallback"

        if not recommendations:
            strategy = "unavailable"

        if self._redis and recommendations:
            try:
                self._redis.setex(
                    cache_key,
                    3600,
                    json.dumps({"item_ids": recommendations, "strategy": strategy}),
                )
            except redis.RedisError as e:
                logger.warning("redis_write_error", error=str(e))

        return recommendations, strategy

    def predict(
        self,
        user_id: int,
        candidate_item_ids: list[int],
        top_k: int,
    ) -> tuple[list[int], str]:
        """Filtra recomendações por um conjunto opcional de candidatos."""
        if not candidate_item_ids:
            return self.recommend(user_id, top_k)

        candidate_total = max(top_k, len(candidate_item_ids))
        recommended, strategy = self.recommend(user_id, candidate_total)
        candidate_set = set(candidate_item_ids)

        filtered = [item_id for item_id in recommended if item_id in candidate_set]
        return filtered[:top_k], strategy


# Singleton de módulo em vez de lru_cache: `Settings` é BaseSettings do
# pydantic v1 e não é hashable, então `lru_cache` sobre uma função que recebe
# Settings levanta TypeError na primeira chamada — o que derrubava /recommend
# com 500. Migrar Settings para pydantic v2 resolveria a raiz, mas está fora
# de escopo e segue registrado como dívida técnica.
_service_instance: RecommendationService | None = None


def get_recommendation_service(settings: Settings) -> RecommendationService:  # noqa: D103
    global _service_instance  # noqa: PLW0603
    if _service_instance is None:
        _service_instance = RecommendationService(settings)
    return _service_instance


def reset_recommendation_service() -> None:
    """Descarta o serviço em cache para que o próximo acesso recarregue o modelo."""
    global _service_instance  # noqa: PLW0603
    _service_instance = None
