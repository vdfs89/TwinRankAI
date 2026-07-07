"""Métricas de avaliação para sistemas de recomendação (ranking).

RMSE/acurácia não fazem sentido aqui — feedback é implícito, então
usamos métricas de ranking top-K: Recall, NDCG, MRR e MAP.
"""

import numpy as np

from reco.models.base import RecommenderModel


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Fração dos itens relevantes que aparecem no top-k recomendado."""
    if not relevant:
        return 0.0
    hits = len(set(recommended[:k]) & relevant)
    return hits / len(relevant)


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Fração do top-k recomendado que é de fato relevante."""
    if k == 0:
        return 0.0
    hits = len(set(recommended[:k]) & relevant)
    return hits / k


def mrr_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Reciprocal rank do primeiro item relevante encontrado no top-k."""
    for rank, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def average_precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Precisão média sobre as posições em que itens relevantes aparecem."""
    if not relevant:
        return 0.0
    hits = 0
    sum_precisions = 0.0
    for rank, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            hits += 1
            sum_precisions += hits / rank
    return sum_precisions / min(len(relevant), k)


def ndcg_at_k(recommended: list[int], relevance_scores: dict[int, float], k: int) -> float:
    """NDCG com relevância diferenciada (transaction > addtocart > view).

    relevance_scores mapeia itemid -> peso de relevância (0 se não recomendado
    ou não interagido).
    """
    dcg = sum(
        relevance_scores.get(item, 0.0) / np.log2(rank + 1)
        for rank, item in enumerate(recommended[:k], start=1)
    )
    ideal_order = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg = sum(rel / np.log2(rank + 1) for rank, rel in enumerate(ideal_order, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_model(
    model: RecommenderModel,
    test_events_by_visitor: dict[int, dict[int, float]],
    k: int,
) -> dict[str, float]:
    """Avalia o modelo em todos os visitantes do conjunto de teste.

    test_events_by_visitor: {visitor_id: {item_id: relevance_score}}
    Retorna as 4 métricas exigidas, agregadas por média.
    """
    recalls, precisions, mrrs, maps, ndcgs = [], [], [], [], []

    for visitor_id, relevance_map in test_events_by_visitor.items():
        recommended = model.predict_top_k(visitor_id, k)
        relevant_set = set(relevance_map.keys())

        recalls.append(recall_at_k(recommended, relevant_set, k))
        precisions.append(precision_at_k(recommended, relevant_set, k))
        mrrs.append(mrr_at_k(recommended, relevant_set, k))
        maps.append(average_precision_at_k(recommended, relevant_set, k))
        ndcgs.append(ndcg_at_k(recommended, relevance_map, k))

    return {
        f"recall_at_{k}": float(np.mean(recalls)),
        f"precision_at_{k}": float(np.mean(precisions)),
        f"mrr_at_{k}": float(np.mean(mrrs)),
        f"map_at_{k}": float(np.mean(maps)),
        f"ndcg_at_{k}": float(np.mean(ndcgs)),
    }
