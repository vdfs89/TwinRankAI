"""Métricas de avaliação para sistemas de recomendação (ranking).

RMSE/acurácia não fazem sentido aqui — feedback é implícito, então
usamos métricas de ranking top-K: Recall, NDCG, MRR e MAP.
"""

import numpy as np
import pandas as pd

from reco.models.base import RecommenderModel

# Critério único de população, compartilhado entre treino e avaliação:
# só visitantes com pelo menos este número de interações no TREINO entram.
# Sem ele, 57% dos visitantes avaliados têm 1 única interação de treino e
# dominam a média, tornando a métrica incomparável com a literatura.
MIN_TRAIN_INTERACTIONS = 5


def build_relevance_lookup(test_events: pd.DataFrame) -> dict[int, dict[int, float]]:
    """Monta {visitor_id: {item_id: relevancia}} a partir dos eventos de teste.

    Pares (visitor, item) repetidos são deduplicados pela relevância MÁXIMA:
    se o usuário deu `view` (1.0) e depois `transaction` (5.0) no mesmo item,
    o par vale 5.0. Antes desta correção prevalecia a última linha lida, o que
    podia rebaixar uma compra a uma visualização no cálculo do NDCG.
    """
    grouped = test_events.groupby(["visitorid", "itemid"])["relevance"].max()
    lookup: dict[int, dict[int, float]] = {}
    for (visitor_id, item_id), relevance in grouped.items():
        lookup.setdefault(int(visitor_id), {})[int(item_id)] = float(relevance)
    return lookup


def _as_visitor_id(value: object) -> object:
    """Normaliza o id para int quando possível, preservando ids não numéricos.

    O dataset RetailRocket usa ids inteiros, mas o demo plugável aceita
    planilhas com ids alfanuméricos (`U01`). Forçar `int()` quebrava esse caso
    com `ValueError`.
    """
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return value


def eligible_visitors(
    train_events: pd.DataFrame,
    min_interactions: int = MIN_TRAIN_INTERACTIONS,
) -> set[object]:
    """Visitantes com histórico de treino suficiente para serem modelados/avaliados."""
    counts = train_events.groupby("visitorid")["itemid"].nunique()
    return {_as_visitor_id(v) for v in counts[counts >= min_interactions].index}


def filter_lookup_by_history(
    lookup: dict[int, dict[int, float]],
    train_events: pd.DataFrame,
    min_interactions: int = MIN_TRAIN_INTERACTIONS,
) -> tuple[dict[int, dict[int, float]], dict[str, int]]:
    """Restringe o lookup aos visitantes elegíveis; devolve também a auditoria."""
    eligible = eligible_visitors(train_events, min_interactions)
    filtered = {v: rel for v, rel in lookup.items() if v in eligible}
    audit = {
        "eval_min_train_interactions": min_interactions,
        "eval_visitors_before_filter": len(lookup),
        "eval_visitors_after_filter": len(filtered),
    }
    return filtered, audit


def train_items_by_visitor(train_events: pd.DataFrame) -> dict[int, set[int]]:
    """Itens já vistos por cada visitante no treino.

    Usado para separar, na avaliação, descoberta real (item novo) de
    repetição (item recomendado que o visitante já tinha visto/comprado).
    """
    grouped = train_events.groupby("visitorid")["itemid"].apply(set)
    return {int(v): items for v, items in grouped.items()}


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


def evaluate_novel_recall_at_k(
    model: RecommenderModel,
    test_events_by_visitor: dict[int, dict[int, float]],
    train_items_by_visitor_map: dict[int, set[int]],
    k: int,
) -> float:
    """Recall@k restrito a itens não vistos pelo visitante no treino.

    Mede descoberta pura: exclui do conjunto de relevantes qualquer item já
    presente no histórico de treino, isolando o quanto do recall_at_k comum
    vem de repetição (recomendar de volta o que o visitante já viu/comprou).
    """
    recalls = []
    for visitor_id, relevance_map in test_events_by_visitor.items():
        seen = train_items_by_visitor_map.get(visitor_id, set())
        novel_relevant = set(relevance_map.keys()) - seen
        recommended = model.predict_top_k(visitor_id, k)
        recalls.append(recall_at_k(recommended, novel_relevant, k))
    return float(np.mean(recalls)) if recalls else 0.0
