"""Smoke tests para as métricas de ranking."""

from reco.training.evaluate import (
    average_precision_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_recall_at_k_hit_completo() -> None:
    """Quando todos os itens relevantes estão no top-k, recall deve ser 1.0."""
    assert recall_at_k([1, 2, 3], {1, 2}, k=3) == 1.0


def test_recall_at_k_sem_relevantes() -> None:
    """Sem itens relevantes, recall deve ser 0.0 (evita divisão por zero)."""
    assert recall_at_k([1, 2, 3], set(), k=3) == 0.0


def test_precision_at_k() -> None:
    """1 de 3 recomendados é relevante -> precision = 1/3."""
    assert abs(precision_at_k([1, 2, 3], {1}, k=3) - (1 / 3)) < 1e-9


def test_mrr_at_k_primeiro_item() -> None:
    """Item relevante na primeira posição -> MRR = 1.0."""
    assert mrr_at_k([5, 1, 2], {5}, k=3) == 1.0


def test_mrr_at_k_sem_hit() -> None:
    """Nenhum item relevante no top-k -> MRR = 0.0."""
    assert mrr_at_k([1, 2, 3], {99}, k=3) == 0.0


def test_average_precision_at_k() -> None:
    """AP considera a precisão em cada posição de acerto."""
    ap = average_precision_at_k([1, 2, 3], {1, 3}, k=3)
    assert 0.0 < ap <= 1.0


def test_ndcg_at_k_ordem_ideal() -> None:
    """Quando a ordem recomendada é a ideal, NDCG deve ser 1.0."""
    relevance = {1: 5.0, 2: 3.0, 3: 1.0}
    assert abs(ndcg_at_k([1, 2, 3], relevance, k=3) - 1.0) < 1e-9


def test_ndcg_at_k_sem_relevancia() -> None:
    """Sem nenhum item relevante, NDCG deve ser 0.0."""
    assert ndcg_at_k([1, 2, 3], {}, k=3) == 0.0
