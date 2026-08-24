"""Interface comum a todos os modelos de recomendação.

Qualquer implementação (baseline ou neural) deve seguir este contrato,
o que permite ao Factory e ao pipeline de avaliação tratá-las de forma
polimórfica.
"""

from typing import Protocol

import pandas as pd


class RecommenderModel(Protocol):
    """Contrato: todo recomendador sabe treinar e retornar top-K itens."""

    def fit(self, train_events: pd.DataFrame) -> None:
        """Treina o modelo a partir de eventos pré-processados (com 'relevance')."""
        ...

    def predict_top_k(self, visitor_id: int, k: int) -> list[int]:
        """Retorna os k itens de maior score previsto para o visitante."""
        ...

    def save(self, path: str) -> None:
        """Save the trained model to disk."""
        ...

    def load(self, path: str) -> None:
        """Carrega um modelo treinado do disco."""
        ...


def native_id(value: object) -> object:
    """Normaliza ids numpy (`numpy.int64`) para tipos nativos de Python.

    `numpy.int64` não é serializável por `json.dumps`, o que já derrubou a
    gravação no cache Redis com `TypeError`. A normalização preserva ids
    alfanuméricos intactos (`"P001"`), usados pelo demo plugável.

    Args:
    ----
        value: id de item vindo do índice do modelo.

    Returns:
    -------
        `int` quando o valor é numérico, o próprio valor caso contrário.

    """
    if isinstance(value, str):
        return value
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return value
