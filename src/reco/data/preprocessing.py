"""Estratégias de pré-processamento de eventos (padrão Strategy).

Permite trocar a forma de atribuir relevância/peso aos eventos implícitos
sem alterar o pipeline de treino ou avaliação (Open/Closed Principle).
"""

from typing import Protocol

import pandas as pd

from reco.settings import Settings


class PreprocessingStrategy(Protocol):
    """Interface comum a qualquer estratégia de preprocessamento de eventos."""

    def transform(self, events: pd.DataFrame) -> pd.DataFrame:
        """Recebe events.csv bruto e retorna DataFrame com coluna 'relevance'."""
        ...


class ImplicitFeedbackWeightingStrategy:
    """Atribui peso de relevância diferenciado por tipo de evento.

    transaction > addtocart > view — usado tanto para negative sampling
    quanto para calcular NDCG ponderado (não apenas binário 0/1).
    """

    def __init__(self, settings: Settings) -> None:
        self._weights = {
            "view": settings.weight_view,
            "addtocart": settings.weight_addtocart,
            "transaction": settings.weight_transaction,
        }

    def transform(self, events: pd.DataFrame) -> pd.DataFrame:
        """Adiciona coluna 'relevance' mapeada a partir do tipo de evento."""
        out = events.copy()
        out["relevance"] = out["event"].map(self._weights)
        return out


class SessionWindowStrategy:
    """Agrupa eventos em sessões por janela de inatividade (para sequência).

    Útil se a Etapa 4 evoluir o two-tower para considerar histórico
    sequencial do usuário, não apenas interações agregadas.
    """

    def __init__(self, session_gap_seconds: int = 1800) -> None:
        self._session_gap_ms = session_gap_seconds * 1000

    def transform(self, events: pd.DataFrame) -> pd.DataFrame:
        """Adiciona coluna 'session_id' baseada em gaps de tempo por visitor."""
        out = events.sort_values(["visitorid", "timestamp"]).copy()
        gap = out.groupby("visitorid")["timestamp"].diff().gt(self._session_gap_ms)
        out["session_id"] = gap.groupby(out["visitorid"]).cumsum()
        return out
