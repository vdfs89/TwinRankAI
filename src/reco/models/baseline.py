"""Baselines para comparação justa com o modelo two-tower.

PopularityRecommender é o piso mínimo (não personaliza).
MatrixFactorizationRecommender via SVD é o baseline clássico de recsys.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


class PopularityRecommender:
    """Recomenda sempre os itens com maior soma de relevância (não-personalizado)."""

    def __init__(self) -> None:
        self._ranked_items: list[int] = []

    def fit(self, train_events: pd.DataFrame) -> None:
        """Ordena itens pela soma de relevância acumulada em todo o dataset."""
        scores = train_events.groupby("itemid")["relevance"].sum()
        self._ranked_items = scores.sort_values(ascending=False).index.tolist()

    def predict_top_k(self, visitor_id: int, k: int) -> list[int]:
        """Ignora o visitor_id (não-personalizado) e retorna o top-k global."""
        return self._ranked_items[:k]

    def save(self, path: str) -> None:
        """Persiste a lista ranqueada de itens em disco."""
        np.save(path, np.array(self._ranked_items))

    def load(self, path: str) -> None:
        """Carrega a lista ranqueada de itens do disco."""
        self._ranked_items = np.load(path).tolist()


class MatrixFactorizationRecommender:
    """Baseline clássico: SVD truncado sobre a matriz visitor-item esparsa."""

    def __init__(self, embedding_dim: int = 64) -> None:
        self._embedding_dim = embedding_dim
        self._svd = TruncatedSVD(n_components=embedding_dim, random_state=42)
        self._visitor_index: dict[int, int] = {}
        self._item_index: dict[int, int] = {}
        self._index_to_item: dict[int, int] = {}
        self._item_factors: np.ndarray | None = None
        self._visitor_factors: np.ndarray | None = None

    def fit(self, train_events: pd.DataFrame) -> None:
        """Constrói a matriz esparsa visitor-item e ajusta o SVD."""
        visitors = train_events["visitorid"].unique()
        items = train_events["itemid"].unique()
        self._visitor_index = {v: i for i, v in enumerate(visitors)}
        self._item_index = {it: i for i, it in enumerate(items)}
        self._index_to_item = {i: it for it, i in self._item_index.items()}

        rows = train_events["visitorid"].map(self._visitor_index)
        cols = train_events["itemid"].map(self._item_index)
        matrix = csr_matrix(
            (train_events["relevance"], (rows, cols)),
            shape=(len(visitors), len(items)),
        )
        self._visitor_factors = self._svd.fit_transform(matrix)
        self._item_factors = self._svd.components_.T

    def predict_top_k(self, visitor_id: int, k: int) -> list[int]:
        """Calcula scores via produto interno entre fator do visitante e itens."""
        if visitor_id not in self._visitor_index or self._item_factors is None:
            return []
        v_idx = self._visitor_index[visitor_id]
        scores = self._item_factors @ self._visitor_factors[v_idx]
        top_indices = np.argsort(-scores)[:k]
        return [self._index_to_item[i] for i in top_indices]

    def save(self, path: str) -> None:
        """Persiste os fatores latentes em disco."""
        np.savez(path, visitor_factors=self._visitor_factors, item_factors=self._item_factors)

    def load(self, path: str) -> None:
        """Carrega os fatores latentes do disco."""
        data = np.load(path)
        self._visitor_factors = data["visitor_factors"]
        self._item_factors = data["item_factors"]
