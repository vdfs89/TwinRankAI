"""Módulo de demonstração do e-commerce com modelo Two-Tower."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from reco.models.two_tower import TwoTowerRecommender


@dataclass
class DemoConfig:
    """Configuração mínima isolada para o demo, sem dependência do ambiente global."""

    random_seed: int = 42
    embedding_dim: int = 16
    negative_samples_per_positive: int = 4
    learning_rate: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 15
    early_stopping_patience: int = 3
    faiss_index_path: Path = Path("dummy_data/item_index.faiss")


def load_demo_data(products_csv: object, orders_csv: object) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega os DataFrames de Produtos e Pedidos a partir de arquivos CSV ou UploadedFile."""
    # pd.read_csv handles both str/Path and file-like objects (UploadedFile)
    prod_df = pd.read_csv(products_csv)
    ord_df = pd.read_csv(orders_csv)

    # Padroniza coluna de usuário se for do dataset RetailRocket original
    if "visitorid" in ord_df.columns and "user_id" not in ord_df.columns:
        ord_df = ord_df.rename(columns={"visitorid": "user_id"})

    return prod_df, ord_df


def train_demo_model(products_df: pd.DataFrame, orders_df: pd.DataFrame) -> TwoTowerRecommender:
    """Treina o modelo TwinRank (Two-Tower) localmente na base de pedidos fornecida.

    Gera automaticamente o índice FAISS.
    """
    settings = DemoConfig()
    model = TwoTowerRecommender(settings)  # type: ignore

    # The TwoTowerRecommender expects 'visitorid' and 'itemid'
    train_events = orders_df.rename(columns={"user_id": "visitorid", "product_id": "itemid"})

    # Train on-the-fly (this builds the FAISS index automatically)
    model.fit(train_events)
    return model


def recommend_for_user(
    model: TwoTowerRecommender, products_df: pd.DataFrame, user_id: str, k: int = 10
) -> list[dict]:
    """Gera as recomendações top-K para um usuário e enriquece com os metadados dos produtos."""
    recommended_item_ids = model.predict_top_k(user_id, k)

    if not recommended_item_ids:
        return []

    # Build response with product metadata
    recommendations = []
    for pid in recommended_item_ids:
        prod_info = products_df[products_df["product_id"] == pid]
        if not prod_info.empty:
            row = prod_info.iloc[0]
            recommendations.append(
                {
                    "product_id": str(pid),
                    "name": str(row.get("name", "Unknown")),
                    "category": str(row.get("category", "Unknown")),
                    "price": float(row.get("price", 0.0)),
                }
            )
    return recommendations
