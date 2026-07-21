"""Módulo de demonstração do e-commerce com modelo Two-Tower."""

from pathlib import Path

import pandas as pd
from src.reco.models.two_tower import TwoTowerRecommender
from src.reco.settings import Settings


def load_demo_data(products_csv: str, orders_csv: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega os DataFrames de Produtos e Pedidos a partir de arquivos CSV."""
    prod_df = pd.read_csv(products_csv)
    ord_df = pd.read_csv(orders_csv)
    return prod_df, ord_df


def train_demo_model(orders_df: pd.DataFrame) -> TwoTowerRecommender:
    """Treina o modelo TwinRank (Two-Tower) localmente na base de pedidos fornecida.

    Gera automaticamente o índice FAISS.
    """
    settings = Settings(
        max_epochs=15,
        batch_size=32,
        embedding_dim=16,
        early_stopping_patience=3,
        redis_url="redis://localhost:6379/0",
        faiss_index_path=Path("dummy_data/item_index.faiss"),
    )
    model = TwoTowerRecommender(settings)

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
