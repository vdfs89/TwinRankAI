from pathlib import Path

import pandas as pd

from reco.models.two_tower import TwoTowerRecommender
from reco.settings import Settings


class PluggableRecommender:
    def __init__(self, products_df: pd.DataFrame, orders_df: pd.DataFrame):
        self.products_df = products_df
        self.orders_df = orders_df

        # Override settings for fast on-the-fly training
        self.settings = Settings(
            max_epochs=15,
            batch_size=32,
            embedding_dim=16,
            early_stopping_patience=3,
            redis_url="redis://localhost:6379/0",
            faiss_index_path=Path("dummy_data/item_index.faiss"),
        )
        self.model = TwoTowerRecommender(self.settings)
        self._prepare_and_train()

    def _prepare_and_train(self):
        # The TwoTowerRecommender expects 'visitorid' and 'itemid'
        train_events = self.orders_df.rename(
            columns={"user_id": "visitorid", "product_id": "itemid"}
        )

        # Train on-the-fly (this builds the FAISS index automatically)
        self.model.fit(train_events)

    def recommend_for_user(self, user_id: str, k: int = 10) -> list[dict]:
        # predict_top_k in TwoTowerRecommender has a type hint of `int` for visitor_id
        # but dynamically accepts the type that was passed in `fit()` (in this case `str`).
        recommended_item_ids = self.model.predict_top_k(user_id, k)

        if not recommended_item_ids:
            return []

        # Build response with product metadata
        recommendations = []
        for pid in recommended_item_ids:
            prod_info = self.products_df[self.products_df["product_id"] == pid]
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
