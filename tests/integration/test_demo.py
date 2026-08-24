"""Testes do demo plugável (upload de CSV pelo usuário no Streamlit).

O demo treina um two-tower on-the-fly sobre uma planilha arbitrária, e por isso
exercita caminhos que o pipeline principal nunca toca: ids alfanuméricos,
bases minúsculas e ausência da coluna `relevance`. Cinco regressões chegaram à
produção por não haver cobertura aqui.
"""

from __future__ import annotations

import pandas as pd
import pytest

from reco.demo.ecommerce_demo import (
    DemoConfig,
    load_demo_data,
    recommend_for_user,
    train_demo_model,
    with_relevance,
)
from reco.models.two_tower import TwoTowerRecommender

PRODUCTS = "dummy_data/products_sample.csv"
ORDERS = "dummy_data/orders_sample.csv"


@pytest.fixture(scope="module")
def demo_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Planilhas de exemplo distribuídas com o projeto."""
    return load_demo_data(PRODUCTS, ORDERS)


def test_with_relevance_usa_pesos_do_projeto_quando_ha_event_type() -> None:
    """Com `event_type`, aplica os mesmos pesos do pré-processamento."""
    events = pd.DataFrame(
        {
            "visitorid": ["U1", "U1", "U1"],
            "itemid": ["P1", "P2", "P3"],
            "event_type": ["view", "addtocart", "transaction"],
        }
    )

    assert with_relevance(events)["relevance"].tolist() == [1.0, 3.0, 5.0]


def test_with_relevance_assume_peso_unitario_sem_event_type() -> None:
    """Planilha de pedidos comum não tem tipo de evento; toda linha vale 1,0."""
    events = pd.DataFrame({"visitorid": ["U1", "U2"], "itemid": ["P1", "P2"]})

    assert with_relevance(events)["relevance"].tolist() == [1.0, 1.0]


def test_demo_treina_e_recomenda_com_ids_alfanumericos(
    demo_data: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Regressão: o demo quebrava antes de chegar a produzir recomendações.

    Cinco causas encadeadas: coluna `relevance` ausente, corte de 5 interações
    mínimas contra uma base cujo maior usuário tem 4, `int()` sobre ids
    `U01`/`P001` em dois pontos distintos, e `DemoConfig` sem
    `dataloader_workers`. Este teste falha se qualquer uma delas voltar.
    """
    products_df, orders_df = demo_data
    model = train_demo_model(products_df, orders_df)

    user_id = orders_df["user_id"].iloc[0]
    recommendations = recommend_for_user(model, products_df, user_id, k=3)

    assert recommendations, "o demo deve produzir recomendações para um usuário do arquivo"
    assert len(recommendations) <= 3
    for item in recommendations:
        assert item["name"]
        assert isinstance(item["price"], float)


def test_fit_aceita_eventos_sem_coluna_relevance() -> None:
    """`fit` não deve exigir `relevance` de uma planilha arbitrária.

    O pipeline principal sempre cria essa coluna no pré-processamento, mas o
    demo recebe CSV do usuário. Sem esse fallback, o treino quebra com
    `KeyError: 'relevance'` — o erro que apareceu em produção.
    """
    events = pd.DataFrame(
        {
            "visitorid": ["U1", "U1", "U1", "U2", "U2", "U2"],
            "itemid": ["P1", "P2", "P3", "P2", "P3", "P4"],
        }
    )

    model = TwoTowerRecommender(DemoConfig())  # type: ignore[arg-type]
    model.fit(events)

    assert model.predict_top_k("U1", 2)
