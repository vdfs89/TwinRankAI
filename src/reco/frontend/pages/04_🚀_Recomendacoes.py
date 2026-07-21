"""Pluggable Recommendations Demo for small e-commerce stores with CSV uploads."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(BASE_DIR / "src"))
from reco.demo.ecommerce_demo import (  # noqa: E402
    load_demo_data,
    recommend_for_user,
    train_demo_model,
)
from reco.frontend.utils import inject_custom_css  # noqa: E402

st.set_page_config(page_title="Recomendações - TwinRank AI", page_icon="🚀", layout="wide")
inject_custom_css()

st.title("🚀 Pluggable Recommendations Demo")
st.markdown(
    "Suba as suas planilhas de **Produtos** e **Pedidos**, e tenha um sistema "
    "de recomendação neural treinado exclusivamente para a sua loja em poucos segundos!"
)

# Sidebar instructions
st.sidebar.header("1. Upload de Dados")
products_file = st.sidebar.file_uploader("Upload products.csv", type=["csv"])
orders_file = st.sidebar.file_uploader("Upload orders.csv", type=["csv"])


@st.cache_resource(show_spinner="Treinando o modelo TwinRank na sua base (Two-Tower + FAISS)...")
def get_recommender(
    products_csv: object, orders_csv: object
) -> tuple[object, pd.DataFrame, pd.DataFrame]:
    """Inicializa e treina o modelo de recomendação com os dados carregados."""
    prod_df, ord_df = load_demo_data(products_csv, orders_csv)
    model = train_demo_model(prod_df, ord_df)
    return model, prod_df, ord_df


# Fallback para dados dummy
if not products_file or not orders_file:
    st.info(
        "Usando dados de exemplo (dummy_data). "
        "Faça o upload das suas planilhas para testar com seus próprios dados."
    )
    products_file = str(BASE_DIR / "dummy_data" / "products_sample.csv")
    orders_file = str(BASE_DIR / "dummy_data" / "orders_sample.csv")

try:
    model, prod_df, ord_df = get_recommender(products_file, orders_file)
    st.sidebar.success("Modelo treinado com sucesso!")

    st.sidebar.header("2. Recomendações")
    unique_users = ord_df["user_id"].unique()
    selected_user = st.sidebar.selectbox("Selecione o Usuário:", unique_users)
    top_k = st.sidebar.slider("Quantidade (Top K):", 1, 20, 5)

    if st.sidebar.button("Gerar Recomendações", type="primary"):
        st.subheader(f"Recomendações para o Usuário `{selected_user}`")

        recos = recommend_for_user(model, prod_df, selected_user, top_k)

        if not recos:
            st.warning(
                "Nenhuma recomendação encontrada. "
                "Tente outro usuário ou use uma amostra maior de dados."
            )
        else:
            st.success("TwinRank trained on your data in ~1.5s ⚡")
            # Exibir como grid/cards
            cols = st.columns(min(len(recos), 4))
            for idx, item in enumerate(recos):
                with cols[idx % 4]:  # noqa: SIM117
                    with st.container(border=True):
                        st.markdown(f"**{item['name']}**")
                        st.caption(f"Categoria: {item['category']}")
                        st.metric(label="Preço", value=f"${item['price']:.2f}")

            st.markdown("### Tabela Detalhada")
            st.dataframe(pd.DataFrame(recos), use_container_width=True)

except Exception as e:
    st.error(f"Erro ao processar os dados: {e}")
