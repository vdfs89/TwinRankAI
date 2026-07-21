import os
import sys
import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
from src.reco.frontend.utils import inject_custom_css
from src.reco.demo.ecommerce_demo import PluggableRecommender

st.set_page_config(page_title="Recommendations - TwinRank AI", page_icon="🚀", layout="wide")
inject_custom_css()

st.title("🚀 Pluggable Recommendations Demo")
st.markdown(
    "Suba as suas planilhas de **Produtos** e **Pedidos**, e tenha um sistema de recomendação neural treinado exclusivamente para a sua loja em poucos segundos!"
)

# Sidebar instructions
st.sidebar.header("1. Upload de Dados")
products_file = st.sidebar.file_uploader("Upload products.csv", type=["csv"])
orders_file = st.sidebar.file_uploader("Upload orders.csv", type=["csv"])

@st.cache_resource(show_spinner="Treinando o modelo TwinRank na sua base (Two-Tower + FAISS)...")
def get_recommender(products_csv, orders_csv):
    prod_df = pd.read_csv(products_csv)
    ord_df = pd.read_csv(orders_csv)
    return PluggableRecommender(prod_df, ord_df), prod_df, ord_df

# Fallback para dados dummy
if not products_file or not orders_file:
    st.info(
        "Usando dados de exemplo (dummy_data). Faça o upload das suas planilhas para testar com seus próprios dados."
    )
    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    )
    products_file = os.path.join(base_dir, "dummy_data", "products_sample.csv")
    orders_file = os.path.join(base_dir, "dummy_data", "orders_sample.csv")

try:
    recommender, prod_df, ord_df = get_recommender(products_file, orders_file)
    st.sidebar.success("Modelo treinado com sucesso!")

    st.sidebar.header("2. Recomendações")
    unique_users = ord_df["user_id"].unique()
    selected_user = st.sidebar.selectbox("Selecione o Usuário:", unique_users)
    top_k = st.sidebar.slider("Quantidade (Top K):", 1, 20, 5)

    if st.sidebar.button("Gerar Recomendações", type="primary"):
        st.subheader(f"Recomendações para o Usuário `{selected_user}`")

        recos = recommender.recommend_for_user(selected_user, top_k)

        if not recos:
            st.warning(
                "Nenhuma recomendação encontrada. Tente outro usuário ou aumente a quantidade de eventos."
            )
        else:
            # Exibir como grid/cards
            cols = st.columns(min(len(recos), 4))
            for idx, item in enumerate(recos):
                with cols[idx % 4]:
                    # Card design manually using markdown/HTML or simple st elements
                    st.markdown(f"**{item['name']}**")
                    st.caption(f"Categoria: {item['category']}")
                    st.text(f"Preço: ${item['price']:.2f}")
                    st.markdown("---")

            st.markdown("### Tabela Detalhada")
            st.dataframe(pd.DataFrame(recos), use_container_width=True)

except Exception as e:
    st.error(f"Erro ao processar os dados: {e}")
