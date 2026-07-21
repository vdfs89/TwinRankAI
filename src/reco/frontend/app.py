import os

import requests
import streamlit as st

# Conf API backend
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="TwinRank AI Dashboard",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 TwinRank AI Recommendations")
st.markdown("Dashboard interativo para consultar o sistema de recomendação via FastAPI.")

# Barra lateral para informações
st.sidebar.header("Configurações")
user_id_input = st.sidebar.number_input("ID do Usuário (Ex: 10)", min_value=0, value=10, step=1)
top_k_input = st.sidebar.slider("Quantidade (Top K)", min_value=1, max_value=50, value=10)

if st.sidebar.button("Gerar Recomendações", type="primary"):
    with st.spinner(f"Buscando recomendações para usuário {user_id_input}..."):
        try:
            # Faz chamada para a API FastAPI
            response = requests.get(
                f"{API_URL}/recommend/{user_id_input}", params={"top_k": top_k_input}
            )
            response.raise_for_status()
            data = response.json()
            item_ids = data.get("item_ids", [])

            st.success(f"Encontradas {len(item_ids)} recomendações!")

            # Mostra como grid ou lista
            st.subheader("Itens Recomendados")
            if not item_ids:
                st.info("Nenhuma recomendação encontrada para este usuário.")
            else:
                # Mostrar os resultados
                cols = st.columns(5)
                for idx, item_id in enumerate(item_ids):
                    col = cols[idx % 5]
                    with col:
                        st.metric(label=f"Rank {idx + 1}", value=f"Item {item_id}")

        except requests.exceptions.RequestException as e:
            st.error(f"Erro ao conectar com a API TwinRank AI: {e}")
