import os  # noqa: D100
import sys

import streamlit as st

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
from reco.frontend.utils import embedding_dim_from_metrics, inject_custom_css

st.set_page_config(page_title="Arquitetura do Modelo - TwinRank AI", page_icon="⚙️", layout="wide")
inject_custom_css()

embedding_dim = embedding_dim_from_metrics()

st.title("⚙️ Arquitetura do Modelo")
st.markdown(
    "Visão geral da arquitetura do sistema de recomendação TwinRank AI, construída para escalabilidade industrial."  # noqa: E501
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 1. Two-Tower Neural Network")
    st.write(
        "A base do TwinRank é a arquitetura **Two-Tower** (Duas Torres), implementada em PyTorch. "
        "Uma torre processa os dados do usuário (Visitor) e a outra os dados do item (Product). "
        "As duas torres mapeiam suas respectivas entradas para um mesmo espaço latente denso "
        f"(Embedding Space) de {embedding_dim} dimensões."
    )

    st.markdown("### 2. Negative Sampling uniforme")
    st.write(
        "Para a rede aprender a diferenciar itens relevantes dos irrelevantes, "
        "cada interação positiva é acompanhada de 4 negativos sorteados "
        "uniformemente do catálogo. O positivo carrega o peso do feedback "
        "implícito (view 1,0 / addtocart 3,0 / transaction 5,0) numa BCE "
        "ponderada, de modo que uma compra pesa mais que uma visualização."
    )

with col2:
    st.markdown("### 3. Approximate Nearest Neighbors (FAISS)")
    st.write(
        "Os embeddings de item são indexados com **FAISS** (Facebook AI "
        "Similarity Search) via `IndexFlatIP`, evitando varrer o catálogo "
        "inteiro a cada requisição para encontrar os Top-K por produto interno. "
        "Medido ponta a ponta no container, um `/recommend` responde em torno "
        "de **1,6 ms** — a busca vetorial é apenas uma parte desse tempo."
    )

    st.markdown("### 4. Cache-Aside Pattern (Redis)")
    st.write(
        "A camada da API construída em FastAPI é protegida por um padrão **Cache-Aside** no Redis. "
        "Requisições de usuários populares ou bots batem diretamente no cache, poupando processamento neural e aliviando a carga do backend e do índice vetorial."  # noqa: E501
    )

st.markdown("---")
st.info(
    "A arquitetura Two-Tower com busca vetorial aproximada é o padrão de "
    "retrieval em recomendação de larga escala, descrito em publicações de "
    "YouTube e Pinterest. Este projeto implementa esse padrão em escala "
    "acadêmica: os números de latência acima são de um container local, "
    "single-worker, não de um teste de carga."
)
