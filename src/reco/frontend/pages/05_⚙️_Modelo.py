import os  # noqa: D100
import sys

import streamlit as st

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
from reco.frontend.utils import inject_custom_css

st.set_page_config(page_title="Arquitetura do Modelo - TwinRank AI", page_icon="⚙️", layout="wide")
inject_custom_css()

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
        "As duas torres mapeiam suas respectivas entradas para um mesmo espaço latente denso (Embedding Space) de 128 dimensões."  # noqa: E501
    )

    st.markdown("### 2. In-Batch Negative Sampling")
    st.write(
        "Para que a rede aprenda a diferenciar itens relevantes dos não relevantes, o TwinRank utiliza a técnica de "  # noqa: E501
        "**In-Batch Negative Sampling**. Durante o treino, para um dado usuário, todos os outros itens do batch que "  # noqa: E501
        "não foram interagidos por ele são considerados amostras negativas. Isso otimiza o uso de memória GPU e acelera a convergência."  # noqa: E501
    )

with col2:
    st.markdown("### 3. Approximate Nearest Neighbors (FAISS)")
    st.write(
        "Em tempo de inferência (produção), calcular a similaridade de um usuário contra 1 milhão de itens via *Dot Product* exato seria inviável (latência alta). "  # noqa: E501
        "Portanto, os embeddings de todos os produtos são cacheados e indexados usando **FAISS** (Facebook AI Similarity Search) via `IndexFlatIP`. "  # noqa: E501
        "Isso reduz a latência da busca dos Top-K itens de centenas de milissegundos para **menos de 1 milissegundo**."  # noqa: E501
    )

    st.markdown("### 4. Cache-Aside Pattern (Redis)")
    st.write(
        "A camada da API construída em FastAPI é protegida por um padrão **Cache-Aside** no Redis. "
        "Requisições de usuários populares ou bots batem diretamente no cache, poupando processamento neural e aliviando a carga do backend e do índice vetorial."  # noqa: E501
    )

st.markdown("---")
st.info(
    "Essa combinação (Two-Tower + In-Batch Negatives + FAISS + Redis) é o padrão da indústria (usado em YouTube, Pinterest, etc.) "  # noqa: E501
    "para servir dezenas de milhares de requisições por segundo com altíssima relevância."
)
