import os  # noqa: D100
import sys

import streamlit as st

# Ensure imports work from src
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
from reco.frontend.utils import inject_custom_css

st.set_page_config(page_title="Início - TwinRank AI", page_icon="🏠", layout="wide")
inject_custom_css()

# Hero Section
logo_path = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
    ),
    "docs",
    "logo.png",
)
if os.path.exists(logo_path):
    st.image(logo_path, width=500)
else:
    st.markdown(
        "<h1 style='font-size: 3.5rem; margin-bottom: 0;'>TwinRank AI</h1>", unsafe_allow_html=True
    )

st.markdown(
    "<h3 style='color: #9CA3AF; margin-top: 0;'>Motor de Recomendação Industrial com Deep Learning</h3>",  # noqa: E501
    unsafe_allow_html=True,
)

st.markdown("---")

# KPIs
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Usuários Processados", value="1.4M", delta="RetailRocket")
with col2:
    st.metric(label="Catálogo de Produtos", value="235K", delta="Live")
with col3:
    st.metric(label="Recall@10 (Two-Tower)", value="0.81", delta="+42% vs Baseline")
with col4:
    st.metric(label="Dimensão de Embeddings", value="128", delta="FAISS Index")

st.markdown("---")
st.markdown("<br>", unsafe_allow_html=True)

# CTAs
st.markdown("### Bem-vindo ao TwinRank AI Portfolio")
st.write(
    "Este portal interativo apresenta a arquitetura, métricas e um caso de uso real "
    "de um sistema de recomendação de duas torres (Two-Tower) com indexação vetorial ultrarrápida."
)

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🚀 Ir para o Live Demo", use_container_width=True):
        st.switch_page("pages/04_🚀_Recomendacoes.py")

with col2:
    st.link_button(
        "📄 Repositório no GitHub", "https://github.com/vdfs89/TwinRankAI", use_container_width=True
    )

with col3:
    st.link_button(
        "📊 Model Card",
        "https://github.com/vdfs89/TwinRankAI/blob/main/docs/model_card.md",
        use_container_width=True,
    )
