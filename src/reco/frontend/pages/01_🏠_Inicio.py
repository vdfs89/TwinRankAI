import os  # noqa: D100
import sys

import streamlit as st

# Ensure imports work from src
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
from reco.frontend.utils import (
    inject_custom_css,
    load_metrics,
    metrics_missing_warning,
)

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

# KPIs — todos os números vêm de reports/metrics.json (pipeline de avaliação).
metrics_data = load_metrics()

if metrics_data is None:
    metrics_missing_warning()
else:
    meta = metrics_data.get("_meta", {})
    two_tower_recall = metrics_data["two_tower"]["recall_at_10"]
    popularity_recall = metrics_data["popularity"]["recall_at_10"]
    lift = two_tower_recall / popularity_recall

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Visitantes no Catálogo",
            value=f"{meta.get('known_visitors', 0) / 1_000_000:.2f}M",
            delta=f"{meta.get('eval_visitors', 0):,} avaliados".replace(",", "."),
        )
    with col2:
        st.metric(
            label="Catálogo de Produtos",
            value=f"{meta.get('catalog_items', 0) / 1_000:.0f}K",
            delta="RetailRocket",
        )
    with col3:
        st.metric(
            label="Recall@10 (Two-Tower)",
            value=f"{two_tower_recall:.5f}".replace(".", ","),
            delta=f"{lift:.0f}x vs Popularity",
        )
    with col4:
        st.metric(
            label="Dimensão de Embeddings",
            value=str(meta.get("embedding_dim", "—")),
            delta="FAISS Index",
        )

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
