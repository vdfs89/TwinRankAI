import os  # noqa: D100
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
from reco.frontend.utils import REPO_ROOT, embedding_dim_from_metrics, inject_custom_css

st.set_page_config(page_title="Embeddings - TwinRank AI", page_icon="🧠", layout="wide")
inject_custom_css()

PROJECTION_PATH = REPO_ROOT / "reports" / "embedding_projection.csv"

embedding_dim = embedding_dim_from_metrics()

st.title("🧠 Neural Embeddings (Projeção 2D)")
st.markdown(
    f"Projeção PCA do espaço latente ({embedding_dim}D → 2D), calculada sobre os "
    "embeddings de item do modelo treinado."
)


@st.cache_data(show_spinner=False)
def load_projection() -> pd.DataFrame | None:
    """Carrega a projeção pré-computada a partir do checkpoint treinado."""
    if not PROJECTION_PATH.exists():
        return None
    return pd.read_csv(PROJECTION_PATH)


df_emb = load_projection()

if df_emb is None:
    st.warning(
        f"Arquivo `{PROJECTION_PATH.name}` não encontrado. "
        "Rode `poetry run python scripts/export_embedding_projection.py` "
        "para gerá-lo a partir do modelo treinado."
    )
    st.stop()

fig = px.scatter(
    df_emb,
    x="x",
    y="y",
    color="cluster",
    hover_name="item_id",
    category_orders={"cluster": sorted(df_emb["cluster"].unique())},
    color_discrete_sequence=[
        "#5B8CFF",
        "#8B5CF6",
        "#10B981",
        "#F59E0B",
        "#EF4444",
        "#06B6D4",
    ],
)

fig.update_traces(marker=dict(size=5, opacity=0.75))
fig.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#F9FAFB", family="Inter"),
    legend=dict(title="Cluster"),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title="PC1"),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title="PC2"),
)

st.plotly_chart(fig, use_container_width=True)

col1, col2, col3 = st.columns(3)
col1.metric("Itens plotados", f"{len(df_emb):,}".replace(",", "."))
col2.metric("Dimensão original", f"{embedding_dim}D")
col3.metric("Agrupamentos", df_emb["cluster"].nunique())

st.caption(
    "Amostra aleatória de itens do catálogo aprendido, projetada com PCA. Os "
    "grupos vêm de KMeans sobre os embeddings — são agrupamentos descobertos no "
    "espaço latente, não categorias de catálogo: o RetailRocket é anonimizado e "
    "não expõe rótulos utilizáveis aqui."
)

st.info(
    "Itens próximos no gráfico foram consumidos por visitantes parecidos — a "
    "única forma de generalização disponível a um modelo de embedding de ID "
    "puro, que aprende por co-ocorrência e não a partir de atributos do produto. "
    "Como a projeção reduz 64 dimensões a 2, distâncias no gráfico são uma "
    "aproximação grosseira das distâncias reais no espaço latente."
)
