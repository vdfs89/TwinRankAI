import json  # noqa: D100
import os
import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
from reco.frontend.utils import inject_custom_css

st.set_page_config(page_title="Métricas - TwinRank AI", page_icon="📊", layout="wide")
inject_custom_css()

st.title("📊 Model Analytics")
st.markdown(
    "Comparativo de performance entre os modelos treinados (Two-Tower vs. Matrix Factorization vs. Popularity)."  # noqa: E501
)

METRICS_PATH = Path("reports/metrics.json")
MODEL_LABELS = {
    "popularity": "Popularity",
    "matrix_factorization": "Matrix Factorization",
    "two_tower": "Two-Tower (TwinRank)",
}

if not METRICS_PATH.exists():
    st.warning(
        f"Arquivo `{METRICS_PATH}` não encontrado. "
        "Rode `dvc repro evaluate` para gerar as métricas."
    )
    st.stop()

metrics_data = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
keys = [k for k in MODEL_LABELS if k in metrics_data]

models = [MODEL_LABELS[k] for k in keys]
recall = [metrics_data[k]["recall_at_10"] for k in keys]
map_k = [metrics_data[k]["map_at_10"] for k in keys]
mrr = [metrics_data[k]["mrr_at_10"] for k in keys]
ndcg = [metrics_data[k]["ndcg_at_10"] for k in keys]

fig = go.Figure(
    data=[
        go.Bar(name="Recall@10", x=models, y=recall, marker_color="#5B8CFF"),
        go.Bar(name="MAP@10", x=models, y=map_k, marker_color="#8B5CF6"),
        go.Bar(name="MRR@10", x=models, y=mrr, marker_color="#10B981"),
        go.Bar(name="NDCG@10", x=models, y=ndcg, marker_color="#F59E0B"),
    ]
)

fig.update_layout(
    barmode="group",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#F9FAFB", family="Inter"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

st.plotly_chart(fig, use_container_width=True)

st.info(
    "**População de avaliação:** apenas visitantes com pelo menos 5 interações no "
    "conjunto de treino (23.476 → 2.920 usuários). O filtro é necessário porque 57% "
    "dos visitantes originais tinham uma única interação de treino e dominavam a "
    "média. Por isso estes números não são comparáveis com execuções anteriores ao "
    "filtro."
)

st.markdown("### Descoberta vs. repetição")
st.write(
    "`Recall@10 (novel)` exclui dos itens relevantes de cada usuário tudo o que já "
    "estava em seu histórico de treino, isolando descoberta real de repetição de "
    "itens conhecidos."
)

novel_rows = [
    {
        "Modelo": MODEL_LABELS[k],
        "Recall@10 (geral)": round(metrics_data[k]["recall_at_10"], 5),
        "Recall@10 (novel)": round(metrics_data[k]["recall_at_10_novel"], 5),
        "Repetição": f"{(1 - metrics_data[k]['recall_at_10_novel'] / metrics_data[k]['recall_at_10']) * 100:.1f}%",  # noqa: E501
    }
    for k in keys
    if metrics_data[k].get("recall_at_10") and "recall_at_10_novel" in metrics_data[k]
]
st.dataframe(novel_rows, use_container_width=True, hide_index=True)

st.write(
    "O Two-Tower aprende representações a partir das interações diretas com negative "
    "sampling, e a indexação por produto interno no FAISS mantém a inferência barata. "
    "Ele lidera tanto no Recall@10 geral quanto na descoberta pura, mas a margem em "
    "descoberta é menor: boa parte do recall geral vem de recomendar de volta itens "
    "que o usuário já havia consumido — comportamento esperado em embeddings de ID "
    "puro, sem features de conteúdo."
)
