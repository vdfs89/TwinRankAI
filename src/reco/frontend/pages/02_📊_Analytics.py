import os
import sys
import streamlit as st
import plotly.graph_objects as go

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from src.reco.frontend.utils import inject_custom_css

st.set_page_config(page_title="Analytics - TwinRank AI", page_icon="📊", layout="wide")
inject_custom_css()

st.title("📊 Model Analytics")
st.markdown("Comparativo de performance entre os modelos treinados (Two-Tower vs. Matrix Factorization vs. Popularity).")

# Dados fictícios para o mockup de comparação (podem ser lidos do JSON na versão final)
models = ['Popularity', 'Matrix Factorization', 'Two-Tower (TwinRank)']
recall = [0.15, 0.45, 0.81]
map_k = [0.08, 0.22, 0.65]
mrr = [0.05, 0.19, 0.52]
ndcg = [0.12, 0.35, 0.77]

fig = go.Figure(data=[
    go.Bar(name='Recall@10', x=models, y=recall, marker_color='#5B8CFF'),
    go.Bar(name='MAP@10', x=models, y=map_k, marker_color='#8B5CF6'),
    go.Bar(name='MRR@10', x=models, y=mrr, marker_color='#10B981'),
    go.Bar(name='NDCG@10', x=models, y=ndcg, marker_color='#F59E0B')
])

fig.update_layout(
    barmode='group',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#F9FAFB', family='Inter'),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("### Por que o Two-Tower supera os baselines?")
st.write(
    "Ao contrário do SVD Clássico (Matrix Factorization) que possui dificuldades severas em lidar com catálogos esparsos e "
    "cold-start de itens long-tail, o modelo de Duas Torres aprende representações ricas a partir das interações "
    "diretas usando Negative Sampling em batch. Aliado à indexação baseada em Produto Interno no FAISS, a inferência é O(1) com Recall altíssimo."
)
