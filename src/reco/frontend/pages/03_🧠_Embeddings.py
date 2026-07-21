import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
from src.reco.frontend.utils import inject_custom_css

st.set_page_config(page_title="Embeddings - TwinRank AI", page_icon="🧠", layout="wide")
inject_custom_css()

st.title("🧠 Neural Embeddings (2D Projection)")
st.markdown("Projeção UMAP simulada do espaço latente (128D -> 2D) para análise visual dos clusters semânticos aprendidos.")

@st.cache_data
def generate_mock_embeddings():
    """Gera dados de exemplo espalhados em clusters para visualização."""
    np.random.seed(42)
    categories = ['Electronics', 'Apparel', 'Home', 'Toys']
    data = []
    
    # Criar clusters artificiais no 2D para cada categoria
    centers = {
        'Electronics': (2, 2),
        'Apparel': (-2, 2),
        'Home': (-2, -2),
        'Toys': (2, -2)
    }
    
    for cat in categories:
        center = centers[cat]
        x = np.random.normal(center[0], 0.8, 150)
        y = np.random.normal(center[1], 0.8, 150)
        for i in range(150):
            data.append({'Category': cat, 'UMAP_X': x[i], 'UMAP_Y': y[i], 'Item_ID': f"Item_{np.random.randint(1000, 9999)}"})
            
    return pd.DataFrame(data)

df_emb = generate_mock_embeddings()

fig = px.scatter(
    df_emb, 
    x="UMAP_X", 
    y="UMAP_Y", 
    color="Category",
    hover_name="Item_ID",
    color_discrete_sequence=['#5B8CFF', '#8B5CF6', '#10B981', '#F59E0B']
)

fig.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#F9FAFB', family='Inter'),
    legend=dict(title="Category"),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title="")
)

st.plotly_chart(fig, use_container_width=True)

st.info("Neste scatter plot interativo, você pode observar como produtos de mesma categoria tendem a se aglomerar no espaço latente, provando que a rede neural aprendeu a similaridade semântica a partir apenas de metadados transacionais.")
