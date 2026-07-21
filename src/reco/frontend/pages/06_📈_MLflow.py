import os
import sys

import pandas as pd
import streamlit as st

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
from reco.frontend.utils import inject_custom_css

st.set_page_config(page_title="MLOps & MLflow - TwinRank AI", page_icon="📈", layout="wide")
inject_custom_css()

st.title("📈 MLOps & Tracking")
st.markdown(
    "Monitoramento de experimentos, versionamento de hiperparâmetros e registro de modelos utilizando **MLflow**."
)

st.write(
    "O TwinRank AI foi construído com as melhores práticas de MLOps. Todas as execuções de treinamento "
    "são rastreadas. Registramos automaticamente as métricas de validação, os hiperparâmetros (como `learning_rate`, "
    "`batch_size`, `embedding_dim`) e persistimos os pesos do modelo (Artifacts)."
)

st.markdown("### Histórico de Experimentos Recentes (Mockup)")

data = {
    "Run ID": ["a1b2c3d4", "f5e6d7c8", "9a8b7c6d", "1a2b3c4d"],
    "Model": ["Two-Tower", "Two-Tower", "Matrix Factorization", "Popularity"],
    "Embed Dim": ["128", "64", "64", "N/A"],
    "Learning Rate": ["0.001", "0.005", "0.01", "N/A"],
    "Recall@10": [0.812, 0.765, 0.450, 0.151],
    "Status": ["✅ COMPLETED", "✅ COMPLETED", "✅ COMPLETED", "✅ COMPLETED"],
}

st.dataframe(pd.DataFrame(data), use_container_width=True)

st.info(
    "Na infraestrutura em Cloud ou Local, acessar http://localhost:5000 exibe o painel oficial do MLflow com os gráficos interativos de cada step das épocas (Loss vs Epoch)."
)
