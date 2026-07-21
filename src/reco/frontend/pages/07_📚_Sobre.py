import os
import sys
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
from src.reco.frontend.utils import inject_custom_css

st.set_page_config(page_title="Sobre - TwinRank AI", page_icon="📚", layout="wide")
inject_custom_css()

st.title("📚 About TwinRank AI")
st.markdown("Contexto sobre o desenvolvimento do projeto e seus criadores.")

st.markdown("### Contexto Acadêmico & Portfolio")
st.write(
    "O **TwinRank AI** foi desenvolvido como resposta ao Tech Challenge (Fase 2) da **FIAP**, focado "
    "na criação de sistemas de Machine Learning profissionais para o mundo real. "
    "O projeto demonstra as habilidades completas de um **Machine Learning Engineer**, cobrindo:"
)

st.markdown(
    """
    - **Data Engineering**: Processamento de grandes volumes de eventos em Pandas.
    - **Machine Learning**: Construção de Redes Neurais Profundas (Two-Tower) com PyTorch.
    - **MLOps**: Rastreabilidade com MLflow e versionamento de artefatos com DVC.
    - **Backend & Serving**: Exposição do modelo com FastAPI, busca vetorial sub-milisegundo (FAISS) e cache (Redis).
    - **Frontend**: Criação de Dashboards executivos com Streamlit.
    - **Deploy**: Containerização full-stack com Docker Compose.
    """
)

st.markdown("---")

st.markdown("### Contato")
st.write("Criado e arquitetado para ser uma solução escalável na era de AI e dados.")
st.link_button("📄 Acessar Repositório no GitHub", "https://github.com/vdfs89/TwinRankAI")
