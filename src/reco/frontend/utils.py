import json  # noqa: D100
import os
from pathlib import Path
from typing import Any

import streamlit as st


def inject_custom_css():  # noqa: ANN201
    """Lê o style.css e injeta no Streamlit."""
    css_file = os.path.join(os.path.dirname(__file__), "styles", "style.css")

    # Injetar logo na sidebar (maior e visível)
    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    logo_path = os.path.join(base_dir, "docs", "logo.png")
    if os.path.exists(logo_path):
        st.sidebar.image(logo_path, use_container_width=True)
        st.sidebar.markdown("<br>", unsafe_allow_html=True)

    if os.path.exists(css_file):
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Esconde a primeira página "App" gerada automaticamente pelo app.py da sidebar
    hide_app_page = """
    <style>
    ul[data-testid="stSidebarNavItems"] li:first-child {
        display: none;
    }
    </style>
    """
    st.markdown(hide_app_page, unsafe_allow_html=True)


# Resolvido a partir do arquivo, não do cwd: o Streamlit Cloud não garante que
# o processo rode a partir da raiz do repositório.
_REPO_ROOT = Path(__file__).resolve().parents[3]
METRICS_PATH = _REPO_ROOT / "reports" / "metrics.json"

MODEL_LABELS = {
    "popularity": "Popularity",
    "matrix_factorization": "Matrix Factorization",
    "two_tower": "Two-Tower (TwinRank)",
}


@st.cache_data(show_spinner=False)
def load_metrics() -> dict[str, Any] | None:
    """Carrega as métricas reais geradas pelo pipeline de avaliação.

    Fonte única de verdade para todas as páginas: nenhuma página deve manter
    número de métrica de modelo hardcoded. Retorna None se o arquivo não
    existir, cabendo à página exibir o aviso.

    Returns
    -------
        Dicionário do `reports/metrics.json`, ou None se ausente.

    """
    if not METRICS_PATH.exists():
        return None
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def metrics_missing_warning() -> None:
    """Exibe o aviso padrão de métricas ausentes."""
    st.warning(
        f"Arquivo `{METRICS_PATH}` não encontrado. "
        "Rode `dvc repro evaluate` para gerar as métricas."
    )


def model_keys(metrics_data: dict[str, Any]) -> list[str]:
    """Retorna as chaves de modelo presentes, ignorando metadados (`_meta`)."""
    return [key for key in MODEL_LABELS if key in metrics_data]


def embedding_dim_from_metrics(default: int = 64) -> int:
    """Dimensão de embedding do run avaliado, com fallback para o default do projeto.

    Evita repetir o valor em texto nas páginas: ele acompanha o que o pipeline
    realmente treinou.

    Args:
    ----
        default: valor usado quando `reports/metrics.json` ainda não existe.

    Returns:
    -------
        Dimensão de embedding registrada no run avaliado.

    """
    metrics_data = load_metrics()
    if metrics_data is None:
        return default
    return int(metrics_data.get("_meta", {}).get("embedding_dim", default))
