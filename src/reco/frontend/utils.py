import os  # noqa: D100

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
