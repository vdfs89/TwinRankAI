import os
import streamlit as st

def inject_custom_css():
    """Lê o style.css e injeta no Streamlit."""
    css_file = os.path.join(os.path.dirname(__file__), "styles", "style.css")
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
