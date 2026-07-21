import os
import streamlit as st

def inject_custom_css():
    """Lê o style.css e injeta no Streamlit."""
    css_file = os.path.join(os.path.dirname(__file__), "styles", "style.css")
    
    # Injetar logo fixa no rodapé do menu lateral
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    logo_path = os.path.join(base_dir, "docs", "logo.png")
    
    logo_css = ""
    if os.path.exists(logo_path):
        import base64
        with open(logo_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        logo_css = f"""
        <style>
            [data-testid="stSidebarNav"]::after {{
                content: "";
                display: block;
                width: 100%;
                height: 100px;
                background-image: url("data:image/png;base64,{encoded}");
                background-size: contain;
                background-repeat: no-repeat;
                background-position: center;
                margin-top: 20px;
            }}
        </style>
        """
        
    if os.path.exists(css_file):
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
            
    if logo_css:
        st.markdown(logo_css, unsafe_allow_html=True)

    # Esconde a primeira página "App" gerada automaticamente pelo app.py da sidebar
    hide_app_page = """
    <style>
    ul[data-testid="stSidebarNavItems"] li:first-child {
        display: none;
    }
    </style>
    """
    st.markdown(hide_app_page, unsafe_allow_html=True)
