import streamlit as st

st.set_page_config(
    page_title="TwinRank AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Redirect to the actual Home page in the pages/ directory
st.switch_page("pages/01_🏠_Home.py")
