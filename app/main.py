import sys
import os
import streamlit as st

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.model_loader import load_model_and_tokenizer
from app.ui.csv_tab import render_csv_tab
from app.ui.manual_tab import render_manual_tab

def main():
    """Main Streamlit application entry point."""
    st.set_page_config(
        page_title="Keyphrase Extraction System",
        page_icon="🔑",
        layout="wide"
    )

    st.title("🔑 Keyphrase Extraction System")
    st.markdown("Extract keyphrases using the MuSe-Rank algorithm via CSV upload or manual input.")

    # Load model once and cache it
    with st.spinner("Loading model... This may take a moment on first run."):
        tokenizer, model, device = load_model_and_tokenizer()
        st.success("Model loaded successfully!")

    tab1, tab2 = st.tabs(["📁 CSV Upload", "✍️ Manual Input"])

    with tab1:
        render_csv_tab(tokenizer, model, device)

    with tab2:
        render_manual_tab(tokenizer, model, device)


if __name__ == "__main__":
    main()