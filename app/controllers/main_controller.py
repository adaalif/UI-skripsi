import streamlit as st
from app.models.model_loader import load_model_and_tokenizer
from app.models.keyphrase_extractor import KeyphraseExtractor
from app.views.manual_tab_view import ManualInputTab

class MainController:
    """
    The main controller that orchestrates the application.
    """
    def __init__(self):
        """
        Initializes the controller, loads the model, and sets up the view.
        """
        st.set_page_config(
            page_title="Keyphrase Extraction System",
            page_icon=None,
            layout="wide"
        )
        
        tokenizer, model, device = self._load_model()
        
        self.extractor = KeyphraseExtractor(tokenizer, model, device)
        self.manual_tab = ManualInputTab(self.extractor)

    @staticmethod
    def _load_model():
        """
        A static method to wrap the model loading spinner.
        """
        with st.spinner("Loading model... This may take a moment on first run."):
            tokenizer, model, device = load_model_and_tokenizer()
            st.success("Model loaded successfully!")
        return tokenizer, model, device

    def run(self):
        """
        Runs the main application loop, rendering the UI.
        """
        st.title("Keyphrase Extraction System")
        st.markdown("Extract keyphrases using the MuSe-Rank algorithm via manual text input.")

        self.manual_tab.render()
