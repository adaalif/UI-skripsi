import streamlit as st
from models.model_loader import load_model_and_tokenizer
from models.keyphrase_model import KeyphraseModel
from views.extraction_view import ExtractionView

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
        
        self.model = KeyphraseModel(tokenizer, model, device)
        self.view = ExtractionView(self)

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

        self.view.render()

    def process_extraction(self, document_text, title, top_k):
        """
        Mediates the extraction request from the View to the Model.
        """
        return self.model.extract(document_text, title, top_k)
