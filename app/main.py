import sys
import os
import streamlit as st
import logging

# Suppress the specific PyTorch warning by raising the logging level for the torch library
logging.getLogger("torch").setLevel(logging.ERROR)

# Add the project root to the Python path to allow for absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.model_loader import load_model_and_tokenizer
from app.core.keyphrase_extraction import KeyphraseExtractor
from app.ui.manual_tab import ManualInputTab

class App:
    """
    The main application class that orchestrates the Streamlit UI and backend logic.
    """
    def __init__(self):
        """
        Initializes the application, sets up page configuration, and loads necessary resources.
        """
        st.set_page_config(
            page_title="Keyphrase Extraction System",
            page_icon=None,
            layout="wide"
        )
        
        # Load the model and tokenizer (cached by Streamlit)
        tokenizer, model, device = self._load_model()
        
        # Instantiate the core extractor and the UI component
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

        # Directly render the manual input interface
        self.manual_tab.render()


if __name__ == "__main__":
    # Create an instance of the App and run it
    app = App()
    app.run()