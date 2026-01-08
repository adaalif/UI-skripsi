import sys
import os
import streamlit as st
import logging

# Suppress the specific PyTorch warning by raising the logging level for the torch library
logging.getLogger("torch").setLevel(logging.ERROR)

# Add the project root to the Python path to allow for absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.controllers.main_controller import MainController

if __name__ == "__main__":
    # Create an instance of the App and run it
    app = MainController()
    app.run()