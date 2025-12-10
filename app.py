"""
Main entry point for the Keyphrase Extraction Streamlit application.

This script imports the main App class from the structured 'app' directory
and runs it. This allows for a clean separation of concerns where the root
file is just a runner, and all the application logic is organized
within the 'app/' package.

To run the application:
streamlit run app.py
"""

import sys
import os

# Ensure the 'app' directory can be imported from
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'app')))

# Import the main App class from the refactored application structure
from main import App

if __name__ == "__main__":
    # Create an instance of the main application class
    app_instance = App()
    # Run the application
    app_instance.run()