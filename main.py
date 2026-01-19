import streamlit as st
from controllers.main_controller import MainController

if __name__ == "__main__":
    app = MainController()
    app.run()
