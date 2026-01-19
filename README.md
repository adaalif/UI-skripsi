# Keyphrase Extraction System - MuSe-Rank

A web-based application for extracting ordered keyphrases from a single document using the MuSe-Rank algorithm. This system is built with Streamlit and follows an MVC (Model-View-Controller) architecture.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Architecture](#architecture)

## 🎯 Overview

This system allows a user to input a document's title and content (e.g., an abstract) to extract a ranked list of keyphrases. The core logic is based on the MuSe-Rank algorithm, which involves:
- Pre-processing the text.
- Extracting candidate phrases using POS tagging and chunking.
- Scoring candidates based on global, theme, and position scores.
- Fusing these scores using Reciprocal Rank Fusion (RRF) to produce a final ranked list.

## ✨ Features

- **Manual Text Input**: A simple UI to paste a document's title and content.
- **Configurable Top-K**: Users can specify how many keyphrases to extract.
- **Ranked Results**: Keyphrases are displayed in a table, sorted by relevance.
- **Download Results**: Extracted keyphrases can be downloaded as a `.txt` file.

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- A stable internet connection (for the first-time model download).

### Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    streamlit run main.py
    ```
    The application will automatically download the required SciBERT model on the first run. This might take a few minutes.

## 💻 Usage

1.  Once the application is running, open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).
2.  The "Manual Input" tab will be displayed.
3.  Fill in the "Judul Dokumen" (Document Title) and "Konten Dokumen" (Document Content) fields.
4.  Adjust the "Jumlah frasa kunci yang diinginkan (Top-K)" to set how many keyphrases you want.
5.  Click the "Ekstrak Frasa Kunci" button.
6.  The results will be displayed in a table, and you will see a button to download the keyphrases.

## 🏗️ Architecture

The application is structured using the Model-View-Controller (MVC) pattern:

-   **Models (`models/`):**
    -   `keyphrase_model.py`: Contains the `KeyphraseModel` class, which implements the core MuSe-Rank algorithm.
    -   `model_loader.py`: Handles the download and caching of the SciBERT model and tokenizer.

-   **Views (`views/`):**
    -   `extraction_view.py`: The `ExtractionView` class, which is responsible for rendering the Streamlit UI components and delegating actions to the Controller.

-   **Controllers (`controllers/`):**
    -   `main_controller.py`: The `MainController` class, which initializes the application, loads the model, and connects the model (extractor) with the view (UI).

-   **Main Entrypoint:**
    -   `main.py`: The main entry point to run the Streamlit application.
