# Keyphrase Extraction System - MDERank-RRF

A web-based application for extracting ordered keyphrases from documents using the MDERank-RRF (Multi-Dimensional Embedding Rank with Reciprocal Rank Fusion) algorithm.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Functional Requirements](#functional-requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Architecture](#architecture)
- [Documentation](#documentation)

## 🎯 Overview

This system implements an advanced keyphrase extraction pipeline that:
- Accepts CSV files containing documents with titles and full-text content
- Extracts candidate phrases using POS tagging and chunking
- Scores candidates using multiple dimensions (global, theme, position)
- Ranks keyphrases using Reciprocal Rank Fusion (RRF)
- Returns ordered keyphrase lists

## ✨ Features

1. **CSV File Upload**: Simple drag-and-drop interface for CSV files
2. **Automatic Column Detection**: Automatically detects title and text columns
3. **Batch Processing**: Process single or multiple documents
4. **Ordered Results**: Keyphrases ranked by importance
5. **Export Results**: Download extraction results as CSV

## 📝 Functional Requirements

The system fulfills the following functional requirements:

| No | Requirement | Implementation |
|---|---|---|
| 1 | Perangkat lunak dapat menerima masukan berupa teks dokumen yang terdiri dari judul dan isi dokumen (full-text) | CSV upload with automatic column detection for title and text fields |
| 2 | Perangkat lunak dapat memberikan hasil ekstraksi frasa kunci yang terurut | RRF-based ranking returns ordered keyphrases by importance |
| 3 | Perangkat lunak dapat melakukan pembacaan file dataset | CSV file reading with pandas, supports various column name formats |

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster processing)
- Stable internet connection (for first-time model download)

### Setup

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. **(Recommended)** Pre-download the SciBERT model to avoid waiting during first app run:
```bash
python download_model.py
```
This downloads the ~442MB model in advance. Otherwise, it will download automatically on first app run (may take 5-15 minutes).

4. Download spaCy model (if using notebook):
```bash
python -m spacy download en_core_web_trf
```

5. Download NLTK data (automatically handled by the app, but can be done manually):
```python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
```

### ⚠️ First-Time Download Note

The SciBERT model (~442MB) is downloaded automatically on first run. This can take:
- **5-15 minutes** on a normal connection
- **Longer** on slow connections (100-200 KB/s)

**Solutions for slow downloads:**
1. Pre-download using `python download_model.py` (recommended)
2. Use a faster internet connection
3. The download will resume if interrupted (HuggingFace supports resume)
4. Model is cached after first download - subsequent runs are instant

## 💻 Usage

### Running the Streamlit Application

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser to the URL shown (typically `http://localhost:8501`)

3. Upload a CSV file with the following structure:
   - **Required columns**: `title` (or `judul`) and `text` (or `content`, `isi`, `abstract`, `body`)
   - **Optional**: Additional columns are preserved but not used

4. Configure processing options:
   - Number of keyphrases to extract (default: 15)
   - Number of rows to process

5. Click "Extract Keyphrases" and wait for processing

6. Review results and download as CSV if needed

### CSV Format Example

```csv
title,text
"Machine Learning in Healthcare","Machine learning algorithms are transforming healthcare..."
"Natural Language Processing","Natural language processing enables computers to understand..."
```

### Using the Jupyter Notebook

1. Open `mderank-rrf (6).ipynb` in Jupyter Notebook or JupyterLab

2. Run cells sequentially to:
   - Set up the environment
   - Load models and datasets
   - Test the extraction pipeline
   - Evaluate on benchmark datasets

## 🏗️ Architecture

### Pipeline Overview

```
Input CSV
    ↓
Document Preprocessing
    ↓
Candidate Phrase Extraction (POS Tagging + Chunking)
    ↓
Multi-Dimensional Scoring
    ├── Global Score (Masking-based)
    ├── Theme Score (Title similarity)
    └── Position Score (Document position)
    ↓
Reciprocal Rank Fusion (RRF)
    ↓
Ordered Keyphrase List
```

### Key Components

1. **Preprocessing**: Text normalization, cleaning, case folding
2. **Candidate Extraction**: NP chunking with POS tagging, lemmatization, deduplication
3. **Scoring**: Three-dimensional scoring using SciBERT embeddings
4. **Ranking**: RRF combines multiple ranking signals
5. **Output**: Ordered list of keyphrases

## 📚 Documentation

### Code Documentation Standards

This project follows FAANG SWE documentation standards:

- **Module-level docstrings**: Describe purpose, usage, and key functions
- **Function docstrings**: Include Args, Returns, Raises sections
- **Inline comments**: Explain complex logic and algorithms
- **Type hints**: Where applicable for better code clarity

### Notebook Structure

The notebook (`mderank-rrf (6).ipynb`) is organized into sections:

1. **Environment Setup**: Library installation and configuration
2. **Model Loading**: SciBERT model and dataset loading
3. **Utility Functions**: Text preprocessing and helper functions
4. **Candidate Extraction**: Phrase extraction algorithms
5. **Scoring Functions**: Multi-dimensional scoring implementation
6. **RRF Implementation**: Reciprocal Rank Fusion algorithm
7. **Evaluation**: Metrics calculation and analysis
8. **Pipeline**: End-to-end processing functions

Each section includes:
- Clear section headers
- Function documentation
- Usage examples
- Test cases

## 🔧 Technical Details

### Models Used

- **SciBERT**: `allenai/scibert_scivocab_uncased` for text embeddings
- **spaCy**: `en_core_web_trf` for POS tagging (notebook only)

### Algorithms

- **Candidate Extraction**: Regex-based NP chunking with grammar rules
- **Global Score**: Masking-based importance using cosine similarity
- **Theme Score**: Title-phrase similarity using CLS token embeddings
- **Position Score**: Inverse position weighting
- **RRF**: Reciprocal Rank Fusion with k=40-60

### Performance

- Processing time: ~2-5 seconds per document (GPU) or ~10-20 seconds (CPU)
- Memory usage: ~2-4 GB (depends on batch size)
- Accuracy: See notebook evaluation results

## 📄 License

This project is for academic/research purposes.

## 👥 Authors

Keyphrase Extraction System - MDERank-RRF Implementation

## 🙏 Acknowledgments

- SciBERT model by AllenAI
- Inspec dataset for evaluation
- HuggingFace Transformers library

