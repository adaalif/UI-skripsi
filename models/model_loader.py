import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel

@st.cache_resource
def load_model_and_tokenizer():
    """
    Load and cache the SciBERT model and tokenizer.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_NAME = 'allenai/scibert_scivocab_uncased'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        use_safetensors=True  
    ).to(device)
    model.eval()
    return tokenizer, model, device