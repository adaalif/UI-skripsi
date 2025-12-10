import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel

@st.cache_resource
def load_model_and_tokenizer():
    """
    Load and cache the SciBERT model and tokenizer.
    
    Uses safetensors format to avoid PyTorch version compatibility issues.
    
    Returns:
        tuple: (tokenizer, model, device) - Tokenizer, model, and device
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_NAME = 'allenai/scibert_scivocab_uncased'
    
    # Use safetensors to avoid torch.load security vulnerability issues
    # Safetensors works with older PyTorch versions
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        use_safetensors=True  # Use safetensors format (safer, works with older torch)
    ).to(device)
    model.eval()
    return tokenizer, model, device