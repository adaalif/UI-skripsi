"""
Pre-download script for SciBERT model.

Run this script before using the Streamlit app to download the model in advance.
This avoids waiting during the first app run.

Usage:
    python download_model.py
"""

import torch
from transformers import AutoTokenizer, AutoModel
import sys

def download_model():
    """
    Pre-download the SciBERT model and tokenizer.
    """
    print("=" * 60)
    print("SciBERT Model Pre-download Script")
    print("=" * 60)
    print("\nThis script will download the SciBERT model (~442MB)")
    print("so you don't have to wait when running the Streamlit app.\n")
    
    MODEL_NAME = 'allenai/scibert_scivocab_uncased'
    
    print(f"📥 Downloading tokenizer from: {MODEL_NAME}")
    print("   This may take a few minutes...\n")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        print("✅ Tokenizer downloaded successfully!\n")
    except Exception as e:
        print(f"❌ Error downloading tokenizer: {e}")
        sys.exit(1)
    
    print(f"📥 Downloading model from: {MODEL_NAME}")
    print("   This may take 5-15 minutes depending on your internet speed...")
    print("   Using safetensors format (safer, compatible with older PyTorch)...\n")
    
    try:
        model = AutoModel.from_pretrained(MODEL_NAME, use_safetensors=True)
        print("✅ Model downloaded successfully!\n")
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        sys.exit(1)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    print("\n" + "=" * 60)
    print("✅ Model pre-download complete!")
    print("   You can now run 'streamlit run app.py' without waiting.")
    print("=" * 60)

if __name__ == "__main__":
    download_model()

