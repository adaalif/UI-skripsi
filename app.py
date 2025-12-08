"""
Streamlit UI Application for Keyphrase Extraction from CSV Files

This module provides a web-based interface for extracting keyphrases from documents
stored in CSV format. It implements the MDERank-RRF algorithm for keyphrase extraction.

Author: Keyphrase Extraction System
Date: 2024
"""

import streamlit as st
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import RegexpParser
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)

try:
    nltk.data.find('chunkers/maxent_ne_chunker')
except LookupError:
    nltk.download('maxent_ne_chunker', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Initialize global components
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

# Initialize NLTK components
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def get_wordnet_pos(treebank_tag):
    """
    Convert NLTK POS tag to WordNet POS tag for lemmatization.
    
    Args:
        treebank_tag (str): NLTK Treebank POS tag
        
    Returns:
        wordnet.POS: WordNet POS tag
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess_text(text):
    """
    Normalize and clean text input.
    
    Args:
        text (str): Raw text input
        
    Returns:
        str: Preprocessed text
    """
    text = text.lower()
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'[^\w\s.,;:!?()-]', '', text)
    return text

def extract_candidate_phrases(text, min_length=1, max_length=5):
    """
    Extract candidate keyphrases from text using POS tagging and chunking.
    
    Args:
        text (str): Preprocessed text
        min_length (int): Minimum phrase length in words
        max_length (int): Maximum phrase length in words
        
    Returns:
        list: List of candidate phrase dictionaries with 'phrase', 'position', 'length'
    """
    text = preprocess_text(text)
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    grammar = """NP: {<NN.*|JJ>*<NN.*>}"""
    cp = RegexpParser(grammar)
    tree = cp.parse(pos_tags)
    candidates = {}
    current_char_index = 0
    
    for subtree in tree:
        if hasattr(subtree, 'label') and subtree.label() == 'NP':
            chunk_leaves = subtree.leaves()
            if not (min_length <= len(chunk_leaves) <= max_length):
                continue
            
            surface_tokens = [w for w, t in chunk_leaves]
            surface_form = " ".join(surface_tokens)
            
            lemma_tokens = []
            for word, tag in chunk_leaves:
                wn_tag = get_wordnet_pos(tag)
                lemma_tokens.append(lemmatizer.lemmatize(word, wn_tag))
            
            lemma_form = " ".join(lemma_tokens)
            
            if (lemma_tokens[0] in stop_words or 
                lemma_tokens[-1] in stop_words):
                continue

            start_idx = text.find(surface_form)
            
            if lemma_form not in candidates:
                candidates[lemma_form] = {
                    'phrase': surface_form,
                    'position': start_idx,
                    'length': len(surface_tokens)
                }
            else:
                existing = candidates[lemma_form]
                if len(surface_form) > len(existing['phrase']):
                    existing['phrase'] = surface_form
                if start_idx != -1 and start_idx < existing['position']:
                    existing['position'] = start_idx

    return list(candidates.values())

def get_pooled_embeddings_batched(texts, tokenizer, model, device, BATCH_SIZE=64, pooling_strategy='mean'):
    """
    Generate embeddings for a batch of texts using mean pooling.
    
    Args:
        texts (list): List of text strings
        tokenizer: HuggingFace tokenizer
        model: HuggingFace model
        device: torch device
        BATCH_SIZE (int): Batch size for processing
        pooling_strategy (str): Pooling strategy ('mean', 'cls', 'max')
        
    Returns:
        np.ndarray: Array of embeddings
    """
    all_embeddings = []
    
    with torch.inference_mode():
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i : i + BATCH_SIZE]
            
            inputs = tokenizer(
                batch_texts, 
                padding='max_length',
                truncation=True, 
                max_length=512, 
                return_tensors='pt'
            ).to(device)
            
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-2]
            attention_mask = inputs['attention_mask']

            if pooling_strategy == 'cls':
                pooled_embeddings = last_hidden_state[:, 0, :]
            elif pooling_strategy == 'mean':
                mask_expanded = attention_mask.unsqueeze(-1).expand_as(last_hidden_state)
                sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
                sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
                pooled_embeddings = sum_embeddings / sum_mask.unsqueeze(-1)
            elif pooling_strategy == 'max':
                mask_expanded = attention_mask.unsqueeze(-1).expand_as(last_hidden_state)
                masked_hidden_state = last_hidden_state.masked_fill(mask_expanded == 0, -1e9)
                pooled_embeddings = torch.max(masked_hidden_state, dim=1)[0]
            
            all_embeddings.append(pooled_embeddings.cpu().numpy())
        
    if not all_embeddings:
        return np.array([])
        
    return np.vstack(all_embeddings)

def calculate_scores_batched(document_text, title, candidates, tokenizer, model, device, BATCH_SIZE=64, pooling_strategy='mean'):
    """
    Calculate global, theme, and position scores for candidate phrases.
    
    Args:
        document_text (str): Full document text
        title (str): Document title
        candidates (list): List of candidate phrase dictionaries
        tokenizer: HuggingFace tokenizer
        model: HuggingFace model
        device: torch device
        BATCH_SIZE (int): Batch size
        pooling_strategy (str): Pooling strategy
        
    Returns:
        tuple: (global_scores, theme_scores, position_scores) dictionaries
    """
    original_text = document_text
    masked_texts = []
    
    for c in candidates:
        phr = c["phrase"]
        n_words = len(phr.split())
        mask_seq = " ".join(["[MASK]"] * n_words)
        
        try:
            pattern = re.compile(r'\b' + re.escape(phr) + r'\b', re.IGNORECASE)
            masked_text = pattern.sub(mask_seq, original_text)
        except:
            masked_text = original_text.replace(phr, mask_seq)
            
        masked_texts.append(masked_text)

    candidate_phrases = [c['phrase'] for c in candidates]

    # Global Score
    global_scores = {}
    try:
        texts_for_global_score = [original_text] + masked_texts
        all_pooled_embeddings_global = get_pooled_embeddings_batched(
            texts_for_global_score, tokenizer, model, device, BATCH_SIZE, pooling_strategy
        )
        
        if all_pooled_embeddings_global.shape[0] > 0:
            original_embedding = all_pooled_embeddings_global[0:1]
            masked_embeddings = all_pooled_embeddings_global[1:]
            
            if masked_embeddings.shape[0] > 0:
                similarities_global = cosine_similarity(
                    masked_embeddings, 
                    original_embedding
                )
                global_scores = {
                    c['phrase']: (1.0 - sim) 
                    for c, sim in zip(candidates, similarities_global.flatten())
                }
            else:
                global_scores = {c['phrase']: 0 for c in candidates}
        else:
             global_scores = {c['phrase']: 0 for c in candidates}
            
    except Exception as e:
        global_scores = {c['phrase']: 0 for c in candidates}

    # Theme Score
    theme_scores = {}
    try:
        texts_for_pooling = [title] + candidate_phrases
        all_pooled_embeddings = get_pooled_embeddings_batched(
            texts_for_pooling, tokenizer, model, device,
            BATCH_SIZE=BATCH_SIZE, pooling_strategy='cls'
        )
    
        if all_pooled_embeddings.shape[0] > 0:
            title_embedding = all_pooled_embeddings[0:1]
            candidate_embeddings = all_pooled_embeddings[1:]
            if candidate_embeddings.shape[0] > 0:
                similarities_theme = cosine_similarity(candidate_embeddings, title_embedding)
                theme_scores = {c['phrase']: max(0.0, float(sim))
                                for c, sim in zip(candidates, similarities_theme.flatten())}
            else:
                theme_scores = {c['phrase']: 0.0 for c in candidates}
        else:
            theme_scores = {c['phrase']: 0.0 for c in candidates}
    except Exception as e:
        theme_scores = {c['phrase']: 0.0 for c in candidates}

    # Position Score
    position_scores = {c['phrase']: 1 / (c['position'] + 1) for c in candidates}
    
    return global_scores, theme_scores, position_scores

def reciprocal_rank_fusion(global_scores, theme_scores, position_scores, k=60):
    """
    Combine three ranking lists using Reciprocal Rank Fusion (RRF).
    
    Args:
        global_scores (dict): Global scores for phrases
        theme_scores (dict): Theme scores for phrases
        position_scores (dict): Position scores for phrases
        k (int): RRF parameter (default: 60)
    
    Returns:
        list: Final ranked list of phrases
    """
    all_phrases = set(global_scores.keys()) | set(theme_scores.keys()) | set(position_scores.keys())
    
    global_ranking = sorted(global_scores.items(), key=lambda x: x[1], reverse=True)
    theme_ranking = sorted(theme_scores.items(), key=lambda x: x[1], reverse=True)
    position_ranking = sorted(position_scores.items(), key=lambda x: x[1], reverse=True)
    
    global_ranks = {phrase: rank + 1 for rank, (phrase, _) in enumerate(global_ranking)}
    theme_ranks = {phrase: rank + 1 for rank, (phrase, _) in enumerate(theme_ranking)}
    position_ranks = {phrase: rank + 1 for rank, (phrase, _) in enumerate(position_ranking)}
    
    rrf_scores = {}
    
    for phrase in all_phrases:
        rrf_score = 0
        
        if phrase in global_ranks:
            rrf_score += 1 / (k + global_ranks[phrase])
        
        if phrase in theme_ranks:
            rrf_score += 1 / (k + theme_ranks[phrase])
        
        if phrase in position_ranks:
            rrf_score += 1 / (k + position_ranks[phrase])
        
        rrf_scores[phrase] = rrf_score
    
    final_ranking = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [phrase for phrase, score in final_ranking]

def extract_keyphrases(document_text, title, tokenizer, model, device, top_k=15):
    """
    Main pipeline function to extract keyphrases from a document.
    
    Args:
        document_text (str): Full document text
        title (str): Document title
        tokenizer: HuggingFace tokenizer
        model: HuggingFace model
        device: torch device
        top_k (int): Number of top keyphrases to return
        
    Returns:
        list: Ordered list of top keyphrases
    """
    if not document_text or not title:
        return []
    
    document_text = preprocess_text(document_text)
    candidates = extract_candidate_phrases(document_text)
    
    if not candidates:
        return []
    
    global_scores, theme_scores, position_scores = calculate_scores_batched(
        document_text, title, candidates, tokenizer, model, device,
        pooling_strategy='mean'
    )
    
    final_ranking = reciprocal_rank_fusion(global_scores, theme_scores, position_scores, k=40)
    
    return final_ranking[:top_k]

def main():
    """
    Main Streamlit application entry point.
    """
    st.set_page_config(
        page_title="Keyphrase Extraction System",
        page_icon="🔑",
        layout="wide"
    )
    
    st.title("🔑 Keyphrase Extraction System")
    st.markdown("### Extract ordered keyphrases using MDERank-RRF algorithm")
    
    # Load model
    with st.spinner("Loading model... This may take a moment on first run."):
        tokenizer, model, device = load_model_and_tokenizer()
        st.success("Model loaded successfully!")
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📝 Manual Input", "📁 CSV Upload"])
    
    # ========== TAB 1: MANUAL INPUT ==========
    with tab1:
        st.header("✍️ Manual Document Input")
        st.markdown("Enter document title and content manually to extract keyphrases.")
        
        # Manual input form
        with st.form("manual_input_form"):
            title_input = st.text_input(
                "📌 Document Title",
                placeholder="e.g., Machine Learning in Healthcare",
                help="Enter the title of your document"
            )
            
            text_input = st.text_area(
                "📄 Document Content",
                placeholder="Enter the full text content of your document here...",
                height=300,
                help="Enter the complete text content of your document"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                top_k_manual = st.number_input(
                    "Number of keyphrases to extract",
                    min_value=5,
                    max_value=50,
                    value=15,
                    step=5,
                    key="manual_top_k"
                )
            
            submit_manual = st.form_submit_button("🚀 Extract Keyphrases", type="primary", use_container_width=True)
        
        # Process manual input
        if submit_manual:
            if not title_input.strip():
                st.error("❌ Please enter a document title.")
            elif not text_input.strip():
                st.error("❌ Please enter document content.")
            else:
                with st.spinner("⏳ Extracting keyphrases... This may take a few seconds."):
                    try:
                        keyphrases = extract_keyphrases(
                            text_input,
                            title_input,
                            tokenizer,
                            model,
                            device,
                            top_k=top_k_manual
                        )
                        
                        # Display results
                        st.success(f"✅ Successfully extracted {len(keyphrases)} keyphrases!")
                        st.divider()
                        
                        st.subheader("📋 Extracted Keyphrases")
                        st.markdown("**Ordered by importance (most important first):**")
                        
                        # Display keyphrases in a nice format
                        for i, phrase in enumerate(keyphrases, 1):
                            st.markdown(f"**{i}.** {phrase}")
                        
                        # Show keyphrases as comma-separated
                        st.divider()
                        st.subheader("📝 Keyphrases (Comma-separated)")
                        keyphrases_text = ", ".join(keyphrases)
                        st.code(keyphrases_text, language=None)
                        
                        # Copy button
                        st.markdown("---")
                        st.download_button(
                            label="📥 Download Keyphrases as Text",
                            data=keyphrases_text,
                            file_name="extracted_keyphrases.txt",
                            mime="text/plain"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error extracting keyphrases: {str(e)}")
                        st.exception(e)
        
        # Show example
        with st.expander("💡 Example Input"):
            st.markdown("""
            **Title:** Machine Learning in Healthcare
            
            **Content:** 
            Machine learning algorithms are transforming healthcare by enabling predictive analytics 
            and personalized treatment recommendations. These advanced computational methods analyze 
            large datasets to identify patterns and make predictions about patient outcomes. 
            Deep learning models, in particular, have shown remarkable success in medical image 
            analysis and drug discovery. The integration of artificial intelligence in clinical 
            decision support systems is revolutionizing how healthcare providers deliver care.
            """)
    
    # ========== TAB 2: CSV UPLOAD ==========
    with tab2:
        st.header("📁 Upload CSV File")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="CSV file should contain 'title' and 'text' (or 'content') columns"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File loaded successfully! Found {len(df)} rows.")
            
            # Display column names
            st.subheader("📊 CSV Structure")
            st.write("**Available columns:**", ", ".join(df.columns.tolist()))
            st.dataframe(df.head(), use_container_width=True)
            
            # Check for required columns
            title_col = None
            text_col = None
            
            # Try to find title column
            for col in df.columns:
                if col.lower() in ['title', 'judul']:
                    title_col = col
                    break
            
            # Try to find text column
            for col in df.columns:
                if col.lower() in ['text', 'content', 'isi', 'abstract', 'body']:
                    text_col = col
                    break
            
            if not title_col or not text_col:
                st.error("❌ Please ensure your CSV has columns named 'title' (or 'judul') and 'text' (or 'content', 'isi', 'abstract', 'body')")
                st.stop()
            
            st.info(f"📝 Using columns: **{title_col}** (title) and **{text_col}** (text)")
            
            # Processing options
            st.header("⚙️ Processing Options")
            col1, col2 = st.columns(2)
            with col1:
                top_k = st.number_input("Number of keyphrases to extract", min_value=5, max_value=50, value=15, step=5)
            with col2:
                process_all = st.checkbox("Process all rows", value=False)
                max_rows = st.number_input("Max rows to process", min_value=1, max_value=len(df), value=min(5, len(df)), disabled=process_all)
            
            # Process button
            if st.button("🚀 Extract Keyphrases", type="primary"):
                rows_to_process = df if process_all else df.head(max_rows)
                
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, row in rows_to_process.iterrows():
                    status_text.text(f"Processing row {idx + 1}/{len(rows_to_process)}...")
                    progress_bar.progress((idx + 1) / len(rows_to_process))
                    
                    title = str(row[title_col]) if pd.notna(row[title_col]) else ""
                    text = str(row[text_col]) if pd.notna(row[text_col]) else ""
                    
                    if not title or not text:
                        results.append({
                            'row': idx + 1,
                            'title': title,
                            'keyphrases': [],
                            'error': 'Missing title or text'
                        })
                        continue
                    
                    try:
                        keyphrases = extract_keyphrases(text, title, tokenizer, model, device, top_k=top_k)
                        results.append({
                            'row': idx + 1,
                            'title': title[:100] + "..." if len(title) > 100 else title,
                            'keyphrases': keyphrases,
                            'error': None
                        })
                    except Exception as e:
                        results.append({
                            'row': idx + 1,
                            'title': title[:100] + "..." if len(title) > 100 else title,
                            'keyphrases': [],
                            'error': str(e)
                        })
                
                progress_bar.empty()
                status_text.empty()
                st.success(f"✅ Processed {len(results)} documents!")
                
                # Display results
                st.header("📋 Extraction Results")
                
                for result in results:
                    with st.expander(f"Row {result['row']}: {result['title']}", expanded=False):
                        if result['error']:
                            st.error(f"Error: {result['error']}")
                        else:
                            if result['keyphrases']:
                                st.write("**Extracted Keyphrases (ordered by importance):**")
                                for i, phrase in enumerate(result['keyphrases'], 1):
                                    st.write(f"{i}. {phrase}")
                            else:
                                st.warning("No keyphrases extracted.")
                
                # Download results
                st.header("💾 Download Results")
                results_df = pd.DataFrame([
                    {
                        'row': r['row'],
                        'title': r['title'],
                        'keyphrases': ', '.join(r['keyphrases']) if r['keyphrases'] else '',
                        'keyphrase_count': len(r['keyphrases'])
                    }
                    for r in results
                ])
                
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name="keyphrase_extraction_results.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")
            st.exception(e)
        
        else:
            st.info("👆 Please upload a CSV file to get started.")
            
            # Show example CSV format
            st.subheader("📝 Expected CSV Format")
            example_df = pd.DataFrame({
                'title': ['Machine Learning in Healthcare', 'Natural Language Processing'],
                'text': [
                    'Machine learning algorithms are transforming healthcare by enabling predictive analytics and personalized treatment recommendations.',
                    'Natural language processing enables computers to understand and generate human language through advanced neural networks.'
                ]
            })
            st.dataframe(example_df, use_container_width=True)
            st.caption("Your CSV should have similar structure with 'title' and 'text' columns")

if __name__ == "__main__":
    main()

