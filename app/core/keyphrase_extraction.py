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
from nltk.stem import WordNetLemmatizer

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

# Initialize NLTK components
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def get_wordnet_pos(treebank_tag):
    """
    Convert NLTK POS tag to WordNet POS tag for lemmatization.
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
    """
    text = text.lower()
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'[^\w\s.,;:!?()-]', '', text)
    return text

def extract_candidate_phrases(text, min_length=1, max_length=5):
    """
    Extract candidate keyphrases from text using POS tagging and chunking.
    """
    text = preprocess_text(text)
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    grammar = """NP: {<NN.*|JJ>*<NN.*>}"""
    cp = RegexpParser(grammar)
    tree = cp.parse(pos_tags)
    candidates = {}
    
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
    Generate embeddings for a batch of texts.
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
            
    except Exception:
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
    except Exception:
        theme_scores = {c['phrase']: 0.0 for c in candidates}

    # Position Score
    position_scores = {c['phrase']: 1 / (c['position'] + 1) for c in candidates}
    
    return global_scores, theme_scores, position_scores

def reciprocal_rank_fusion(global_scores, theme_scores, position_scores, k=60):
    """
    Combine three ranking lists using Reciprocal Rank Fusion (RRF).
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
    return final_ranking

def extract_keyphrases(document_text, title, tokenizer, model, device, top_k=15):
    """
    Main pipeline function to extract keyphrases from a document.
    
    Returns:
        list: Ordered list of (keyphrase, score) tuples.
    """
    # The title is often a good source for the main keyphrase, but we need the abstract for context
    if not document_text:
        return []
    
    # If no title is provided, use the first few sentences of the document as a proxy.
    if not title:
        title = " ".join(document_text.split('.')[:2])

    
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