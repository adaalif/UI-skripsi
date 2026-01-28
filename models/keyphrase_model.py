import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.tag import pos_tag
from nltk.chunk import RegexpParser
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

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


class KeyphraseModel:
    def __init__(self, tokenizer, model, device):
        """
        inisialisasi extractor make preloaded model, tokenizer, sama device
        """
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def _get_wordnet_pos(self, treebank_tag):
        """
        Convert NLTK POS tag ke wordnet buat lemma
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

    def _preprocess_text(self, text):
        """
        normalisasi dan bersihin input text
        """
        text = text.lower()
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'[^\w\s.,;:!?()-]', '', text)
        return text

    def _extract_candidate_phrases(self, text, min_length=1, max_length=5):
        """
        ekstract kandidate keyphrase
        """
        text = self._preprocess_text(text)
        tokens = nltk.word_tokenize(text)
        pos_tags = pos_tag(tokens)
        grammar = """NP: {<NN.*|JJ>*<NN.*>}"""
        chunk_phrases = RegexpParser(grammar)
        tree = chunk_phrases.parse(pos_tags)
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
                    wn_tag = self._get_wordnet_pos(tag)
                    lemma_tokens.append(self.lemmatizer.lemmatize(word, wn_tag))
                
                lemma_form = " ".join(lemma_tokens)
                
                if (lemma_tokens[0] in self.stop_words or 
                    lemma_tokens[-1] in self.stop_words):
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

    def _get_pooled_embeddings_batched(self, texts, BATCH_SIZE=64, pooling_strategy='mean'):
        """
        bikin embedding
        """
        all_embeddings = []
        
        with torch.inference_mode():
            for i in range(0, len(texts), BATCH_SIZE):
                batch_texts = texts[i : i + BATCH_SIZE]
                
                inputs = self.tokenizer(
                    batch_texts, 
                    padding='max_length',
                    truncation=True, 
                    max_length=512, 
                    return_tensors='pt'
                ).to(self.device)
                
                outputs = self.model(**inputs, output_hidden_states=True)
                last_hidden_state = outputs.hidden_states[-2]
                attention_mask = inputs['attention_mask']

                if pooling_strategy == 'mean':
                    mask_expanded = attention_mask.unsqueeze(-1).expand_as(last_hidden_state)
                    sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
                    sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
                    pooled_embeddings = sum_embeddings / sum_mask.unsqueeze(-1)
                elif pooling_strategy == 'cls':
                    pooled_embeddings = last_hidden_state[:, 0, :]
                
                all_embeddings.append(pooled_embeddings.cpu().numpy())
            
        if not all_embeddings:
            return np.array([])
            
        return np.vstack(all_embeddings)

    def _calculate_scores_batched(self, document_text, title, candidates, BATCH_SIZE=64, pooling_strategy='mean'):
        """
        itung semua skornya 
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

        # skor global
        global_scores = {}
        try:
            texts_for_global_score = [original_text] + masked_texts
            all_pooled_embeddings_global = self._get_pooled_embeddings_batched(
                texts_for_global_score, BATCH_SIZE, pooling_strategy
            )
            
            if all_pooled_embeddings_global.shape[0] > 0:
                original_embedding = all_pooled_embeddings_global[0:1]
                masked_embeddings = all_pooled_embeddings_global[1:]
                
                if masked_embeddings.shape[0] > 0:
                    similarities_global = cosine_similarity(masked_embeddings, original_embedding)
                    global_scores = {c['phrase']: (1.0 - sim) for c, sim in zip(candidates, similarities_global.flatten())}
                else:
                    global_scores = {c['phrase']: 0 for c in candidates}
            else:
                 global_scores = {c['phrase']: 0 for c in candidates}
        except Exception:
            global_scores = {c['phrase']: 0 for c in candidates}

        # skor tema
        theme_scores = {}
        try:
            texts_for_pooling = [title] + candidate_phrases
            all_pooled_embeddings = self._get_pooled_embeddings_batched(
                texts_for_pooling, BATCH_SIZE=BATCH_SIZE, pooling_strategy='cls'
            )
        
            if all_pooled_embeddings.shape[0] > 0:
                title_embedding = all_pooled_embeddings[0:1]
                candidate_embeddings = all_pooled_embeddings[1:]
                if candidate_embeddings.shape[0] > 0:
                    similarities_theme = cosine_similarity(candidate_embeddings, title_embedding)
                    theme_scores = {c['phrase']: max(0.0, float(sim)) for c, sim in zip(candidates, similarities_theme.flatten())}
                else:
                    theme_scores = {c['phrase']: 0.0 for c in candidates}
            else:
                theme_scores = {c['phrase']: 0.0 for c in candidates}
        except Exception:
            theme_scores = {c['phrase']: 0.0 for c in candidates}

        # skor posisi
        position_scores = {c['phrase']: 1 / (c['position'] + 1) for c in candidates}
        
        return global_scores, theme_scores, position_scores

    def _reciprocal_rank_fusion(self, global_scores, theme_scores, position_scores, k=60):
        """
        ini rrf 
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

    def extract(self, document_text, title, top_k=15, progress_bar=None, status_text=None):
        """
        ini kek digabung semua ke main pipelinenya 
        """
        if not document_text:
            return []
        
        

        if status_text: status_text.text("Tahap 1/4: Pra-pemrosesan teks...")
        if progress_bar: progress_bar.progress(10)
        document_text = self._preprocess_text(document_text)
        
        if status_text: status_text.text("Tahap 2/4: Mencari kandidat frasa...")
        if progress_bar: progress_bar.progress(30)
        candidates = self._extract_candidate_phrases(document_text)
        if not candidates:
            return []
        
  
        if status_text: status_text.text("Tahap 3/4: Menghitung skor ")
        if progress_bar: progress_bar.progress(50)
        global_scores, theme_scores, position_scores = self._calculate_scores_batched(
            document_text, title, candidates, pooling_strategy='mean'
        )
        
        if status_text: status_text.text("Tahap 4/4: Memberi peringkat")
        if progress_bar: progress_bar.progress(90)
        final_ranking = self._reciprocal_rank_fusion(global_scores, theme_scores, position_scores, k=60)
        
        detailed_results = []
        for phrase, rrf_score in final_ranking[:top_k]:
            detailed_results.append({
                "Keyphrase": phrase,
                "Skor Akhir": rrf_score,
                "Skor Global": global_scores.get(phrase, 0.0),
                "Skor Tema": theme_scores.get(phrase, 0.0),
                "Skor Posisi": position_scores.get(phrase, 0.0),
            })
            
        return detailed_results
