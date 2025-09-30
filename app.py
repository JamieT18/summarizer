import os
from flask import Flask, render_template, request
import re
from typing import List
import numpy as np
import torch
from transformers import pipeline
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from langdetect import detect
import nltk

# Ensure required NLTK data is available
def ensure_nltk_data():
    try:
        stopwords.words('english')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

ensure_nltk_data()

MAX_WORDS = 5000  # Server-side input cap

class TextSummarizer:
    def __init__(self, language='english', abstractive_model_name="facebook/bart-large-cnn"):
        self.language = language
        self.stop_words = set(stopwords.words(language)) if language in stopwords.fileids() else set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.abstractive_model_name = abstractive_model_name
        self.abstractive_model = None
        self.abstractive_model_loaded = False
        self.sentence_embedder = None

    def autodetect_language(self, text: str) -> str:
        try:
            return detect(text)
        except Exception:
            return 'english'

    def _preprocess_text(self, text: str) -> str:
        text = re.sub(r'\[[0-9]*\]', ' ', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.,]', '', text)
        text = text.lower()
        return text.strip()

    def _filter_short_sentences(self, sentences: List[str], min_length: int = 20) -> List[str]:
        return [s for s in sentences if len(s) > min_length]

    def extractive_summarize(self, text: str, summary_length_ratio: float = 0.2, use_embeddings=True) -> (str, List[float]):
        if not text.strip():
            return "No text to summarize.", []
        processed_text = self._preprocess_text(text)
        sentences = sent_tokenize(processed_text)
        sentences = self._filter_short_sentences(sentences)
        if len(sentences) <= 1:
            return text.strip(), [1.0]

        if use_embeddings:
            # Semantic similarity matrix
            if self.sentence_embedder is None:
                self.sentence_embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            embeddings = self.sentence_embedder.encode(sentences)
            similarity_matrix = cosine_similarity(embeddings)
        else:
            # TF-IDF fallback
            from sklearn.feature_extraction.text import TfidfVectorizer
            tfidf_vectorizer = TfidfVectorizer(stop_words=self.language)
            tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
            similarity_matrix = cosine_similarity(tfidf_matrix)

        scores = self._pagerank(similarity_matrix)
        num_sentences = max(1, int(len(sentences) * summary_length_ratio))
        ranked_sentence_indices = np.argsort(scores)[::-1]
        summary_indices = sorted(ranked_sentence_indices[:num_sentences])
        summary = [sentences[i] for i in summary_indices]
        return ' '.join(summary), scores.tolist()

    def _pagerank(self, similarity_matrix: np.ndarray, eps: float = 0.0001, d: float = 0.85) -> np.ndarray:
        N = similarity_matrix.shape[0]
        scores = np.ones(N) / N
        similarity_matrix = similarity_matrix / (similarity_matrix.sum(axis=1, keepdims=True) + 1e-8)
        for _ in range(100):
            prev_scores = scores.copy()
            scores = (1 - d) / N + d * similarity_matrix.T.dot(scores)
            if np.linalg.norm(scores - prev_scores) < eps:
                break
        return scores

    def _chunk_text_by_tokens(self, text: str, max_tokens: int = 512, overlap_tokens: int = 50) -> List[str]:
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_tokens = 0
        for sentence in sentences:
            tokens = len(word_tokenize(sentence))
            if current_tokens + tokens <= max_tokens:
                current_chunk.append(sentence)
                current_tokens += tokens
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    overlap = []
                    overlap_count = 0
                    for s in reversed(current_chunk):
                        t = len(word_tokenize(s))
                        if overlap_count + t <= overlap_tokens:
                            overlap.insert(0, s)
                            overlap_count += t
                        else:
                            break
                    current_chunk = overlap + [sentence]
                    current_tokens = sum(len(word_tokenize(s)) for s in current_chunk)
                else:
                    current_chunk = [sentence]
                    current_tokens = tokens
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        return chunks

    def _load_abstractive_model(self):
        if not self.abstractive_model_loaded:
            try:
                self.abstractive_model = pipeline(
                    "summarization", model=self.abstractive_model_name
                )
                self.abstractive_model_loaded = True
            except Exception as e:
                print(f"Error loading summarization model: {e}")
                self.abstractive_model = None

    def abstractive_summarize(self, text: str, min_length: int = 50, max_length: int = 150) -> str:
        if not text.strip():
            return "No text to summarize."
        self._load_abstractive_model()
        if self.abstractive_model is None:
            return "Abstractive summarization model could not be loaded. Please check your installation."
        chunks = self._chunk_text_by_tokens(text, max_tokens=512, overlap_tokens=50)
        try:
            summaries = self.abstractive_model(
                chunks, max_length=max_length, min_length=min_length, do_sample=False
            )
            combined_summary = self._smart_merge_summaries([s['summary_text'] for s in summaries])
            return combined_summary
        except Exception as e:
            return f"Abstractive summarization failed: {e}"

    def _smart_merge_summaries(self, summaries: List[str]) -> str:
        # Simple deduplication + second-pass summarization if needed
        merged = ' '.join(dict.fromkeys(summaries))  # deduplicate
        if len(merged.split()) > 250:
            # Second-pass summarization for coherence
            self._load_abstractive_model()
            try:
                result = self.abstractive_model(
                    merged, max_length=150, min_length=50, do_sample=False
                )
                return result[0]['summary_text']
            except Exception:
                return merged
        return merged

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'summarizersecret')

MODELS = [
    ("facebook/bart-large-cnn", "BART Large CNN"),
    ("google/pegasus-xsum", "Pegasus XSUM"),
    ("t5-base", "T5 Base"),
]

@app.route('/', methods=['GET', 'POST'])
def index():
    summary = ""
    original_text = ""
    scores = []
    method = "Extractive"
    model_choice = MODELS[0][0]
    summary_length_ratio = 0.2
    min_length = 50
    max_length = 150
    error = None
    compression = 0.0
    detected_language = 'english'

    if request.method == "POST":
        method = request.form.get("method", "Extractive")
        model_choice = request.form.get("abstractive_model", MODELS[0][0])
        summary_length_ratio = float(request.form.get("summary_length_ratio", 0.2))
        min_length = int(request.form.get("min_length", 50))
        max_length = int(request.form.get("max_length", 150))
        text_input = request.form.get("text_input", "")
        file = request.files.get("file_input")
        if file and file.filename:
            try:
                original_text = file.read().decode("utf-8")
            except Exception as e:
                error = f"Error reading file: {e}"
        else:
            original_text = text_input

        word_count = len(original_text.split())
        if word_count > MAX_WORDS:
            error = f"Warning: Input exceeds recommended limit ({MAX_WORDS} words). Accuracy may be reduced. Processing anyway."
        summarizer = TextSummarizer(abstractive_model_name=model_choice)
        detected_language = summarizer.autodetect_language(original_text)
        summarizer.language = detected_language
        summarizer.stop_words = set(stopwords.words(detected_language)) if detected_language in stopwords.fileids() else set(stopwords.words('english'))
        try:
            if method == "Abstractive":
                summary = summarizer.abstractive_summarize(
                    original_text, min_length=min_length, max_length=max_length
                )
            else:
                summary, scores = summarizer.extractive_summarize(
                    original_text, summary_length_ratio
                )
            if original_text.strip():
                compression = round(100 * len(summary.split()) / max(1, len(original_text.split())), 2)
        except Exception as e:
            error = f"Summarization error: {e}"

    return render_template(
        "index.html",
        summary=summary,
        original_text=original_text,
        scores=scores,
        method=method,
        model_choice=model_choice,
        summary_length_ratio=summary_length_ratio,
        min_length=min_length,
        max_length=max_length,
        error=error,
        models=MODELS,
        compression=compression,
        detected_language=detected_language
    )

if __name__ == "__main__":
    app.run(debug=True)
