import nltk, joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans

nltk.download("punkt")

embedder = SentenceTransformer("all-MiniLM-L6-v2")
VOCAB_SIZE = 256
CHUNK_SIZE = 5

def chunk_text(text):
    s = nltk.sent_tokenize(text)
    return [" ".join(s[i:i+CHUNK_SIZE]) for i in range(0, len(s), CHUNK_SIZE)]

def train_tokenizer(texts):
    chunks = []
    for t in texts:
        chunks += chunk_text(t)
    X = embedder.encode(chunks, show_progress_bar=True)
    kmeans = MiniBatchKMeans(VOCAB_SIZE).fit(X)
    joblib.dump(kmeans, "models/tokenizer.km")
    return kmeans

def encode(text, kmeans):
    chunks = chunk_text(text)
    X = embedder.encode(chunks)
    return kmeans.predict(X)
