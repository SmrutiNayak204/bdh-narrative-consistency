import os
import nltk
import joblib
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from tokenizer import encode
from bdh import BDH, BDHConfig

# Ensure NLTK works on Render
nltk.data.path.append("./nltk_data")

print("Loading models...")

# -----------------------------
# Load semantic encoder (cached locally)
# -----------------------------
embedder = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device="cpu",
    cache_folder="./hf_cache"
)

# -----------------------------
# Load BDH
# -----------------------------
config = BDHConfig(
    vocab_size=256,
    n_embd=64,
    n_layer=3,
    n_head=2,
    mlp_internal_dim_multiplier=32
)

bdh = BDH(config)
bdh.load_state_dict(torch.load("models/bdh.pt", map_location="cpu"))
bdh.eval()

# -----------------------------
# Load canon (early Edmond)
# -----------------------------
with open("data/novels/monte_cristo.txt", encoding="utf-8") as f:
    canon = f.read()

canon = canon[:8000]
canon_embed = embedder.encode(canon, normalize_embeddings=True)

# -----------------------------
# Load tokenizer
# -----------------------------
kmeans = joblib.load("models/tokenizer.km")

# -----------------------------
# Run BDH safely
# -----------------------------
def run_bdh(text):
    tokens = encode(text, kmeans)

    if len(tokens) < 10:
        return torch.zeros(64)

    idx = torch.tensor([tokens[:300]], dtype=torch.long)

    with torch.no_grad():
        _, _, h = bdh(idx)

    return h.squeeze(0).float()

# -----------------------------
# Feature extractor
# -----------------------------
def extract_features(book_name, backstory):
    # ---- Semantic similarity ----
    back_embed = embedder.encode(backstory, normalize_embeddings=True)
    semantic = float(np.dot(canon_embed, back_embed))

    # ---- BDH belief drift ----
    H_back = run_bdh(backstory)
    H_canon = run_bdh(canon)

    drift = torch.norm(H_back - H_canon).item()
    drift_norm = min(drift / 250.0, 1.0)

    return semantic, drift_norm

