import joblib
from tokenizer import encode
from joblib import load

# Load Monte Cristo
with open("data/novels/monte_cristo.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Use only Edmond’s early life (first 10%)
early = text[: int(0.1 * len(text))]

# Load tokenizer
kmeans = load("models/tokenizer.km")

# Encode
tokens = encode(early, kmeans)

# Save canonical belief state
joblib.dump(tokens, "models/monte_tokens.km")

print("Cached Edmond early-life tokens.")
