import os
import joblib
from tokenizer import train_tokenizer

# Load both novels
with open("data/novels/monte_cristo.txt", "r", encoding="utf-8") as f:
    monte = f.read()

with open("data/novels/castaways.txt", "r", encoding="utf-8") as f:
    castaways = f.read()

print("Training tokenizer on novels...")
kmeans = train_tokenizer([monte, castaways])

os.makedirs("models", exist_ok=True)
joblib.dump(kmeans, "models/tokenizer.km")

print("Tokenizer saved to models/tokenizer.km")
