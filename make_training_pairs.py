import random
import joblib
from tokenizer import encode

# Load canonical Edmond early life
canon = open("data/novels/monte_cristo.txt", encoding="utf-8").read()
canon = canon[: int(0.1 * len(canon))]

kmeans = joblib.load("models/tokenizer.km")

sentences = canon.split(".")
sentences = [s.strip() for s in sentences if len(s) > 50]

pairs = []

# Generate consistent samples
for _ in range(200):
    s = " ".join(random.sample(sentences, 3))
    pairs.append((s, 1))   # consistent

# Generate contradictory samples
for _ in range(200):
    fake = random.choice([
        "Edmond was born a noble.",
        "Edmond ruled France.",
        "Edmond hated his father.",
        "Edmond married a different woman.",
        "Edmond was a villain."
    ])
    pairs.append((fake, 0))   # contradict

joblib.dump(pairs, "models/training_pairs.km")
print("Generated narrative training pairs")
