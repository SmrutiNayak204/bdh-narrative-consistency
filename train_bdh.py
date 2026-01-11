import torch
import joblib
import numpy as np
from tokenizer import encode
from bdh import BDH, BDHConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Load tokenizer
# -------------------------------
kmeans = joblib.load("models/tokenizer.km")

# -------------------------------
# Load novels
# -------------------------------
with open("data/novels/monte_cristo.txt", "r", encoding="utf-8") as f:
    monte = f.read()

with open("data/novels/castaways.txt", "r", encoding="utf-8") as f:
    castaways = f.read()

novels = [monte, castaways]

# -------------------------------
# Encode novels
# -------------------------------
print("Encoding novels into narrative tokens...")
seqs = [encode(n, kmeans) for n in novels]

# -------------------------------
# Batch generator
# -------------------------------
def make_batch(batch_size=16, block_size=64):
    X = []
    Y = []

    for _ in range(batch_size):
        s = seqs[np.random.randint(len(seqs))]
        i = np.random.randint(0, len(s) - block_size - 1)
        X.append(s[i:i+block_size])
        Y.append(s[i+1:i+block_size+1])

    X = torch.tensor(np.array(X), dtype=torch.long, device=device)
    Y = torch.tensor(np.array(Y), dtype=torch.long, device=device)
    return X, Y

# -------------------------------
# Initialize BDH
# -------------------------------
config = BDHConfig(
    vocab_size=256,
    n_embd=96,
    n_layer=4,
    n_head=4,
    mlp_internal_dim_multiplier=64
)

model = BDH(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

print("Training BDH on narrative data...")

# -------------------------------
# Train
# -------------------------------
for step in range(3000):
    X, Y = make_batch()
    logits, loss, _ = model(X, Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"Step {step:4d} | Loss {loss.item():.4f}")

# -------------------------------
# Save model
# -------------------------------
torch.save(model.state_dict(), "models/bdh.pt")
print("BDH saved to models/bdh.pt")