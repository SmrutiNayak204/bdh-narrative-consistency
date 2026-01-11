import pandas as pd
from consistency_engine import extract_features

df = pd.read_csv("data/train.csv")

vals = []

for _, row in df.iterrows():
    feats = extract_features(row["book_name"], row["content"], row["char"])
    vals.append((feats[0], feats[2], row["label"]))

for d, c, label in vals[:20]:
    print(label, " delta=", round(d,2), " char_drift=", round(c,2))
