import pandas as pd
import joblib
from consistency_engine import extract_features

# Load classifier
clf = joblib.load("models/classifier.pt")

# Load test set
df = pd.read_csv("data/test.csv")

predictions = []

for _, row in df.iterrows():
    backstory = row["content"]
    book = row["book_name"]

    
    feats = extract_features(book, backstory, backstory)
    pred = clf.predict([feats])[0]

    # Convert 0/1 to required labels
    label = "consistent" if pred == 1 else "contradict"

    predictions.append([row["id"], label])

# Save submission
out = pd.DataFrame(predictions, columns=["id", "label"])
out.to_csv("submission.csv", index=False)

print("submission.csv created")