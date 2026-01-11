import joblib
import numpy as np
from consistency_engine import extract_features

pairs = joblib.load("models/training_pairs.km")

X = []
Y = []

for text, label in pairs:
    feats = extract_features("monte", text)
    X.append(feats)
    Y.append(label)

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(X, Y)

joblib.dump(clf, "models/classifier.pt")
print("Classifier retrained on narrative canon")
