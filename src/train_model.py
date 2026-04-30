import pandas as pd
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

from preprocess import clean_text
from features import get_tfidf_features

# -------------------------
# Load dataset
# -------------------------
df = pd.read_csv("data/social_data.csv")

# -------------------------
# Preprocessing
# -------------------------
df["clean_text"] = df["text"].apply(clean_text)

# -------------------------
# Feature extraction
# -------------------------
X, vectorizer = get_tfidf_features(df["clean_text"])
y = df["sentiment"]

# -------------------------
# Train model
# -------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# -------------------------
# Predictions
# -------------------------
y_pred = model.predict(X)

# -------------------------
# Evaluation
# -------------------------
print("\nAccuracy:", accuracy_score(y, y_pred))
print("\nClassification Report:\n")
print(classification_report(y, y_pred))

# -------------------------
# Confusion Matrix
# -------------------------
cm = confusion_matrix(y, y_pred, labels=["positive", "negative", "neutral"])

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=["positive", "negative", "neutral"],
            yticklabels=["positive", "negative", "neutral"])

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("outputs/confusion_matrix.png")
plt.show()

# -------------------------
# Save model
# -------------------------
pickle.dump(model, open("models/model.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))

print("\nModel saved successfully!")