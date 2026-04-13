import json
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import re

# ─── Load Dataset ────────────────────────────────────────────────────────────
with open("intents.json", "r") as f:
    data = json.load(f)

# ─── Preprocess Text ─────────────────────────────────────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ─── Build Training Data ─────────────────────────────────────────────────────
X = []  # patterns
y = []  # intent labels

for intent in data["intents"]:
    tag = intent["tag"]
    if tag == "unknown":
        continue
    for pattern in intent["patterns"]:
        X.append(clean_text(pattern))
        y.append(tag)

print(f"✅ Total training samples: {len(X)}")
print(f"✅ Unique intents: {len(set(y))}")

# ─── Train/Test Split ─────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─── Build Pipeline (TF-IDF + Logistic Regression) ───────────────────────────
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),   # unigrams and bigrams
        max_features=5000,
        sublinear_tf=True
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        C=10,
        solver="lbfgs",
        multi_class="auto"
    ))
])

# ─── Train Model ─────────────────────────────────────────────────────────────
pipeline.fit(X_train, y_train)

# ─── Evaluate ────────────────────────────────────────────────────────────────
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n🎯 Model Accuracy: {acc * 100:.2f}%\n")
print("📊 Classification Report:")
print(classification_report(y_test, y_pred))

# ─── Save Model ──────────────────────────────────────────────────────────────
with open("chatbot_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

with open("intents_data.pkl", "wb") as f:
    pickle.dump(data, f)

print("✅ Model saved as chatbot_model.pkl")
print("✅ Intents data saved as intents_data.pkl")
