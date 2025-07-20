import os
import json
import pickle
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer

# 📦 Setup
nltk.download('punkt')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTENTS_PATH = os.path.join(BASE_DIR, "intents.json")

# 📂 Load intents safely
try:
    with open(INTENTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
except Exception as e:
    print(f"❌ Error loading intents.json: {e}")
    exit()

# 🧠 Preprocessing
stemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')
X, y = [], []

# 🛡 Robust loop with skip warnings
for intent in data["intents"]:
    patterns = intent.get("patterns")
    if not patterns:
        print(f"⚠️ Skipping intent: {intent.get('tag', 'unknown')} — missing 'patterns'")
        continue
    for pattern in patterns:
        tokens = tokenizer.tokenize(pattern.lower())
        stemmed = [stemmer.stem(token) for token in tokens]
        X.append(" ".join(stemmed))
        y.append(intent["tag"])

# 📊 Train model
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

clf = MultinomialNB()
clf.fit(X_vec, y)

# 💾 Save artifacts
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

with open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "wb") as f:
    pickle.dump(vectorizer, f)

with open(os.path.join(MODEL_DIR, "classifier.pkl"), "wb") as f:
    pickle.dump(clf, f)

print("✅ Model training complete. Files saved in /model")