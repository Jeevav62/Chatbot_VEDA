import os
import json
import pickle
import random
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import nltk
import spacy
from textblob import TextBlob
from sympy import sympify, solve, Eq, symbols
import re

# 📚 NLP model setup
nltk.download('punkt')
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError("Run: python -m spacy download en_core_web_sm")

# 📂 File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
INTENTS_PATH = os.path.join(BASE_DIR, "intents.json")

# 📦 Load models and data
vectorizer = pickle.load(open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "rb"))
classifier = pickle.load(open(os.path.join(MODEL_DIR, "classifier.pkl"), "rb"))
with open(INTENTS_PATH, "r", encoding="utf-8") as f:
    intents_data = json.load(f)

# 🧠 NLP preprocessing
stemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')
chat_history = []

# 🎭 Sentiment analysis
def detect_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    return "positive" if polarity > 0.3 else "negative" if polarity < -0.3 else "neutral"

# 🔍 Entity recognition
def extract_entities(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents]

# 🎯 Intent detection
def predict_intents(text, top_k=2, threshold=0.3):
    tokens = tokenizer.tokenize(text.lower())
    stemmed = [stemmer.stem(t) for t in tokens]
    processed = " ".join(stemmed)
    vec = vectorizer.transform([processed])
    probs = classifier.predict_proba(vec)[0]
    tags = classifier.classes_
    sorted = probs.argsort()[::-1]
    return [(tags[i], probs[i]) for i in sorted if probs[i] >= threshold][:top_k] or [("fallback", 1.0)]

# 🧮 Math expression detection
def is_math_expression(text):
    math_keywords = ["calculate", "solve", "evaluate", "times", "plus", "minus", "multiplied", "divided"]
    has_math_symbols = bool(re.search(r"[\d\+\-\*/\^\(\)=x]", text))
    return any(word in text.lower() for word in math_keywords) or has_math_symbols

# 🧠 Evaluate math input
def evaluate_math(text):
    try:
        # Replace verbal math operators with symbols
        cleaned = text.lower()
        cleaned = cleaned.replace("plus", "+").replace("minus", "-")
        cleaned = cleaned.replace("times", "*").replace("multiplied by", "*")
        cleaned = cleaned.replace("divided by", "/").replace("^", "**")

        # Check for equation
        if "=" in cleaned:
            x = symbols("x")
            lhs, rhs = cleaned.split("=")
            equation = Eq(sympify(lhs.strip()), sympify(rhs.strip()))
            sol = solve(equation, x)
            return f"🧮 Solved equation: x = {sol[0]}"
        else:
            # Evaluate plain math expression
            result = sympify(cleaned)
            return f"🧮 Result: {result}"
    except Exception:
        return "⚠️ I couldn't evaluate that. Please check your equation or expression syntax."

# 💬 Response generator
def generate_response(tag, entities=None, sentiment="neutral"):
    entities = entities or []

    if tag == "time":
        now = datetime.now().strftime("%I:%M %p")
        return f"⏰ Current time is {now}."

    if tag == "date":
        today = datetime.now().strftime("%A, %d %B %Y")
        return f"📅 Today's date is {today}."

    if tag == "math":
        return "🧠 Send me any math expression or equation you'd like to solve!"

    for intent in intents_data.get("intents", []):
        if intent.get("tag") == tag:
            reply = random.choice(intent.get("responses", []))
            if sentiment == "positive":
                reply = "😄 That's great! " + reply
            elif sentiment == "negative":
                reply = "I'm here for you — " + reply
            if entities:
                reply += f" (I noticed: {', '.join(entities)})"
            return reply

    return "🤖 I’m not sure how to respond to that. Can you try rephrasing?"

# 🚀 Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    chat_history.append(user_input)

    if is_math_expression(user_input):
        response = evaluate_math(user_input)
        return jsonify({"response": response})

    intents = predict_intents(user_input)
    sentiment = detect_sentiment(user_input)
    entities = extract_entities(user_input)

    replies = [generate_response(tag, entities, sentiment) for tag, _ in intents]
    final_reply = " ".join(replies)

    return jsonify({"response": final_reply})

if __name__ == "__main__":
    app.run(debug=True)