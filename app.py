from flask import Flask, request, jsonify, render_template
import pickle
import random
import re
import os

app = Flask(__name__)

# ─── Load Model & Intents ─────────────────────────────────────────────────────
MODEL_PATH = "chatbot_model.pkl"
INTENTS_PATH = "intents_data.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ Model not found! Please run: python train_model.py first.")

with open(MODEL_PATH, "rb") as f:
    pipeline = pickle.load(f)

with open(INTENTS_PATH, "rb") as f:
    intents_data = pickle.load(f)

# ─── Text Cleaner ─────────────────────────────────────────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ─── Get Response by Intent Tag ───────────────────────────────────────────────
def get_response(tag):
    for intent in intents_data["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    # fallback to unknown
    for intent in intents_data["intents"]:
        if intent["tag"] == "unknown":
            return random.choice(intent["responses"])
    return "I'm sorry, I couldn't understand that. Please try again."

# ─── Predict Intent ───────────────────────────────────────────────────────────
def predict_intent(user_message):
    cleaned = clean_text(user_message)
    predicted_tag = pipeline.predict([cleaned])[0]
    probabilities = pipeline.predict_proba([cleaned])[0]
    confidence = max(probabilities)

    # If confidence is too low, return unknown
    if confidence < 0.35:
        predicted_tag = "unknown"

    response = get_response(predicted_tag)
    return {
        "intent": predicted_tag,
        "confidence": round(float(confidence), 4),
        "response": response
    }

# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' field in request body"}), 400

    user_message = data["message"].strip()

    if not user_message:
        return jsonify({"error": "Message cannot be empty"}), 400

    result = predict_intent(user_message)

    return jsonify({
        "user_message": user_message,
        "intent": result["intent"],
        "confidence": result["confidence"],
        "response": result["response"]
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "loaded"})

@app.route("/intents", methods=["GET"])
def list_intents():
    tags = [i["tag"] for i in intents_data["intents"]]
    return jsonify({"intents": tags, "total": len(tags)})

# ─── Run Server ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
