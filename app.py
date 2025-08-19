from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Enable cross-origin requests

# ---------------------------
# Load Q&A JSON
# ---------------------------
try:
    with open("data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
except FileNotFoundError:
    data = []

questions = [item['input'] for item in data]
answers = [item['output'] for item in data]

# Generic Q&A
generic_qa = [
    {"input": "hello", "output": "Hi! I'm Harsh Deep, a Python developer and tech enthusiast."},
    {"input": "hi", "output": "Hi! How can I help you today?"},
    {"input": "hey", "output": "Hey there! I'm Harsh Deep, a Python developer."},
    {"input": "help me", "output": "Sure! You can ask me about my skills, experience, AI projects, or contact info."},
    {"input": "who are you", "output": "I'm Harsh Deep, Python developer and tech enthusiast."},
    {"input": "bye", "output": "Goodbye! Feel free to reach out anytime."}
]

for item in generic_qa:
    questions.append(item["input"])
    answers.append(item["output"])

# ---------------------------
# TF-IDF vectorizer for lightweight embeddings
# ---------------------------
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)
SIMILARITY_THRESHOLD = 0.2  # Adjusted for TF-IDF

# ---------------------------
# Function to get answer
# ---------------------------
def get_answer(user_query):
    user_vec = vectorizer.transform([user_query])
    scores = cosine_similarity(user_vec, question_vectors)
    top_idx = np.argmax(scores)
    top_score = scores[0][top_idx]
    if top_score >= SIMILARITY_THRESHOLD:
        return answers[top_idx]
    else:
        return ("I can assist you better on "
                "<a href='https://wa.me/917009349232' target='_blank' "
                "style='text-decoration: underline; display: inline-flex; align-items: center; color:white;'>"
                "WhatsApp "
                "<img src='https://upload.wikimedia.org/wikipedia/commons/6/6b/WhatsApp.svg' "
                "alt='WhatsApp' style='width:24px; height:24px; margin-left:5px;'>"
                "</a>")

# ---------------------------
# API Route
# ---------------------------
@app.route('/ask', methods=['POST'])
def ask():
    data_in = request.json
    msg = data_in.get("message", "")
    reply = get_answer(msg)
    return jsonify({"reply": reply})

# ---------------------------
# Run the app on Render's assigned port
# ---------------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
