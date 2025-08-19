from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from sentence_transformers import SentenceTransformer, util
import torch

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# ---------------------------
# Load Q&A JSON
# ---------------------------
with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

questions = [item['input'] for item in data]
answers = [item['output'] for item in data]

# Add generic Q&A
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
# Load embedding model
# ---------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')
question_embeddings = model.encode(questions, convert_to_tensor=True)

SIMILARITY_THRESHOLD = 0.55

def get_answer(user_query):
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, question_embeddings)
    top_score, top_idx = torch.max(cos_scores, dim=1)
    top_score = top_score.item()
    
    if top_score >= SIMILARITY_THRESHOLD:
        return answers[top_idx.item()]
    else:
        # Fallback with WhatsApp link
        return ("I can assist you better on "
        "<a href='https://wa.me/917009349232' target='_blank' "
        "style='text-decoration: underline; display: inline-flex; align-items: center; color:white; '>"
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
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
