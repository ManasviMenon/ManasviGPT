print("FLASK APP STARTED")

from dotenv import load_dotenv
load_dotenv()

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from query_chatbot import answer_question

app = Flask(__name__)

# ✅ Allow your frontend domains
CORS(app, resources={
    r"/chat": {
        "origins": [
            "https://manasvigpt.online",
            "https://www.manasvigpt.online",
            "https://manasvimenon.github.io"   # optional if you still use GH pages
        ]
    }
})

@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    # ✅ Preflight request
    if request.method == "OPTIONS":
        return ("", 204)

    data = request.get_json(silent=True) or {}
    user_input = data.get("question")

    if not user_input:
        return jsonify({"error": "No question provided"}), 400

    response = answer_question(user_input)
    return jsonify({"answer": response})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)