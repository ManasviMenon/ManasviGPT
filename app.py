print("FLASK APP STARTED")

from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found in .env")


from flask import Flask, request, jsonify  # type: ignore
from query_chatbot import answer_question
import os

app = Flask(__name__)

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "ManasviGPT API is running. Use POST /chat"}), 200

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True, silent=True) or {}
    user_input = data.get("question")

    if not user_input:
        return jsonify({"error": "No question provided"}), 400

    response = answer_question(user_input)
    return jsonify({"answer": response})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render uses PORT
    app.run(host="0.0.0.0", port=port)
