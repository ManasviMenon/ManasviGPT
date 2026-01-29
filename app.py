from flask import Flask, request, jsonify  # type: ignore
from query_chatbot import answer_question
import os

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("question")

    if not user_input:
        return jsonify({"error": "No question provided"}), 400

    response = answer_question(user_input)
    return jsonify({"answer": response})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render uses PORT
    app.run(host="0.0.0.0", port=port)
