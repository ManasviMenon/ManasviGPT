import os
import faiss
import pickle
import numpy as np
from dotenv import load_dotenv
import requests
from sentence_transformers import SentenceTransformer # type: ignore

# ---------- LOAD ENV ----------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ---------- LOAD FAISS ----------
FAISS_FOLDER = "faiss_index"
INDEX_PATH = f"{FAISS_FOLDER}/faiss_index.bin"
TEXTS_PATH = f"{FAISS_FOLDER}/texts.pkl"

# Load FAISS index
index = faiss.read_index(INDEX_PATH)

# Load texts corresponding to embeddings
with open(TEXTS_PATH, "rb") as f:
    texts = pickle.load(f)

print(f"✅ FAISS index loaded with {len(texts)} texts")

# ---------- LOAD EMBEDDING MODEL ----------
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------- USER QUESTION ----------
question = input("Enter your question: ")

# Convert question to embedding
question_vector = model.encode([question], convert_to_numpy=True).astype("float32")

# Retrieve top-k relevant chunks
k = 5  # number of chunks to retrieve
distances, indices = index.search(question_vector, k)

retrieved_chunks = [texts[i] for i in indices[0]]
print("\nRetrieved Chunks:")
for i, chunk in enumerate(retrieved_chunks, 1):
    print(f"{i}. {chunk}\n")

# ---------- SEND TO GROQ ----------
# ---------- SEND TO GROQ ----------
context = "\n\n".join(retrieved_chunks)

url = "https://api.groq.com/openai/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

data = {
    "model": "llama-3.1-8b-instant",
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer strictly using the provided context."
        },
        {
            "role": "user",
            "content": f"""
Context:
{context}

Question:
{question}
"""
        }
    ],
    "temperature": 0.2
}

response = requests.post(url, headers=headers, json=data)

if response.status_code == 200:
    answer = response.json()["choices"][0]["message"]["content"]
    print("\n🟢 Groq Answer:\n", answer)
else:
    print("\n🔴 Error:", response.status_code, response.text)


