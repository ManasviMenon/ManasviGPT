# ManasviGPT : RAG Portfolio Chatbot

A personal portfolio chatbot built on a full RAG pipeline live at [manasvigpt.online](https://manasvigpt.online)

## How It Works
1. Knowledge base split into chunks and embedded using `all-MiniLM-L6-v2`
2. Chunks indexed with FAISS for fast semantic retrieval
3. User query retrieves the most relevant context chunks
4. Context passed to Llama 3.1 via Groq API to generate a grounded response
5. Served via Flask with CORS, deployed on Render

## Tech Stack
Python · FAISS · Sentence Transformers · Groq API · Flask · Render

## Files
| File | Description |
|------|-------------|
| `app.py` | Flask server, CORS config, and API routes |
| `query_chatbot.py` | Full RAG pipeline — retrieval, intent detection, and generation |
| `build_faiss.py` | Builds and saves the FAISS vector index |
| `requirements.txt` | Dependencies |
| `runtime.txt` | Python runtime for Render |
