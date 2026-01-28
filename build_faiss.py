import os
import pickle
import numpy as np
import faiss # type: ignore

from langchain_text_splitters import RecursiveCharacterTextSplitter # type: ignore
from langchain_core.documents import Document # type: ignore
from sentence_transformers import SentenceTransformer # type: ignore



from sentence_transformers import SentenceTransformer # type: ignore

# ---------- CONFIG ----------
DATA_FOLDER = "D:/Chatbotfolder"
FAISS_FOLDER = "faiss_index"

FILENAME_TO_SECTION = {
    "Project 1 Description.txt": "taxi_project",
    "Project 2 Description.txt": "airbnb_project",
    "Work Experience AIESEC.txt": "aiesec",
    "Work Experience CoinDCX.txt": "coindcx",
    "Manasvi Menon Resume.txt": "experience",
    "Manasvi Menon Background and Values.txt": "background",
    "Skills and Tools.txt": "skills",
    "FAQs.txt": "faq"
}


# ---------- LOAD FILES ----------
documents = []

for file in os.listdir(DATA_FOLDER):
    if file.endswith(".txt"):
        section = FILENAME_TO_SECTION.get(file, "experience")

        with open(os.path.join(DATA_FOLDER, file), "r", encoding="utf-8") as f:
            documents.append(
                Document(
                    page_content=f.read(),
                    metadata={"section": section}
                )
            )

print(f"Loaded {len(documents)} documents")

# ---------- SPLIT ----------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=120
)

chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

texts = [
    {
        "text": chunk.page_content,
        "section": chunk.metadata["section"]
    }
    for chunk in chunks
]

# ---------- EMBEDDINGS (LOCAL – HUGGING FACE) ----------
model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings_array = model.encode(
    [t["text"] for t in texts],
    show_progress_bar=True,
    convert_to_numpy=True
).astype("float32")


# ---------- FAISS ----------
dim = embeddings_array.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings_array)

# ---------- SAVE ----------
os.makedirs(FAISS_FOLDER, exist_ok=True)

faiss.write_index(index, os.path.join(FAISS_FOLDER, "faiss_index.bin"))

with open(os.path.join(FAISS_FOLDER, "texts.pkl"), "wb") as f:
    pickle.dump(texts, f)

print("✅ FAISS index created successfully using local embeddings (NO Ollama, NO OpenAI)")
