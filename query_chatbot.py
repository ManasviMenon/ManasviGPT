import os
import requests # type: ignore
import unicodedata
from dotenv import load_dotenv  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
from sentence_transformers import SentenceTransformer
import numpy as np

# ----------- LOAD EMBEDDING MODEL ONCE -----------
_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(
            "all-MiniLM-L6-v2",  # smaller, ~120MB",
            device="cpu"
        )
    return _embedder


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found in .env")

import faiss # type: ignore
import pickle

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    return text.lower().strip()


PRIORITY_FAQ = {
    "what role does she wish to work in?": "Manasvi wishes to work in data-driven and analytics-focused industries. She is interested in roles in data, analytics, business operations, strategy, and growth. This includes Data Analyst, Business Analyst, Product Analytics, Analytics Specialist, Business Development, GTM Strategy, Growth and Revenue Strategy, Strategy & Operations, Sales Operations, Commercial Analytics, Project/Program Management, and Founder’s Office roles. This reflects her career aspirations and interests, not her past work experience.",
    "What roles is she interested in?":"Manasvi wishes to work in data-driven and analytics-focused industries. She is interested in roles in data, analytics, business operations, strategy, and growth. This includes Data Analyst, Business Analyst, Product Analytics, Analytics Specialist, Business Development, GTM Strategy, Growth and Revenue Strategy, Strategy & Operations, Sales Operations, Commercial Analytics, Project/Program Management, and Founder’s Office roles. This reflects her career aspirations and interests, not her past work experience.",
    "What roles is she open to?":"Manasvi wishes to work in data-driven and analytics-focused industries. She is interested in roles in data, analytics, business operations, strategy, and growth. This includes Data Analyst, Business Analyst, Product Analytics, Analytics Specialist, Business Development, GTM Strategy, Growth and Revenue Strategy, Strategy & Operations, Sales Operations, Commercial Analytics, Project/Program Management, and Founder’s Office roles. This reflects her career aspirations and interests, not her past work experience.", 
    "what is her leadership experience?": "Manasvi has held leadership roles such as Vice President at AIESEC, leading teams of up to 60 members and cross-functional teams, mentoring, and managing strategic projects. She also led a direct team of 15 and an entity of 60+, securing B2B partnerships with multinational brands, achieving 100% sustainability in strategic partnerships, and driving 92% revenue growth.",
    "Academic background?": "Manasvi completed her undergraduate studies in Economics with majors in Statistics and Finance. She also studied quantitative analysis, econometrics, programming, and regression analysis, which provided a strong foundation for data-driven decision-making and analytics. Currently studying Data Science and Analytics where she is  studying subjects like Machine Learning, Natural Language Processing, Big Data Engineering, Statistics.",
    "who is Manasvi ?": "Manasvi Menon is a final-year postgraduate student currently based in Sydney, Australia. She is pursuing a Master’s degree in Data Science and Analytics at the University of Technology Sydney (UTS). She has a strong academic and analytical background, combined with professional experience in startups, FinTech, and Not-for-Profit organizations.",
    "what motivates her?": "I am motivated by opportunities to create impact, lead others, build solutions from scratch, and continuously grow personally and professionally.",
    "what leadership experience does Manasvi have?": "Manasvi has held leadership roles such as Vice President at AIESEC, leading teams of up to 60 members and cross-functional teams, mentoring, and managing strategic projects. She also led a direct team of 15 and an entity of 60+, securing B2B partnerships with multinational brands, achieving 100% sustainability in strategic partnerships, and driving 92% revenue growth.",
    "tell me about her leadership experience": "Manasvi has held leadership roles such as Vice President at AIESEC, leading teams of up to 60 members and cross-functional teams, mentoring, and managing strategic projects. She also led a direct team of 15 and an entity of 60+, securing B2B partnerships with multinational brands, achieving 100% sustainability in strategic partnerships, and driving 92% revenue growth.",
    "what are her interests and hobbies?": "Manasvi is a national-level debater who has represented institutions across multiple competitive debating tournaments in India. She is an avid reader and writer, with a strong interest in ideas, storytelling, and critical thinking. Outside of academics, she enjoys hiking and travelling to remote locations, drawn to experiences that challenge her comfort zone and push her limits. She is also a keen tennis enthusiast.",
    "Extra-curriculars": "Manasvi is a national-level debater who has represented institutions across multiple competitive debating tournaments in India. She is an avid reader and writer, with a strong interest in ideas, storytelling, and critical thinking. Outside of academics, she enjoys hiking and travelling to remote locations, drawn to experiences that challenge her comfort zone and push her limits. She is also a keen tennis enthusiast."
}

PRIORITY_FAQ = {
    normalize_text(k): v
    for k, v in PRIORITY_FAQ.items()
}


_faq_embeddings = None
_faq_keys = None

def get_faq_embeddings():
    global _faq_embeddings, _faq_keys
    if _faq_embeddings is None:
        _faq_keys = list(PRIORITY_FAQ.keys())
        _faq_embeddings = get_embedder().encode(_faq_keys, convert_to_numpy=True)
    return _faq_embeddings, _faq_keys


def search_priority_faq_semantic(question, threshold=0.65):
    question_vec = get_embedder().encode([normalize_text(question)], convert_to_numpy=True)
    faq_embeddings, faq_keys = get_faq_embeddings()
    sims = cosine_similarity(question_vec, faq_embeddings)[0]
    best_idx = np.argmax(sims)
    if sims[best_idx] >= threshold:
        return PRIORITY_FAQ[faq_keys[best_idx]]

    return None



# ----------- LOAD FAISS INDEX & TEXTS -----------
_index = None
_texts = None

def get_faiss():
    global _index, _texts
    if _index is None or _texts is None:
        _index = faiss.read_index("faiss_index/faiss_index.bin")
        with open("faiss_index/texts.pkl", "rb") as f:
            _texts = pickle.load(f)
    return _index, _texts

# ----------- LOAD LOCAL EMBEDDING MODEL -----------

def embed_query(query):
    return get_embedder().encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

def detect_intent(question):
    q = question.lower()

    project_keywords = [
        "project", "pipeline", "etl", "elt", "airbnb", "taxi",
        "databricks", "spark", "gcp", "dbt", "sql", "ml",
        "dataset", "analytics", "model"
    ]

    experience_keywords = [
        "intern", "experience", "worked", "responsible",
        "led", "managed", "team", "organisation", "company"
    ]

    if any(pk in q for pk in project_keywords):
        return "project"

    if any(ek in q for ek in experience_keywords):
        return "experience"

    return "general"
groq_cache = {}

def groq_answer_cached(question, context_chunks):
    key = normalize_text(question)
    if key in groq_cache:
        return groq_cache[key]
    answer = groq_answer(question, context_chunks)
    groq_cache[key] = answer
    return answer

# ----------- RETRIEVE RELEVANT CHUNKS -----------
def retrieve_chunks(query, top_k=20, section=None, max_distance=1.5):
    query_vec = embed_query(query)
    index, texts = get_faiss()
    D, I = index.search(query_vec, k=top_k)

    chunks = []

    # ⬇️ DISTANCE FILTER IS APPLIED HERE
    for dist, idx in zip(D[0], I[0]):
        if idx >= len(texts):
            continue

        # FAISS distance filter (lower = better)
        if dist > max_distance:
            continue

        chunk = texts[idx]

        # Section filtering
        if section and isinstance(chunk, dict):
            if chunk.get("section") != section:
                continue

        # Extract text safely
        if isinstance(chunk, dict):
            chunks.append(chunk.get("text", ""))
        else:
            chunks.append(chunk)

    return chunks



# ----------- QUERY LLM -----------
def groq_answer(question, context_chunks):
    context = "\n\n".join(context_chunks)

    system_prompt = """
You are an expert assistant answering questions strictly using ONLY the provided context.

Rules (must follow exactly):
1. Use ONLY the information present in the context below.
2. Do NOT infer, assume, or add skills, experience, or details not explicitly stated.
3. Respect section relevance — do not mix projects, education, or experience unless the context includes them.
4. Merge overlapping or repeated information into one concise explanation.
5. Do not repeat text verbatim unless necessary for clarity.
6. Write in a professional, recruiter-friendly tone.
7. Only respond with:
   "I don't have enough information to answer that."
   IF the question cannot be reasonably answered even with careful inference.
8: Never invent roles, exposure, or work history.
"""

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama-3.1-8b-instant",
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""
Context:
{context}

Question:
{question}
"""
            }
        ]
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code != 200:
        return f"Groq error: {response.text}"

    return response.json()["choices"][0]["message"]["content"]

# ----------------- SECTION DETECTION -----------------
def preprocess_question(question):
    # Replace pronouns with the name to improve retrieval
    question = question.replace("her", "Manasvi Menon")
    question = question.replace("she", "Manasvi Menon")
    return question


# ----------- ANSWER FUNCTION (WITH STRICT SECTION ISOLATION) -----------
def answer_question(question):
    faq_answer = (
        search_priority_faq_semantic(question)
        if detect_intent(question) not in ["project", "experience"]
        else None
    )
    if faq_answer:
        return faq_answer

    question = preprocess_question(question)
    intent = detect_intent(question)
    top_k = 20

    if intent == "project":
        chunks = (
            retrieve_chunks("taxi project", top_k=top_k, section="taxi_project")
            + retrieve_chunks("airbnb project", top_k=top_k, section="airbnb_project")
        )

    elif intent == "experience":
        chunks = (
            retrieve_chunks(question, top_k=top_k, section="aiesec")
            + retrieve_chunks(question, top_k=top_k, section="coindcx")
            + retrieve_chunks(question, top_k=top_k, section="experience")
        )

    else:
        chunks = retrieve_chunks(question, top_k=top_k)

    # Deduplicate
    chunks = list(dict.fromkeys(chunks))

    # FALLBACK MUST BE INSIDE FUNCTION
    if not chunks:
        return groq_answer_cached(
            question,
            ["Use reasonable inference based on the provided profile, without inventing facts."]
        )

    return groq_answer_cached(question, chunks)
