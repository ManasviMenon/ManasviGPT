import os
import requests  # type: ignore
import unicodedata
from dotenv import load_dotenv  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
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

import faiss  # type: ignore
import pickle


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.replace("'", "'").replace("\u201c", '"').replace("\u201d", '"')
    return text.lower().strip()


PRIORITY_FAQ = {
    "what role does she wish to work in?": "Manasvi wishes to work in data-driven and analytics-focused industries. She is interested in roles in data, analytics, business operations, strategy, and growth. This includes Data Analyst, Business Analyst, Product Analytics, Analytics Specialist, Business Development, GTM Strategy, Growth and Revenue Strategy, Strategy & Operations, Sales Operations, Commercial Analytics, Project/Program Management, and Founder's Office roles. This reflects her career aspirations and interests, not her past work experience.",
    "What roles is she interested in?": "Manasvi wishes to work in data-driven and analytics-focused industries. She is interested in roles in data, analytics, business operations, strategy, and growth. This includes Data Analyst, Business Analyst, Product Analytics, Analytics Specialist, Business Development, GTM Strategy, Growth and Revenue Strategy, Strategy & Operations, Sales Operations, Commercial Analytics, Project/Program Management, and Founder's Office roles. This reflects her career aspirations and interests, not her past work experience.",
    "What roles is she open to?": "Manasvi wishes to work in data-driven and analytics-focused industries. She is interested in roles in data, analytics, business operations, strategy, and growth. This includes Data Analyst, Business Analyst, Product Analytics, Analytics Specialist, Business Development, GTM Strategy, Growth and Revenue Strategy, Strategy & Operations, Sales Operations, Commercial Analytics, Project/Program Management, and Founder's Office roles. This reflects her career aspirations and interests, not her past work experience.",
    "what is her leadership experience?": "Manasvi has held leadership roles such as Vice President at AIESEC, leading teams of up to 60 members and cross-functional teams, mentoring, and managing strategic projects. She also led a direct team of 15 and an entity of 60+, securing B2B partnerships with multinational brands, achieving 100% sustainability in strategic partnerships, and driving 92% revenue growth.",
    "Academic background?": "Manasvi completed her undergraduate studies in Economics with majors in Statistics and Finance. She also studied quantitative analysis, econometrics, programming, and regression analysis, which provided a strong foundation for data-driven decision-making and analytics. Currently studying Data Science and Analytics where she is  studying subjects like Machine Learning, Natural Language Processing, Big Data Engineering, Statistics.",
    "who is Manasvi ?": "Manasvi Menon is a final-year postgraduate student currently based in Sydney, Australia. She is pursuing a Master's degree in Data Science and Analytics at the University of Technology Sydney (UTS). She has a strong academic and analytical background, combined with professional experience in startups, FinTech, and Not-for-Profit organizations.",
    "what motivates her?": "I am motivated by opportunities to create impact, lead others, build solutions from scratch, and continuously grow personally and professionally.",
    "what leadership experience does Manasvi have?": "Manasvi has held leadership roles such as Vice President at AIESEC, leading teams of up to 60 members and cross-functional teams, mentoring, and managing strategic projects. She also led a direct team of 15 and an entity of 60+, securing B2B partnerships with multinational brands, achieving 100% sustainability in strategic partnerships, and driving 92% revenue growth.",
    "tell me about her leadership experience": "Manasvi has held leadership roles such as Vice President at AIESEC, leading teams of up to 60 members and cross-functional teams, mentoring, and managing strategic projects. She also led a direct team of 15 and an entity of 60+, securing B2B partnerships with multinational brands, achieving 100% sustainability in strategic partnerships, and driving 92% revenue growth.",
    "what are her interests and hobbies?": "Manasvi is a national-level debater who has represented institutions across multiple competitive debating tournaments in India. She is an avid reader and writer, with a strong interest in ideas, storytelling, and critical thinking. Outside of academics, she enjoys hiking and travelling to remote locations, drawn to experiences that challenge her comfort zone and push her limits. She is also a keen tennis enthusiast.",
    "Extra-curriculars": "Manasvi is a national-level debater who has represented institutions across multiple competitive debating tournaments in India. She is an avid reader and writer, with a strong interest in ideas, storytelling, and critical thinking. Outside of academics, she enjoys hiking and travelling to remote locations, drawn to experiences that challenge her comfort zone and push her limits. She is also a keen tennis enthusiast.",
    "Is she available to work immediately": "Yes",
    "What is her GPA": "Bachelor's Degree in Economics and Statistics: 9.1/10 CGPA and Master's Degree in Data Science: 6.5/7"
}

PRIORITY_FAQ = {normalize_text(k): v for k, v in PRIORITY_FAQ.items()}

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


SCOPE_ANCHORS = [
    "Manasvi's education and academic background",
    "Manasvi's work experience and professional background",
    "Manasvi's projects, technical work, and tools",
    "Manasvi's skills in data science, analytics, machine learning, NLP",
    "Manasvi's leadership, teamwork, and achievements",
    "Manasvi's hobbies and interests mentioned in her profile",
    "Why Manasvi is a good fit for a role based on her profile",
    # ── ADDED: hiring/synthesis anchors ──
    "Who is Manasvi"
    "Where does she live"
    "Candidate overview and introduction"
    "Why Manasvi should be hired and what she brings to a team",
    "Manasvi's unique strengths and value as a candidate",
    "What makes Manasvi stand out professionally as a job applicant",
]

# ── ADDED: out-of-scope anchors for contrastive filtering ──
OUT_OF_SCOPE_ANCHORS = [
    "how attractive or pretty does someone look physically",
    "romantic relationship status and dating life",
    "cooking recipes and food preparation",
    "weather forecast and climate",
    "celebrity gossip and entertainment news",
    "physical body features like height weight skin color",
    "jokes memes and funny content",
    "sports scores and game results",
    "stock prices and cryptocurrency",
    "political opinions and news events",
]

_scope_embeds = None
_out_of_scope_embeds = None  # ADDED

def get_scope_embeddings():
    global _scope_embeds
    if _scope_embeds is None:
        _scope_embeds = get_embedder().encode(SCOPE_ANCHORS, convert_to_numpy=True)
    return _scope_embeds

# ── ADDED ──
def get_out_of_scope_embeddings():
    global _out_of_scope_embeds
    if _out_of_scope_embeds is None:
        _out_of_scope_embeds = get_embedder().encode(OUT_OF_SCOPE_ANCHORS, convert_to_numpy=True)
    return _out_of_scope_embeds


def is_in_scope(question: str, threshold: float = 0.35) -> bool:
    q = normalize_text(question)

    whitelist_patterns = [    "who is", "where is", "where does", "where did",
    "what is her", "what is his", "what does she",
    "how old", "where was", "what nationality",
    "what is she", "what is manasvi", "what has she",
    "what did she", "tell me about", "how did she",
    "what are her", "what are manasvi", "has she",
    "is she", "does she", "did she", "can she",
    "how long", "when did", "when does",]
    if any(p in q for p in whitelist_patterns):
        return True
    # Strip name for in-scope check so "Manasvi" doesn't inflate scores
    q_no_name = q.replace("manasvi menon", "").replace("manasvi", "").strip()
    q_for_in = q_no_name if len(q_no_name) > 4 else q

    qv_in = get_embedder().encode([q_for_in], convert_to_numpy=True)
    qv_out = get_embedder().encode([q], convert_to_numpy=True)  # full question for out-of-scope

    in_scope_sims = cosine_similarity(qv_in, get_scope_embeddings())[0]
    best_in = float(np.max(in_scope_sims))

    out_scope_sims = cosine_similarity(qv_out, get_out_of_scope_embeddings())[0]
    best_out = float(np.max(out_scope_sims))

    if best_in < threshold:
        return False
    if best_out > best_in:
        return False
    return True

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
    return get_embedder().encode([query], convert_to_numpy=True).astype("float32")


def detect_intent(question):
    q = question.lower()

    # ── ADDED: synthesis intent ──
    synthesis_keywords = [
        "hire", "recommend", "why should", "strengths", "stand out",
        "suitable", "fit for", "value", "unique", "best candidate",
        "what makes her", "overall", "summary", "overview",
    ]
    if any(sk in q for sk in synthesis_keywords):
        return "synthesis"

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

def groq_answer_cached(question, context_chunks, intent="general"):  # ADDED intent param
    key = normalize_text(question)
    if key in groq_cache:
        return groq_cache[key]
    answer = groq_answer(question, context_chunks, intent=intent)  # ADDED intent
    groq_cache[key] = answer
    return answer


# ----------- RETRIEVE RELEVANT CHUNKS -----------
def retrieve_chunks(query, top_k=20, section=None, max_distance=1.5):
    query_vec = embed_query(query)
    index, texts = get_faiss()
    D, I = index.search(query_vec, k=top_k)

    chunks = []

    for dist, idx in zip(D[0], I[0]):
        if idx >= len(texts):
            continue

        if dist > max_distance:
            continue

        chunk = texts[idx]

        if section and isinstance(chunk, dict):
            if chunk.get("section") != section:
                continue

        if isinstance(chunk, dict):
            chunks.append(chunk.get("text", ""))
        else:
            chunks.append(chunk)

    return chunks


def context_relevance_score(question: str, chunks: list[str]) -> float:
    if not chunks:
        return 0.0

    q = normalize_text(question)
    qv = get_embedder().encode([q], convert_to_numpy=True)

    sample = chunks[:8]
    cv = get_embedder().encode([normalize_text(c) for c in sample], convert_to_numpy=True)

    sims = cosine_similarity(qv, cv)[0]
    return float(np.max(sims))


# ----------- QUERY LLM -----------
def groq_answer(question, context_chunks, intent="general"):  # ADDED intent param
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

    # ── ADDED: synthesis mode injection ──
    if intent == "synthesis":
        system_prompt += """
SYNTHESIS MODE — the recruiter is asking you to make a case for Manasvi as a candidate.
- Draw on ALL sections of the context: education, skills, experience, projects, leadership.
- Connect the dots across sections (e.g. her analytical degree + data science masters
  + hands-on projects + leadership = a well-rounded data professional).
- Structure your answer: short opening → 3-4 concrete evidence-backed reasons → closing sentence.
- Be persuasive but grounded — every claim must be traceable to the context.
"""

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama-3.1-8b-instant",
        "temperature": 0.4 if intent == "synthesis" else 0.2,  # ADDED: slightly more creative for synthesis
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
    question = question.replace("her", "Manasvi Menon")
    question = question.replace("she", "Manasvi Menon")
    return question


# ----------- ANSWER FUNCTION (WITH STRICT SECTION ISOLATION) -----------
def answer_question(question):
    # Block obvious off-topic BEFORE FAQ check
    if not is_in_scope(question):
        return "I'm here to answer questions about Manasvi's professional profile. That question is outside my scope!"

    # Priority FAQ checked before RAG pipeline
    if detect_intent(question) not in ["project", "experience", "synthesis"]:
        faq_answer = search_priority_faq_semantic(question)
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
    elif intent == "synthesis":
        chunks = retrieve_chunks(question, top_k=top_k * 2)
    else:
        chunks = retrieve_chunks(question, top_k=top_k)

    chunks = list(dict.fromkeys(chunks))

    score = context_relevance_score(question, chunks)
    if score < 0.52:
        return "I don't have enough information to answer that."

    if not chunks:
        return "I don't have enough information to answer that."

    return groq_answer_cached(question, chunks, intent=intent)