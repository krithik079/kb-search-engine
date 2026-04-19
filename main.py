import os
from fastapi import FastAPI
from pydantic import BaseModel

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# =========================
# Load FAISS DB
# =========================
DB_PATH = "faiss_index"

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = FAISS.load_local(
    DB_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

# =========================
# FastAPI App
# =========================
app = FastAPI()

class QueryRequest(BaseModel):
    query: str


# =========================
# Clean Answer Builder
# =========================
def build_answer(docs):
    """
    Convert retrieved chunks into a clean answer
    """
    text = " ".join([doc.page_content for doc in docs])

    # Split into sentences properly
    sentences = text.replace("\n", " ").split(".")

    clean_sentences = []
    for s in sentences:
        s = s.strip()

        # Keep only meaningful sentences
        if len(s) > 50:
            clean_sentences.append(s)

    # Take first 2–3 good sentences
    answer = ". ".join(clean_sentences[:3]).strip()

    if not answer.endswith("."):
        answer += "."

    return answer


# =========================
# API Endpoint
# =========================
@app.post("/ask")
def ask_question(request: QueryRequest):
    docs = db.similarity_search(request.query, k=5)

    if not docs:
        return {"answer": "No relevant information found."}

    # Step 1: Combine top results
    combined_text = " ".join([doc.page_content for doc in docs])

    # Step 2: Split into sentences
    sentences = combined_text.split(".")

    # Step 3: Find definition sentence
    definition = None
    for sentence in sentences:
        s = sentence.lower()
        if "photonic crystal" in s and ("is" in s or "are" in s):
            definition = sentence.strip()
            break

    # Step 4: fallback
    if not definition:
        definition = sentences[0].strip()

    return {"answer": definition + "."}