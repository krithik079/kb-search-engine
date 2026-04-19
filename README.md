# 📚 Knowledge Base Search Engine (RAG System)

A simple Retrieval-Augmented Generation (RAG) system built using **FastAPI + FAISS + HuggingFace embeddings** to answer questions from PDF/text documents.

---

## 🚀 Features

- 📄 Load documents (PDF / TXT)
- ✂️ Split documents into chunks
- 🔎 Generate embeddings using HuggingFace
- 🧠 Store vectors using FAISS
- ⚡ FastAPI endpoint for querying
- 🎯 Smart answer extraction (returns definition-style answers)

---

## 🏗️ Tech Stack

- Python
- FastAPI
- LangChain
- FAISS (Vector Database)
- HuggingFace Embeddings (all-MiniLM-L6-v2)

---

## 📁 Project Structure
kb-search-engine/
│
├── data/ # Input documents (PDF/TXT)
├── faiss_index/ # Generated vector DB (auto-created)
├── ingest.py # Document processing + DB creation
├── main.py # FastAPI app
├── requirements.txt
└── README.md


---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository

```bash
git clone https://github.com/krithik079/kb-search-engine.git
cd kb-search-engine

#Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

#Install dependencies
pip install -r requirements.txt
pip install pypdf faiss-cpu langchain-huggingface

#Create vector database
python ingest.py

#Run FastAPI server
python -m uvicorn main:app --reload

#Open API Docs
http://127.0.0.1:8000/docs

#Example Query
{
  "query": "What are photonic crystals?"
}

#Example Output
Photonic crystals are periodic dielectric structures that can manipulate the flow of light in a controlled manner.

#How It Works
Documents are loaded from the data/ folder
Text is split into smaller chunks
Embeddings are generated using HuggingFace model
FAISS stores the vectors for similarity search
On query:
Top relevant chunks are retrieved
Sentences are filtered to extract a clean definition-style answer

# Improvements (Future Scope)
Add LLM (OpenAI / local models) for better answers
Improve ranking with re-ranking models
Add UI (Streamlit / React)
Support more document types

#Author

Krithik Kumar


