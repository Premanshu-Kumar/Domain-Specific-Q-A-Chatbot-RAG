# 🤖 Domain-Specific Q&A Chatbot — RAG

> **A full-stack AI-powered knowledge assistant that uses Retrieval-Augmented Generation (RAG) to answer questions from custom documents with relevant, context-aware responses and source citations.**

![Status](https://img.shields.io/badge/Status-Phase%201%20Completed-success)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![React](https://img.shields.io/badge/React-18%2B-61DAFB)
![Flask](https://img.shields.io/badge/Flask-Backend-black)
![ChromaDB](https://img.shields.io/badge/VectorStore-ChromaDB-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

This project is a **domain-specific AI chatbot built using Retrieval-Augmented Generation (RAG)**.

Instead of relying entirely on an LLM's pre-trained knowledge, the system retrieves relevant information from a custom document collection and uses that context to generate grounded, fact-checked responses.

Users can upload documents such as:
* 📄 Company handbooks & SOPs
* 📚 Academic materials & lecture notes
* 🔧 Technical documentation & API guides
* 📦 Product manuals & troubleshooting docs
* ❓ FAQ documents
* 📝 Policies and reports

The goal is to build a practical **AI Knowledge Assistant** that can understand and answer questions from private or domain-specific information.

---

## 🎯 Why This Project?

Traditional chatbots depend primarily on predefined responses or general-purpose model knowledge.

This project addresses that limitation by combining:

```text
Custom Knowledge Base
        +
Semantic Retrieval
        +
Large Language Model
        =
Context-Aware AI Assistant
```

### Key Objectives
* Build a complete end-to-end RAG pipeline
* Understand document processing and embeddings
* Implement semantic vector search with ChromaDB
* Integrate an LLM for answer generation
* Provide source-grounded responses
* Build a modern full-stack AI application
* Deploy the system as a production-ready application

---

## 🏗️ System Architecture

The application follows a **client-server architecture** with a RAG pipeline at its core.

```text
                         ┌───────────────────────┐
                         │     React Frontend    │  (Phase 3)
                         │                       │
                         │  Chat • Upload • UI   │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │       Flask API       │
                         │                       │
                         │   REST API + RAG      │
                         └───────────┬───────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
                    ▼                ▼                ▼
             ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
             │  Document   │  │   Vector    │  │     LLM     │
             │ Processing  │  │    Store    │  │ Generation  │
             │ (Ingestion) │  │  (ChromaDB) │  │  (Phase 2)  │
             └──────┬──────┘  └──────┬──────┘  └──────▲──────┘
                    │                │                │
                    ▼                ▼                │
             ┌─────────────┐  ┌─────────────┐         │
             │ Text        │  │ Embeddings  │─────────┘
             │ Chunking    │  │ + Retrieval │
             └─────────────┘  └─────────────┘
```

---

## 🔄 RAG Pipeline

### 1. Document Ingestion
Users upload supported documents (`.pdf`, `.txt`, `.docx`).
```text
PDF / TXT / DOCX ──► Text Extraction & Normalization
```

### 2. Text Processing & Cleaning
The extracted content is cleaned and divided into smaller, semantically coherent chunks.
```text
Raw Text ──► Cleaning ──► Recursive Chunking ──► Metadata Enrichment
```

### 3. Embedding Generation
Each chunk is converted into a numerical vector representation using an embedding model (e.g. `sentence-transformers/all-MiniLM-L6-v2` or OpenAI).
```text
Text Chunk ──► Embedding Model ──► Vector Representation (Dense Vector)
```

### 4. Vector Storage
The generated embeddings are stored in a persistent ChromaDB vector collection with cosine similarity indexing.

### 5. Query Retrieval
When the user asks a question:
```text
User Question ──► Query Embedding ──► Similarity Search ──► Top-K Relevant Chunks
```

### 6. Answer Generation (Phase 2)
The retrieved context is combined with anti-hallucination prompt instructions and passed to the LLM.
```text
Question + Retrieved Context ──► LLM ──► Context-Aware, Grounded Answer
```

### 7. Response with Citations
The generated answer is returned along with the exact source documents cited.

---

## ✨ Features

### 📄 Knowledge Base Upload
Upload custom documents that become part of the chatbot's knowledge base.
* **Supported formats:** PDF, TXT, DOCX

### 🧠 Retrieval-Augmented Generation
Retrieve relevant sections from the knowledge base before generating an answer, guaranteeing high accuracy and domain relevance.

### 💬 Interactive Chat Interface (Phase 3)
A modern React-based interface for asking questions, viewing conversation history, and receiving real-time AI responses.

### 📚 Source References
Every response can cite the exact source files and chunks used to formulate the answer:
```text
Answer: Employees are entitled to 20 days of annual leave according to the company leave policy.
Sources: 📄 Employee_Handbook.pdf 📄 Leave_Policy.pdf
```

### 🧩 Conversation Memory
Supports multi-turn conversations and contextual follow-up questions.

### 👍 Response Feedback
Users can provide feedback on generated responses (👍 Helpful / 👎 Not Helpful) to track retrieval quality.

---

## 🛠️ Tech Stack

| Category | Technologies |
| :--- | :--- |
| **Frontend** | React 18, JavaScript / TypeScript, HTML5, Vanilla CSS / Tailwind |
| **Backend** | Python 3.9+, Flask, Flask-CORS |
| **AI / LLM** | LangChain, Hugging Face Transformers, Sentence Transformers, OpenAI API |
| **Vector Store** | ChromaDB / FAISS |
| **Document Processing** | PyPDF, python-docx |
| **Testing** | pytest |
| **Containerization** | Docker, Docker Compose |
| **Version Control** | Git, GitHub |

---

## 📂 Project Structure

```text
Domain-Specific Q&A Chatbot — RAG/
├── backend/
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py           # REST endpoints (/upload, /search, /documents, /health)
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py         # Centralized configuration & environment variables
│   ├── models/
│   │   └── __init__.py
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── chunking.py         # Text cleaning and recursive chunking
│   │   ├── embeddings.py       # Embeddings manager & ChromaDB vector store
│   │   ├── generation.py       # Prompt templates & LLM generation
│   │   ├── ingestion.py        # PDF, TXT, DOCX text extractors
│   │   ├── pipeline.py         # Complete ingestion & indexing pipeline
│   │   └── retrieval.py        # Semantic retrieval wrapper
│   └── app.py                  # Flask application entry point & factory
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py        # Comprehensive unit & integration tests (12 passing tests)
├── .env.example                # Example environment configuration
├── .gitignore                  # Git ignore rules
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 🗺️ Development Roadmap

The project is developed in **4 major phases**:

### 🟢 Phase 1 — Foundation & Core Backend `[COMPLETED]`
* [x] Initialize Git repository
* [x] Set up Python virtual environment
* [x] Create backend project structure
* [x] Configure `requirements.txt`
* [x] Implement document upload
* [x] Extract text from PDF/TXT/DOCX files
* [x] Implement text cleaning
* [x] Implement document chunking with metadata enrichment
* [x] Generate dense embeddings (`sentence-transformers/all-MiniLM-L6-v2`)
* [x] Set up ChromaDB persistent vector store
* [x] Build document indexing pipeline
* [x] Core Flask REST API endpoints
* [x] 12 Passing unit & integration tests

---

### 🔵 Phase 2 — RAG Pipeline & API `[NEXT]`
* [ ] Implement query embeddings & cosine similarity search
* [ ] Retrieve Top-K relevant chunks with score filtering
* [ ] Build context generation & anti-hallucination prompt template
* [ ] Integrate LLM (OpenAI / HuggingFace / Groq)
* [ ] Implement grounded response generation
* [ ] Implement conversational `/api/chat` endpoint with source citations
* [ ] Add error handling and input validation

---

### 🟣 Phase 3 — Frontend & Integration
* [ ] Initialize React application
* [ ] Design chatbot interface
* [ ] Create message components
* [ ] Create document upload interface
* [ ] Connect React with Flask REST API
* [ ] Implement chat state management & loading skeletons
* [ ] Display AI responses and source references drawer
* [ ] Implement responsive design & dark mode

---

### 🔴 Phase 4 — Polish, Testing & Deployment
* [ ] UI/UX polishing & micro-animations
* [ ] Implement multi-turn conversation memory
* [ ] Add response feedback (👍 / 👎)
* [ ] Comprehensive end-to-end testing
* [ ] Dockerize frontend and backend (`Dockerfile`, `docker-compose.yml`)
* [ ] Performance optimization & caching
* [ ] Complete deployment & demo

---

## ⚙️ Getting Started

### Prerequisites
* Python 3.9+
* Node.js & npm (for Phase 3)
* Git

### 1. Clone the Repository
```bash
git clone https://github.com/Premanshu-Kumar/Domain-Specific-Q-A-Chatbot-RAG.git
cd Domain-Specific-Q-A-Chatbot-RAG
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
*(Optionally add your `OPENAI_API_KEY` or custom configuration)*

### 3. Setup Python Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Run Tests
```bash
pytest -v
```

### 5. Start Backend Server
```bash
python backend/app.py
```
The API server will run at `http://localhost:5000`.

---

## 💻 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | API status and available endpoints |
| `GET` | `/api/health` | Service health check |
| `POST` | `/api/upload` | Upload & index documents (`.pdf`, `.txt`, `.docx`) |
| `POST` | `/api/search` | Semantic search over indexed documents |
| `GET` | `/api/documents` | List indexed documents & collection statistics |
| `POST` | `/api/chat` | Chat with RAG knowledge assistant *(Phase 2)* |

---

## 📊 RAG Evaluation Metrics

| Metric | Purpose |
| :--- | :--- |
| **Retrieval Accuracy** | Measures relevance of retrieved chunks |
| **Context Precision** | Measures proportion of retrieved context that is useful |
| **Context Recall** | Measures whether all required ground truth facts were retrieved |
| **Answer Relevance** | Measures alignment between the generated answer and the user question |
| **Faithfulness** | Measures whether the answer is strictly derived from retrieved context |
| **Latency** | Measures end-to-end response time |

---

## 💼 Resume Highlights

> **Developed a full-stack Retrieval-Augmented Generation (RAG) knowledge assistant using React, Flask, LangChain, ChromaDB, and Sentence Transformers, enabling context-aware question answering over custom document collections with source-grounded responses and anti-hallucination guardrails.**

---

## 📜 License

This project is licensed under the **MIT License** - see the `LICENSE` file for details.

---

## 👨‍💻 Author

**Premanshu Kumar**
* GitHub: [@Premanshu-Kumar](https://github.com/Premanshu-Kumar)
