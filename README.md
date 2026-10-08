# 🤖 Domain-Specific Q&A Chatbot — RAG + JEV

> An intelligent, domain-specific AI knowledge assistant that combines **Retrieval-Augmented Generation (RAG)** with **JEV-based orchestration** to provide grounded, context-aware answers from private document collections.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![React](https://img.shields.io/badge/React-18%2B-61DAFB)
![Flask](https://img.shields.io/badge/Flask-Backend-black)
![RAG](https://img.shields.io/badge/AI-RAG-purple)
![JEV](https://img.shields.io/badge/AI-JEV-orange)
![Status](https://img.shields.io/badge/Phase%202-Completed-success)
![License](https://img.shields.io/badge/License-Apache%202.0-blue)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Why This Project](#-why-this-project)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [RAG Pipeline](#-rag-pipeline)
- [JEV Integration](#-jev-integration)
- [Project Roadmap](#-project-roadmap)
- [Project Structure](#-project-structure)
- [Technology Stack](#-technology-stack)
- [Current Status](#-current-status)
- [Getting Started](#-getting-started)
- [API Endpoints](#-api-endpoints)
- [Example Workflow](#-example-workflow)
- [Evaluation](#-evaluation)
- [Future Enhancements](#-future-enhancements)
- [Resume Description](#-resume-description)
- [License](#-license)

---

# 🚀 Overview

The **Domain-Specific Q&A Chatbot** is a full-stack AI application designed to answer questions from a user's private knowledge base.

Instead of relying only on an LLM's pre-trained knowledge, the system retrieves relevant information from uploaded documents and uses that information to generate **grounded responses with source references**.

The project is being evolved from a traditional RAG chatbot into an **Agentic RAG system using JEV orchestration**.

### Current Evolution

```text
Traditional RAG
      ↓
Reliable RAG
      ↓
JEV Orchestration
      ↓
Agentic RAG
      ↓
Evaluation
      ↓
Production AI Knowledge Assistant
```

---

# 🎯 Why This Project?

General-purpose AI models can sometimes:

- Generate unsupported information
- Lack access to private documents
- Fail to retrieve domain-specific information
- Provide answers without verifiable sources

This project addresses these problems by combining:

### 🔹 Retrieval-Augmented Generation

Retrieve relevant document chunks before generating an answer.

### 🔹 Grounded Generation

The LLM generates answers using retrieved context instead of relying purely on internal knowledge.

### 🔹 Source Attribution

Relevant document sources are returned alongside answers.

### 🔹 JEV Orchestration

The next development stage introduces JEV as an orchestration layer that can determine **how a user query should be handled**.

---

# ✨ Key Features

## ✅ Currently Implemented

### 📄 Document Knowledge Base

- Upload PDF/TXT documents
- Extract document text
- Split documents into chunks
- Generate embeddings
- Store embeddings in ChromaDB
- Maintain document metadata

### 🔎 Semantic Retrieval

- Convert user questions into embeddings
- Perform similarity search
- Retrieve Top-K relevant chunks
- Apply similarity score filtering

### 🧠 Grounded RAG

- Retrieved chunks are passed to the LLM
- Context-aware prompt construction
- Anti-hallucination instructions
- Grounded responses

### 💬 Conversational Chat

- `/api/chat` endpoint
- Conversation history support
- Context-aware responses
- Source citations
- Input validation
- Error handling

### 🔌 Multiple LLM Providers

The current implementation supports automatic provider selection between configured providers such as:

- OpenAI
- Groq

A stub/fallback mode is also available when an API key is not configured.

---

# 🤖 Planned JEV Capabilities

JEV will be introduced as an orchestration layer rather than replacing the existing RAG pipeline.

Planned capabilities include:

- Query intent detection
- Intelligent workflow selection
- RAG tool routing
- Query rewriting
- Multi-step retrieval
- Agentic reasoning
- Conversation-aware routing
- Source verification
- Future summary/comparison workflows

---

# 🏗️ System Architecture

## Current Architecture — Phase 2

```text
                 ┌─────────────────────┐
                 │   React Frontend    │
                 │     Phase 3         │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │     Flask API       │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │    RAG Pipeline     │
                 └──────────┬──────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
        Document       Embeddings      Retrieval
        Processing        Model          Layer
              │             │             │
              └─────────────┼─────────────┘
                            ▼
                       ChromaDB
                            │
                            ▼
                          LLM
                            │
                            ▼
                  Grounded Response
```

---

# 🤖 Target Architecture — RAG + JEV

The next architecture introduces JEV between the API and the existing RAG system.

```text
                       USER
                         │
                         ▼
                ┌─────────────────┐
                │  React Frontend │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │    Flask API    │
                └────────┬────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   JEV Orchestrator  │
              │                     │
              │ • Intent Detection  │
              │ • Query Routing     │
              │ • Workflow Select.  │
              │ • Agent Control     │
              └──────────┬──────────┘
                         │
             ┌───────────┼───────────┐
             │           │           │
             ▼           ▼           ▼
        ┌────────┐  ┌──────────┐ ┌────────────┐
        │RAG Tool│  │ Summary  │ │ Comparison │
        │        │  │ Workflow │ │  Workflow  │
        └────┬───┘  └──────────┘ └────────────┘
             │
             ▼
       ┌──────────────┐
       │ RAG Pipeline │
       └──────┬───────┘
              │
       ┌──────┼─────────┐
       ▼      ▼         ▼
   Retriever  Context   Generator
       │      │         │
       └──────┼─────────┘
              ▼
             LLM
              │
              ▼
     Grounded Answer
       + Citations
```

---

# 🔍 RAG Pipeline

The current RAG pipeline follows:

```text
User Question
      │
      ▼
Question Embedding
      │
      ▼
Vector Similarity Search
      │
      ▼
Top-K Relevant Chunks
      │
      ▼
Similarity Filtering
      │
      ▼
Context Construction
      │
      ▼
Grounded Prompt
      │
      ▼
LLM Generation
      │
      ▼
Answer + Sources
```

The existing RAG implementation remains the foundation of the project.

---

# 🤖 JEV Integration

## Why JEV?

JEV is planned as an **orchestration layer** on top of the existing RAG system.

The goal is not to replace RAG.

Instead:

```text
                JEV
                 │
       ┌─────────┼─────────┐
       ▼         ▼         ▼
      RAG     Summary   Comparison
      Tool     Tool       Tool
       │
       ▼
 Existing RAG Pipeline
```

This allows the system to evolve from:

> "Always perform RAG"

into:

> "Understand the user's request and select the appropriate workflow."

---

# 🛣️ Project Roadmap

## Phase 1 — Project Foundation ✅

**Status: Completed**

- Project structure
- Flask backend
- React frontend foundation
- Environment configuration
- Document ingestion foundation
- Basic API structure
- Vector database setup
- Initial RAG components

---

# Phase 2 — Core RAG Engine ✅

**Status: Completed**

### Implemented

- Query embeddings
- Cosine similarity search
- Top-K retrieval
- Similarity score filtering
- Context construction
- Anti-hallucination prompting
- LLM generation
- OpenAI/Groq provider support
- Grounded responses
- `/api/chat`
- Source citations
- Input validation
- Error handling
- Conversation history
- Stub mode without API key

---

# Phase 3 — JEV Orchestration 🚧

**Status: Planned / Next Major Phase**

This phase introduces JEV without rewriting the existing RAG implementation.

### Planned

- JEV foundation
- JEV orchestrator
- RAG tool integration
- Intelligent query routing
- API integration
- Agent memory
- Initial Agentic RAG
- Testing
- RAG vs JEV+RAG evaluation

### Target Flow

```text
/api/chat
    ↓
JEV Orchestrator
    ↓
Selected Tool
    ↓
RAG / Workflow
    ↓
Grounded Response
```

---

# Phase 4 — Advanced Agentic RAG 🚧

**Status: Planned**

- Query rewriting
- Hybrid retrieval
- Semantic + keyword search
- Reranking
- Multi-step retrieval
- Context compression
- Source verification
- Agentic reasoning
- Improved memory
- Failure recovery

---

# Phase 5 — Premium Frontend 🚧

**Status: Planned**

- Modern chat interface
- Document management dashboard
- Upload interface
- Source citation cards
- Retrieval score visualization
- Conversation history
- Agent activity indicator
- JEV workflow status
- Responsive design
- Dark/light theme
- Streaming responses

---

# Phase 6 — Evaluation & Analytics 🚧

**Status: Planned**

Metrics:

- Retrieval quality
- Answer relevance
- Faithfulness
- Citation accuracy
- Hallucination rate
- JEV routing accuracy
- Response latency
- Query success rate

---

# Phase 7 — Production & Deployment 🚧

**Status: Planned**

- Docker
- Production configuration
- Authentication
- Database integration
- API security
- Rate limiting
- Logging
- Monitoring
- Error tracking
- CI/CD
- Cloud deployment

---

# 📁 Project Structure

```text
Domain-Specific-Q-A-Chatbot-RAG/
│
├── backend/
│   ├── agents/                  # Phase 3 — JEV
│   │   ├── __init__.py
│   │   ├── orchestrator.py
│   │   ├── router.py
│   │   └── tools.py
│   │
│   ├── api/
│   │   └── routes.py
│   │
│   ├── config/
│   │   └── settings.py
│   │
│   ├── rag/
│   │   ├── chunking.py
│   │   ├── embeddings.py
│   │   ├── generation.py
│   │   ├── ingestion.py
│   │   ├── pipeline.py
│   │   └── retrieval.py
│   │
│   └── app.py
│
├── tests/
│   ├── test_pipeline.py
│   ├── test_phase2.py
│   ├── test_jev_orchestrator.py
│   ├── test_jev_router.py
│   └── test_agentic_rag.py
│
├── requirements.txt
├── .env.example
├── docker-compose.yml
└── README.md
```

> `agents/` and the JEV-specific tests are part of the planned Phase 3 implementation.

---

# 🧰 Technology Stack

### Backend
- Python
- Flask
- REST API

### AI / RAG
- Retrieval-Augmented Generation
- Sentence Transformers
- Embeddings
- ChromaDB
- LangChain components where applicable
- OpenAI
- Groq
- JEV orchestration

### Frontend
- React
- JavaScript / TypeScript
- HTML5
- CSS3

### DevOps
- Docker
- Docker Compose
- GitHub
- CI/CD — planned
- Cloud deployment — planned

---

# 📊 Evaluation Strategy

The project will compare:

```text
Traditional RAG
       VS
JEV + RAG
```

| Metric | Basic RAG | JEV + RAG |
|---|---:|---:|
| Retrieval Quality | ✓ | ✓ |
| Answer Relevance | ✓ | ✓ |
| Faithfulness | ✓ | ✓ |
| Citation Accuracy | ✓ | ✓ |
| Tool Selection | — | ✓ |
| Query Refinement | — | ✓ |
| Multi-step Retrieval | — | ✓ |
| Hallucination Rate | ✓ | ✓ |
| Latency | ✓ | ✓ |

---

# 💼 Resume Description

> **Built an Agentic Domain-Specific Q&A Assistant using Retrieval-Augmented Generation (RAG), vector search, LLMs, and JEV-based orchestration to provide grounded, citation-backed responses from private document collections.**

---

# 🔮 Future Enhancements

- Multi-agent architecture
- Advanced hybrid retrieval
- Cross-document reasoning
- Automatic document summarization
- Knowledge graph integration
- Personalized knowledge bases
- Advanced evaluation datasets
- Streaming responses
- Voice interaction
- Multimodal document understanding
- Enterprise authentication
- Role-based access control
- Observability and monitoring

---

# 📌 Current Status

```text
Phase 1  ████████████████████  Completed
Phase 2  ████████████████████  Completed
Phase 3  ░░░░░░░░░░░░░░░░░░░░  Planned
Phase 4  ░░░░░░░░░░░░░░░░░░░░  Planned
Phase 5  ░░░░░░░░░░░░░░░░░░░░  Planned
Phase 6  ░░░░░░░░░░░░░░░░░░░░  Planned
Phase 7  ░░░░░░░░░░░░░░░░░░░░  Planned
```

### Current Milestone

**Phase 2 — Core RAG Engine: Completed ✅**

### Next Milestone

**Phase 3 — JEV Orchestration 🚧**

---

# 📜 License

This project is licensed under the **Apache License 2.0**.

See the [`LICENSE`](LICENSE) file for the full license text.

---

# 👨‍💻 Author

**Premanshu Kumar**

B.Tech — Computer Science Engineering  
Data Science Specialization

GitHub: `Premanshu-Kumar`

---

⭐ If you find this project useful, consider giving the repository a star!
