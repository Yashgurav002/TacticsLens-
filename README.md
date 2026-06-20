<div align="center">

# ⚽ TacticLens

### RAG-powered tactical football analysis — ask questions, get scout-level answers

[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-Hugging%20Face%20Spaces-yellow?style=for-the-badge)](https://yash-2002-tacticlens.hf.space)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-Vite-61DAFB?style=flat&logo=react&logoColor=black)](https://react.dev/)
[![LangChain](https://img.shields.io/badge/🦜🔗-LangChain-1C3C3C?style=flat)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-FF6F61?style=flat)](https://www.trychroma.com/)
[![Groq](https://img.shields.io/badge/Groq-Llama%203.1-F55036?style=flat)](https://groq.com/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)

**[🚀 Try the live app](https://yash-2002-tacticlens.hf.space)** · **[📦 View on Hugging Face](https://huggingface.co/spaces/Yash-2002/tacticlens)**

</div>

---

## What is this?

TacticLens is a Retrieval-Augmented Generation chatbot that lets you have a real conversation about specific football matches — not generic football trivia, but actual tactical breakdowns grounded in event-level data: who assisted whom, shot xG, formations, substitution timing, and match flow.

Ask it things like:
- *"How did Argentina break down France's defense in the 2022 final?"*
- *"Who created the most chances for Barcelona against Eibar?"*
- *"Compare Croatia's pressing in the 2018 final to their group stage games."*

Under the hood, it's a from-scratch RAG pipeline built on real StatsBomb open event data — not a wrapper around an LLM's vague pre-trained football knowledge.

---

## How it works

```
┌─────────────┐      ┌──────────────┐      ┌────────────────┐      ┌─────────┐
│  React UI   │ ───▶ │   FastAPI     │ ───▶ │  Two-Tier RAG   │ ───▶ │  Groq   │
│  (chat)     │ ◀─── │   Backend     │ ◀─── │   Retrieval     │ ◀─── │ Llama3.1│
└─────────────┘      └──────────────┘      └────────────────┘      └─────────┘
                                                     │
                                            ┌────────┴────────┐
                                            │    ChromaDB      │
                                            │  3,563 chunks    │
                                            │  across 5 matches│
                                            └──────────────────┘
```

**The retrieval isn't a single similarity search — it's a two-step pipeline:**

1. **Direct metadata fetch** — every query first pulls the `MATCH_SUMMARY` chunk for the selected match by `match_id`, guaranteeing the LLM always has full match context (score, teams, key stats) regardless of how the semantic search performs.
2. **MMR semantic search** — on top of that, Maximal Marginal Relevance retrieval pulls the 5 most relevant *player and event-level* chunks, tuned for diversity over raw similarity so the model isn't fed five near-duplicate passages about the same shot.

This combination means the bot doesn't lose the forest for the trees — it always knows the final score even while reasoning about a single 60th-minute through-ball.

---

## Engineering decisions worth knowing about

| Decision | Why |
|---|---|
| **Player + event-type chunking** | Instead of naive fixed-size chunking, data is grouped by player and event type so semantically related actions (e.g. all of a player's key passes) stay together in retrieval. |
| **Assist resolution via `key_pass_id`** | StatsBomb's event data links assists to their originating pass through an ID reference, not inline. The ingestion pipeline resolves this cross-reference at index time so the bot can answer "who assisted X" directly. |
| **`match_id` metadata filtering** | Retrieval is scoped to the selected match via Chroma metadata filters, preventing cross-match hallucination when multiple matches are indexed. |
| **MMR over plain similarity** | Plain top-k similarity search tends to return near-duplicate chunks about the same high-salience event. MMR trades a little relevance for diversity, giving the LLM a broader slice of the match. |
| **Groq (Llama 3.1 8B) over OpenAI** | Free tier, and meaningfully faster inference — important for a chat UI where response latency directly affects perceived quality. |
| **Built from scratch before LangChain** | The retrieval logic was first hand-built (manual chunking, embedding, and retrieval) before formalizing it with LangChain abstractions — so the architecture reflects actual understanding of the RAG internals, not just framework defaults. |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React + Vite |
| Backend | FastAPI + Uvicorn |
| RAG Orchestration | LangChain |
| Vector Store | ChromaDB (persisted, pre-built index) |
| Embeddings | `BAAI/bge-small-en-v1.5` |
| LLM (production) | Llama 3.1 8B Instant via Groq API |
| LLM (local dev) | Mistral via Ollama |
| Deployment | Docker → Hugging Face Spaces |
| Data Source | [StatsBomb Open Data](https://github.com/statsbomb/open-data) |

---

## Matches currently indexed

| Match | Competition | Result |
|---|---|---|
| Barcelona vs Eibar | La Liga | 5–0 |
| Barcelona vs Leganés | La Liga | 2–0 |
| Celta Vigo vs Barcelona | La Liga | 2–2 |
| Argentina vs France | World Cup 2022 Final | 3–3 (AET, Argentina won 4–2 on pens) |
| France vs Croatia | World Cup 2018 Final | 4–2 |

---

## Project Structure

```
TacticsLens/
├── backend/
│   ├── main.py            # FastAPI app + lifespan RAG chain init
│   ├── ingest.py           # StatsBomb data → ChromaDB ingestion pipeline
│   ├── pipeline.py         # Two-tier retrieval + RAG chain construction
│   ├── config.py           # Model, embedding, and chunking configuration
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── chroma_db/          # Pre-built vector index (committed via Git LFS)
│   └── static/              # Production React build, served by FastAPI
├── frontend/
│   └── src/
│       ├── App.jsx
│       └── api.js
└── data/
    └── open-data/           # StatsBomb open-data (not committed — see below)
```

---

## Running it locally

**1. Clone the repo**
```bash
git clone https://github.com/Yashgurav002/TacticsLens-.git
cd TacticsLens
```

**2. Get the data**

Match data isn't committed to this repo (it's a public dataset, no reason to duplicate it). Clone it separately into `data/open-data/`:
```bash
git clone https://github.com/statsbomb/open-data.git data/open-data
```

**3. Backend setup**
```bash
cd backend
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Create a `.env` file with your Groq API key:
```
GROQ_API_KEY=your_key_here
```

Build the vector index, then run the server:
```bash
python ingest.py
uvicorn main:app --reload --port 8000
```

**4. Frontend setup**
```bash
cd frontend
npm install
npm run dev
```

---

## Deployment

The app ships as a single Docker container — FastAPI serves both the API and the built React frontend from `backend/static/`, so there's no separate frontend host or CORS configuration needed in production. It's currently deployed on **Hugging Face Spaces** using the free Docker SDK tier.

---

## Roadmap

- [ ] Expand match coverage beyond the current 5 (full competition-level ingestion)
- [ ] Add streaming token responses to the chat UI
- [ ] Player-level comparison queries across multiple matches
- [ ] Visual pass-map / shot-map rendering alongside text answers

---

## About

Built by **Yash Gurav** — AI/ML engineer focused on practical RAG systems, not toy demos.
🐦 X/Twitter: [@yash_gurav_2002](https://x.com/yash_gurav_2002)
- 💼 LinkedIn: [yash-gurav](https://www.linkedin.com/in/yash-gurav-58bbba21a/)

---

<div align="center">

*If this project is useful or interesting to you, a ⭐ on the repo is appreciated.*

</div>