# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from contextlib import asynccontextmanager
import os

from config import USE_LOCAL, LLM_MODEL
from pipeline import build_rag_chain, query as rag_query

# ------------------------------------------------
# GLOBAL — chain loaded once when server starts
# ------------------------------------------------
rag_chain = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_chain
    print("Loading RAG chain...")
    rag_chain = build_rag_chain()
    print("RAG chain ready!")
    yield


# ------------------------------------------------
# APP SETUP
# ------------------------------------------------
app = FastAPI(
    title="TacticLens API",
    description="Football analytics powered by RAG",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------
# REQUEST / RESPONSE MODELS
# ------------------------------------------------
class ChatRequest(BaseModel):
    question: str
    match_context: str | None = None
    match_id: str | None = None
    history: list[dict] | None = []


class ChatResponse(BaseModel):
    answer: str
    question: str


# ------------------------------------------------
# ROUTES
# ------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": LLM_MODEL,
        "mode": "local" if USE_LOCAL else "production",
    }


@app.get("/matches")
def get_matches():
    matches = [
        {
            "match_id": 303731,
            "home_team": "Barcelona",
            "away_team": "Eibar",
            "home_score": 5,
            "away_score": 0,
            "match_date": "2020-02-22",
            "competition": "La Liga",
            "season": "2019/2020",
        },
        {
            "match_id": 303532,
            "home_team": "Barcelona",
            "away_team": "Leganés",
            "home_score": 2,
            "away_score": 0,
            "match_date": "2020-06-16",
            "competition": "La Liga",
            "season": "2019/2020",
        },
        {
            "match_id": 303516,
            "home_team": "Celta Vigo",
            "away_team": "Barcelona",
            "home_score": 2,
            "away_score": 2,
            "match_date": "2020-06-27",
            "competition": "La Liga",
            "season": "2019/2020",
        },
        {
            "match_id": 3869685,
            "home_team": "Argentina",
            "away_team": "France",
            "home_score": 3,
            "away_score": 3,
            "match_date": "2022-12-18",
            "competition": "FIFA World Cup",
            "season": "2022",
            "note": "AET — Argentina won 4-2 on penalties",

        },
        {
            "match_id": 8658,
            "home_team": "France",
            "away_team": "Croatia",
            "home_score": 4,
            "away_score": 2,
            "match_date": "2018-07-15",
            "competition": "FIFA World Cup",
            "season": "2018",
        },
    ]
    return {"matches": matches}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    global rag_chain

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        question = request.question
        match_ctx = request.match_context or ""
        q_lower = question.lower()

        if any(w in q_lower for w in ["lineup", "starting", "11", "squad", "team sheet", "who played", "who started"]):
            question = f"Starting lineup and squad for {match_ctx}: {question}. List the starting 11 players for both teams."

        elif any(w in q_lower for w in ["assist", "assisted", "who helped", "who set up"]):
            question = f"Goals and assists in {match_ctx}: {question}. Who assisted the goals scored in this match."

        elif any(w in q_lower for w in ["score", "result", "winner", "final", "goals"]):
            question = f"Match result and goals for {match_ctx}: {question}."

        elif match_ctx:
            question = f"Regarding the match {match_ctx}: {question}."

        answer = rag_query(question, request.history or [], match_id=request.match_id)
        return ChatResponse(
            answer=answer,
            question=request.question,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------
# SERVE REACT FRONTEND
# Must be LAST — catches all remaining routes
# Only mount if static folder exists (not in dev)
# ------------------------------------------------
if os.path.exists("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")


# ------------------------------------------------
# RUN
# ------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)