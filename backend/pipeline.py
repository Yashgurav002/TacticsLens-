# pipeline.py
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from config import (
    USE_LOCAL,
    LLM_MODEL,
    OLLAMA_BASE_URL,
    CHROMA_PATH,
    CHROMA_COLLECTION,
    EMBED_MODEL_NAME,
    TOP_K,
)

# ------------------------------------------------
# STEP 1 — Load embedding model (same as ingest.py)
# Must be identical so vectors match
# ------------------------------------------------
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ------------------------------------------------
# STEP 2 — Connect to ChromaDB
# ------------------------------------------------
def get_vectorstore():
    embedding_model = get_embedding_model()
    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION,
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_model,
    )
    return vectorstore


# ------------------------------------------------
# STEP 3 — Load LLM
# Ollama locally, Groq in production
# ------------------------------------------------
def get_llm():
    if USE_LOCAL:
        return ChatOllama(
            model=LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,  # low temp = more factual answers
        )
    else:
        from config import GROQ_API_KEY
        return ChatGroq(
            model=LLM_MODEL,
            api_key=GROQ_API_KEY,
            temperature=0.1,
        )


# ------------------------------------------------
# STEP 4 — Build the RAG prompt
# ------------------------------------------------
def get_prompt():
    template = """You are TacticLens, an expert football analyst AI.
You answer questions about football matches using ONLY the match data provided below.

Rules:
- Always use exact player names, minute numbers, and statistics from the data
- For "who scored" questions, list ALL goalscorers with their minutes
- For counting questions, use the exact numbers from the data
- If asked about a specific match, only use data from that match
- If the data doesn't contain enough information, say exactly what you do and don't know
- Keep answers concise but complete

Previous conversation:
{history}

Match Data:
{context}

Question: {question}

Answer:"""
    return ChatPromptTemplate.from_template(template)
# ------------------------------------------------
# STEP 5 — Format retrieved documents into a string
# ------------------------------------------------
def format_docs(docs):
    return "\n\n---\n\n".join([doc.page_content for doc in docs])


# ------------------------------------------------
# STEP 6 — Build the full RAG chain
# ------------------------------------------------
def build_rag_chain():
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": TOP_K,
            "fetch_k": 20,
            "lambda_mult": 0.7,
        },
    )
    llm = get_llm()
    prompt = get_prompt()

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
            "history": lambda x: "",  # placeholder, filled by query()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# ------------------------------------------------
# STEP 7 — Query function (used by FastAPI later)
# ------------------------------------------------
def query(question: str, history: list[dict] = []) -> str:
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "fetch_k": 20, "lambda_mult": 0.7},
    )
    llm = get_llm()
    prompt = get_prompt()

    # Format history as readable text
    history_text = ""
    if history:
        lines = []
        for msg in history:
            role = "User" if msg["role"] == "user" else "TacticLens"
            lines.append(f"{role}: {msg['content']}")
        history_text = "\n".join(lines)

    docs = retriever.invoke(question)
    context = format_docs(docs)

    formatted = prompt.format_messages(
        context=context,
        question=question,
        history=history_text,
    )
    response = llm.invoke(formatted)
    return response.content


# ------------------------------------------------
# SANITY CHECK
# Run: python pipeline.py
# ------------------------------------------------
if __name__ == "__main__":
    print("Loading pipeline...")
    print("(First run may take 30s to load embedding model)\n")

    test_questions = [
        "How many shots did Barcelona take against Eibar?",
        "Which player made the most passes in the Barcelona vs Leganés match?",
        "Who scored in the Celta Vigo vs Barcelona match?",
    ]

    chain = build_rag_chain()

    for question in test_questions:
        print(f"Q: {question}")
        print(f"A: {chain.invoke(question)}")
        print("-" * 60)