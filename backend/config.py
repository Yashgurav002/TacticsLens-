import os
from dotenv import load_dotenv

load_dotenv()

# ----------------------------
# ENVIRONMENT TOGGLE
# True  = local dev (Ollama + local embeddings)
# False = production (Groq API + HuggingFace)
USE_LOCAL=False

LLM_PROVIDER = "ollama" if USE_LOCAL else "groq"
LLM_MODEL = "mistral" if USE_LOCAL else "llama-3.1-8b-instant"
OLLAMA_BASE_URL = "http://localhost:11434"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    
# EMBEDDING SETTINGS
# Same model used locally and in production
# HuggingFace downloads it automatically
# ----------------------------
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# CHROMADB SETTINGS
# ----------------------------
CHROMA_PATH = "./chroma_db"
CHROMA_COLLECTION = "tacticlens"

# STATSBOMB DATA PATH
# ----------------------------
DATA_PATH = os.getenv("DATA_PATH", "../data/open-data")


CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K = 8          # how many chunks to retrieve per query

# SANITY CHECK
# Run: python config.py
# ----------------------------
if __name__ == "__main__":
    print("=== TacticLens Config ===")
    print(f"Mode        : {'LOCAL (Ollama)' if USE_LOCAL else 'PRODUCTION (Groq)'}")
    print(f"LLM Model   : {LLM_MODEL}")
    print(f"Embed Model : {EMBED_MODEL_NAME}")
    print(f"ChromaDB    : {CHROMA_PATH}")
    print(f"Data Path   : {DATA_PATH}")
    print(f"Chunk Size  : {CHUNK_SIZE} tokens")
    print(f"Top K       : {TOP_K} chunks retrieved")
    print("========================")
