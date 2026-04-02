import os
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from pathlib import Path
load_dotenv()
current_dir = Path(__file__).resolve().parent
CHROMA_PATH = str(current_dir.parent / "vector_store" / "chroma_db")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

# Run this to see all valid tags
all_docs = vector_db.get(include=["metadatas"])
unique_countries = set(m.get("country") for m in all_docs["metadatas"])
print(f"The countries in your DB are: {unique_countries}")