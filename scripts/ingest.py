import os
import time
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# PATH SETUP
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
RAW_DATA_PATH = project_root / "data" / "raw_pdfs"
CHROMA_PATH = project_root / "vector_store" / "chroma_db"
PROGRESS_FILE = project_root / "ingestion_progress.txt"

def get_processed_files():
    if not PROGRESS_FILE.exists():
        return set()
    with open(PROGRESS_FILE, "r") as f:
        return set(line.strip() for line in f)

def mark_as_processed(filename):
    with open(PROGRESS_FILE, "a") as f:
        f.write(f"{filename}\n")

def ingest_docs_with_gemini():
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not found.")
        return

    processed_files = get_processed_files()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_version="v1")
    
    # Initialize Vector DB (Connects to existing if it exists)
    vector_db = Chroma(persist_directory=str(CHROMA_PATH), embedding_function=embeddings)

    # 1. Loop through PDFs in the raw_pdfs folder
    pdf_files = [f for f in os.listdir(str(RAW_DATA_PATH)) if f.endswith(".pdf")]

    for file in pdf_files:
        if file in processed_files:
            print(f" Skipping {file} (already fully processed)")
            continue

        print(f"\n Processing: {file}")
        
        # --- SELECTIVE DELETE ---
        # This removes any existing chunks for THIS specific file to avoid duplicates 
        # in case the last run crashed halfway through this file.
        try:
            vector_db.delete(where={"source": file})
            print(f"Cleaned existing partial data for {file}")
        except Exception:
            pass

        # Metadata Logic
        filename_clean = file.replace(".pdf", "")
        parts = filename_clean.split("_")
        
        # Ensuring "india" becomes "India" to match main.py logic
        country_name = parts[0].strip().title() 
        visa_category = parts[1].strip().title() if len(parts) > 1 else "General"

        # Load and Tag
        loader = PyMuPDFLoader(os.path.join(str(RAW_DATA_PATH), file))
        file_docs = loader.load()
        for doc in file_docs:
            doc.metadata["country"] = country_name
            doc.metadata["visa_type"] = visa_category
            doc.metadata["source"] = file

        # Chunking THIS file
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        file_chunks = text_splitter.split_documents(file_docs)
        print(f" Created {len(file_chunks)} chunks for {file}")

        # Batch Upload with Retries
        batch_size = 20
        for i in range(0, len(file_chunks), batch_size):
            batch = file_chunks[i : i + batch_size]
            print(f" Uploading chunks {i} to {i + len(batch)} for {file}...")
            
            # Simple retry loop for network glitches
            success = False
            for attempt in range(5):
                try:
                    vector_db.add_documents(batch)
                    print(f"  [Batch {i//batch_size + 1}] Uploaded successfully.")
                    success = True
                    break
                except Exception as e:
                    wait = (attempt +1) * 60
                    print(f" Rate limit hit: {e}. Retrying in {wait}s...")
                    time.sleep(wait)
            
            if not success:
                print(f" Failed to upload batch for {file}. Stopping script.")
                return

            time.sleep(45) # Short rest between batches

        # Mark file as complete in our log
        mark_as_processed(file)
        print(f" Successfully finished: {file}")

    print(f"\n All files processed! Vector store ready at {CHROMA_PATH}")

if __name__ == "__main__":
    ingest_docs_with_gemini()