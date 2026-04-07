import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pathlib import Path

load_dotenv()

current_dir = Path(__file__).resolve().parent
CHROMA_PATH= str(current_dir.parent/"vector_store"/"chroma_db")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_db = Chroma(persist_directory=str(CHROMA_PATH), embedding_function=embeddings)
def test_query(query,country):
    print(f"\n...Retrieving query.. '{query}' in '{country}'...")
    
     # 1. Detect Intent: We look for keywords to decide the 'Focus'
    if any(word in query for word in ["study", "student", "university", "college", "degree"]):
        visa_focus = "student"
    elif any(word in query for word in ["visitor", "tourism", "vacation", "holiday", "tourist"]):
        visa_focus = "visitor"
    elif any(word in query for word in ["work", "job", "employment", "hire", "salary"]):
        visa_focus = "work"
    else:
        visa_focus = "" # Fallback to general search if intent is unclear
    search_string = f"{country} {visa_focus} {query}"    
    # We use similarity_search_with_relevance_scores to see the RAW distance
    results = vector_db.similarity_search_with_relevance_scores(
        search_string, 
        k=3, 
        filter={"country": country}
    )
    if not results:
        print("RESULT: ❌ No documents found matching that country filter!")
        return
    
    
    for i, (doc, score)in enumerate(results):
        similarity = max(0, (1 - score)) * 100
        print(f"\n[Match #{i+1}] Confidence: {similarity:.2f}%")
        print(f"Source: {doc.metadata.get('source')}")
        print(f"Snippet: {doc.page_content[:150]}...")
    
if __name__ == "__main__":
    test_query("tourist visa requirements","Usa")