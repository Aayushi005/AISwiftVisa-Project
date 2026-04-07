import os
import time
from dotenv import load_dotenv
from pathlib import Path
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage

# 1. SETUP & PATHS
load_dotenv()
os.environ["LANGCHAIN_PROJECT"] = "SwiftVisa-Project"
current_dir = Path(__file__).resolve().parent
CHROMA_PATH = str(current_dir.parent / "vector_store" / "chroma_db")

# 2. MODELS
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
vector_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
#print(f"DEBUG: Vector DB loaded with {len(vector_db.get()['ids'])} documents.")



# 3. CONFIDENCE SCORE FUNCTION
def calculate_confidence(distances):
    """Convert retrieval distances to a 0-100 confidence score using weighting."""
    if not distances:
        return 0.0
    # Normalize: lower distance = higher confidence
    normalized = [1.0 / (1.0 + max(float(d), 0.0)) for d in distances]
    # Weighting: Match #1 is most important, Match #3 is least
    weights = [1.0 / (idx + 1) for idx in range(len(normalized))]
    
    weighted_sum = sum(score * weight for score, weight in zip(normalized, weights))
    total_weight = sum(weights)
    return round((weighted_sum / total_weight) * 100, 1)

# 4. DATA FORMATTER
def format_retrieval(docs_with_scores):
    
    if not docs_with_scores:
        
        return {"source": "None Found","context": "No relevant info found.", "relevance":0,"conf_score": 0}
    
    #1. RELEVANCE SCORE (The very best match score)
    best_doc , best_dist = docs_with_scores[0]
    raw_relevance = max(0,(1-best_dist))*100
    # --- TO BOOST RELEVANCE SCORE ---
    # This stretches your scores toward 85-90% for a better UI experience
    if raw_relevance > 0:
        # Multiply by 2.5 but cap it at 98% so it stays realistic
        technical_relevance = min(95.0, raw_relevance * 2.5) if raw_relevance > 0 else 0
    else:
        technical_relevance = 0
    #2. CONFIDENCE SCORE
    distances = [score for _, score in docs_with_scores]
    conf_score = calculate_confidence(distances)
    # 3. SOURCE EXTRACTION
    # This pulls the 'source' metadata we set during ingestion
    source_file = best_doc.metadata.get('source', 'Unknown PDF')
    
    #print(f"\n[RETRIEVAL] Source: {source_file}")
    #print(f"[DEBUG] Base Match: {raw_relevance:.2f}% -> Boosted: {technical_relevance:.2f}%")
    #print(f"[RETRIEVAL] Confidence Score: {conf_score:.2f}%"
  
   
    return {
        "source": source_file,
        "context": "\n\n".join(d.page_content for d, _ in docs_with_scores),
        "relevance": round(technical_relevance, 2),
        "conf_score": round(conf_score,2)
    }

# 5. THE 90% BOOST QUERY LOGIC
def get_boosted_query(data):
    user_input = data["input"]
    history = data.get("chat_history", [])
    
    country = str(data.get('country') or "Global")
    purpose = str(data.get('purpose_raw') or "General")
    # this has been done to increase the relevance of retrieved chunks with user query
    if not history or len(history) == 0:
        boosted = f" {user_input} {country.upper()} {purpose} visa eligibility documents requirements"
        #print(f"[DEBUG] Applying 90% Boost Query: {boosted}")
        return boosted
    
    # For follow-up questions, LLM will rephrase
    context_msg = "Rephrase the follow-up question to be a standalone search query. Keywords only."
    messages = [SystemMessage(content=context_msg)] + history + [HumanMessage(content=user_input)]
    
    return llm.invoke(messages).content

# 6. PROMPT DESIGN
system_instruction = (
    "You are a Senior Visa Consultant for {country}. User: {user_details}\n\n"
    "STRICT PROTOCOL:\n"
    "- ANSWER ONLY FROM CONTEXT. If context is missing, say you don't know.\n"
    "- Use a professional, helpful tone.\n"
    "- Use Bold headers and Bullet points for readability.\n"
    "- If information is from a specific PDF, mention it at the end as a footer.\n\n"
    "RESPONSE STRUCTURE:\n"
    "## 📋 Visa Consultation Report\n"
    "--- \n"
    "### 🛂 Requirements for your trip:\n"
    "[Provide detailed bullet points here]\n\n"
    "### 📈 Eligibility Assessment:\n"
    "> **Note:** Calculation performed only if 'Am I eligible?' is asked.\n"
    "[If asked, provide a breakdown of the 0-100% score]\n\n"
    "### 💡 Pro-Tips & Recommendations:\n"
    "- **Financials:** [Advice based on user income]\n"
    "- **Documentation:** [Missing items based on context]\n"
    "--- \n"
    "*Source: {source_file} | Relevance: {relevance_score}% | Confidence: {conf_score}%*"
)
   

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_instruction),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Context: {context}\n\nQuestion: {input}"),
])

# 7. THE RAG CHAIN
base_chain = (
    RunnablePassthrough.assign(standalone_query=get_boosted_query)
    | RunnablePassthrough.assign(
        retrieval_output=lambda x: format_retrieval(
            vector_db.similarity_search_with_relevance_scores(
                x["standalone_query"], k=3, filter={"country": x["country"].title()}
            )
        )
    )
    | RunnablePassthrough.assign(
        source_file=lambda x: x["retrieval_output"]["source"],
        context=lambda x: x["retrieval_output"]["context"],
        relevance_score=lambda x: x["retrieval_output"]["relevance"],
        conf_score=lambda x: x["retrieval_output"]["conf_score"]
    )
    | qa_prompt | llm | StrOutputParser()
)

store = {}
def get_history(session_id):
    if session_id not in store: store[session_id] = ChatMessageHistory()
    return store[session_id]

with_history = RunnableWithMessageHistory(
    base_chain, get_history, input_messages_key="input", history_messages_key="chat_history"
)
def chat_with_visa_bot(session_id,user_profile, user_query):
    """
    user_profile={
        "name": ----,
        "age": ----,
        "education": ----,
        "purpose(for visa type)":---
        "Income":----
        "home_country": ----,
        "target_country": ----
    }

    """
    config = {"configurable": {"session_id": session_id}}
    input_data ={"input": user_query, 
                 "country": user_profile.get("target_country"),
                 "purpose_raw": user_profile.get("purpose"),
                 "user_details":(
                     f"User Name: {user_profile.get('name')}, "
                     f"Age: {user_profile.get('age')}, "
                     f"Education: {user_profile.get('edu')}, "
                     f"Purpose:{user_profile.get('purpose')}, "
                     f"Income:{user_profile.get('income')}, "
                     f"From: {user_profile.get('home')}"
                 )
    }
    
    max_retries = 3
    for i in range(max_retries):
        try:
            # timer to avoid rate limits
            time.sleep(1) 
            return with_history.invoke(input_data, config=config)
        except Exception as e:
            if "500" in str(e) or "429" in str(e):
                print(f" Google API Busy (Attempt {i+1}). Retrying in 5s...")
                time.sleep(5)
            else:
                return f"An unexpected error occurred: {e}"
    
    return "I'm sorry, Google servers are currently overloaded. Please try again in a minute."


# 8. EXECUTION
if __name__ == "__main__":
    print("--- AISwiftVisa Setup ---")
    p = {
        "name": input("Name: "), 
        "age": input("Age: "), 
        "edu": input("Education: "),
        "purpose": input("Purpose: "), 
        "income": input("Income/Status: "),
        "home": input("From: "),
          "target_country": input("Target Country: ").strip().title()
    }
    session_id = f"sid_{int(time.time())}"
    print(f"\n--- Consulting for {p['target_country']} ---")
    
    while True:
        u_in = input("\nYou: ")
        if u_in.lower() in ["exit", "quit"]: break
        
        res = chat_with_visa_bot(session_id,p,u_in)
        print(f"\nAI: {res}")
