import os
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from operator import itemgetter
from pathlib import Path
#  LOAD API key
load_dotenv()
os.environ["LANGCHAIN_PROJECT"] ="SwiftVisa-Project"

   
# Getting the directory of this script
current_dir = Path(__file__).resolve().parent

# Going to the project root (/scripts)
project_root = current_dir.parent 

#  creating my path for vector_db
CHROMA_PATH = project_root / "vector_store" / "chroma_db"


CHROMA_PATH_STR = str(CHROMA_PATH)


# 2. MODELS & RETRIEVER

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0,max_tokens=None,timeout=None,max_retries=2)

# Load the Vector DB
vector_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

# 3. PROMPT DESIGN
#to create context I am rephrasing the 
contextualize_prompt = (
    "Transform the chat history and latest user message into a "
    "concise, 3-5 word search query for a vector database. "
    "Focus on: Country + Visa Type + Specific Requirement. "
    "Output ONLY the keywords."
)
system_instruction = (
    "You are a Senior Visa Officer. {user_details}\n\n"
    
    "PHASE 1 (Initial Inquiry): If the user is asking about requirements for the first time, "
    "list the requirements from the PDF ONLY. Ensure you display the Technical RAG Score prominently. "
    "To achieve a high score, focus on the specific visa eligibility criteria in the text.\n\n"
    
    "PHASE 2 (Conclusion): ONLY calculate the Eligibility Confidence Score (0-100%) if the user "
    "explicitly asks 'Am I eligible?' or at the very end of the conversation. "
    "Use this weighting: 25% (Identity,valid passport), 25% (Funds), 25% Ties(Job,Family), 25% Purpose(Acceptance letter,Job offer,Invitation letter).\n\n"
    
    "- Eligibility Confidence: [Calculated ONLY on request]\n\n"
    
    "REQUIRED FORMAT:\n"
    "### 1. **Technical RAG Score**: {relevance_score}%\n"
    "### 2. **Visa Requirements**: [Directly from Context]\n"
    "### 3. **Eligibility Analysis**: [Calculated ONLY if 'eligibility' is asked .If eligibility is not requested, simply hide this section or say 'Awaiting user profile completion']\n"
    
    "TECHNICAL METRICS:\n"
    "- Technical RAG Score: {relevance_score}%\n"
    "### 4. **Source**: [Filename]"
)
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_instruction),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Context: {context}\n\nQuestion: {input}"),
])




# 4. SESSION HISTORY STORE
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 5. I have created this function so that it gets  the most relevant ie first chunk (context that we get from vector_store)
def format_docs(docs_with_scores):
    
    
    if not docs_with_scores:
        return {"context":"No relevant information found.","relevance_score":0}
    #docs_with_scores is a list [(doc,score),(doc,score),.....]
    doc, score = docs_with_scores[0]
    
    #doc,score = first_tuple
    # Distance to Similarity conversion: (0.0 distance = 100% similarity)
    similarity = max(0, (1 - score)) * 100
    #checking if i am getting the chunk (for 500 ERROR)
    print(f" Source: {doc.metadata.get('source', 'Unknown')}")
    
    print(f"RAG relevance score: {similarity:.2f}%")
    # Combine all k=3 chunks for AI to read
    full_context = "\n\n".join(doc.page_content for doc, _ in docs_with_scores)
    return {
        "context":full_context,
        "relevance_score": round(similarity,2)
    }
    

    
   
def get_single_query(data):
    #if history is absent use the input
    if not data.get("chat_history"):
        return f"{data['country']} {data['input']} official eligibility requirements"
    #if history is present LLM will merge the country context into question
    messages = [SystemMessage(content=contextualize_prompt)] +data["chat_history"] + [HumanMessage(content=data["input"])]
    standalone_q=  llm.invoke(messages).content

    #print(f"\n Original query : {data['input']}")
    #print(f"\n Reformed query: {standalone_q}")
    return standalone_q
# Base chain without history

base_rag_chain = (
    RunnablePassthrough.assign(
        standalone_query = get_single_query
    )    
    | RunnablePassthrough.assign(
        retrieval_data = lambda x: format_docs(
            vector_db.similarity_search_with_relevance_scores(
                # for filter of country
                x["standalone_query"],
                k=3,
                filter={"country": x["country"].strip().title()}
            )
        )
    )
    | RunnablePassthrough.assign(
        context = lambda x: x["retrieval_data"]["context"],
        relevance_score = lambda x: x["retrieval_data"]["relevance_score"]
    )
    | qa_prompt
    | llm
    | StrOutputParser()
)

# 6. WRAP CHAIN  WITH SESSION HISTORY
with_history_chain = RunnableWithMessageHistory(
    base_rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# 7. CHAT FUNCTION 
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
                 "country": user_profile["target_country"],
                 "user_details":(
                     f"User Name: {user_profile["name"]},"
                     f"Age: {user_profile["age"]},"
                     f"Education: {user_profile["education"]},"
                     f"Purpose:{user_profile["purpose"]}"
                     f"Income:{user_profile["income"]}"
                     f"From: {user_profile["home_country"]}"
                 )
    }
    
    max_retries = 3
    for i in range(max_retries):
        try:
            # timer to avoid rate limits
            time.sleep(1) 
            return with_history_chain.invoke(input_data, config=config)
        except Exception as e:
            if "500" in str(e) or "429" in str(e):
                print(f" Google API Busy (Attempt {i+1}). Retrying in 5s...")
                time.sleep(5)
            else:
                return f"An unexpected error occurred: {e}"
    
    return "I'm sorry, Google servers are currently overloaded. Please try again in a minute."

# 8. EXECUTION LOOP
if __name__ == "__main__":
    print("--- AISwiftVisa: User profile setup--- ")
    profile={
        "name": input("Enter name:"),
        "age" : input("Enter age:"),
        "education" : input("Enter education:"),
        "purpose" : input("Enter the purpose for which you want to visit:"),
        "income" : input("Enter if you are employed or not,if employed mention your income:"),
        "home_country": input("Enter your country:"),
        "target_country": input(" Enter country you want to visit:").strip().title()
    }
    session_id = f"user_{int(time.time())}"
    print(f"\n--- Consulting for {profile['target_country']} ---")

   

    while True:
        u_input = input("\nYou: ")
        if u_input.lower() in ["exit", "quit"]: break
        
        response = chat_with_visa_bot(session_id,profile, u_input)
        print(f"\nAI: {response}")