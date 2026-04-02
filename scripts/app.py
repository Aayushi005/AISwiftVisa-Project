import streamlit as st
import time
import newMain
from newMain import chat_with_visa_bot # Import from newMain


st.set_page_config(page_title="AISwiftVisa", layout="wide",page_icon="✈️")

# 2. CUSTOM CSS 
st.markdown("""
    <style>
    /* overall Background */
    .stApp {
        background-color: #F7F9FC;
        
    }
    
    /* Sidebar Styling - Soft Indigo*/
    section[data-testid="stSidebar"] {
        background-color: #1E293B !important;
        color: white;
    }
    /* Force Sidebar labels to be white/readable */
    section[data-testid="stSidebar"] label {
        color: #E2E8F0 !important;
        font-weight: 500;
    }        
    /* Main Header Styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1E293B;
        text-align: center;
        padding: 20px;
        background: white;
        border-radius: 15px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        margin-bottom: 25px;
    }
    /* Chat Bubble - Rounded & Shadowed */
    .stChatMessage {
        background-color: white !important;
        border-radius: 15px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #E2E8F0 !important;
        padding: 15px;
    }
    /* User Message - Light Indigo Tint */
    .stChatMessage[data-testid="stChatMessageUser"] {
        background-color: #EEF2FF !important;
        border: 1px solid #C7D2FE !important;
    }
    /* Input Box Styling */
    .stChatInputContainer {
        padding-bottom: 20px;
        background-color: transparent !important;
    }
    
    /* Hide the top black bar/padding */
    .block-container {
        padding-top: 2rem !important;
    }
    /* Target the button container and the button itself specifically */
    div.stButton > button:first-child {
        background-color: #FFFFFF !important; /* Pure White Background */
        color: #1E293B !important;           /* Dark Blue/Black Text */
        border: 2px solid #4F46E5 !important; /* Indigo Border */
        border-radius: 10px !important;
        width: 100% !important;
        height: 3em !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Hover State: Inverse colors */
    div.stButton > button:hover {
        background-color: #4F46E5 !important; /* Indigo Background */
        color: #FFFFFF !important;           /* White Text */
        border: 2px solid #FFFFFF !important;
    }
    </style>
    """, unsafe_allow_html=True)
# --- SIDEBAR: USER PROFILE ---
with st.sidebar:
    st.title(" 👤 User Profile")
    st.divider()
    st.info("Fill this out before starting the chat.")
    name = st.text_input("Full Name")
    age = st.number_input("Age", min_value=1, max_value=100, value=22)
    edu = st.selectbox("Education", ["High School", "Graduation", "Masters", "PhD", "Other"])
    income = st.text_input("Employment/Income Status" ,placeholder="e.g. Unemployed/employed")
    target_country = st.selectbox("Target Country", ["Usa", "Uk", "Canada"])
    purpose = st.text_input("Purpose of Visit (e.g., Study/Tourist/Work)")
    home = st.text_input("Your Country")
    
    user_profile = {
        "name": name,
          "age": age, 
          "edu": edu,
        "income": income, 
        "target_country": target_country,
        "purpose": purpose,
        "home": home
    }
    
    st.divider()
    if st.button("CLEAR CHAT HISTORY"):
        st.session_state.messages = []
        st.session_state.session_id = f"st_{int(time.time())}"
        st.success("Chat history wiped!") # Shows a green checkmark briefly
        time.sleep(0.5)
        st.rerun()
#---MAIN CHAT AREA---
st.title(f" AISwiftVisa: {target_country.upper()} Consultant")

# --- CHAT SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = f"st_{int(time.time())}"

# Display chat history
for message in st.session_state.messages:
    avatar = "🤖" if message["role"] == "assistant" else "👤"
    with st.chat_message(message["role"], avatar=avatar):
        # If metrics exist in the message, display them
        if message["role"] == "assistant" and "metrics" in message:
            m = message["metrics"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Relevance", f"{m['relevance']}%")
            c2.metric("Confidence", f"{m['conf']}%")
            c3.metric("Source", m['source'])
            st.divider()
        st.markdown(message["content"])

# --- CHAT INPUT ---
if prompt := st.chat_input("Ask about your visa..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
       st.markdown(prompt)

    # Call main logic
    with st.spinner("Analyzing PDF and Profile..."):
        # 1. Get Answer
        answer = chat_with_visa_bot(st.session_state.session_id, user_profile, prompt)
        
    # Add AI response to history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer
       
    })
    
    # Display AI response
    with st.chat_message("assistant", avatar="🤖"):
        
        st.markdown(answer)