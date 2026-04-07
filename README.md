# SwiftVisa AI: Intelligent Visa Consultation Agent
**SwiftVisa AI** is a sophisticated, agentic AI system designed to simulate a Consular Officer. Unlike standard chatbots, SwiftVisa use Policy Corpus,RAG and LLM to conduct dynamic interviews, ensuring all critical eligibility pillars (Finances, Home Ties, Education) are verified before providing a final assessment.

---

## Key Features
* **Agentic Workflow:** Built with **LangChain** to manage complex conversational loops and state transitions.
* **Policy-Grounded Reasoning (RAG):** Utilizes **ChromaDB** to anchor AI advice in actual immigration policies, reducing hallucinations.
* **Evaluation Dashboard:** Generates a final report highlighting **Recommendations**, **Financials**, and **Missing Documentation** based on the interview.
* **High-Speed Inference:** Powered by **Llama-3** via the **Groq** LPUs for near-instantaneous responses.

---

## Tech Stack
| Component | Technology |
| **Frontend** | Streamlit |
| **LLM Engine** | Llama-3 (via Groq API) |
| **Vector Database** | ChromaDB |
| **Framework** | LangChain |
| **Deployment** | Render |

---

## Project Structure
* app.py : The streamlit frontend and UI logic
* main.py : The RAG pipeline and logic implementation
* ingest.py : Contains the logic for document ingestion and ChromaDB management.
* vector_store/ : Contains the embeddings of the immigration policy documents

## Getting Started
### Prerequisites
* Python 3.9+
* A Groq API Key

### Installation

1. **Clone the repository:**
   git clone [https://github.com/your_username/AISwiftVisa-Project.git](https://github.com/your-username/AISwiftVisa-Project.git)
   

2. **Install the dependencies**
   pip install -r requirements.txt

3. **Set up envirionment variables**
   GROQ_API_KEY=your_key_here

4. **Run the application**
   streamlit run app.py