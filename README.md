# 🤖 AI-Powered Chatbot with Audio RAG + LangGraph + FastAPI

This repository contains a modular **AI Chatbot system** that integrates **FastAPI**, **LangGraph**, **OpenAI Whisper**, **PyAnnote Diarization**, **ChromaDB**, and **React (Vite)** frontend.  
It supports both **textual chat** and **audio-based transcript analysis**, complete with **retrieval-augmented generation (RAG)**, **web search (Tavily)**, and **email automation (SendGrid)**.

---

## 📂 Project Structure

airagchatbot/
│
├── backend/
│ ├── main.py # FastAPI app entry (endpoints + CORS)
│ ├── orchestrator.py # LangGraph audio pipeline orchestration
│ ├── graph.py # Chat router graph for text chat
│ ├── agent_tools.py # RAG, web, and email tool definitions
│ ├── chroma_store.py # Vector DB setup (Chroma)
│ ├── db.py # MongoDB connection and collections
│ ├── utils.py # FFmpeg, Whisper, diarization, email helpers
│ ├── requirements.txt # Python dependencies
│ └── init.py
│
├── frontend/
│ ├── src/ # React app source (Vite)
│ ├── index.html
│ ├── package.json
│ ├── vite.config.js
│ └── ...
│
├── chroma/ # ChromaDB local vector store
├── .venv/ # Python virtual environment (ignored)
├── .env # API keys and secrets (ignored)
├── .gitignore
└── README.md

markdown
Copy code

---

## 🚀 Features

### 🧠 Chat Graph
- Built with **LangGraph** routing logic  
- Detects intents using regex:
  - **Audio / Transcript queries**
  - **Summary / Action item requests**
  - **Email sending / address completion**
  - **Web search (Tavily)**
  - **Fallback chat**

### 🎧 Audio Pipeline
- Upload any **meeting/audio file**
- Converts to `.wav` with **FFmpeg**
- Transcribes with **OpenAI Whisper**
- Performs **speaker diarization (PyAnnote)**
- Summarizes & indexes via **ChromaDB**
- Optionally sends a summary via **SendGrid**

### 🧩 Integrations
- 🧠 **OpenAI APIs** — LLM & Whisper
- 🔎 **Tavily Search API** — real-time contextual search
- 💌 **SendGrid** — automated email sending
- 🗃️ **ChromaDB + MongoDB** — hybrid vector + document persistence

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/airagchatbot.git
cd airagchatbot
2. Backend setup
bash
Copy code
cd backend
python -m venv .venv
source (.\.venv\Scripts\activate)
pip install -r requirements.txt
3. Environment variables
Create a .env file in /backend:

bash
Copy code
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key
SENDGRID_API_KEY=your_sendgrid_key
MONGO_URI=mongodb://localhost:27017
CHROMA_PATH=../chroma
4. Run the backend
bash
Copy code
uvicorn main:app --reload
🖥️ Frontend Setup
bash
Copy code
cd ../frontend
npm install
npm run dev
Open http://localhost:5173/ (default Vite port).

🔌 API Endpoints
Endpoint	Method	Description
/chat	POST	Handles single chat messages
/chat_stream	POST	Streams chat responses
/upload_audio	POST	Uploads and processes audio
/jobs/{job_id}	GET	Fetches job status and results
/health	GET	Health check

🧬 Function Flow

d:\Downloads\mermaid-diagram-f4bNveb-HcE8ZEGYk8S86-low.png

🧰 Key Modules
File	Responsibility
main.py	FastAPI entry + routing
graph.py	LangGraph chat node router
orchestrator.py	Audio-RAG pipeline builder
agent_tools.py	Tools for RAG, web, email
chroma_store.py	Vector DB management
db.py	MongoDB data persistence
utils.py	FFmpeg, Whisper, diarization, mail

🧪 Testing
Use cURL or Postman:

bash
Copy code
curl -X POST http://127.0.0.1:8000/upload_audio \
  -F "file=@meeting.mp3"
Monitor job progress:

bash
Copy code
curl http://127.0.0.1:8000/jobs/<job_id>
🛠️ Tech Stack
Category	Technologies
Backend	FastAPI, LangGraph, OpenAI, PyAnnote
Vector DB	ChromaDB
Database	MongoDB
Search	Tavily API
Email	SendGrid
Frontend	React, Vite, Axios

📄 License
This project is licensed under the MIT License.

🧑‍💻 Author
Raj Shekhar
Full-Stack Developer & AI Engineer

🌐 GitHub
💼 LinkedIn