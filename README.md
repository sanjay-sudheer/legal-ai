# ⚖️ LexAI — Legal Intelligence Platform

A multi-agent AI system for analyzing legal documents. Upload a contract, summons, or any legal document and ask questions in plain English — LexAI automatically dispatches specialized agents to explain clauses, detect risks, parse court documents, and draft formal responses.

![Tech Stack](https://img.shields.io/badge/LLM-Llama_3.3_70b-orange?style=flat-square)
![Backend](https://img.shields.io/badge/Backend-Flask-black?style=flat-square)
![Frontend](https://img.shields.io/badge/Frontend-Next.js-blue?style=flat-square)
![Embeddings](https://img.shields.io/badge/Embeddings-LegalBERT-purple?style=flat-square)
![Vector DB](https://img.shields.io/badge/VectorDB-FAISS-green?style=flat-square)

---

## ✨ Features

- **Multi-agent orchestration** — a central orchestrator classifies intent and routes queries to the right specialist agents automatically
- **Risk detection** — scans every clause in the background, categorizes risks as Critical / High / Medium / Low with recommendations
- **Clause explanation** — translates dense legalese into plain English
- **Summons parsing** — extracts parties, court, deadlines, case numbers, and filing dates from court documents
- **Response drafting** — generates formal legal responses with configurable tone (formal / assertive / conciliatory)
- **Document analysis** — provides an AI brief with document type, parties, obligations, rights, and action items
- **PDF export** — download any drafted response as a formatted PDF

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Next.js Frontend                  │
│  Chat Interface · Risk Panel · Analysis Panel        │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP
┌──────────────────────▼──────────────────────────────┐
│              Flask Orchestrator (orchestrator.py)    │
│  Intent Classification → Agent Dispatch → Synthesis  │
└──┬──────────┬──────────┬──────────┬─────────────────┘
   │          │          │          │
   ▼          ▼          ▼          ▼
Legal      Risk       Summons   Response
Simplifier Detector   Handler   Generator
Agent      Agent      Agent     Agent
   │          │          │          │
   └──────────┴──────────┴──────────┘
                    │
         ┌──────────▼──────────┐
         │  Document Processing │
         │  LegalBERT + FAISS   │
         └─────────────────────┘
```

### Agents

| Agent | File | Triggered by |
|---|---|---|
| **Legal Simplifier** | `legal_simplifier_agent.py` | Clause explanation, plain English queries |
| **Risk Detector** | `risk_detector_agent.py` | Risk and vulnerability queries |
| **Summons Handler** | `summons_handler_agent.py` | Court documents, deadlines, case info |
| **Response Generator** | `response_generator_agent.py` | Drafting formal responses |

---

## 📁 Project Structure

```
lexai/
├── backend/
│   ├── orchestrator.py              # Flask server + agent orchestration
│   ├── document_processing_module.py # LegalBERT embeddings + FAISS indexing
│   ├── legal_simplifier_agent.py
│   ├── risk_detector_agent.py
│   ├── summons_handler_agent.py
│   ├── response_generator_agent.py
│   └── .env                         # GROQ_API_KEY goes here
│
└── frontend/
    ├── pages/
    │   ├── index.js                 # Main UI
    │   └── _app.js
    ├── styles/
    │   └── globals.css
    ├── package.json
    └── next.config.js
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Node.js 18+
- A [Groq API key](https://console.groq.com) (free tier works)

### 1. Clone the repo

```bash
git clone https://github.com/sanjay-sudheer/legal-ai.git
cd legal-ai
```

### 2. Backend setup

```bash
cd backend

# Install Python dependencies
pip install flask flask-cors python-dotenv openai faiss-cpu torch \
            transformers sentence-transformers PyMuPDF python-docx

# Create .env file
echo "GROQ_API_KEY=your_key_here" > .env

# Start the Flask server
python orchestrator.py
```

The backend will start on `http://localhost:5000`.

### 3. Frontend setup

```bash
cd frontend

# Install dependencies
npm install

# Start the dev server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## 🔌 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/status` | Health check, model info, DB stats |
| `POST` | `/upload` | Upload PDF / DOCX / TXT files |
| `POST` | `/ask` | Send a query (orchestrator picks agents) |
| `GET` | `/documents` | List indexed documents |
| `DELETE` | `/documents` | Clear the vector database |
| `GET` | `/scan-status` | Background risk scan progress |

### POST /ask

```json
{
  "query": "What are the main risks in this contract?",
  "tone": "formal"
}
```

**Response:**
```json
{
  "success": true,
  "intents": ["risk"],
  "agents_used": ["Risk Detector"],
  "synthesis": "The contract contains several high-risk clauses...",
  "results": { ... }
}
```

### POST /upload

Accepts `multipart/form-data` with one or more files attached as `files`. Returns clause count and triggers a background risk scan.

---

## ⚙️ Configuration

All configuration is via environment variables in `backend/.env`:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile    # optional, this is the default
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq API / Llama-3.3-70b-versatile |
| Embeddings | LegalBERT (`nlpaueb/legal-bert-base-uncased`) |
| Vector DB | FAISS (in-process, persisted to disk) |
| Backend | Python / Flask / Flask-CORS |
| Frontend | Next.js 14 / React |
| PDF parsing | PyMuPDF |
| DOCX parsing | python-docx |

---

## 📄 Supported File Types

- **PDF** — contracts, court documents, agreements
- **DOCX** — Word documents
- **TXT** — plain text legal documents

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## ⚠️ Disclaimer

LexAI is an AI-assisted tool for informational purposes only. It does not constitute legal advice. Always consult a qualified lawyer for legal matters.

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.