"""
Agent Orchestration + Executor Module
======================================
Flask server — single entry point for the entire Legal AI system.

Imports and calls the ACTUAL agent files:
  - legal_simplifier_agent.py   → LegalSimplifierAgent
  - risk_detector_agent.py      → RiskDetectorAgent
  - summons_handler_agent.py    → SummonsHandlerAgent
  - response_generator_agent.py → ResponseGeneratorAgent
  - document_processing_module.py → DocumentProcessingModule

All agents must be in the same folder as this file.

Setup:
  pip install flask flask-cors python-dotenv
  Set GROQ_API_KEY in .env file

Run:
  python orchestrator.py

API Endpoints:
  POST /upload       — Upload PDF / DOCX / TXT files
  POST /ask          — Ask anything (orchestrator picks agents)
  GET  /status       — Health check
  GET  /documents    — List ingested docs
  DELETE /documents  — Clear vector DB
"""

import os
import sys
import json
import time
import logging
import tempfile
import threading
from pathlib import Path
from dataclasses import asdict

# ── Load .env before anything else ───────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

# ── Silence HuggingFace noise ─────────────────────────────────────────────
os.environ["TRANSFORMERS_VERBOSITY"]        = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"]  = "1"
os.environ["TOKENIZERS_PARALLELISM"]        = "false"

from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

# ── Import actual agent modules ───────────────────────────────────────────
from document_processing_module import (
    DocumentProcessingModule,
    FAISS_INDEX_PATH,
    METADATA_PATH,
)
from legal_simplifier_agent   import LegalSimplifierAgent
from risk_detector_agent      import RiskDetectorAgent
from summons_handler_agent     import SummonsHandlerAgent
from response_generator_agent  import ResponseGeneratorAgent

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "your-key-here")  # Set this in your .env file or environment variables
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL    = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
UPLOAD_FOLDER = Path(tempfile.gettempdir()) / "legal_ai_uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_EXT   = {".pdf", ".docx", ".txt"}

if not GROQ_API_KEY:
    print("\n[ERROR] GROQ_API_KEY not set. Add it to your .env file.\n")
    sys.exit(1)

# ── Intent classifier prompt ─────────────────────────────────────────────
CLASSIFIER_PROMPT = """\
User message: "{query}"

Which legal tasks are needed? Pick one or more:
simplify  - user wants explanation, summary, or clarification of legal text
risk      - user wants risks, problems, or issues identified
summons   - user asks about court, parties, deadlines, case number, allegations
draft     - user wants to write or draft a legal response or letter

Reply with ONLY a comma-separated list of applicable tasks.
Example: simplify,risk
"""

SYNTHESIS_PROMPT = """\
A user asked: "{query}"

Multiple legal agents produced these results:
{results}

Write a single clear response (under 250 words) that:
- Directly answers the user's question
- Highlights the most important findings from each agent
- Ends with one clear recommendation

Plain English only. No agent names or technical labels.
"""

# ── LLM caller (just for classify + synthesize) ───────────────────────────
class _LLM:
    def __init__(self):
        self.client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)

    def call(self, prompt: str, max_tokens=200) -> str:
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0,
                )
                text = (resp.choices[0].message.content or "").strip()
                if text:
                    return text
                time.sleep(5 * (attempt + 1))
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    wait = 15 * (attempt + 1)
                    logger.warning(f"Rate limited — waiting {wait}s")
                    time.sleep(wait)
                else:
                    logger.error(f"LLM error: {e}")
                    return ""
        return ""


# ── DB ready check (lightweight) ──────────────────────────────────────────
def _db_ready() -> bool:
    return os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH)


def _clause_count() -> int:
    if not _db_ready():
        return 0
    try:
        data = json.load(open(METADATA_PATH))
        return len(data)
    except Exception:
        return 0


def _doc_sources() -> list:
    if not _db_ready():
        return []
    try:
        data = json.load(open(METADATA_PATH))
        seen, sources = set(), []
        for c in data:
            s = c.get("source", "unknown")
            if s not in seen:
                seen.add(s)
                sources.append(s)
        return sources
    except Exception:
        return []


# ── Agent factory — creates fresh instances (avoids stale FAISS handles) ──
def _make_simplifier():
    return LegalSimplifierAgent(
        index_path=FAISS_INDEX_PATH,
        meta_path=METADATA_PATH,
        groq_api_key=GROQ_API_KEY,
    )

def _make_risk():
    return RiskDetectorAgent(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        base_url=GROQ_BASE_URL,
        index_path=FAISS_INDEX_PATH,
        meta_path=METADATA_PATH,
    )

def _make_summons():
    return SummonsHandlerAgent(api_key=GROQ_API_KEY)

def _make_response_generator():
    return ResponseGeneratorAgent()


# ── Intent Classifier ─────────────────────────────────────────────────────
class IntentClassifier:
    VALID = {"simplify", "risk", "summons", "draft"}

    def __init__(self, llm: _LLM):
        self.llm = llm

    def classify(self, query: str) -> list:
        raw     = self.llm.call(CLASSIFIER_PROMPT.format(query=query), max_tokens=40)
        intents = [t.strip().lower() for t in raw.split(",") if t.strip().lower() in self.VALID]
        return intents or ["simplify"]   # safe default


# ── Agent Executor ────────────────────────────────────────────────────────
class AgentExecutor:
    """
    Calls the real agent classes.
    Each method wraps one agent, catches all exceptions,
    and always returns a safe dict — never raises.
    """

    # ── Legal Simplifier ─────────────────────────────────────────────────
    def run_simplify(self, query: str) -> dict:
        try:
            agent  = _make_simplifier()
            result = agent.query(query, level="detailed")   # returns QueryResult
            return {
                "agent":  "Legal Simplifier",
                "answer": result.direct_answer,
                "clauses_used": [
                    {
                        "text":        c.original_clause[:300],
                        "legal_types": c.legal_types,
                        "explanation": c.plain_english,
                    }
                    for c in (result.supporting_clauses or [])
                ],
            }
        except FileNotFoundError:
            return {"agent": "Legal Simplifier", "error": "Vector DB not ready."}
        except Exception as e:
            logger.error(f"Simplifier error: {e}")
            return {"agent": "Legal Simplifier", "error": str(e)}

    # ── Risk Detector ─────────────────────────────────────────────────────
    def run_risk(self, query: str) -> dict:
        try:
            agent  = _make_risk()
            # Use targeted query (not full scan — too slow for a web request)
            answer = agent.query(query)     # returns plain string
            return {
                "agent":  "Risk Detector",
                "answer": answer,
            }
        except FileNotFoundError:
            return {"agent": "Risk Detector", "error": "Vector DB not ready."}
        except Exception as e:
            logger.error(f"Risk detector error: {e}")
            return {"agent": "Risk Detector", "error": str(e)}

    # ── Summons Handler ───────────────────────────────────────────────────
    def run_summons(self, query: str) -> dict:
        try:
            agent  = _make_summons()
            result = agent.analyze(query=query)   # returns SummonsAnalysis dataclass
            data   = asdict(result)               # safe — all fields are plain types
            return {
                "agent":   "Summons Handler",
                "answer":  (
                    f"Document type: {result.document_type}. "
                    f"Plaintiff: {result.plaintiff}. "
                    f"Defendant: {result.defendant}. "
                    f"Court: {result.court}. "
                    f"Case: {result.case_number}. "
                    f"Status: {result.overall_status}. "
                    f"Response required: {'Yes' if result.response_required else 'No'}."
                ),
                "parsed":  data,
            }
        except FileNotFoundError:
            return {"agent": "Summons Handler", "error": "Vector DB not ready."}
        except ValueError as e:
            return {"agent": "Summons Handler", "error": str(e)}
        except Exception as e:
            logger.error(f"Summons handler error: {e}")
            return {"agent": "Summons Handler", "error": str(e)}

    # ── Response Generator ────────────────────────────────────────────────
    def run_draft(self, query: str, tone: str = "formal") -> dict:
        try:
            agent   = _make_response_generator()
            context = agent.db.get_full_document_context()
            draft   = agent.generate_draft(
                context=context,
                instruction=query,
                tone=tone,
            )
            return {
                "agent":  "Response Generator",
                "answer": draft,
                "tone":   tone,
            }
        except FileNotFoundError:
            return {"agent": "Response Generator", "error": "Vector DB not ready."}
        except Exception as e:
            logger.error(f"Response generator error: {e}")
            return {"agent": "Response Generator", "error": str(e)}


# ── Orchestrator ──────────────────────────────────────────────────────────
class Orchestrator:
    def __init__(self):
        self.llm        = _LLM()
        self.classifier = IntentClassifier(self.llm)
        self.executor   = AgentExecutor()
        self._lock      = threading.Lock()

    def process(self, query: str, tone: str = "formal") -> dict:
        if not _db_ready():
            return {
                "success":    False,
                "error":      "No documents ingested yet. Upload a document first.",
                "intents":    [],
                "agents_used": [],
                "results":    {},
                "synthesis":  "",
            }

        # 1. Classify intent
        intents = self.classifier.classify(query)

        # 2. Run each agent
        results     = {}
        agents_used = []

        for intent in intents:
            if intent == "simplify":
                out = self.executor.run_simplify(query)
            elif intent == "risk":
                out = self.executor.run_risk(query)
            elif intent == "summons":
                out = self.executor.run_summons(query)
            elif intent == "draft":
                out = self.executor.run_draft(query, tone=tone)
            else:
                continue

            results[intent] = out
            agents_used.append(out.get("agent", intent))
            time.sleep(2)   # rate limit buffer between agents

        if not results:
            return {
                "success":    False,
                "error":      "Could not determine what to do with your query.",
                "intents":    intents,
                "agents_used": [],
                "results":    {},
                "synthesis":  "",
            }

        # 3. Synthesize if multiple agents ran
        if len(results) > 1:
            results_text = "\n\n".join(
                f"[{v.get('agent', k).upper()}]\n{v.get('answer', v.get('error', ''))}"
                for k, v in results.items()
            )
            synthesis = self.llm.call(
                SYNTHESIS_PROMPT.format(query=query, results=results_text),
                max_tokens=400,
            )
        else:
            # Single agent — use its answer directly as synthesis
            single = list(results.values())[0]
            synthesis = single.get("answer", single.get("error", ""))

        return {
            "success":     True,
            "query":       query,
            "intents":     intents,
            "agents_used": agents_used,
            "results":     results,
            "synthesis":   synthesis,
        }


# ── Flask App ─────────────────────────────────────────────────────────────
app          = Flask(__name__)
CORS(app)
orchestrator = Orchestrator()


@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "status":       "online",
        "db_ready":     _db_ready(),
        "clause_count": _clause_count(),
        "documents":    _doc_sources(),
        "model":        GROQ_MODEL,
        "agents": [
            "Legal Simplifier",
            "Risk Detector",
            "Summons Handler",
            "Response Generator",
        ],
    })


@app.route("/documents", methods=["GET"])
def list_documents():
    return jsonify({
        "documents":    _doc_sources(),
        "clause_count": _clause_count(),
        "db_ready":     _db_ready(),
    })


@app.route("/documents", methods=["DELETE"])
def clear_documents():
    try:
        for path in [FAISS_INDEX_PATH, METADATA_PATH]:
            if os.path.exists(path):
                os.remove(path)
        return jsonify({"success": True, "message": "Vector DB cleared."})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/upload", methods=["POST"])
def upload():
    if "files" not in request.files:
        return jsonify({"success": False, "error": "No files provided. Use key 'files'."}), 400

    files     = request.files.getlist("files")
    results   = []
    processor = DocumentProcessingModule()

    for file in files:
        if not file.filename:
            continue

        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXT:
            results.append({
                "file":    file.filename,
                "success": False,
                "error":   f"Unsupported type '{ext}'. Use PDF, DOCX, or TXT.",
            })
            continue

        save_path = UPLOAD_FOLDER / file.filename
        try:
            file.save(str(save_path))
            clauses = processor.ingest(str(save_path))
            results.append({
                "file":           file.filename,
                "success":        True,
                "clauses_added":  len(clauses) if clauses else 0,
            })
        except Exception as e:
            results.append({"file": file.filename, "success": False, "error": str(e)})

    success_count = sum(1 for r in results if r["success"])
    return jsonify({
        "success":        success_count > 0,
        "files_uploaded": success_count,
        "total_files":    len(results),
        "clause_count":   _clause_count(),
        "details":        results,
    })


@app.route("/ask", methods=["POST"])
def ask():
    """
    POST /ask
    Body JSON:
      query : str  — user's question or instruction  (required)
      tone  : str  — formal | assertive | conciliatory  (optional, for drafting)

    Response JSON:
      success      : bool
      query        : str
      intents      : list[str]   — what the orchestrator detected
      agents_used  : list[str]   — which agents actually ran
      results      : dict        — per-agent detailed output
      synthesis    : str         — unified plain-English answer (show this to user)
    """
    data  = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    tone  = (data.get("tone")  or "formal").strip().lower()

    if not query:
        return jsonify({"success": False, "error": "Field 'query' is required."}), 400

    if tone not in ("formal", "assertive", "conciliatory"):
        tone = "formal"

    try:
        result = orchestrator.process(query, tone=tone)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Orchestrator error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ── Entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Legal AI — Flask Server")
    print("  http://localhost:5000")
    print("=" * 60)
    print(f"  Model    : {GROQ_MODEL}")
    print(f"  DB ready : {_db_ready()}")
    if _db_ready():
        print(f"  Clauses  : {_clause_count()}")
        print(f"  Docs     : {', '.join(_doc_sources())}")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)