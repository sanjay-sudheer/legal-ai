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
from risk_detector_agent      import RiskDetectorAgent, CACHE_FILE as RISK_CACHE_FILE, SUMMARY_PROMPT as RISK_SUMMARY_PROMPT
from summons_handler_agent     import SummonsHandlerAgent
from response_generator_agent  import ResponseGeneratorAgent

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────
GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL    = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
UPLOAD_FOLDER = Path(tempfile.gettempdir()) / "legal_ai_uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_EXT   = {".pdf", ".docx", ".txt"}

if not GROQ_API_KEY:
    print("\n[ERROR] GROQ_API_KEY not set. Add it to your .env file.\n")
    sys.exit(1)

# ── Background scan state (shared across threads) ─────────────────────────
_scan_state = {
    "running":   False,    # True while background scan is in progress
    "done":      False,    # True once scan has completed at least once
    "total":     0,        # total clauses to scan
    "current":   0,        # clauses scanned so far
    "failed":    0,        # clauses that errored
    "summary":   "",       # executive summary from last scan
    "error":     "",       # last scan-level error (if any)
}
_scan_lock = threading.Lock()

# ── Intent classifier prompt ─────────────────────────────────────
CLASSIFIER_PROMPT = """You are a legal AI coordinator helping a non-lawyer. They sent this message:

"{query}"

Decide which agents to activate to give the MOST HELPFUL, UNDERSTANDABLE answer.
Think about what they ACTUALLY NEED, not just what they literally asked.

Agents available:
  simplify  - Reads actual clauses and explains them in plain language.
              Use when: user wants to understand something, asks what X means,
              asks about consequences, implications, fairness, rights, or what happens if...
  risk      - Scans ALL clauses for legal risks, unfair terms, hidden obligations.
              Use when: user asks about risks, problems, dangers, what could go wrong,
              or whether the contract is safe/fair/good/bad.
  summons   - Extracts court info, parties, deadlines, case numbers.
              Use when: user mentions court, lawsuit, summons, hearing, deadline, filing.
  draft     - Writes a formal legal response letter.
              Use when: user wants to write, draft, compose, reply, or respond.

PAIRING RULES — follow these carefully:
- "what are the risks" or "is this safe" → risk,simplify  (risks must be EXPLAINED)
- "what does X clause mean" or "explain X" → simplify
- "what happens if they terminate/breach/fail" → risk,simplify
- "is this fair / should I sign" → risk,simplify
- "who are the parties / deadline / court" → summons
- "draft a reply / write a response" → draft,simplify
- "explain my rights / what can they do to me" → simplify

Reply with ONLY a comma-separated list. No explanation.
Examples: risk,simplify  |  summons  |  draft,simplify  |  simplify
"""

SYNTHESIS_PROMPT = """You are a senior lawyer giving advice to a client (non-lawyer) in plain English.

The client asked: "{query}"

Your specialist agents found this:
{results}

Write a clear, warm, HUMAN response following these rules STRICTLY:
1. Start with a direct answer using SPECIFIC names, dates, and facts from the document — never use generic terms like "the principal" if you know the actual name.
2. If the agent results contain a stated REASON or PURPOSE (e.g. "because the principal is residing outside India", "due to professional commitments abroad"), you MUST state that reason explicitly in your answer. NEVER say the document doesn't provide a reason if a reason appears anywhere in the results.
3. Explain what the 2-3 most important findings mean in practical terms for this person.
4. If there are risks, explain them in real-world terms.
5. Flag any urgent actions clearly.
6. End with one concrete next step.

ABSOLUTE RULES:
- Search the ENTIRE results text for any sentence containing "because", "reason", "residing", "professional", "committed", "unable" — if found, include it.
- Use real names from the document, not placeholders.
- Never use legal jargon without explaining it.
- Length: 200-320 words.
"""

RISK_EXPLAIN_PROMPT = """A legal risk scan found these issues in a contract:
{risk_summary}

The user asked: "{query}"

Explain these risks in plain English:
- What does each one MEAN for the person in real life?
- What could go WRONG if not addressed?
- Which are most urgent?

No jargon. Under 200 words.
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
    """Returns number of scannable clauses (excludes full_doc and table sentinel records)."""
    if not _db_ready():
        return 0
    try:
        data = json.load(open(METADATA_PATH))
        return sum(
            1 for e in data
            if e.get("clause_index", 0) >= 0
            and "table_data" not in e.get("legal_types", [])
        )
    except Exception:
        return 0

def _total_indexed() -> int:
    """Returns total records in FAISS including sentinels — used for UI display."""
    if not _db_ready():
        return 0
    try:
        return len(json.load(open(METADATA_PATH)))
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
    def run_simplify(self, query: str, focus_on_risks: bool = False,
                     doc_id: str = None, doc_name: str = None) -> dict:
        try:
            agent = _make_simplifier()

            effective_query = query
            if focus_on_risks and os.path.exists(RISK_CACHE_FILE):
                risk_agent = _make_risk()
                if len(risk_agent.cache._data) > 0:
                    high_risks = sorted(
                        [r for r in risk_agent.cache._data.values()
                         if r.get("risk_level") in ("high", "critical")],
                        key=lambda r: {"critical": 3, "high": 2}.get(r.get("risk_level", ""), 0),
                        reverse=True
                    )[:3]
                    if high_risks:
                        risk_topics = "; ".join(
                            r.get("summary", "") for r in high_risks if r.get("summary")
                        )
                        effective_query = (
                            f"{query}. Focus on explaining these specific high-risk areas "
                            f"in plain English a non-lawyer understands: {risk_topics}"
                        )

            result = agent.query(effective_query, level="detailed",
                                 doc_id=doc_id, doc_name=doc_name)
            return {
                "agent":  "Legal Simplifier",
                "answer": result.direct_answer,
                "clauses_used": [
                    {
                        "text":        c.original_clause[:800],
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
    def run_risk(self, query: str, doc_id: str = None, doc_name: str = None) -> dict:
        try:
            agent = _make_risk()

            if os.path.exists(RISK_CACHE_FILE) and len(agent.cache._data) > 0:
                cached_risks = list(agent.cache._data.values())

                # Filter to the active document if specified
                if doc_id or doc_name:
                    cached_risks = [
                        r for r in cached_risks
                        if (not doc_id   or r.get("doc_id")   == doc_id)
                        and (not doc_name or r.get("doc_name") == doc_name)
                    ]

                total = len(cached_risks)
                query_lower = query.lower()
                keywords = query_lower.replace("?", "").split()
                relevant = [
                    r for r in cached_risks
                    if r.get("risk_level") in ("high", "critical")
                    or any(kw in r.get("clause_text", "").lower() for kw in keywords if len(kw) > 4)
                ]
                if not relevant:
                    relevant = [r for r in cached_risks if r.get("risk_level") in ("high", "medium")]
                relevant.sort(key=lambda r: {"critical":3,"high":2,"medium":1,"low":0}.get(r.get("risk_level","low"),0), reverse=True)

                counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
                for r in cached_risks:
                    lvl = r.get("risk_level", "low")
                    counts[lvl] = counts.get(lvl, 0) + 1

                with _scan_lock:
                    scan_status = f" (scan in progress)" if _scan_state["running"] else ""

                risk_lines = []
                for r in relevant[:8]:
                    lvl   = r.get("risk_level", "").upper()
                    summ  = r.get("summary", "")
                    rec   = r.get("recommendation", "")
                    types = ", ".join(r.get("risk_types", [])) or "general"
                    if summ:
                        risk_lines.append(f"[{lvl}] {summ}  →  {rec}  (category: {types})")

                answer = (
                    f"Risk analysis{scan_status}: {counts['critical']} critical · "
                    f"{counts['high']} high · {counts['medium']} medium · {counts['low']} low\n\n"
                    + ("\n".join(risk_lines) if risk_lines else "No specific risks matched.")
                )
                if _scan_state.get("summary"):
                    answer += f"\n\nExecutive assessment: {_scan_state['summary']}"

                return {
                    "agent": "Risk Detector", "answer": answer,
                    "from_cache": True, "total_clauses": total,
                    "counts": counts, "top_risks": relevant[:8],
                }

            with _scan_lock:
                scan_running = _scan_state["running"]
            status_note = " (scan in progress)" if scan_running else ""
            answer = agent.query(query) + status_note
            return {"agent": "Risk Detector", "answer": answer, "from_cache": False}

        except FileNotFoundError:
            return {"agent": "Risk Detector", "error": "Vector DB not ready."}
        except Exception as e:
            logger.error(f"Risk detector error: {e}")
            return {"agent": "Risk Detector", "error": str(e)}

    # ── Summons Handler ───────────────────────────────────────────────────
    def run_summons(self, query: str, doc_id: str = None, doc_name: str = None) -> dict:
        try:
            agent  = _make_summons()
            result = agent.analyze(query=query)
            data   = asdict(result)
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
    def run_draft(self, query: str, tone: str = "formal",
                  doc_id: str = None, doc_name: str = None) -> dict:
        try:
            agent   = _make_response_generator()
            # Get context scoped to the active document
            context = agent.db.get_document_context(doc_id=doc_id, doc_name=doc_name)
            draft   = agent.generate_draft(context=context, instruction=query, tone=tone)
            return {"agent": "Response Generator", "answer": draft, "tone": tone}
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

    def process(self, query: str, tone: str = "formal",
                doc_id: str = None, doc_name: str = None) -> dict:
        if not _db_ready():
            return {
                "success":    False,
                "error":      "No documents ingested yet. Upload a document first.",
                "intents":    [],
                "agents_used": [],
                "results":    {},
                "synthesis":  "",
            }

        # ── Step 1: Classify intent ────────────────────────────────────
        intents = self.classifier.classify(query)
        logger.info(f"[ORCHESTRATOR] Query: {query!r} → Classifier intents: {intents}")

        # ── Step 2: Smart pairing rules ────────────────────────────────
        q_lower = query.lower()

        if "risk" in intents and "simplify" not in intents:
            intents.append("simplify")
        if "draft" in intents and "simplify" not in intents:
            intents.append("simplify")

        understanding_words = [
            "understand", "mean", "what is", "what does", "explain",
            "fair", "safe", "good", "bad", "should i sign", "my rights",
            "what happens", "consequences", "implications", "worried", "concern",
            "what can they", "allowed to", "penalty", "liable", "why", "who",
            "how much", "when", "deadline", "how long", "can i", "can they",
        ]
        if any(kw in q_lower for kw in understanding_words) and "simplify" not in intents:
            intents.append("simplify")

        ORDER = {"simplify": 0, "risk": 1, "summons": 2, "draft": 3}
        intents = sorted(set(intents), key=lambda x: ORDER.get(x, 99))
        logger.info(f"[ORCHESTRATOR] Final agent plan: {intents} | doc_id={doc_id} | doc_name={doc_name}")

        # ── Step 3: Run agents ──────────────────────────────────────────────
        results     = {}
        agents_used = []

        for intent in intents:
            if intent == "simplify":
                out = self.executor.run_simplify(
                    query,
                    focus_on_risks=("risk" in intents),
                    doc_id=doc_id, doc_name=doc_name
                )
            elif intent == "risk":
                out = self.executor.run_risk(query, doc_id=doc_id, doc_name=doc_name)
            elif intent == "summons":
                out = self.executor.run_summons(query, doc_id=doc_id, doc_name=doc_name)
            elif intent == "draft":
                out = self.executor.run_draft(query, tone=tone, doc_id=doc_id, doc_name=doc_name)
            else:
                continue

            results[intent] = out
            agents_used.append(out.get("agent", intent))
            if intent != intents[-1]:
                time.sleep(2)

        if not results:
            return {
                "success":    False,
                "error":      "Could not determine what to do with your query.",
                "intents":    intents,
                "agents_used": [],
                "results":    {},
                "synthesis":  "",
            }

        # ── Step 4: Synthesize ──────────────────────────────────────────────
        results_text = "\n\n".join(
            f"[{v.get('agent', k).upper()} FINDINGS]\n{v.get('answer', v.get('error', ''))}"
            for k, v in results.items()
        )
        synthesis = self.llm.call(
            SYNTHESIS_PROMPT.format(query=query, results=results_text),
            max_tokens=500,
        )
        if not synthesis:
            synthesis = max(
                results.values(),
                key=lambda v: len(v.get("answer", "")),
                default={}
            ).get("answer", "Sorry, could not generate a response.")

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
        "status":         "online",
        "db_ready":       _db_ready(),
        "clause_count":   _clause_count(),      # scannable clauses only
        "total_indexed":  _total_indexed(),     # all FAISS records including sentinels
        "documents":      _doc_sources(),
        "model":          GROQ_MODEL,
        "agents": [
            "Legal Simplifier",
            "Risk Detector",
            "Summons Handler",
            "Response Generator",
        ],
    })


@app.route("/documents", methods=["GET"])
def list_documents():
    """Returns all ingested documents with their doc_id and doc_name for frontend scoping."""
    try:
        if not _db_ready():
            return jsonify({"documents": [], "clause_count": 0, "db_ready": False})
        data  = json.load(open(METADATA_PATH))
        seen, docs = set(), []
        for entry in data:
            did = entry.get("doc_id", "")
            if did not in seen:
                seen.add(did)
                docs.append({
                    "doc_id":   did,
                    "doc_name": entry.get("doc_name", ""),
                    "source":   entry.get("source", ""),
                })
        return jsonify({
            "documents":    docs,
            "clause_count": _clause_count(),
            "db_ready":     True,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/documents", methods=["DELETE"])
def clear_documents():
    try:
        # Clear vector DB
        for path in [FAISS_INDEX_PATH, METADATA_PATH]:
            if os.path.exists(path):
                os.remove(path)

        # Clear risk cache (RISK_CACHE_FILE imported at top — guaranteed same path)
        if os.path.exists(RISK_CACHE_FILE):
            os.remove(RISK_CACHE_FILE)

        # Reset scan state
        with _scan_lock:
            _scan_state.update({"running": False, "done": False, "current": 0,
                                 "total": 0, "failed": 0, "summary": "", "error": ""})

        return jsonify({
            "success": True,
            "message": "Vector DB and risk cache cleared."
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


def _background_risk_scan():
    """
    Runs the full RiskDetectorAgent.scan() in a background thread after upload.
    Updates _scan_state so /scan-status can report live progress to the frontend.
    Results are saved to risk_cache.json by the agent itself.
    """
    global _scan_state
    with _scan_lock:
        if _scan_state["running"]:
            logger.info("Scan already running — skipping duplicate launch")
            return
        _scan_state.update({"running": True, "done": False, "current": 0,
                             "failed": 0, "summary": "", "error": ""})

    logger.info("[BG SCAN] Starting full risk scan in background thread...")
    try:
        agent   = _make_risk()
        clauses = agent.db.all_clauses()
        total   = len(clauses)

        with _scan_lock:
            _scan_state["total"] = total

        counts  = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        results = []

        for i, clause in enumerate(clauses):
            # Backfill fields that old index records may be missing
            clause.setdefault("doc_name", "")
            clause.setdefault("id", "")

            # Check cache BEFORE _analyze so we know whether to sleep
            was_cached = agent.cache.get(clause["clause_text"]) is not None
            try:
                result = agent._analyze(clause)
                counts[result.risk_level] = counts.get(result.risk_level, 0) + 1
                results.append(result)
            except Exception as e:
                logger.warning(f"[BG SCAN] Clause {i+1} failed: {e}")
                with _scan_lock:
                    _scan_state["failed"] += 1
            with _scan_lock:
                _scan_state["current"] = i + 1

            if not was_cached:
                time.sleep(2)

        # Generate executive summary
        results.sort(key=lambda r: {"critical":3,"high":2,"medium":1,"low":0}.get(r.risk_level,0), reverse=True)
        top_text = "\n".join(
            f"- [{r.risk_level.upper()}] {r.summary}"
            for r in results[:4] if r.summary
        ) or "No major risks found."

        # Collect all unique doc names scanned
        doc_names = list({r.doc_name for r in results if r.doc_name}) or ["the document"]
        display_name = ", ".join(doc_names)

        summary = agent.llm.call(
            RISK_SUMMARY_PROMPT.format(
                doc_name=display_name,
                critical=counts["critical"], high=counts["high"],
                medium=counts["medium"],    low=counts["low"],
                top=top_text,
            ),
            max_tokens=150,
        )

        with _scan_lock:
            _scan_state.update({
                "running": False,
                "done":    True,
                "summary": summary,
                "error":   "",
            })
        logger.info(f"[BG SCAN] Complete. {total} clauses scanned. Critical:{counts['critical']} High:{counts['high']}")

    except Exception as e:
        logger.error(f"[BG SCAN] Fatal error: {e}")
        with _scan_lock:
            _scan_state.update({"running": False, "done": False, "error": str(e)})


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
                "clauses_added":   sum(
                    1 for c in (clauses or [])
                    if c.get("clause_index", 0) >= 0
                    and "table_data" not in c.get("legal_types", [])
            ),
            })
        except Exception as e:
            results.append({"file": file.filename, "success": False, "error": str(e)})

    success_count = sum(1 for r in results if r["success"])

    # ── Kick off background risk scan if at least one file ingested ────────
    if success_count > 0:
        t = threading.Thread(target=_background_risk_scan, daemon=True)
        t.start()
        logger.info("[UPLOAD] Background risk scan thread started")

    return jsonify({
        "success":        success_count > 0,
        "files_uploaded": success_count,
        "total_files":    len(results),
        "clause_count":   _clause_count(),
        "scan_started":   success_count > 0,
        "details":        results,
    })


@app.route("/ask", methods=["POST"])
def ask():
    """
    POST /ask
    Body JSON:
      query    : str  — user's question or instruction  (required)
      tone     : str  — formal | assertive | conciliatory  (optional, for drafting)
      doc_id   : str  — filter responses to a specific document (optional)
      doc_name : str  — filter by document filename stem e.g. 'general_power_of_attorney' (optional)

    Response JSON:
      success      : bool
      query        : str
      intents      : list[str]   — what the orchestrator detected
      agents_used  : list[str]   — which agents actually ran
      results      : dict        — per-agent detailed output
      synthesis    : str         — unified plain-English answer (show this to user)
    """
    data     = request.get_json(silent=True) or {}
    query    = (data.get("query") or "").strip()
    tone     = (data.get("tone")  or "formal").strip().lower()
    doc_id   = (data.get("doc_id")   or "").strip() or None
    doc_name = (data.get("doc_name") or "").strip() or None

    if not query:
        return jsonify({"success": False, "error": "Field 'query' is required."}), 400

    if tone not in ("formal", "assertive", "conciliatory"):
        tone = "formal"

    try:
        result = orchestrator.process(query, tone=tone, doc_id=doc_id, doc_name=doc_name)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Orchestrator error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/scan-status", methods=["GET"])
def scan_status():
    """
    Poll this endpoint to get live risk scan progress after upload.
    Frontend can use this to show a progress bar.
    """
    with _scan_lock:
        state = dict(_scan_state)
    pct = 0
    if state["total"] > 0:
        pct = round((state["current"] / state["total"]) * 100)
    return jsonify({
        "running":   state["running"],
        "done":      state["done"],
        "total":     state["total"],
        "current":   state["current"],
        "failed":    state["failed"],
        "percent":   pct,
        "summary":   state["summary"],
        "error":     state["error"],
        "cache_file": RISK_CACHE_FILE,
    })


@app.route("/risks", methods=["GET"])
def get_risks():
    """
    GET /risks
    Returns all cached risks from the background scan, sorted by severity.
    Used by the frontend Risk panel to display all risks with colour coding.

    Response JSON:
      success      : bool
      risks        : list[dict]  — all risk entries, sorted critical → low
      counts       : dict        — { critical, high, medium, low }
      total        : int
      scan_done    : bool
      scan_running : bool
      summary      : str         — executive summary from scan
    """
    try:
        if not os.path.exists(RISK_CACHE_FILE):
            with _scan_lock:
                running = _scan_state["running"]
            return jsonify({
                "success": True,
                "risks": [],
                "counts": {"critical": 0, "high": 0, "medium": 0, "low": 0},
                "total": 0,
                "scan_done": False,
                "scan_running": running,
                "summary": "",
            })

        agent        = _make_risk()
        cached_risks = list(agent.cache._data.values())

        ORDER = {"critical": 3, "high": 2, "medium": 1, "low": 0}
        cached_risks.sort(key=lambda r: ORDER.get(r.get("risk_level", "low"), 0), reverse=True)

        counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for r in cached_risks:
            lvl = r.get("risk_level", "low")
            counts[lvl] = counts.get(lvl, 0) + 1

        # Clean up each risk entry for frontend consumption
        risks_out = []
        for r in cached_risks:
            risks_out.append({
                "risk_level":      r.get("risk_level", "low"),
                "summary":         r.get("summary", ""),
                "recommendation":  r.get("recommendation", ""),
                "risk_types":      r.get("risk_types", []),
                "clause_text":     r.get("clause_text", "")[:400],  # truncate for UI
            })

        with _scan_lock:
            state = dict(_scan_state)

        return jsonify({
            "success":      True,
            "risks":        risks_out,
            "counts":       counts,
            "total":        len(cached_risks),
            "scan_done":    state["done"],
            "scan_running": state["running"],
            "summary":      state.get("summary", ""),
        })

    except Exception as e:
        logger.error(f"GET /risks error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


ANALYSIS_PROMPT = """You are a senior legal analyst. Based on the document clauses and risk scan below, provide a structured overview of this legal document for a non-lawyer.

Document clauses sample:
{clauses}

Risk scan summary:
{risk_summary}

Return a JSON object with EXACTLY these fields (no markdown, no code fences, just raw JSON):
{{
  "document_type": "e.g. Employment Contract / NDA / Service Agreement / Summons etc.",
  "parties": ["Party A name", "Party B name"],
  "purpose": "1-2 sentence plain English summary of what this document is about",
  "key_obligations": ["obligation 1", "obligation 2", "obligation 3"],
  "key_rights": ["right 1", "right 2"],
  "duration": "contract term or N/A",
  "governing_law": "jurisdiction or N/A",
  "overall_risk_level": "low | medium | high | critical",
  "risk_summary": "2-3 sentence plain English risk overview",
  "action_items": ["urgent action 1", "action 2"],
  "plain_summary": "3-4 sentence plain English summary a non-lawyer would understand"
}}"""


@app.route("/analysis", methods=["GET"])
def get_analysis():
    """
    GET /analysis
    Returns a document overview: clause types, risk distribution, scan status.
    Used by the ANALYSIS panel in the frontend.
    """
    try:
        if not _db_ready():
            return jsonify({
                "success": False, "error": "No documents ingested yet.",
                "clause_count": 0, "documents": [], "clause_types": {},
                "risk_distribution": {"critical": 0, "high": 0, "medium": 0, "low": 0},
                "risk_summary": "", "scan_done": False, "scan_running": False,
            })

        metadata     = json.load(open(METADATA_PATH))
        clause_count = len(metadata)

        type_counts = {}
        for clause in metadata:
            for t in (clause.get("legal_types") or []):
                t = t.strip().lower()
                if t:
                    type_counts[t] = type_counts.get(t, 0) + 1
        type_counts = dict(sorted(type_counts.items(), key=lambda x: x[1], reverse=True))

        risk_dist = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        if os.path.exists(RISK_CACHE_FILE):
            agent  = _make_risk()
            cached = list(agent.cache._data.values())
            for r in cached:
                lvl = r.get("risk_level", "low")
                risk_dist[lvl] = risk_dist.get(lvl, 0) + 1

        with _scan_lock:
            risk_summary = _scan_state.get("summary", "")
            scan_done    = _scan_state.get("done", False)
            scan_running = _scan_state.get("running", False)

        return jsonify({
            "success": True, "clause_count": clause_count,
            "documents": _doc_sources(), "clause_types": type_counts,
            "risk_distribution": risk_dist, "risk_summary": risk_summary,
            "scan_done": scan_done, "scan_running": scan_running,
        })
    except Exception as e:
        logger.error(f"Analysis endpoint error: {e}")
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