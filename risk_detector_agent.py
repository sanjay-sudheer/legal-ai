"""
Risk Detector Agent
--------------------
Reads clauses from the shared FAISS vector DB and detects legal risks.
Works with any Groq model (llama, gpt, etc.) via OpenAI-compatible API.

Key fixes vs original:
  - Multi-document support: all_clauses(), search(), scan(), query() accept
    doc_id / doc_name filters — 6-doc index no longer cross-contaminates
  - Skips full_document (clause_index == -1) and table_data records during scan
  - RiskResult stores doc_name for per-document cache scoping
  - ANALYZE_PROMPT now accepts doc_type hint for domain-specific risk labels
  - max_tokens raised to 250 — SUMMARY/RECOMMENDATION were being cut off
  - DATE parsing extended to "15th January, 2025" style (summons)
  - Cache key uses 500 chars (was 350 — PSA/insurance clauses are longer)
  - scan() generates per-document summary
  - QUERY_PROMPT enforces specific fact citation (amounts, rates, dates)
"""

import os
import re
import json
import time
import hashlib
import argparse
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import List, Optional
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import faiss
from openai import OpenAI

os.environ["TRANSFORMERS_VERBOSITY"]       = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"]       = "false"

from document_processing_module import LegalBERTEmbedder, FAISS_INDEX_PATH, METADATA_PATH

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────

GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
TOP_K         = 5
RISK_ORDER    = {"low": 0, "medium": 1, "high": 2, "critical": 3}
RISK_ICONS    = {"critical": "!!!", "high": " !!", "medium": "  !", "low": "  i"}
CACHE_FILE    = "risk_cache.json"

# ── Prompts ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a senior Indian legal risk analyst specialising in contracts, "
    "insurance policies, court summons, lease agreements, affidavits, and "
    "power of attorney documents. Be precise, concise, and accurate."
)

ANALYZE_PROMPT = """\
Analyze this legal clause for risks.

DOCUMENT TYPE: {doc_type}
CLAUSE: {clause}

Domain-specific risk guidance by document type:
- "civil court summons": flag missed deadlines, ex-parte risk, unrealistic defences
- "life insurance policy": flag exclusions, lapse conditions, suicide clause, tax traps
- "leave and licence agreement": flag one-sided eviction, no-subletting, pet restrictions, deposit forfeiture
- "affidavit / sworn statement": flag false-statement liability, concealed encumbrances, undisclosed POA
- "general power of attorney": flag unfettered discretion, no price floor, irrevocability risk, gift power
- "professional services agreement": flag unlimited liability exposure, IP ownership gap, no termination protection, 18% interest compounding

Answer in EXACTLY this format — one value per line, no extra text:
RISK_LEVEL: low|medium|high|critical
ONE_SIDED: yes|no
RISK_TYPES: comma separated (e.g. financial, liability, procedural, disclosure, deadline, irrevocability) or none
VAGUE_TERMS: comma separated vague/undefined terms or none
HIDDEN_OBLIGATIONS: comma separated hidden duties or none
SUMMARY: one plain-English sentence under 25 words describing the specific risk
RECOMMENDATION: one plain-English sentence under 25 words on what to do
"""

QUERY_PROMPT = """\
Relevant clauses from the document:

{clauses}

User question: {query}

Answer in plain English under 180 words:
- State the specific risk level (low / medium / high / critical) for this topic
- Cite EXACT figures from the document: amounts (e.g. Rs. 18,00,000), rates (e.g. 18% per annum), \
dates (e.g. 15th January 2025), periods (e.g. 60 days notice, 30 days grace period), percentages
- Explain the real-world consequence if the risk materialises
- Say clearly what the person should do about it
- If the document is a summons, include the specific hearing date, deadline, and ex-parte risk
- If the document is an insurance policy, include the specific exclusion or condition that applies
"""

SUMMARY_PROMPT = """\
Risk scan complete for: {doc_name}
Critical: {critical} | High: {high} | Medium: {medium} | Low: {low}

Top risks:
{top}

Write exactly 3 sentences:
1. Overall risk level of this document and why
2. The single most dangerous clause and its real-world consequence
3. The most important action the reader must take immediately
"""

# ── Month map for named-date parsing ────────────────────────────────────────

MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

# ── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class DeadlineInfo:
    date: str
    days_remaining: int
    urgency: str

@dataclass
class RiskResult:
    clause_id: str
    clause_text: str
    legal_types: list
    source: str
    doc_name: str
    risk_level: str
    one_sided: bool
    risk_types: list
    vague_terms: list
    hidden_obligations: list
    summary: str
    recommendation: str

@dataclass
class ScanReport:
    doc_name: str
    total: int
    critical: int
    high: int
    medium: int
    low: int
    executive_summary: str
    risks: list = field(default_factory=list)

# ── FAISS Reader ──────────────────────────────────────────────────────────────

class VectorDB:
    def __init__(self, index_path=FAISS_INDEX_PATH, meta_path=METADATA_PATH):
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FileNotFoundError("Vector DB not found. Ingest documents first.")
        self.index    = faiss.read_index(index_path)
        self.metadata = json.load(open(meta_path))

    def all_clauses(self, doc_id: str = None, doc_name: str = None) -> list:
        """
        Return clause records for risk scanning.
        Excludes full_document records (clause_index == -1) and table_data records
        — these are retrieval helpers, not individual analysable clauses.
        """
        out = []
        for entry in self.metadata:
            ci = entry.get("clause_index", 0)
            lt = entry.get("legal_types", [])
            # Skip full-doc and table sentinel records
            if ci == -1 or ci < -1 or "table_data" in lt:
                continue
            if doc_id   and entry.get("doc_id")   != doc_id:
                continue
            if doc_name and entry.get("doc_name") != doc_name:
                continue
            out.append(entry)
        return out

    def list_documents(self) -> list:
        seen, docs = set(), []
        for entry in self.metadata:
            did = entry.get("doc_id", "")
            if did not in seen:
                seen.add(did)
                docs.append({
                    "doc_id":   did,
                    "doc_name": entry.get("doc_name", ""),
                    "source":   entry.get("source", ""),
                })
        return docs

    def search(self, embedding: np.ndarray, top_k: int = TOP_K,
               doc_id: str = None, doc_name: str = None) -> list:
        fetch_k = top_k * 8 if (doc_id or doc_name) else top_k
        fetch_k = min(fetch_k, max(1, self.index.ntotal))
        distances, indices = self.index.search(embedding, fetch_k)
        out = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            rec = dict(self.metadata[idx])
            rec["distance"] = float(dist)
            if doc_id   and rec.get("doc_id")   != doc_id:
                continue
            if doc_name and rec.get("doc_name") != doc_name:
                continue
            out.append(rec)
            if len(out) >= top_k:
                break
        return out

# ── LLM Client ────────────────────────────────────────────────────────────────

class LLMClient:
    def __init__(self, api_key: str, model: str, base_url=GROQ_BASE_URL):
        if not api_key:
            raise ValueError("Set GROQ_API_KEY env variable or pass --apikey.")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model  = model

    def call(self, prompt: str, max_tokens: int = 300) -> str:
        for attempt in range(4):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=0,
                )
                text = (resp.choices[0].message.content or "").strip()
                if text:
                    return text
                time.sleep(8 * (attempt + 1))
            except Exception as e:
                msg = str(e)
                if "429" in msg or "rate" in msg.lower():
                    wait = 15 * (attempt + 1)
                    logger.warning(f"Rate limited — waiting {wait}s")
                    time.sleep(wait)
                else:
                    logger.warning(f"LLM error: {e}")
                    return ""
        return ""

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_kv(text: str) -> dict:
    result = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        result[key.strip().upper()] = val.strip()
    return result

def kv_to_list(val: str) -> list:
    if not val or val.lower() in ("none", "n/a", "-", ""):
        return []
    return [v.strip() for v in val.split(",") if v.strip()]

def extract_dates(text: str) -> list:
    """Extract dates in multiple formats from document text."""
    dates = []
    # "15th January, 2025" / "15 January 2025"
    named = re.findall(
        r'\b(\d{1,2})(?:st|nd|rd|th)?\s+'
        r'(January|February|March|April|May|June|July|'
        r'August|September|October|November|December)'
        r'[,\s]+(\d{4})\b',
        text, re.IGNORECASE
    )
    for day, month, year in named:
        dates.append(f"{day.zfill(2)} {month.capitalize()} {year}")
    # Numeric
    for pattern in [r"\b\d{1,2}/\d{1,2}/\d{4}\b", r"\b\d{1,2}-\d{1,2}-\d{4}\b"]:
        dates.extend(re.findall(pattern, text))
    return list(dict.fromkeys(dates))

def compute_deadline(date_str: str) -> Optional[DeadlineInfo]:
    date = None
    m = re.match(
        r'(\d{1,2})\s+(January|February|March|April|May|June|July|'
        r'August|September|October|November|December)\s+(\d{4})',
        date_str, re.IGNORECASE
    )
    if m:
        try:
            month = MONTH_MAP.get(m.group(2).lower())
            date  = datetime(int(m.group(3)), month, int(m.group(1)))
        except (ValueError, TypeError):
            pass
    if not date:
        for fmt in ("%d/%m/%Y", "%d-%m-%Y"):
            try:
                date = datetime.strptime(date_str, fmt); break
            except ValueError:
                continue
    if not date:
        return None
    delta = (date - datetime.now()).days
    urgency = (
        "expired"  if delta < 0  else
        "critical" if delta <= 2 else
        "high"     if delta <= 7 else
        "medium"   if delta <= 21 else "low"
    )
    return DeadlineInfo(date_str, delta, urgency)

def infer_doc_type(legal_types: list, source: str) -> str:
    lt  = " ".join(legal_types).lower()
    src = source.lower()
    if "court_procedure" in lt or "summons" in src:     return "civil court summons"
    if "insurance_benefit" in lt or "insurance" in src: return "life insurance policy"
    if "rental_terms" in lt or "lease" in src:          return "leave and licence agreement"
    if "affidavit" in lt or "affidavit" in src:         return "affidavit / sworn statement"
    if "property_authority" in lt or "power_of_attorney" in src or "gpa" in src:
        return "general power of attorney"
    if "scope_of_services" in lt or "professional_services" in src or "psa" in src:
        return "professional services agreement"
    return "legal document"

# ── Risk Cache ────────────────────────────────────────────────────────────────

class RiskCache:
    def __init__(self, path: str = CACHE_FILE):
        self.path   = path
        self._data: dict = {}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                self._data = json.load(open(self.path, encoding="utf-8"))
            except Exception:
                self._data = {}

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)

    @staticmethod
    def _key(clause_text: str) -> str:
        return hashlib.sha256(clause_text[:500].strip().encode()).hexdigest()

    def get(self, clause_text: str) -> Optional[dict]:
        return self._data.get(self._key(clause_text))

    def set(self, clause_text: str, result: dict):
        self._data[self._key(clause_text)] = result
        self._save()

    def clear(self):
        self._data = {}
        if os.path.exists(self.path):
            os.remove(self.path)
        print("[Cache] Cleared.")

    def stats(self) -> str:
        return f"[Cache] {len(self._data)} entries → {self.path}"

# ── Risk Detector Agent ───────────────────────────────────────────────────────

class RiskDetectorAgent:

    def __init__(self, api_key=GROQ_API_KEY, model=DEFAULT_MODEL,
                 base_url=GROQ_BASE_URL,
                 index_path=FAISS_INDEX_PATH, meta_path=METADATA_PATH):
        self.db       = VectorDB(index_path, meta_path)
        self.embedder = LegalBERTEmbedder()
        self.llm      = LLMClient(api_key=api_key, model=model, base_url=base_url)
        self.cache    = RiskCache()

    def _analyze(self, clause: dict) -> RiskResult:
        cached = self.cache.get(clause["clause_text"])
        if cached:
            # Backfill any fields that were added after the cache entry was written.
            # Old cache entries (pre doc_name) will be missing these keys — fill
            # them with safe defaults so RiskResult(**cached) never KeyErrors.
            cached.setdefault("doc_name",           clause.get("doc_name", ""))
            cached.setdefault("source",              clause.get("source", ""))
            cached.setdefault("clause_id",           clause.get("id", ""))
            cached.setdefault("clause_text",         clause.get("clause_text", ""))
            cached.setdefault("legal_types",         clause.get("legal_types", []))
            cached.setdefault("risk_level",          "medium")
            cached.setdefault("one_sided",           False)
            cached.setdefault("risk_types",          [])
            cached.setdefault("vague_terms",         [])
            cached.setdefault("hidden_obligations",  [])
            cached.setdefault("summary",             "")
            cached.setdefault("recommendation",      "")
            # Only pass fields that RiskResult accepts — strip any unknown extras
            valid_fields = {f.name for f in RiskResult.__dataclass_fields__.values()} \
                           if hasattr(RiskResult, '__dataclass_fields__') \
                           else set(cached.keys())
            safe = {k: v for k, v in cached.items() if k in valid_fields}
            return RiskResult(**safe)

        snippet  = clause["clause_text"][:800].replace("\n", " ").strip()
        doc_type = infer_doc_type(clause.get("legal_types", []), clause.get("source", ""))

        raw = self.llm.call(
            ANALYZE_PROMPT.format(clause=snippet, doc_type=doc_type),
            max_tokens=250
        )
        kv = parse_kv(raw)

        rl = kv.get("RISK_LEVEL", "medium").strip().lower()
        if rl not in RISK_ORDER:
            for level in ("critical", "high", "low", "medium"):
                if level in raw.lower():
                    rl = level; break
            else:
                rl = "medium"

        result = RiskResult(
            clause_id          = clause.get("id", ""),
            clause_text        = clause["clause_text"],
            legal_types        = clause.get("legal_types", []),
            source             = clause.get("source", ""),
            doc_name           = clause.get("doc_name", ""),
            risk_level         = rl,
            one_sided          = kv.get("ONE_SIDED", "no").lower() == "yes",
            risk_types         = kv_to_list(kv.get("RISK_TYPES", "")),
            vague_terms        = kv_to_list(kv.get("VAGUE_TERMS", "")),
            hidden_obligations = kv_to_list(kv.get("HIDDEN_OBLIGATIONS", "")),
            summary            = kv.get("SUMMARY", "").strip('"'),
            recommendation     = kv.get("RECOMMENDATION", "").strip('"'),
        )
        self.cache.set(clause["clause_text"], asdict(result))
        return result

    def scan(self, min_risk: str = "low",
             doc_id: str = None, doc_name: str = None) -> ScanReport:
        clauses = self.db.all_clauses(doc_id=doc_id, doc_name=doc_name)
        min_idx = RISK_ORDER.get(min_risk, 0)
        counts  = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        results = []
        hits    = 0

        for i, clause in enumerate(clauses):
            was_cached = self.cache.get(clause["clause_text"]) is not None
            print(f"  Analyzing {i+1}/{len(clauses)} "
                  f"[{'cached' if was_cached else 'LLM'}]...", end="\r")
            result = self._analyze(clause)
            counts[result.risk_level] = counts.get(result.risk_level, 0) + 1
            if RISK_ORDER.get(result.risk_level, 0) >= min_idx:
                results.append(result)
            hits += was_cached
            if not was_cached:
                time.sleep(2)

        print()
        print(self.cache.stats() + f"  |  hits: {hits}/{len(clauses)}")
        results.sort(key=lambda r: RISK_ORDER.get(r.risk_level, 0), reverse=True)

        top_text = "\n".join(
            f"- [{r.risk_level.upper()}] {r.summary}"
            for r in results[:4] if r.summary
        ) or "No major risks found."

        display_name = doc_name or doc_id or "all documents"
        summary = self.llm.call(
            SUMMARY_PROMPT.format(
                doc_name=display_name,
                critical=counts["critical"], high=counts["high"],
                medium=counts["medium"],    low=counts["low"],
                top=top_text,
            ),
            max_tokens=150,
        )

        return ScanReport(
            doc_name          = display_name,
            total             = len(clauses),
            critical          = counts["critical"],
            high              = counts["high"],
            medium            = counts["medium"],
            low               = counts["low"],
            executive_summary = summary,
            risks             = [asdict(r) for r in results],
        )

    def query(self, user_query: str,
              doc_id: str = None, doc_name: str = None) -> str:
        embedding    = self.embedder.embed([user_query])
        clauses      = self.db.search(embedding, top_k=TOP_K,
                                      doc_id=doc_id, doc_name=doc_name)
        if not clauses:
            return "No relevant clauses found in the document."
        clauses_text = "\n---\n".join(
            f"[{', '.join(c.get('legal_types', ['general']))}]\n{c['clause_text'][:1000]}"
            for c in clauses
        )
        return self.llm.call(
            QUERY_PROMPT.format(clauses=clauses_text, query=user_query),
            max_tokens=250,
        )

    @staticmethod
    def show_risk(r):
        if isinstance(r, dict):
            r = RiskResult(**r)
        icon    = RISK_ICONS.get(r.risk_level, "?")
        sep     = "-" * 68
        preview = r.clause_text[:250] + ("..." if len(r.clause_text) > 250 else "")
        print(f"\n{sep}")
        print(f"  [{icon}] {r.risk_level.upper()}  |  "
              f"{', '.join(r.risk_types) or 'general'}  |  {r.doc_name}")
        print(f"  Category : {', '.join(r.legal_types) or 'general'}")
        print(f"\nCLAUSE      : {preview}")
        print(f"RISK        : {r.summary}")
        print(f"ACTION      : {r.recommendation}")
        if r.vague_terms:
            print(f"VAGUE TERMS : {', '.join(r.vague_terms)}")
        if r.hidden_obligations:
            print(f"HIDDEN OBL. : {'; '.join(r.hidden_obligations)}")
        if r.one_sided:
            print("  >> ONE-SIDED: favours one party")

    @staticmethod
    def show_report(report: ScanReport):
        sep = "=" * 68
        print(f"\n{sep}\n  RISK REPORT — {report.doc_name}\n{sep}")
        print(f"  Total: {report.total}  |  CRITICAL: {report.critical}  "
              f"HIGH: {report.high}  MEDIUM: {report.medium}  LOW: {report.low}")
        print(f"\n{sep}\n  EXECUTIVE SUMMARY\n{sep}")
        print(f"\n{report.executive_summary or 'No summary.'}")
        if report.risks:
            print(f"\n{sep}\n  FLAGGED CLAUSES ({len(report.risks)})\n{sep}")
            for r in report.risks:
                RiskDetectorAgent.show_risk(r)
        print(f"\n{sep}\n")

# ── Interactive CLI ───────────────────────────────────────────────────────────

def run_interactive(agent: RiskDetectorAgent):
    print(f"\n{'='*68}\n  Risk Detector Agent\n"
          f"  Commands: scan | scan high | list-docs | use <doc_name> | quit\n{'='*68}\n")
    for d in agent.db.list_documents():
        print(f"  Doc: {d['doc_name']}")
    print()
    active_doc = None

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye."); break

        if not user: continue
        if user.lower() == "quit": break
        if user.lower() == "list-docs":
            for d in agent.db.list_documents():
                print(f"  {d['doc_name']}  —  {d['source']}")
            continue
        if user.lower().startswith("use "):
            active_doc = user[4:].strip()
            print(f"[Scoped to: '{active_doc}']\n"); continue
        if user.lower().startswith("scan"):
            parts    = user.lower().split()
            min_risk = parts[1] if len(parts) > 1 and parts[1] in RISK_ORDER else "low"
            report   = agent.scan(min_risk=min_risk, doc_name=active_doc)
            RiskDetectorAgent.show_report(report)
        else:
            ans = agent.query(user, doc_name=active_doc)
            print(f"\n{'-'*68}\n{ans}\n{'-'*68}\n")

# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Risk Detector Agent")
    ap.add_argument("--scan",        action="store_true")
    ap.add_argument("--query",       type=str)
    ap.add_argument("--doc-name",    type=str, default="")
    ap.add_argument("--min-risk",    type=str, default="low",
                    choices=["low", "medium", "high", "critical"])
    ap.add_argument("--report",      type=str)
    ap.add_argument("--model",       type=str, default=DEFAULT_MODEL)
    ap.add_argument("--apikey",      type=str, default="")
    ap.add_argument("--base-url",    type=str, default=GROQ_BASE_URL)
    ap.add_argument("--index",       type=str, default=FAISS_INDEX_PATH)
    ap.add_argument("--meta",        type=str, default=METADATA_PATH)
    ap.add_argument("--clear-cache", action="store_true")
    ap.add_argument("--cache-stats", action="store_true")
    ap.add_argument("--list-docs",   action="store_true")
    args = ap.parse_args()

    agent = RiskDetectorAgent(
        api_key=args.apikey or GROQ_API_KEY, model=args.model,
        base_url=args.base_url, index_path=args.index, meta_path=args.meta,
    )

    if args.cache_stats:  print(agent.cache.stats()); raise SystemExit(0)
    if args.clear_cache:  agent.cache.clear()
    if args.list_docs:
        for d in agent.db.list_documents():
            print(f"  {d['doc_name']}  —  {d['source']}")
        raise SystemExit(0)

    doc_name = args.doc_name or None
    if args.scan:
        report = agent.scan(min_risk=args.min_risk, doc_name=doc_name)
        RiskDetectorAgent.show_report(report)
        if args.report:
            with open(args.report, "w") as f:
                json.dump(asdict(report), f, indent=2)
            print(f"[Saved → {args.report}]")
    elif args.query:
        print(f"\n{'-'*68}\n{agent.query(args.query, doc_name=doc_name)}\n{'-'*68}\n")
    else:
        run_interactive(agent)