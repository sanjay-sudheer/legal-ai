"""
Risk Detector Agent
--------------------
Reads clauses from the shared FAISS vector DB and detects legal risks.
Works with any Groq model (llama, gpt, etc.) via OpenAI-compatible API.

Usage:
  python risk_detector_agent.py --scan
  python risk_detector_agent.py --scan --min-risk high
  python risk_detector_agent.py --query "what are the termination risks?"
  python risk_detector_agent.py --scan --model openai/gpt-oss-120b
  python risk_detector_agent.py  (interactive)
"""

import os
import re
import json
import time
import argparse
import logging
from dataclasses import dataclass, asdict, field
from typing import Literal

import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# Silence HuggingFace output
os.environ["TRANSFORMERS_VERBOSITY"]       = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"]       = "false"

from document_processing_module import LegalBERTEmbedder, FAISS_INDEX_PATH, METADATA_PATH

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Config ─────────────────────────────────────────────────────────────────

GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "your-key-here")  # Set this in your .env file or environment variables
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
TOP_K         = 3
RISK_ORDER    = {"low": 0, "medium": 1, "high": 2, "critical": 3}
RISK_ICONS    = {"critical": "!!!", "high": " !!", "medium": "  !", "low": "  i"}

# ── Prompts ─────────────────────────────────────────────────────────────────
#
# Strategy: ask for plain key=value lines instead of JSON.
# Models output this format far more reliably than raw JSON,
# and we parse it ourselves. Zero JSON formatting errors.
#
SYSTEM_PROMPT = "You are a legal risk analyst. Be concise and accurate."

ANALYZE_PROMPT = """\
Analyze this contract clause for legal risks:

CLAUSE: {clause}

Answer in exactly this format, one value per line, no extra text:
RISK_LEVEL: low|medium|high|critical
ONE_SIDED: yes|no
RISK_TYPES: comma separated list or none
VAGUE_TERMS: comma separated list or none
HIDDEN_OBLIGATIONS: comma separated list or none
SUMMARY: one sentence under 15 words
RECOMMENDATION: one sentence under 15 words
"""

QUERY_PROMPT = """\
These are relevant contract clauses:
{clauses}

Question: {query}

Answer in plain English under 120 words. State the risk level and what to do.
"""

SUMMARY_PROMPT = """\
Contract risk scan results:
Critical: {critical}, High: {high}, Medium: {medium}, Low: {low}

Top risks found:
{top}

Write a 3-sentence plain English summary: overall risk level, biggest concern, recommendation.
"""

# ── Data Classes ────────────────────────────────────────────────────────────

@dataclass
class RiskResult:
    clause_id: str
    clause_text: str
    legal_types: list
    source: str
    risk_level: str
    one_sided: bool
    risk_types: list
    vague_terms: list
    hidden_obligations: list
    summary: str
    recommendation: str

@dataclass
class ScanReport:
    total: int
    critical: int
    high: int
    medium: int
    low: int
    executive_summary: str
    risks: list = field(default_factory=list)

# ── FAISS Reader ────────────────────────────────────────────────────────────

class VectorDB:
    def __init__(self, index_path=FAISS_INDEX_PATH, meta_path=METADATA_PATH):
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(
                "Vector DB not found. Run document_processing_module.py first."
            )
        self.index    = faiss.read_index(index_path)
        self.metadata = json.load(open(meta_path))

    def all_clauses(self):
        return self.metadata

    def search(self, embedding: np.ndarray, top_k=TOP_K):
        distances, indices = self.index.search(embedding, top_k)
        out = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:
                rec = dict(self.metadata[idx])
                rec["distance"] = float(dist)
                out.append(rec)
        return out

# ── LLM Client ──────────────────────────────────────────────────────────────

class LLMClient:
    def __init__(self, api_key: str, model: str, base_url=GROQ_BASE_URL):
        if not api_key:
            raise ValueError("Set GROQ_API_KEY env variable or pass --apikey.")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model  = model

    def call(self, prompt: str, max_tokens=300) -> str:
        """Call the LLM with automatic retry on rate limit or empty response."""
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
                text = resp.choices[0].message.content or ""
                text = text.strip()
                if text:
                    return text
                # empty response — back off and retry
                wait = 8 * (attempt + 1)
                print(f"\n  [Empty response] Waiting {wait}s...", end="")
                time.sleep(wait)
            except Exception as e:
                msg = str(e)
                if "429" in msg or "rate" in msg.lower():
                    wait = 15 * (attempt + 1)
                    print(f"\n  [Rate limited] Waiting {wait}s...", end="")
                    time.sleep(wait)
                else:
                    logging.warning(f"LLM error: {e}")
                    return ""
        return ""

# ── Key=Value Parser ─────────────────────────────────────────────────────────

def parse_kv(text: str) -> dict:
    """
    Parse key=value lines like:
      RISK_LEVEL: high
      ONE_SIDED: yes
      VAGUE_TERMS: reasonable, appropriate
    Returns a dict with normalized values.
    """
    result = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip().upper()
        val = val.strip()
        result[key] = val
    return result

def kv_to_list(val: str) -> list:
    """Turn 'a, b, c' or 'none' into a list."""
    if not val or val.lower() in ("none", "n/a", "-", ""):
        return []
    return [v.strip() for v in val.split(",") if v.strip()]

# ── Risk Detector Agent ──────────────────────────────────────────────────────

class RiskDetectorAgent:
    def __init__(self, api_key=GROQ_API_KEY, model=DEFAULT_MODEL,
                 base_url=GROQ_BASE_URL,
                 index_path=FAISS_INDEX_PATH, meta_path=METADATA_PATH):
        self.db       = VectorDB(index_path, meta_path)
        self.embedder = LegalBERTEmbedder()
        self.llm      = LLMClient(api_key=api_key, model=model, base_url=base_url)

    # ── Analyze one clause ──────────────────────────────────────────────────

    def _analyze(self, clause: dict) -> RiskResult:
        # Trim clause to 350 chars to keep tokens low
        snippet = clause["clause_text"][:350].replace("\n", " ").strip()
        raw     = self.llm.call(ANALYZE_PROMPT.format(clause=snippet), max_tokens=180)
        kv      = parse_kv(raw)

        rl = kv.get("RISK_LEVEL", "medium").strip().lower()
        if rl not in RISK_ORDER:
            # Try to find a risk word anywhere in the response
            for level in ("critical", "high", "low", "medium"):
                if level in raw.lower():
                    rl = level
                    break
            else:
                rl = "medium"

        return RiskResult(
            clause_id=clause.get("id", ""),
            clause_text=clause["clause_text"],
            legal_types=clause.get("legal_types", []),
            source=clause.get("source", ""),
            risk_level=rl,
            one_sided=kv.get("ONE_SIDED", "no").lower() == "yes",
            risk_types=kv_to_list(kv.get("RISK_TYPES", "")),
            vague_terms=kv_to_list(kv.get("VAGUE_TERMS", "")),
            hidden_obligations=kv_to_list(kv.get("HIDDEN_OBLIGATIONS", "")),
            summary=kv.get("SUMMARY", "").strip('"'),
            recommendation=kv.get("RECOMMENDATION", "").strip('"'),
        )

    # ── Full scan ───────────────────────────────────────────────────────────

    def scan(self, min_risk: str = "low") -> ScanReport:
        clauses  = self.db.all_clauses()
        min_idx  = RISK_ORDER.get(min_risk, 0)
        counts   = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        results  = []

        for i, clause in enumerate(clauses):
            print(f"  Analyzing {i+1}/{len(clauses)}...", end="\r")
            result = self._analyze(clause)
            counts[result.risk_level] = counts.get(result.risk_level, 0) + 1
            if RISK_ORDER.get(result.risk_level, 0) >= min_idx:
                results.append(result)
            time.sleep(2)  # 30 req/min limit = 1 per 2s

        print()
        results.sort(key=lambda r: RISK_ORDER.get(r.risk_level, 0), reverse=True)

        top_text = "\n".join(
            f"- [{r.risk_level.upper()}] {r.summary}"
            for r in results[:4] if r.summary
        ) or "No major risks found."

        summary = self.llm.call(
            SUMMARY_PROMPT.format(
                critical=counts["critical"], high=counts["high"],
                medium=counts["medium"],    low=counts["low"],
                top=top_text,
            ),
            max_tokens=120,
        )

        return ScanReport(
            total=len(clauses),
            critical=counts["critical"],
            high=counts["high"],
            medium=counts["medium"],
            low=counts["low"],
            executive_summary=summary,
            risks=[asdict(r) for r in results],
        )

    # ── Targeted query ──────────────────────────────────────────────────────

    def query(self, user_query: str) -> str:
        embedding = self.embedder.embed([user_query])
        clauses   = self.db.search(embedding, top_k=TOP_K)
        if not clauses:
            return "No relevant clauses found in the document."

        clauses_text = "\n---\n".join(
            f"[{', '.join(c.get('legal_types', ['general']))}]\n{c['clause_text'][:220]}"
            for c in clauses
        )
        return self.llm.call(
            QUERY_PROMPT.format(clauses=clauses_text, query=user_query),
            max_tokens=200,
        )

    # ── Display ─────────────────────────────────────────────────────────────

    @staticmethod
    def show_risk(r):
        if isinstance(r, dict):
            r = RiskResult(**r)
        icon = RISK_ICONS.get(r.risk_level, "?")
        sep  = "-" * 68
        print(f"\n{sep}")
        print(f"  [{icon}] {r.risk_level.upper()}  |  {', '.join(r.risk_types) or 'general'}")
        print(f"  Category : {', '.join(r.legal_types) or 'general'}")
        preview = r.clause_text[:220] + ("..." if len(r.clause_text) > 220 else "")
        print(f"\nCLAUSE      : {preview}")
        print(f"RISK        : {r.summary}")
        print(f"ACTION      : {r.recommendation}")
        if r.vague_terms:
            print(f"VAGUE TERMS : {', '.join(r.vague_terms)}")
        if r.hidden_obligations:
            print(f"HIDDEN OBL. : {'; '.join(r.hidden_obligations)}")
        if r.one_sided:
            print("  >> ONE-SIDED: favors one party")

    @staticmethod
    def show_report(report: ScanReport):
        sep = "=" * 68
        print(f"\n{sep}")
        print("  RISK DETECTION REPORT")
        print(sep)
        print(f"  Total clauses : {report.total}")
        print(f"  CRITICAL: {report.critical}  HIGH: {report.high}  "
              f"MEDIUM: {report.medium}  LOW: {report.low}")
        print(f"\n{sep}\n  EXECUTIVE SUMMARY\n{sep}")
        print(f"\n{report.executive_summary or 'No summary generated.'}")
        if report.risks:
            print(f"\n{sep}\n  FLAGGED CLAUSES ({len(report.risks)})\n{sep}")
            for r in report.risks:
                RiskDetectorAgent.show_risk(r)
        print(f"\n{sep}\n")

# ── Interactive CLI ──────────────────────────────────────────────────────────

def run_interactive(agent: RiskDetectorAgent):
    sep = "=" * 68
    print(f"\n{sep}")
    print("  Risk Detector Agent")
    print("  Commands: scan | scan high | scan critical | <question> | quit")
    print(f"{sep}\n")

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user:
            continue
        if user.lower() == "quit":
            break

        if user.lower().startswith("scan"):
            parts    = user.lower().split()
            min_risk = parts[1] if len(parts) > 1 and parts[1] in RISK_ORDER else "low"
            report   = agent.scan(min_risk=min_risk)
            RiskDetectorAgent.show_report(report)
        else:
            answer = agent.query(user)
            print(f"\n{'-'*68}\n{answer}\n{'-'*68}\n")

# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Risk Detector Agent")
    ap.add_argument("--scan",     action="store_true")
    ap.add_argument("--query",    type=str)
    ap.add_argument("--min-risk", type=str, default="low",
                    choices=["low", "medium", "high", "critical"])
    ap.add_argument("--report",   type=str, help="Save JSON report to file")
    ap.add_argument("--model",    type=str, default=DEFAULT_MODEL,
                    help="Model name, e.g. llama-3.3-70b-versatile or openai/gpt-oss-120b")
    ap.add_argument("--apikey",   type=str, default="")
    ap.add_argument("--base-url", type=str, default=GROQ_BASE_URL)
    ap.add_argument("--index",    type=str, default=FAISS_INDEX_PATH)
    ap.add_argument("--meta",     type=str, default=METADATA_PATH)
    args = ap.parse_args()

    agent = RiskDetectorAgent(
        api_key=args.apikey or GROQ_API_KEY,
        model=args.model,
        base_url=args.base_url,
        index_path=args.index,
        meta_path=args.meta,
    )

    if args.scan:
        report = agent.scan(min_risk=args.min_risk)
        RiskDetectorAgent.show_report(report)
        if args.report:
            with open(args.report, "w") as f:
                json.dump(asdict(report), f, indent=2)
            print(f"[Saved to {args.report}]")
    elif args.query:
        print(f"\n{'-'*68}\n{agent.query(args.query)}\n{'-'*68}\n")
    else:
        run_interactive(agent)