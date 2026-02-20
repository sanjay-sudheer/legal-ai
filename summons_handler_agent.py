"""
Summons Handler Agent (Intelligent Vector DB Version)
------------------------------------------------------
Reads legal documents from FAISS vector DB and performs:

• Summons / Appeal classification
• Party extraction
• Deadline detection + expiry logic
• Procedural intelligence
• Limitation risk detection
• Litigation severity scoring
• Priority action generation
• Structured JSON output

Token-efficient KV parsing.
"""

import os
import re
import json
import argparse
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import List
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import faiss
from openai import OpenAI

from document_processing_module import (
    LegalBERTEmbedder,
    FAISS_INDEX_PATH,
    METADATA_PATH,
)

logging.basicConfig(level=logging.WARNING)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "your-key-here")  # Set this in your .env file or environment variables
BASE_URL      = "https://api.groq.com/openai/v1"
MODEL         = "llama-3.3-70b-versatile"
TOP_K         = 5

SYSTEM_PROMPT = "You are a precise legal summons analysis engine."

ANALYSIS_PROMPT = """\
Analyze these legal clauses:

CLAUSES:
{clauses}

Respond strictly in this format:

DOCUMENT_TYPE: summons | appeal | legal_notice | demand_letter | unknown
PLAINTIFF: name or unknown
DEFENDANT: name or unknown
COURT: name or unknown
CASE_NUMBER: number or unknown
JURISDICTION: location or unknown
CASE_TYPE: civil | criminal | arbitration | unknown
DEADLINES: comma separated dates or none
RESPONSE_REQUIRED: yes | no
ALLEGATIONS: one short sentence
DEMANDS: one short sentence
LEGAL_GROUNDS: short phrase
REQUIRED_ACTIONS: comma separated actions
"""

DATE_PATTERNS = [
    r"\b\d{1,2}/\d{1,2}/\d{4}\b",
    r"\b\d{1,2}-\d{1,2}-\d{4}\b",
    r"\b\d{1,2} [A-Za-z]+ \d{4}\b",
]

# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class DeadlineInfo:
    date: str
    days_remaining: int
    urgency: str

@dataclass
class SummonsAnalysis:
    document_type: str
    plaintiff: str
    defendant: str
    court: str
    case_number: str
    jurisdiction: str
    case_type: str
    deadlines: List[DeadlineInfo] = field(default_factory=list)
    response_required: bool = False
    allegations: str = ""
    demands: str = ""
    legal_grounds: str = ""
    required_actions: List[str] = field(default_factory=list)
    priority_actions: List[str] = field(default_factory=list)
    litigation_risk_score: int = 0
    overall_status: str = "normal"

# ─────────────────────────────────────────────
# VECTOR DB
# ─────────────────────────────────────────────

class VectorDB:
    def __init__(self, index_path=FAISS_INDEX_PATH, meta_path=METADATA_PATH):
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FileNotFoundError("Vector DB not found.")
        self.index = faiss.read_index(index_path)
        self.metadata = json.load(open(meta_path))

    def search(self, embedding: np.ndarray, top_k=TOP_K):
        distances, indices = self.index.search(embedding, top_k)
        results = []
        for idx in indices[0]:
            if idx != -1:
                results.append(self.metadata[idx])
        return results

# ─────────────────────────────────────────────
# LLM CLIENT
# ─────────────────────────────────────────────

class LLMClient:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key, base_url=BASE_URL)
        self.model = MODEL

    def call(self, prompt, max_tokens=300):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────

def parse_kv(text):
    result = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        result[key.strip().upper()] = val.strip()
    return result

def extract_dates(text):
    dates = []
    for pattern in DATE_PATTERNS:
        dates.extend(re.findall(pattern, text))
    return list(set(dates))

def compute_deadline(date_str):
    try:
        date = datetime.strptime(date_str, "%d/%m/%Y")
    except:
        try:
            date = datetime.strptime(date_str, "%d-%m-%Y")
        except:
            return None

    delta = (date - datetime.now()).days

    if delta < 0:
        urgency = "expired"
    elif delta <= 2:
        urgency = "critical"
    elif delta <= 7:
        urgency = "high"
    elif delta <= 21:
        urgency = "medium"
    else:
        urgency = "low"

    return DeadlineInfo(date_str, delta, urgency)

# ─────────────────────────────────────────────
# MAIN AGENT
# ─────────────────────────────────────────────

class SummonsHandlerAgent:

    def __init__(self, api_key=GROQ_API_KEY):
        self.db = VectorDB()
        self.embedder = LegalBERTEmbedder()
        self.llm = LLMClient(api_key)

    def analyze(self, query: str = "legal summons or appeal") -> SummonsAnalysis:

        embedding = self.embedder.embed([query])
        clauses = self.db.search(embedding, top_k=TOP_K)

        if not clauses:
            raise ValueError("No relevant clauses found.")

        combined_text = "\n\n---\n\n".join(
            c["clause_text"][:500] for c in clauses
        )

        response = self.llm.call(
            ANALYSIS_PROMPT.format(clauses=combined_text)
        )

        kv = parse_kv(response)

        # Extract deadlines from all text
        all_text = " ".join(c["clause_text"] for c in clauses)
        found_dates = extract_dates(all_text)

        deadlines = []
        for d in found_dates:
            info = compute_deadline(d)
            if info:
                deadlines.append(info)

        deadlines.sort(key=lambda x: x.days_remaining)

        # Intelligent priority actions
        priority_actions = []
        risk_score = 0

        for d in deadlines:
            if d.days_remaining < 0:
                priority_actions.append(
                    f"EXPIRED: Deadline was {d.date} ({abs(d.days_remaining)} days overdue)"
                )
                risk_score += 40
            elif d.urgency == "critical":
                priority_actions.append(
                    f"CRITICAL: Immediate action required before {d.date}"
                )
                risk_score += 30
            elif d.urgency == "high":
                priority_actions.append(
                    f"HIGH: Prepare response before {d.date}"
                )
                risk_score += 20
            elif d.urgency == "medium":
                priority_actions.append(
                    f"MEDIUM: Plan response before {d.date}"
                )
                risk_score += 10

        document_type = kv.get("DOCUMENT_TYPE", "unknown").lower()
        case_number = kv.get("CASE_NUMBER", "").lower()

        # Appeal intelligence
        if "appeal" in case_number:
            document_type = "appeal"
            risk_score += 15

            if any(d.urgency == "expired" for d in deadlines):
                priority_actions.append(
                    "Check if condonation of delay application is required."
                )
                risk_score += 25

        # Severity classification
        if risk_score >= 70:
            overall_status = "severe"
        elif risk_score >= 40:
            overall_status = "high_risk"
        elif risk_score >= 20:
            overall_status = "moderate"
        else:
            overall_status = "low"

        return SummonsAnalysis(
            document_type=document_type,
            plaintiff=kv.get("PLAINTIFF", "unknown"),
            defendant=kv.get("DEFENDANT", "unknown"),
            court=kv.get("COURT", "unknown"),
            case_number=kv.get("CASE_NUMBER", "unknown"),
            jurisdiction=kv.get("JURISDICTION", "unknown"),
            case_type=kv.get("CASE_TYPE", "unknown"),
            deadlines=deadlines,
            response_required=kv.get("RESPONSE_REQUIRED", "no") == "yes",
            allegations=kv.get("ALLEGATIONS", ""),
            demands=kv.get("DEMANDS", ""),
            legal_grounds=kv.get("LEGAL_GROUNDS", ""),
            required_actions=[
                a.strip() for a in kv.get("REQUIRED_ACTIONS", "").split(",") if a.strip()
            ],
            priority_actions=priority_actions,
            litigation_risk_score=risk_score,
            overall_status=overall_status,
        )

    def to_json(self, result: SummonsAnalysis):
        return json.dumps(asdict(result), indent=2)

# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summons Handler Agent")
    parser.add_argument("--query", type=str, default="legal summons or appeal")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    agent = SummonsHandlerAgent()
    result = agent.analyze(query=args.query)

    print("\n=== SUMMONS ANALYSIS ===\n")
    print(agent.to_json(result))
