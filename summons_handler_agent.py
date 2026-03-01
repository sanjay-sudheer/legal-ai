"""
Summons Handler Agent
---------------------
Reads legal documents from FAISS vector DB and performs:
  • Document classification (summons, legal notice, etc.)
  • Full party, court, and case extraction
  • Deadline detection — supports "15th January, 2025" style dates
  • RERA number extraction
  • Amount paid / disputed / interest rate extraction
  • All three reliefs extracted as a list (not collapsed into one sentence)
  • Litigation severity scoring and priority actions
  • Draft response generation (formal + assertive)
  • Multi-document support: all searches scoped to active document

Fixes vs original:
  - compute_deadline() now parses "15th January, 2025" (was only dd/mm/yyyy)
  - extract_dates() regex fixed for "th"/"st"/"rd" suffixes + comma
  - VectorDB.search() accepts doc_id / doc_name filter
  - SummonsAnalysis dataclass has: rera_number, amount_paid, amount_disputed,
    interest_rate, reliefs (list), written_statement_deadline, hearing_time
  - ANALYSIS_PROMPT asks for each field separately; max_tokens raised to 450
  - RELIEFS field in prompt asks for pipe-separated list, not "one sentence"
  - draft_response() method added for formal and assertive templates
  - No bare except: — specific ValueError catches
  - SYSTEM_PROMPT extended with Indian legal context
"""

import os
import re
import json
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

from document_processing_module import (
    LegalBERTEmbedder,
    FAISS_INDEX_PATH,
    METADATA_PATH,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
BASE_URL     = "https://api.groq.com/openai/v1"
MODEL        = "llama-3.3-70b-versatile"
TOP_K        = 6   # increased — need summons, party, relief, deadline, RERA clauses

SYSTEM_PROMPT = (
    "You are a precise Indian legal summons analysis engine. "
    "Extract exact names, dates, amounts, case numbers, and RERA numbers from the text. "
    "Never guess — if information is not in the text, write 'unknown'. "
    "For reliefs/demands, list ALL of them separately using | as separator."
)

# ── Prompts ───────────────────────────────────────────────────────────────────

ANALYSIS_PROMPT = """\
Analyze these legal document clauses and extract every piece of information requested.
Read ALL text carefully before answering — facts may be spread throughout.

CLAUSES:
{clauses}

Respond STRICTLY in this format (one value per line, no extra text):

DOCUMENT_TYPE: summons | appeal | legal_notice | demand_letter | unknown
PLAINTIFF: full name of person filing the case or unknown
DEFENDANT: full company/person name being sued or unknown
COURT: full court name and location (e.g. Civil Judge Senior Division, Pune) or unknown
CASE_NUMBER: full case number including year (e.g. O.S. No. 2847/2024) or unknown
CNR_NUMBER: CNR number if present (e.g. MHPU01-002847-2024) or unknown
JURISDICTION: city and state or unknown
CASE_TYPE: civil | criminal | arbitration | unknown
FIRST_HEARING_DATE: exact date — look for "first hearing" or "appear before" (e.g. 15th January, 2025) or unknown
HEARING_TIME: time of first hearing (e.g. 10:30 A.M.) or unknown
WRITTEN_STATEMENT_DEADLINE: exact last date for filing written statement (e.g. 14th February, 2025) or unknown
WRITTEN_STATEMENT_DAYS: number of days allowed from first hearing (e.g. 30) or unknown
RERA_NUMBER: RERA registration number — look carefully for "P" followed by digits (e.g. P52100047821) or unknown
TOTAL_CONSIDERATION: total contract/sale price agreed between parties or unknown
AMOUNT_PAID: money already paid by plaintiff — look for "paid" or "85%" clues (e.g. Rs. 1,21,12,500) or unknown
AMOUNT_PAID_PERCENTAGE: percentage of total price already paid (e.g. 85%) or unknown
AMOUNT_DISPUTED: money being claimed back/recovered (e.g. Rs. 45,00,000) or unknown
INTEREST_RATE: interest rate claimed on disputed amount (e.g. 18% per annum) or unknown
RELIEFS: pipe-separated list of ALL reliefs/demands, each as a complete phrase (e.g. specific performance directing defendant to execute sale deed | money decree Rs. 45,00,000 with 18% interest | damages Rs. 15,00,000 for mental agony | costs of suit)
RESPONSE_REQUIRED: yes | no
ALLEGATIONS: one sentence — what did defendant allegedly do wrong?
LEGAL_GROUNDS: statute or legal basis (e.g. specific performance under Transfer of Property Act, breach of contract)
EX_PARTE_CONSEQUENCE: what happens if defendant doesn't appear? (e.g. court may pass ex-parte decree)
REQUIRED_ACTIONS: comma separated list of immediate actions for defendant (e.g. appear in court, file vakalatnama, file written statement, gather RERA documents)
"""

DRAFT_FORMAL_PROMPT = """\
Draft a formal legal response to this civil court summons on behalf of the DEFENDANT.

CASE DETAILS:
{case_summary}

DOCUMENT CLAUSES FOR CONTEXT:
{clauses}

Write a formal legal response/appearance notice addressed to the Court Registrar that:
1. Begins with the case number, court name, and CNR number in the header
2. States defendant's full name and that they appear through a duly authorised advocate
3. Acknowledges the first hearing date and time
4. Reserves the right to file a detailed Written Statement within the prescribed 30-day period
5. States defendant denies the allegations and will produce all relevant documents at the hearing
6. Is properly signed with place and date

Use formal Indian civil court language. Include specific names, dates, and amounts from the case details above. Under 300 words.
"""

DRAFT_ASSERTIVE_PROMPT = """\
Draft an assertive legal reply to this civil court summons on behalf of the DEFENDANT, disputing the claim and requesting additional time.

CASE DETAILS:
{case_summary}

DOCUMENT CLAUSES FOR CONTEXT:
{clauses}

Write an assertive response that:
1. Opens by identifying the case number, CNR, court, and parties clearly
2. States that the claim is misconceived, false, and filed to harass the defendant
3. Points out that defendant has not received adequate time to gather complex documents related to RERA registration and possession timelines
4. Requests a 60-day extension for filing the Written Statement citing complexity and volume of construction-related documents
5. Reserves all rights including the right to challenge the jurisdiction and maintainability
6. States that defendant will produce complete records of the project at the appropriate stage
7. Is firm, confident, and professionally legal in tone

Use specific names, amounts (Rs. 45,00,000 disputed, Rs. 15,00,000 damages claimed), and dates from the case details. Under 350 words.
"""

# ── Month map for named-date parsing ─────────────────────────────────────────

MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class DeadlineInfo:
    date: str
    days_remaining: int
    urgency: str         # expired | critical | high | medium | low

@dataclass
class SummonsAnalysis:
    document_type: str
    plaintiff: str
    defendant: str
    court: str
    case_number: str
    cnr_number: str
    jurisdiction: str
    case_type: str
    # Hearing specifics
    first_hearing_date: str
    hearing_time: str
    written_statement_deadline: str
    written_statement_days: str
    # Financial specifics
    rera_number: str
    total_consideration: str
    amount_paid: str
    amount_paid_percentage: str
    amount_disputed: str
    interest_rate: str
    # Claims
    reliefs: List[str]          # each relief as separate item
    allegations: str
    legal_grounds: str
    ex_parte_consequence: str
    # Procedural
    required_actions: List[str]
    priority_actions: List[str]
    deadlines: List[DeadlineInfo]
    response_required: bool
    litigation_risk_score: int
    overall_status: str

# ── Vector DB ─────────────────────────────────────────────────────────────────

class VectorDB:
    def __init__(self, index_path=FAISS_INDEX_PATH, meta_path=METADATA_PATH):
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FileNotFoundError("Vector DB not found. Ingest documents first.")
        self.index    = faiss.read_index(index_path)
        self.metadata = json.load(open(meta_path))

    def search(self, embedding: np.ndarray, top_k: int = TOP_K,
               doc_id: str = None, doc_name: str = None) -> list:
        """Search with optional document scoping — critical for multi-doc index."""
        fetch_k = top_k * 8 if (doc_id or doc_name) else top_k
        fetch_k = min(fetch_k, max(1, self.index.ntotal))

        distances, indices = self.index.search(embedding, fetch_k)
        results = []
        for idx in indices[0]:
            if idx == -1:
                continue
            rec = dict(self.metadata[idx])
            if doc_id   and rec.get("doc_id")   != doc_id:   continue
            if doc_name and rec.get("doc_name") != doc_name: continue
            results.append(rec)
            if len(results) >= top_k:
                break
        return results

    def get_full_document(self, doc_id: str = None, doc_name: str = None) -> str:
        """
        Return the full text of a document (clause_index == -1 record).
        Falls back to concatenating all clauses if no full-doc record exists.
        This gives the summons agent the complete document for comprehensive extraction.
        """
        candidates = [
            e for e in self.metadata
            if (not doc_id   or e.get("doc_id")   == doc_id)
            and (not doc_name or e.get("doc_name") == doc_name)
        ]
        # Prefer dedicated full-document record
        full = next((e for e in candidates if e.get("clause_index") == -1), None)
        if full:
            return full["clause_text"]
        # Fallback
        return "\n\n---\n\n".join(
            e["clause_text"][:1000] for e in candidates
            if e.get("clause_index", 0) >= 0
        )

# ── LLM Client ────────────────────────────────────────────────────────────────

class LLMClient:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Set GROQ_API_KEY env variable or pass --apikey.")
        self.client = OpenAI(api_key=api_key, base_url=BASE_URL)
        self.model  = MODEL

    def call(self, prompt: str, max_tokens: int = 450) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0,
                max_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return ""

# ── Utilities ─────────────────────────────────────────────────────────────────

def parse_kv(text: str) -> dict:
    result = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        result[key.strip().upper()] = val.strip()
    return result

def extract_dates(text: str) -> list:
    """
    Extract dates in multiple formats.
    Handles: '15th January, 2025', '15 January 2025', '15/01/2025', '15-01-2025'
    Returns normalised strings like '15 January 2025'.
    """
    dates = []

    # Named month: '15th January, 2025' or '15 January 2025'
    named = re.findall(
        r'\b(\d{1,2})(?:st|nd|rd|th)?\s+'
        r'(January|February|March|April|May|June|July|'
        r'August|September|October|November|December)'
        r'[,\s]+(\d{4})\b',
        text, re.IGNORECASE
    )
    for day, month, year in named:
        dates.append(f"{int(day):02d} {month.capitalize()} {year}")

    # Numeric formats
    for pattern in [r"\b\d{1,2}/\d{1,2}/\d{4}\b", r"\b\d{1,2}-\d{1,2}-\d{4}\b"]:
        dates.extend(re.findall(pattern, text))

    return list(dict.fromkeys(dates))   # deduplicate, preserve order

def compute_deadline(date_str: str) -> Optional[DeadlineInfo]:
    """Parse a date string into DeadlineInfo with days_remaining and urgency."""
    date = None

    # "15 January 2025" format
    m = re.match(
        r'(\d{1,2})\s+(January|February|March|April|May|June|July|'
        r'August|September|October|November|December)\s+(\d{4})',
        date_str, re.IGNORECASE
    )
    if m:
        month = MONTH_MAP.get(m.group(2).lower())
        if month:
            try:
                date = datetime(int(m.group(3)), month, int(m.group(1)))
            except ValueError:
                pass

    # Numeric formats
    if not date:
        for fmt in ("%d/%m/%Y", "%d-%m-%Y"):
            try:
                date = datetime.strptime(date_str, fmt)
                break
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

def parse_reliefs(raw: str) -> List[str]:
    """
    Parse pipe-separated or numbered reliefs into a clean list.
    Handles: 'specific performance | money decree | damages' 
    and: '(1) specific performance (2) money decree'
    """
    if not raw or raw.lower() in ("unknown", "none", ""):
        return []
    # Try pipe separator first
    if "|" in raw:
        return [r.strip() for r in raw.split("|") if r.strip()]
    # Try numbered: (1) ... (2) ...
    numbered = re.split(r'\(\d+\)', raw)
    if len(numbered) > 1:
        return [r.strip() for r in numbered if r.strip()]
    # Fallback: comma
    return [r.strip() for r in raw.split(",") if r.strip()]

# ── Main Agent ────────────────────────────────────────────────────────────────

class SummonsHandlerAgent:

    def __init__(self, api_key: str = GROQ_API_KEY):
        self.db       = VectorDB()
        self.embedder = LegalBERTEmbedder()
        self.llm      = LLMClient(api_key)

    def analyze(self, query: str = "civil court summons plaintiff defendant hearing",
                doc_id: str = None, doc_name: str = None) -> SummonsAnalysis:
        """
        Analyze a summons document from the vector DB.
        Uses smart excerpting to ensure ALL critical facts survive even for long docs:
        - First 3000 chars (parties, case, dates, reliefs — always at the top)
        - Last 1000 chars (endorsement, additional reliefs, RERA, financial details)
        - Middle section specifically searched for RELIEFS and financial amounts
        """
        # Step 1: Get full document text
        full_text = self.db.get_full_document(doc_id=doc_id, doc_name=doc_name)

        if not full_text:
            embedding = self.embedder.embed([query])
            clauses   = self.db.search(embedding, top_k=TOP_K,
                                       doc_id=doc_id, doc_name=doc_name)
            if not clauses:
                raise ValueError("No relevant clauses found in the document.")
            full_text = "\n\n---\n\n".join(c["clause_text"][:1200] for c in clauses)

        # Step 2: Smart excerpting — always capture parties + reliefs + financial
        # For long docs, take first 3000 chars + last 1000 chars + any relief section
        if len(full_text) <= 6000:
            excerpt = full_text
        else:
            head = full_text[:3000]
            tail = full_text[-1000:]
            # Also extract any block containing RELIEFS or financial details
            import re as _re
            relief_m = _re.search(
                r'(RELIEFS CLAIMED.*?(?=\n[A-Z]{4,}|\Z))',
                full_text, _re.DOTALL | _re.IGNORECASE
            )
            mid = relief_m.group(1)[:1500] if relief_m else ""
            excerpt = head + "\n\n[...]\n\n" + mid + "\n\n[...]\n\n" + tail

        # Step 3: LLM extraction — structured field-by-field
        response = self.llm.call(
            ANALYSIS_PROMPT.format(clauses=excerpt),
            max_tokens=600  # increased from 450 — more fields, longer reliefs list
        )
        kv = parse_kv(response)

        # Step 4: Extract dates from full text (not just LLM output)
        found_dates = extract_dates(full_text)
        deadlines = []
        for d in found_dates:
            info = compute_deadline(d)
            if info:
                deadlines.append(info)
        deadlines.sort(key=lambda x: x.days_remaining)

        # Step 5: Build priority actions and risk score
        priority_actions = []
        risk_score       = 0

        for d in deadlines:
            if d.days_remaining < 0:
                priority_actions.append(
                    f"EXPIRED: Deadline {d.date} was {abs(d.days_remaining)} days ago"
                )
                risk_score += 40
            elif d.urgency == "critical":
                priority_actions.append(f"CRITICAL: Act immediately — deadline {d.date}")
                risk_score += 30
            elif d.urgency == "high":
                priority_actions.append(f"HIGH: Prepare before {d.date}")
                risk_score += 20
            elif d.urgency == "medium":
                priority_actions.append(f"MEDIUM: Plan response before {d.date}")
                risk_score += 10

        doc_type = kv.get("DOCUMENT_TYPE", "unknown").lower()
        if "appeal" in kv.get("CASE_NUMBER", "").lower():
            doc_type = "appeal"
            risk_score += 15
            if any(d.urgency == "expired" for d in deadlines):
                priority_actions.append(
                    "Check if condonation of delay application is required (Order 22 CPC)"
                )
                risk_score += 25

        if kv.get("RESPONSE_REQUIRED", "no").lower() == "yes":
            ex_parte = kv.get("EX_PARTE_CONSEQUENCE",
                               "Court may pass ex-parte decree against defendant")
            priority_actions.insert(0, f"APPEAR IN COURT: {ex_parte}")

        overall_status = (
            "severe"    if risk_score >= 70 else
            "high_risk" if risk_score >= 40 else
            "moderate"  if risk_score >= 20 else "low"
        )

        return SummonsAnalysis(
            document_type              = doc_type,
            plaintiff                  = kv.get("PLAINTIFF", "unknown"),
            defendant                  = kv.get("DEFENDANT", "unknown"),
            court                      = kv.get("COURT", "unknown"),
            case_number                = kv.get("CASE_NUMBER", "unknown"),
            cnr_number                 = kv.get("CNR_NUMBER", "unknown"),
            jurisdiction               = kv.get("JURISDICTION", "unknown"),
            case_type                  = kv.get("CASE_TYPE", "unknown"),
            first_hearing_date         = kv.get("FIRST_HEARING_DATE", "unknown"),
            hearing_time               = kv.get("HEARING_TIME", "unknown"),
            written_statement_deadline = kv.get("WRITTEN_STATEMENT_DEADLINE", "unknown"),
            written_statement_days     = kv.get("WRITTEN_STATEMENT_DAYS", "unknown"),
            rera_number                = kv.get("RERA_NUMBER", "unknown"),
            total_consideration        = kv.get("TOTAL_CONSIDERATION", "unknown"),
            amount_paid                = kv.get("AMOUNT_PAID", "unknown"),
            amount_paid_percentage     = kv.get("AMOUNT_PAID_PERCENTAGE", "unknown"),
            amount_disputed            = kv.get("AMOUNT_DISPUTED", "unknown"),
            interest_rate              = kv.get("INTEREST_RATE", "unknown"),
            reliefs                    = parse_reliefs(kv.get("RELIEFS", "")),
            allegations                = kv.get("ALLEGATIONS", ""),
            legal_grounds              = kv.get("LEGAL_GROUNDS", ""),
            ex_parte_consequence       = kv.get("EX_PARTE_CONSEQUENCE", ""),
            required_actions           = [
                a.strip() for a in kv.get("REQUIRED_ACTIONS", "").split(",") if a.strip()
            ],
            priority_actions           = priority_actions,
            deadlines                  = deadlines,
            response_required          = kv.get("RESPONSE_REQUIRED", "no").lower() == "yes",
            litigation_risk_score      = risk_score,
            overall_status             = overall_status,
        )

    def draft_response(self, analysis: SummonsAnalysis,
                       tone: str = "formal",
                       doc_id: str = None, doc_name: str = None) -> str:
        """
        Generate a draft response to the summons.
        tone: 'formal' (standard appearance + reservation of rights)
              'assertive' (dispute claim + request extension)
        """
        case_summary = (
            f"Case: {analysis.case_number} | CNR: {analysis.cnr_number}\n"
            f"Court: {analysis.court}\n"
            f"Plaintiff: {analysis.plaintiff}\n"
            f"Defendant: {analysis.defendant}\n"
            f"First Hearing: {analysis.first_hearing_date} at {analysis.hearing_time}\n"
            f"Written Statement Due: {analysis.written_statement_deadline} "
            f"({analysis.written_statement_days} days)\n"
            f"Allegations: {analysis.allegations}\n"
            f"Reliefs Claimed: {'; '.join(analysis.reliefs)}\n"
            f"Amount in Dispute: {analysis.amount_disputed}\n"
            f"Interest Rate: {analysis.interest_rate}"
        )
        full_text = self.db.get_full_document(doc_id=doc_id, doc_name=doc_name)
        context   = full_text[:2000] if full_text else ""

        prompt = (
            DRAFT_FORMAL_PROMPT if tone == "formal" else DRAFT_ASSERTIVE_PROMPT
        ).format(case_summary=case_summary, clauses=context)

        return self.llm.call(prompt, max_tokens=500)

    def to_json(self, result: SummonsAnalysis) -> str:
        return json.dumps(asdict(result), indent=2)

    def print_analysis(self, result: SummonsAnalysis):
        sep = "=" * 68
        print(f"\n{sep}\n  SUMMONS ANALYSIS\n{sep}")
        print(f"  Document Type  : {result.document_type}")
        print(f"  Case Number    : {result.case_number}  (CNR: {result.cnr_number})")
        print(f"  Court          : {result.court}")
        print(f"  Plaintiff      : {result.plaintiff}")
        print(f"  Defendant      : {result.defendant}")
        print(f"  Jurisdiction   : {result.jurisdiction}")
        print(f"\n  HEARING DETAILS")
        print(f"  First Hearing  : {result.first_hearing_date} at {result.hearing_time}")
        print(f"  Written Stmt   : due {result.written_statement_deadline} "
              f"({result.written_statement_days} days)")
        if result.rera_number != "unknown":
            print(f"\n  RERA Number    : {result.rera_number}")
        print(f"\n  FINANCIAL DETAILS")
        print(f"  Total Price    : {result.total_consideration}")
        print(f"  Amount Paid    : {result.amount_paid} ({result.amount_paid_percentage})")
        print(f"  Amount Disputed: {result.amount_disputed}")
        print(f"  Interest Rate  : {result.interest_rate}")
        print(f"\n  RELIEFS CLAIMED")
        for i, r in enumerate(result.reliefs, 1):
            print(f"    {i}. {r}")
        print(f"\n  ALLEGATIONS    : {result.allegations}")
        print(f"  LEGAL GROUNDS  : {result.legal_grounds}")
        print(f"  EX-PARTE RISK  : {result.ex_parte_consequence}")
        print(f"\n  RISK SCORE     : {result.litigation_risk_score} — {result.overall_status.upper()}")
        if result.priority_actions:
            print(f"\n  PRIORITY ACTIONS:")
            for a in result.priority_actions:
                print(f"    >> {a}")
        if result.deadlines:
            print(f"\n  DETECTED DEADLINES:")
            for d in result.deadlines:
                print(f"    {d.urgency.upper():8} | {d.date} | {d.days_remaining} days remaining")
        print(f"\n{sep}\n")

# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summons Handler Agent")
    parser.add_argument("--query",     type=str,
                        default="civil court summons plaintiff defendant hearing date")
    parser.add_argument("--doc-name",  type=str, default="",
                        help="Filter to specific document by filename stem")
    parser.add_argument("--json",      action="store_true", help="Output raw JSON")
    parser.add_argument("--draft",     type=str, choices=["formal", "assertive"],
                        help="Generate a draft response in given tone")
    parser.add_argument("--apikey",    type=str, default="")
    args = parser.parse_args()

    api_key  = args.apikey or GROQ_API_KEY
    agent    = SummonsHandlerAgent(api_key=api_key)
    doc_name = args.doc_name or None

    result = agent.analyze(query=args.query, doc_name=doc_name)

    if args.json:
        print(agent.to_json(result))
    else:
        agent.print_analysis(result)

    if args.draft:
        print(f"\n{'='*68}\n  DRAFT RESPONSE ({args.draft.upper()})\n{'='*68}")
        draft = agent.draft_response(result, tone=args.draft, doc_name=doc_name)
        print(draft)