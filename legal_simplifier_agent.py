"""
Legal Simplifier Agent
-----------------------
Connects to the FAISS knowledge base built by the Document Processing Module.
Uses Groq (via OpenAI-compatible API) as the LLM to explain legal clauses in plain English.

Supports three explanation levels:
  - brief    : 1-2 sentence summary for quick understanding
  - detailed : full breakdown with definitions, obligations, and implications
  - expert   : legal-accurate plain English with technical notes preserved

Usage:
  python legal_simplifier_agent.py
  python legal_simplifier_agent.py --query "can they cancel this contract?" --level detailed
"""

import os
import json
import argparse
import logging
from dataclasses import dataclass, asdict
from typing import Literal
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import faiss
from openai import OpenAI

from document_processing_module import LegalBERTEmbedder, FAISS_INDEX_PATH, METADATA_PATH

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your-key-here")  # Set this in your .env file or environment variables
GROQ_MODEL   = "llama-3.3-70b-versatile"             # fast + capable on Groq free tier
TOP_K_CLAUSES = 5  # Increased: more context = better answers

ExplanationLevel = Literal["brief", "detailed", "expert"]


# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a friendly legal assistant who helps everyday people understand "
    "contracts and legal documents. You explain things clearly, like you're "
    "talking to a smart friend who has never studied law. Avoid jargon unless "
    "you immediately define it in simple terms. Always be accurate — never guess."
)

BRIEF_PROMPT = (
    'Here is a legal clause:\n"""\n{clause}\n"""\n\n'
    'Give a 1-2 sentence plain English summary of what this clause means in practice. '
    'Start with "This means..." or "Basically...". No bullet points.'
)

DETAILED_PROMPT = (
    'Here is a legal clause:\n"""\n{clause}\n"""\n\n'
    "Explain this clause to someone with no legal background. Structure your answer as:\n\n"
    "**What it says (in plain English):**\n"
    "A clear, simple explanation of the clause.\n\n"
    "**Key terms defined:**\n"
    "Define any legal or technical words used.\n\n"
    "**What this means for you:**\n"
    "Practical implications — what each party must do, what rights they have, what could go wrong.\n\n"
    "**A real-world example:**\n"
    "Give a concrete everyday example illustrating this clause.\n\n"
    "Keep the tone conversational and friendly."
)

EXPERT_PROMPT = (
    'Here is a legal clause:\n"""\n{clause}\n"""\n\n'
    "Provide a legally accurate plain-English explanation. Structure your answer as:\n\n"
    "**Plain English Summary:**\n"
    "Clear explanation preserving all legal nuances.\n\n"
    "**Legal Significance:**\n"
    "What legal obligations, rights, or liabilities this creates and for which party.\n\n"
    "**Technical Terms:**\n"
    "Precise definitions of any legal terms used.\n\n"
    "**Practical Implications:**\n"
    "What each party should know and watch out for.\n\n"
    "**Edge Cases / Risks:**\n"
    "Potential issues or ambiguities in this clause."
)

QUERY_EXPANSION_PROMPT = (
    'A user asked this question about a legal document:\n"{query}"\n\n'
    "Rewrite this as a precise legal search phrase (5-15 words) that will retrieve the exact clause answering this question.\n\n"
    "Document type hints:\n"
    "- POA questions (why/who/authority/revoke) → 'reason principal attorney residing outside professional commitments'\n"
    "- Insurance questions (benefit/rider/claim/premium) → 'sum assured death benefit rider critical illness waiver premium grace period'\n"
    "- Summons/court questions (deadline/hearing/appear/relief) → 'first hearing date written statement ex-parte filing deadline'\n"
    "- Rental/lease questions (rent/deposit/sublet/pet/notice) → 'licence fee security deposit monthly rent subletting termination notice'\n"
    "- Affidavit questions (title/loan/encumbrance/false) → 'encumbrance free title deed home loan bank affidavit false statement'\n"
    "- Contract/PSA questions (payment/IP/arbitration/notice) → 'professional fee payment schedule interest arbitration deliverables'\n\n"
    "Return ONLY the search phrase, nothing else."
)

DIRECT_ANSWER_PROMPT = (
    'A user asked: "{query}"\n\n'
    "Read the following document content COMPLETELY before answering. "
    "Pay special attention to lines tagged [REASON] and [PARTIES] — these contain the most important facts.\n\n"
    "DOCUMENT CONTENT:\n---\n{clauses}\n---\n\n"
    "Now answer the user's question. STRICT RULES:\n"
    "1. If you see a [REASON] tagged sentence, you MUST include that specific reason in your answer verbatim.\n"
    "2. Use the actual names from [PARTIES] — never say 'Party A' or 'the principal' if you know the real name.\n"
    "3. If the document states WHY something was done, quote or paraphrase it directly — do NOT say 'the document doesn't state why'.\n"
    "4. If the document states WHAT authority is given, summarize the key powers.\n"
    "5. Answer in plain, friendly English as if explaining to a friend.\n\n"
    'End with a "Key takeaway:" line with the single most important point.'
)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class SimplifiedClause:
    original_clause: str
    legal_types: list
    source: str
    explanation_level: str
    plain_english: str
    similarity_score: float


@dataclass
class QueryResult:
    user_query: str
    direct_answer: str
    supporting_clauses: list


# ---------------------------------------------------------------------------
# FAISS Knowledge Base Reader
# ---------------------------------------------------------------------------

class KnowledgeBaseReader:
    """Read-only access to the FAISS index built by DocumentProcessingModule."""

    def __init__(self, index_path=FAISS_INDEX_PATH, meta_path=METADATA_PATH):
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"Knowledge base not found at '{index_path}' / '{meta_path}'.\n"
                "Run the Document Processing Module first to ingest documents."
            )
        self.index = faiss.read_index(index_path)
        with open(meta_path, "r") as f:
            self.metadata = json.load(f)
        logger.info(f"Loaded knowledge base: {self.index.ntotal} clauses.")

    def list_documents(self) -> list[dict]:
        """Return all unique documents in the knowledge base."""
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

    def search(self, embedding: np.ndarray, top_k: int = TOP_K_CLAUSES,
               doc_id: str = None, doc_name: str = None):
        # Fetch more when filtering so we still get top_k after filter
        fetch_k = top_k * 8 if (doc_id or doc_name) else top_k
        fetch_k = min(fetch_k, max(1, self.index.ntotal))

        distances, indices = self.index.search(embedding, fetch_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            entry = dict(self.metadata[idx])
            entry["distance"] = float(dist)
            if doc_id and entry.get("doc_id") != doc_id:
                continue
            if doc_name and entry.get("doc_name") != doc_name:
                continue
            results.append(entry)
            if len(results) >= top_k:
                break
        return results


# ---------------------------------------------------------------------------
# Groq LLM Client (OpenAI-compatible)
# ---------------------------------------------------------------------------

class GroqClient:
    def __init__(self, api_key=GROQ_API_KEY, model=GROQ_MODEL):
        if not api_key:
            raise ValueError(
                "Groq API key not set.\n"
                "Set it with:  export GROQ_API_KEY='your-key-here'\n"
                "Or pass --apikey YOUR_KEY on the command line."
            )
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        self.model = model
        logger.info(f"Groq client ready: {model}")

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
        )
        return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Legal Term Extractor (lightweight, no LLM needed)
# ---------------------------------------------------------------------------

LEGAL_TERMS = [
    "indemnify", "indemnification", "hold harmless", "exonerate", "liability",
    "breach", "default", "arbitration", "jurisdiction", "governing law",
    "subrogation", "waiver", "consideration", "force majeure", "termination",
    "cure period", "liquidated damages", "pro rata", "in arrears", "warrant",
    "covenant", "lien", "subcontract", "assignee", "indemnitor", "indemnitee",
    "tort", "negligence", "material breach", "intellectual property", "proprietary",
    "royalty-free", "nonexclusive", "irrevocable", "severability", "novation",
]

def extract_legal_terms(text: str):
    text_lower = text.lower()
    return [term for term in LEGAL_TERMS if term in text_lower]


# ---------------------------------------------------------------------------
# Legal Simplifier Agent
# ---------------------------------------------------------------------------

class LegalSimplifierAgent:
    """
    Main agent: retrieves relevant clauses from the FAISS KB and uses Groq
    to explain them in plain English at the requested detail level.
    """

    def __init__(
        self,
        index_path=FAISS_INDEX_PATH,
        meta_path=METADATA_PATH,
        groq_api_key=GROQ_API_KEY,
        top_k=TOP_K_CLAUSES,
    ):
        self.kb = KnowledgeBaseReader(index_path, meta_path)
        self.embedder = LegalBERTEmbedder()
        self.llm = GroqClient(api_key=groq_api_key)
        self.top_k = top_k

    def simplify_clause(self, clause: dict, level: ExplanationLevel = "detailed") -> SimplifiedClause:
        clause_text = clause["clause_text"]

        if level == "brief":
            prompt = BRIEF_PROMPT.format(clause=clause_text)
        elif level == "expert":
            prompt = EXPERT_PROMPT.format(clause=clause_text)
        else:
            prompt = DETAILED_PROMPT.format(clause=clause_text)

        explanation = self.llm.generate(prompt)

        return SimplifiedClause(
            original_clause=clause_text,
            legal_types=clause.get("legal_types", []),
            source=clause.get("source", "unknown"),
            explanation_level=level,
            plain_english=explanation,
            similarity_score=clause.get("distance", 0.0),
        )

    def query(self, user_query: str, level: ExplanationLevel = "detailed",
              top_k: int = None, doc_id: str = None, doc_name: str = None) -> QueryResult:
        top_k = top_k or self.top_k
        logger.info(f"Query: '{user_query}' | level={level} | doc_id={doc_id} | doc_name={doc_name}")

        # Step 1: Expand the user's casual question into legal search terms
        expanded = self.llm.generate(QUERY_EXPANSION_PROMPT.format(query=user_query))
        logger.info(f"Expanded query: '{expanded}'")

        # Step 2: Embed and retrieve from FAISS — scoped to the active document
        embedding = self.embedder.embed([expanded])
        raw_clauses = self.kb.search(embedding, top_k=top_k, doc_id=doc_id, doc_name=doc_name)

        if not raw_clauses:
            return QueryResult(
                user_query=user_query,
                direct_answer="I couldn't find any relevant clauses in the document for your question.",
                supporting_clauses=[],
            )

        # Step 3: Build clause context
        clause_parts = []
        for c in raw_clauses:
            legal_types = c.get('legal_types', ['general'])
            text = c['clause_text']
            is_full_doc = 'full_document' in legal_types or c.get('clause_index') == -1

            if is_full_doc:
                extracted = self._extract_key_sentences(text, user_query)
                clause_parts.append(f"[{', '.join(legal_types)}]\n{extracted}")
            else:
                clause_parts.append(f"[{', '.join(legal_types)}]\n{text[:1500]}")

        clauses_text = "\n\n---\n\n".join(clause_parts)

        direct_answer = self.llm.generate(
            DIRECT_ANSWER_PROMPT.format(query=user_query, clauses=clauses_text)
        )

        # Step 4: Individually simplify each supporting clause
        simplified = [self.simplify_clause(c, level=level) for c in raw_clauses]

        return QueryResult(
            user_query=user_query,
            direct_answer=direct_answer,
            supporting_clauses=simplified,
        )

    @staticmethod
    def _extract_key_sentences(full_text: str, query: str) -> str:
        """
        Universal key-sentence extractor for full-document records.
        Works across ALL document types (POA, insurance, summons, rental, affidavit, PSA).

        Extracts by priority:
        1. REASON/PURPOSE sentences (why the doc exists)
        2. PARTIES sentences (who is involved)
        3. Numeric/date facts (amounts, dates, percentages, deadlines)
        4. Query-keyword matching sentences
        5. Document start (structural context)
        """
        import re

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+|\n(?=[A-Z])', full_text)
        selected = []
        seen = set()

        def add(tag: str, sent: str):
            s = sent.strip()
            if s and s not in seen and len(s) > 20:
                selected.append(f"[{tag}] {s}" if tag else s)
                seen.add(s)

        # ── Priority 1: REASON / PURPOSE sentences ────────────────────────
        reason_patterns = [
            # POA
            r'granted because', r'residing outside', r'residing abroad',
            r'professional commitment', r'unable to be present', r'outside india',
            r'houston', r'power of attorney is granted',
            # Insurance
            r'policy is issued', r'sum assured', r'in consideration of',
            # PSA
            r'desirous of availing', r'engaged in the business',
            r'navi mumbai', r'coastal development',
            # Affidavit
            r'executing this affidavit for the purpose',
            r'submitting.*to.*bank', r'home loan application',
            # Summons
            r'suit has been filed', r'cause of action',
            r'failure to hand over possession',
            # Rental
            r'licensor is the absolute owner', r'willing to grant',
        ]
        for sent in sentences:
            sl = sent.lower()
            if any(re.search(p, sl) for p in reason_patterns) and sent.strip() not in seen:
                add("REASON", sent)

        # ── Priority 2: PARTIES / IDENTITY sentences ─────────────────────
        party_patterns = [
            r's/o|d/o|w/o', r'hereinafter referred to as',
            r'life assured', r'nominee', r'licensor|licensee',
            r'plaintiff|defendant', r'service provider|client',
            r'deponent', r'principal.*attorney',
        ]
        party_sents = []
        for sent in sentences:
            sl = sent.lower()
            if any(re.search(p, sl) for p in party_patterns) and sent.strip() not in seen:
                party_sents.append(sent.strip())
                seen.add(sent.strip())
        if party_sents:
            selected.append("[PARTIES]\n" + " ".join(party_sents[:5]))

        # ── Priority 3: KEY NUMERIC FACTS (amounts, dates, rates, periods) ─
        numeric_patterns = [
            # Dates
            r'\b\d{1,2}(st|nd|rd|th)?\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b',
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',
            # Money amounts
            r'rs\.?\s*[\d,]+', r'rupees.*lakh', r'₹\s*[\d,]+',
            # Percentages and rates
            r'\d+\.?\d*\s*%\s*per\s*(annum|day|month)',
            r'18%|10%|500.*per day',
            # Periods and deadlines
            r'\d+\s*(days?|months?|years?)\s*(grace|notice|within|from)',
            r'grace period', r'first hearing', r'last date.*filing',
            # Policy/contract specific numbers
            r'policy number|policy no', r'case no|cnr', r'rera no',
            r'sum assured.*rs', r'maturity.*\d{4}',
        ]
        for sent in sentences:
            sl = sent.lower()
            if any(re.search(p, sl) for p in numeric_patterns) and sent.strip() not in seen:
                add("FACT", sent)
                if len(selected) > 20:
                    break

        # ── Priority 4: Query-keyword matching ────────────────────────────
        query_words = [w.lower() for w in re.split(r'\W+', query) if len(w) > 3]
        for sent in sentences:
            sl = sent.lower()
            if any(qw in sl for qw in query_words) and sent.strip() not in seen:
                add("", sent)
                if len("\n".join(selected)) > 2200:
                    break

        # ── Priority 5: Document start for structural context ─────────────
        result = "\n\n".join(selected)
        if len(result) < 2000:
            result = full_text[:700] + "\n\n...\n\n" + result

        return result[:3500]

    @staticmethod
    def print_result(result: QueryResult, show_clauses: bool = True):
        divider = "=" * 70
        print(f"\n{divider}")
        print(f"  YOUR QUESTION: {result.user_query}")
        print(divider)
        print("\nANSWER:\n")
        print(result.direct_answer)

        if show_clauses and result.supporting_clauses:
            print(f"\n{divider}")
            print("  SUPPORTING CLAUSES (with plain English explanations)")
            print(divider)
            for i, clause in enumerate(result.supporting_clauses, 1):
                tags = ", ".join(clause.legal_types)
                print(f"\n-- Clause {i} [{tags}] (distance: {clause.similarity_score:.2f}) --")
                preview = clause.original_clause[:400]
                if len(clause.original_clause) > 400:
                    preview += "..."
                print(f"\nORIGINAL:\n{preview}")
                print(f"\nEXPLANATION ({clause.explanation_level.upper()}):\n{clause.plain_english}")

        print(f"\n{divider}\n")

    def to_json(self, result: QueryResult) -> str:
        return json.dumps(asdict(result), indent=2)


# ---------------------------------------------------------------------------
# Interactive CLI
# ---------------------------------------------------------------------------

def interactive_mode(agent: LegalSimplifierAgent):
    print("\n" + "=" * 70)
    print("  Legal Simplifier Agent -- Ask anything about your document")
    print("  Commands: 'level brief|detailed|expert'  |  'quit' to exit")
    print("=" * 70 + "\n")

    level = "detailed"

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if user_input.lower().startswith("level "):
            new_level = user_input.split()[-1].lower()
            if new_level in ("brief", "detailed", "expert"):
                level = new_level
                print(f"[Mode changed to: {level}]\n")
            else:
                print("[Valid levels: brief, detailed, expert]\n")
            continue

        result = agent.query(user_input, level=level)
        agent.print_result(result, show_clauses=(level != "brief"))


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Legal Simplifier Agent")
    parser.add_argument("--query",  type=str,  help="Single query (non-interactive mode)")
    parser.add_argument("--level",  type=str,  default="detailed",
                        choices=["brief", "detailed", "expert"])
    parser.add_argument("--json",   action="store_true", help="Output as JSON")
    parser.add_argument("--index",  type=str,  default=FAISS_INDEX_PATH)
    parser.add_argument("--meta",   type=str,  default=METADATA_PATH)
    parser.add_argument("--apikey", type=str,  default="",
                        help="Groq API key (overrides GROQ_API_KEY env var)")
    args = parser.parse_args()

    api_key = args.apikey or GROQ_API_KEY
    agent = LegalSimplifierAgent(
        index_path=args.index,
        meta_path=args.meta,
        groq_api_key=api_key,
    )

    if args.query:
        result = agent.query(args.query, level=args.level)
        if args.json:
            print(agent.to_json(result))
        else:
            agent.print_result(result)
    else:
        interactive_mode(agent)