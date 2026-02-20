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
TOP_K_CLAUSES = 3

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
    "Rewrite this as a legal search phrase (5-10 words) that would match relevant "
    "clauses in a contract. Return only the search phrase, nothing else."
)

DIRECT_ANSWER_PROMPT = (
    'A user asked: "{query}"\n\n'
    "Based on these relevant contract clauses:\n---\n{clauses}\n---\n\n"
    "Answer their question directly in plain, friendly English — like explaining to "
    "a friend. If the answer is clearly in the clauses, say so confidently. "
    "If not, say the document doesn't seem to cover that.\n\n"
    'End with a "Key takeaway:" line summarizing the most important point.'
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

    def search(self, embedding: np.ndarray, top_k: int = TOP_K_CLAUSES):
        distances, indices = self.index.search(embedding, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            entry = dict(self.metadata[idx])
            entry["distance"] = float(dist)
            results.append(entry)
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

    def query(self, user_query: str, level: ExplanationLevel = "detailed", top_k: int = None) -> QueryResult:
        top_k = top_k or self.top_k
        logger.info(f"Query: '{user_query}' | level={level}")

        # Step 1: Expand the user's casual question into legal search terms
        expanded = self.llm.generate(QUERY_EXPANSION_PROMPT.format(query=user_query))
        logger.info(f"Expanded query: '{expanded}'")

        # Step 2: Embed and retrieve from FAISS
        embedding = self.embedder.embed([expanded])
        raw_clauses = self.kb.search(embedding, top_k=top_k)

        if not raw_clauses:
            return QueryResult(
                user_query=user_query,
                direct_answer="I couldn't find any relevant clauses in the document for your question.",
                supporting_clauses=[],
            )

        # Step 3: Generate a direct plain-English answer grounded in retrieved clauses
        clauses_text = "\n\n---\n\n".join(
            f"[{', '.join(c.get('legal_types', ['general']))}]\n{c['clause_text']}"
            for c in raw_clauses
        )
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