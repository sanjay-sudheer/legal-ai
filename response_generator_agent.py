"""
Response Generator Agent (Single-Document Mode)
------------------------------------------------
• Assumes ONLY ONE document exists in FAISS DB
• Automatically retrieves entire document context
• No manual query needed
• Interactive drafting mode
• User-guided / Auto / Hybrid
• Clean structured output
"""

import os
import json
import logging
from dataclasses import dataclass, asdict, field
from typing import List
from dotenv import load_dotenv
load_dotenv()

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

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your-key-here")  # Set this in your .env file or environment variables
BASE_URL = "https://api.groq.com/openai/v1"
MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """
You are an expert legal drafting engine.

Always structure responses properly:
• Court/Header
• Subject
• Reference
• Body
• Legal Grounds (if needed)
• Prayer/Relief (if applicable)
• Closing

Maintain professional formatting.
"""

# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class DraftVersion:
    version_number: int
    mode: str
    content: str

@dataclass
class ResponseOutput:
    tone: str
    drafts: List[DraftVersion] = field(default_factory=list)

# ─────────────────────────────────────────────
# VECTOR DB
# ─────────────────────────────────────────────

class VectorDB:
    def __init__(self):
        if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(METADATA_PATH):
            raise FileNotFoundError("Vector DB not found.")

        self.index = faiss.read_index(FAISS_INDEX_PATH)
        self.metadata = json.load(open(METADATA_PATH))

    def get_document_context(self, doc_id: str = None, doc_name: str = None) -> str:
        """
        Return the full text of a specific document.
        If doc_id or doc_name is given, filter to that document.
        Falls back to the full_document record if present, else concatenates all clauses.
        """
        candidates = self.metadata
        if doc_id:
            candidates = [c for c in candidates if c.get("doc_id") == doc_id]
        elif doc_name:
            candidates = [c for c in candidates if c.get("doc_name") == doc_name]

        # Prefer the dedicated full_document record
        full_doc = next(
            (c for c in candidates if c.get("clause_index") == -1), None
        )
        if full_doc:
            return full_doc["clause_text"][:8000]

        # Fallback: concatenate clause records (excluding table extras)
        return "\n\n---\n\n".join(
            c["clause_text"][:1200]
            for c in candidates
            if c.get("clause_index", 0) >= 0
        )

    def get_full_document_context(self) -> str:
        """Legacy method — returns all documents. Use get_document_context() for multi-doc."""
        return self.get_document_context()

# ─────────────────────────────────────────────
# LLM CLIENT
# ─────────────────────────────────────────────

class LLMClient:
    def __init__(self):
        self.client = OpenAI(api_key=GROQ_API_KEY, base_url=BASE_URL)
        self.model = MODEL

    def call(self, prompt, max_tokens=1800):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

# ─────────────────────────────────────────────
# MAIN AGENT
# ─────────────────────────────────────────────

class ResponseGeneratorAgent:

    def __init__(self):
        self.db = VectorDB()
        self.llm = LLMClient()

    def generate_draft(self, context, instruction, tone):

        prompt = f"""
CASE CONTEXT:
{context}

TONE: {tone}

DRAFTING INSTRUCTIONS:
{instruction}

Draft a complete legally appropriate response.
"""

        return self.llm.call(prompt)

    def interactive(self):

        print("\n=== RESPONSE GENERATOR AGENT ===\n")

        context = self.db.get_full_document_context()

        print("Document context loaded from Vector DB.\n")

        print("How would you like the reply drafted?")
        print("1. Provide custom instructions")
        print("2. Create automatically (AI decides)")
        print("3. Hybrid (1 user-guided + 1 AI alternative)\n")

        choice = input("Select option (1/2/3): ").strip()

        # Tone selection (clean mapping)
        tone_map = {
            "1": "formal",
            "2": "assertive",
            "3": "conciliatory"
        }

        print("\nSelect tone:")
        print("1. Formal")
        print("2. Assertive")
        print("3. Conciliatory")

        tone_choice = input("Select option (1/2/3): ").strip()
        tone = tone_map.get(tone_choice, "formal")

        drafts = []

        # ───────── USER GUIDED ─────────
        if choice == "1":

            user_instruction = input("\nEnter your drafting instructions:\n")

            draft = self.generate_draft(context, user_instruction, tone)

            drafts.append(
                DraftVersion(
                    version_number=1,
                    mode="user-guided",
                    content=draft
                )
            )

        # ───────── AUTO MODE ─────────
        elif choice == "2":

            # Generate 2 automatic variations
            for i in range(1, 3):

                auto_instruction = f"""
                Draft Version {i}.
                Create a legally strong and defensible reply.
                Ensure this version is stylistically distinct.
                """

                draft = self.generate_draft(context, auto_instruction, tone)

                drafts.append(
                    DraftVersion(
                        version_number=i,
                        mode="auto",
                        content=draft
                    )
                )

        # ───────── HYBRID MODE ─────────
        elif choice == "3":

            user_instruction = input("\nEnter your drafting instructions:\n")

            # User draft
            draft1 = self.generate_draft(context, user_instruction, tone)

            drafts.append(
                DraftVersion(
                    version_number=1,
                    mode="user-guided",
                    content=draft1
                )
            )

            # AI alternative
            auto_instruction = """
            Draft a legally strong alternative version with a slightly different structure.
            """

            draft2 = self.generate_draft(context, auto_instruction, tone)

            drafts.append(
                DraftVersion(
                    version_number=2,
                    mode="auto-variation",
                    content=draft2
                )
            )

        else:
            print("Invalid selection.")
            return

        result = ResponseOutput(
            tone=tone,
            drafts=drafts
        )

        print("\n=== GENERATED DRAFTS ===\n")
        print(json.dumps(asdict(result), indent=2))


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    agent = ResponseGeneratorAgent()
    agent.interactive()