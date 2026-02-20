"""
Document Processing Module
--------------------------
Handles ingestion and interpretation of legal documents.
- Extracts raw text from PDF/DOCX/TXT files
- Parses and segments clauses
- Classifies clauses by legal type (liability, confidentiality, termination, etc.)
- Generates LegalBERT embeddings
- Stores and retrieves embeddings via FAISS
"""

import os
import re
import json
import uuid
import logging
from pathlib import Path
from typing import Optional


import numpy as np
import faiss

# Document parsing
import pdfplumber
import docx

# LegalBERT embeddings
from transformers import AutoTokenizer, AutoModel
import torch

logging.getLogger("mlx").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LEGALBERT_MODEL = "nlpaueb/legal-bert-base-uncased"
EMBEDDING_DIM = 768  # LegalBERT hidden size
FAISS_INDEX_PATH = "legal_faiss.index"
METADATA_PATH = "legal_metadata.json"

# Clause classification keywords (lightweight rule-based; swap for a fine-tuned
# classifier in production)
CLAUSE_PATTERNS: dict[str, list[str]] = {
    "liability": [
        r"\bliabilit(y|ies)\b", r"\bindemnif(y|ication)\b",
        r"\bdamages\b", r"\bnegligence\b", r"\blimit(ation)? of liability\b",
    ],
    "confidentiality": [
        r"\bconfidential(ity)?\b", r"\bnon-disclosure\b", r"\bproprietary\b",
        r"\btrade secret\b", r"\bnda\b",
    ],
    "termination": [
        r"\btermination\b", r"\bterminat(e|ing)\b", r"\bexpiration\b",
        r"\bcancellation\b", r"\bnotice of termination\b",
    ],
    "payment": [
        r"\bpayment\b", r"\binvoice\b", r"\bfee(s)?\b",
        r"\bcompensation\b", r"\bremuneration\b", r"\bprice\b",
    ],
    "intellectual_property": [
        r"\bintellectual property\b", r"\bcopyright\b", r"\bpatent\b",
        r"\btrademark\b", r"\blicens(e|ing)\b", r"\bip rights\b",
    ],
    "governing_law": [
        r"\bgoverning law\b", r"\bjurisdiction\b", r"\bdispute resolution\b",
        r"\barbitration\b", r"\blitigation\b",
    ],
    "force_majeure": [
        r"\bforce majeure\b", r"\bact of god\b", r"\bunforeseeable\b",
        r"\bcircumstances beyond\b",
    ],
    "warranty": [
        r"\bwarrant(y|ies)\b", r"\brepresentation(s)?\b",
        r"\bfitness for purpose\b", r"\bdisclaimer\b",
    ],
}


# ---------------------------------------------------------------------------
# Text Extraction
# ---------------------------------------------------------------------------

class TextExtractor:
    """Extracts raw text from PDF, DOCX, or plain-text files."""

    def extract(self, file_path: str) -> str:
        path = Path(file_path)
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._extract_pdf(file_path)
        elif suffix in (".docx", ".doc"):
            return self._extract_docx(file_path)
        elif suffix in (".txt", ".md"):
            return path.read_text(encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    def _extract_pdf(self, path: str) -> str:
        text_parts = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t)
        return "\n".join(text_parts)

    def _extract_docx(self, path: str) -> str:
        doc = docx.Document(path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


# ---------------------------------------------------------------------------
# Clause Parser
# ---------------------------------------------------------------------------

class ClauseParser:
    """Segments document text into individual clauses."""

    # Split on numbered/lettered section headers or double newlines
    SECTION_RE = re.compile(
        r'(?:(?:^|\n)(?:\d+\.|\([a-z]\)|[A-Z][A-Z\s]{2,}:))',
        re.MULTILINE,
    )

    def parse(self, text: str) -> list[str]:
        # First try splitting on section headers
        segments = self.SECTION_RE.split(text)
        clauses = [s.strip() for s in segments if len(s.strip()) > 60]

        # Fallback: split on paragraph boundaries
        if len(clauses) < 2:
            clauses = [p.strip() for p in re.split(r'\n{2,}', text) if len(p.strip()) > 60]

        logger.info(f"Parsed {len(clauses)} clauses.")
        return clauses


# ---------------------------------------------------------------------------
# Clause Classifier
# ---------------------------------------------------------------------------

class ClauseClassifier:
    """Classifies a clause into one or more legal types using regex patterns."""

    def classify(self, clause_text: str) -> list[str]:
        text_lower = clause_text.lower()
        matched = []
        for label, patterns in CLAUSE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    matched.append(label)
                    break  # one match per label is enough
        return matched if matched else ["general"]


# ---------------------------------------------------------------------------
# Embedding Generator (LegalBERT)
# ---------------------------------------------------------------------------

class LegalBERTEmbedder:
    """Generates sentence-level embeddings using LegalBERT (mean pooling)."""

    def __init__(self, model_name: str = LEGALBERT_MODEL):
        logger.info(f"Loading LegalBERT model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def embed(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                output = self.model(**encoded)
            # Mean pooling over token embeddings
            attention_mask = encoded["attention_mask"]
            token_embeddings = output.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = (token_embeddings * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)
            all_embeddings.append(embeddings.cpu().numpy())
        return np.vstack(all_embeddings).astype("float32")


# ---------------------------------------------------------------------------
# FAISS Knowledge Base
# ---------------------------------------------------------------------------

class FAISSKnowledgeBase:
    """
    Stores and retrieves clause embeddings using a FAISS flat L2 index.
    Metadata (clause text, labels, document source) is stored in a parallel JSON file.
    """

    def __init__(self, dim: int = EMBEDDING_DIM, index_path: str = FAISS_INDEX_PATH, meta_path: str = METADATA_PATH):
        self.dim = dim
        self.index_path = index_path
        self.meta_path = meta_path
        self._load_or_create()

    def _load_or_create(self):
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            logger.info("Loading existing FAISS index and metadata.")
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "r") as f:
                self.metadata: list[dict] = json.load(f)
        else:
            logger.info("Creating new FAISS index.")
            self.index = faiss.IndexFlatL2(self.dim)
            self.metadata = []

    def add(self, embeddings: np.ndarray, records: list[dict]):
        """Add embeddings and their metadata records to the knowledge base."""
        assert len(embeddings) == len(records), "Embedding/record count mismatch."
        self.index.add(embeddings)
        self.metadata.extend(records)
        self._save()
        logger.info(f"Added {len(records)} clauses. Total stored: {self.index.ntotal}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        """Return the top-k most similar clauses to a query embedding."""
        if self.index.ntotal == 0:
            return []
        distances, indices = self.index.search(query_embedding, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            entry = dict(self.metadata[idx])
            entry["distance"] = float(dist)
            results.append(entry)
        return results

    def _save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def __len__(self):
        return self.index.ntotal


# ---------------------------------------------------------------------------
# Document Processing Module (Orchestrator)
# ---------------------------------------------------------------------------

class DocumentProcessingModule:
    """
    Orchestrates the full pipeline:
      file → extract text → parse clauses → classify → embed → store in FAISS
    """

    def __init__(
        self,
        index_path: str = FAISS_INDEX_PATH,
        meta_path: str = METADATA_PATH,
        legalbert_model: str = LEGALBERT_MODEL,
    ):
        self.extractor = TextExtractor()
        self.parser = ClauseParser()
        self.classifier = ClauseClassifier()
        self.embedder = LegalBERTEmbedder(legalbert_model)
        self.kb = FAISSKnowledgeBase(
            dim=EMBEDDING_DIM,
            index_path=index_path,
            meta_path=meta_path,
        )

    def ingest(self, file_path: str, doc_id: Optional[str] = None) -> list[dict]:
        """
        Full ingestion pipeline for a single document.

        Returns a list of clause records that were stored.
        """
        doc_id = doc_id or str(uuid.uuid4())
        logger.info(f"Ingesting document: {file_path}  (doc_id={doc_id})")

        # 1. Extract text
        raw_text = self.extractor.extract(file_path)
        logger.info(f"Extracted {len(raw_text)} characters of text.")

        # 2. Parse clauses
        clauses = self.parser.parse(raw_text)

        # 3. Classify
        records = []
        for i, clause_text in enumerate(clauses):
            labels = self.classifier.classify(clause_text)
            records.append({
                "id": str(uuid.uuid4()),
                "doc_id": doc_id,
                "source": file_path,
                "clause_index": i,
                "clause_text": clause_text,
                "legal_types": labels,
            })

        # 4. Embed
        texts = [r["clause_text"] for r in records]
        embeddings = self.embedder.embed(texts)

        # 5. Store in FAISS
        self.kb.add(embeddings, records)

        logger.info(f"Ingestion complete for {file_path}. {len(records)} clauses stored.")
        return records

    def query(self, query_text: str, top_k: int = 5) -> list[dict]:
        """
        Semantic search over the knowledge base.

        Returns top-k clause records most similar to the query.
        """
        logger.info(f"Querying knowledge base: '{query_text[:80]}...'")
        embedding = self.embedder.embed([query_text])
        results = self.kb.search(embedding, top_k=top_k)
        return results

    def ingest_directory(self, directory: str, extensions: tuple = (".pdf", ".docx", ".txt")) -> list[dict]:
        """Batch ingest all matching files in a directory."""
        all_records = []
        for path in Path(directory).rglob("*"):
            if path.suffix.lower() in extensions:
                try:
                    records = self.ingest(str(path))
                    all_records.extend(records)
                except Exception as e:
                    logger.error(f"Failed to ingest {path}: {e}")
        return all_records

    @property
    def total_clauses(self) -> int:
        return len(self.kb)


# ---------------------------------------------------------------------------
# CLI / Quick Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    module = DocumentProcessingModule()

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python document_processing_module.py ingest <file_or_dir>")
        print("  python document_processing_module.py query  '<search text>'")
        sys.exit(0)

    command = sys.argv[1]

    if command == "ingest":
        target = sys.argv[2]
        if os.path.isdir(target):
            records = module.ingest_directory(target)
        else:
            records = module.ingest(target)
        print(f"\nIngested {len(records)} clauses. Knowledge base total: {module.total_clauses}")
        for r in records[:5]:
            print(f"  [{', '.join(r['legal_types'])}] {r['clause_text'][:100]}...")

    elif command == "query":
        query_text = sys.argv[2]
        results = module.query(query_text, top_k=5)
        print(f"\nTop {len(results)} results for: '{query_text}'\n")
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{', '.join(r['legal_types'])}] dist={r['distance']:.4f}")
            print(f"     Source: {r['source']}")
            print(f"     {r['clause_text'][:150]}...\n")