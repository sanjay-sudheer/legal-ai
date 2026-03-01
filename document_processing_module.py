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
    # ── Universal / Preamble ──────────────────────────────────────────────────
    "recital_preamble": [
        r"\bwhereas\b", r"\bknow all men\b", r"\bthis (agreement|deed|power|policy)\b",
        r"\bdo hereby\b", r"\bnominate.*appoint\b", r"\bprincipal\b.*\battorney\b",
        r"\bresidin(g|g outside)\b", r"\bprofessional commitment\b",
        r"\bunable to be present\b", r"\bpurpose of this\b",
        r"\bgranted because\b", r"\bpower of attorney is granted\b",
        r"\bacting on.*behalf\b", r"\bauthority granted\b",
        r"\brecital(s)?\b", r"\bnow therefore\b", r"\bin witness whereof\b",
    ],
    "parties_identity": [
        r"\bs/o\b", r"\bd/o\b", r"\bw/o\b", r"\baged \d+ years\b",
        r"\baadhaar\b", r"\bpan:\b", r"\bpassport\b",
        r"\bhereinafter referred to as\b", r"\bhereby appoint\b",
        r"\bconstitute and appoint\b", r"\blife assured\b",
        r"\blicensor\b", r"\blicensee\b", r"\bdeponent\b",
        r"\bservice provider\b", r"\bclient\b.*\bparty\b",
        r"\bplaintiff\b", r"\bdefendant\b",
    ],
    "liability": [
        r"\bliabilit(y|ies)\b", r"\bindemnif(y|ication)\b",
        r"\bdamages\b", r"\bnegligence\b", r"\blimit(ation)? of liability\b",
        r"\bhold harmless\b", r"\baggregate liability\b",
        r"\bmental agony\b", r"\bcompensation.*damages\b",
    ],
    "confidentiality": [
        r"\bconfidential(ity)?\b", r"\bnon-disclosure\b", r"\bproprietary\b",
        r"\btrade secret\b", r"\bnda\b",
    ],
    "termination": [
        r"\btermination\b", r"\bterminat(e|ing)\b", r"\bexpiration\b",
        r"\bcancellation\b", r"\bnotice of termination\b",
        r"\brevok(e|ed|ing)\b", r"\buntil revoked\b",
        r"\bconvenience\b.*\bterminate\b", r"\bterminate.*notice\b",
        r"\bvacate\b", r"\bhand over.*possession\b",
    ],
    "payment": [
        r"\bpayment\b", r"\binvoice\b", r"\bfee(s)?\b",
        r"\bcompensation\b", r"\bremuneration\b", r"\bprice\b",
        r"\bstamp dut(y|ies)\b", r"\bregistration charges\b",
        r"\bsale consideration\b", r"\blicence fee\b",
        r"\bsecurity deposit\b", r"\bmonthly rent\b",
        r"\bpremium\b", r"\binstalment\b", r"\bmilestone\b",
        r"\bmobilisation advance\b", r"\bpayment schedule\b",
        r"\blate payment\b", r"\bpenalt(y|ies)\b.*pay",
        r"\binterest.*per annum\b", r"\b18%\b", r"\bgst\b",
    ],
    "property_authority": [
        r"\bimmovable property\b", r"\bplot no\b", r"\bkhasra\b",
        r"\bsale deed\b", r"\bconveyance deed\b", r"\btransfer\b",
        r"\bnegotiate.*sale\b", r"\bregister.*document\b",
        r"\bdda\b", r"\bdevelopment authorit\b",
        r"\bflat no\b", r"\bsurvey no\b", r"\bcarpet area\b",
        r"\bsq\.?\s*ft\b", r"\bsq\.?\s*yd\b",
        r"\bencumbrance\b", r"\bmortgage\b", r"\blien\b",
        r"\btitle deed\b", r"\bpatta\b", r"\bchitta\b",
    ],
    "intellectual_property": [
        r"\bintellectual property\b", r"\bcopyright\b", r"\bpatent\b",
        r"\btrademark\b", r"\blicens(e|ing)\b", r"\bip rights\b",
        r"\bdeliverable(s)?\b", r"\bwork product\b", r"\bpre-existing\b",
    ],
    "governing_law": [
        r"\bgoverning law\b", r"\bjurisdiction\b", r"\bdispute resolution\b",
        r"\barbitration\b", r"\blitigation\b",
        r"\bregistration act\b", r"\bstamp act\b",
        r"\bpowers of attorney act\b", r"\bseat of arbitration\b",
        r"\bcourts at\b", r"\bexclusive jurisdiction\b",
        r"\bmediation\b",
    ],
    "force_majeure": [
        r"\bforce majeure\b", r"\bact of god\b", r"\bunforeseeable\b",
        r"\bcircumstances beyond\b",
    ],
    "warranty": [
        r"\bwarrant(y|ies)\b", r"\brepresentation(s)?\b",
        r"\bfitness for purpose\b", r"\bdisclaimer\b",
        r"\bratif(y|ication)\b", r"\bconfirm.*whatsoever\b",
    ],
    "validity_revocation": [
        r"\bvalid and subsisting\b", r"\buntil revoked\b", r"\brevocation\b",
        r"\bremains valid\b", r"\bshall remain\b.*\bvalid\b",
        r"\bregistered.*sub-registrar\b",
        r"\bapostil(le|led)\b", r"\bhague convention\b",
        r"\bexecuted outside india\b",
    ],
    # ── Insurance-specific ───────────────────────────────────────────────────
    "insurance_benefit": [
        r"\bsum assured\b", r"\bdeath benefit\b", r"\bmaturity\b",
        r"\brider\b", r"\baccidental death\b", r"\bcritical illness\b",
        r"\bwaiver of premium\b", r"\bpaid.?up\b", r"\bsurrender value\b",
        r"\bloan.*policy\b", r"\bpolicy loan\b",
        r"\bnominee\b", r"\blife assured\b", r"\bendowment\b",
        r"\bbonus\b.*\breversionary\b", r"\bfree.?look\b",
    ],
    "insurance_exclusion": [
        r"\bexclusion(s)?\b", r"\bnot cover\b", r"\bsuicide clause\b",
        r"\bwar\b.*\baviation\b", r"\bnuclear\b", r"\bself.?inflicted\b",
        r"\blapse\b", r"\bgrace period\b", r"\bpremium.*unpaid\b",
        r"\brevival\b",
    ],
    "insurance_tax": [
        r"\bincome tax\b", r"\bsection 80c\b", r"\bsection 10\(10d\)\b",
        r"\btax.?free\b", r"\bexempt\b.*\btax\b", r"\birdai\b",
    ],
    # ── Court / Summons-specific ─────────────────────────────────────────────
    "court_procedure": [
        r"\bsummons\b", r"\bplaintiff\b", r"\bdefendant\b",
        r"\bwritten statement\b", r"\bex.?parte\b", r"\bvakalatnama\b",
        r"\bcivil judge\b", r"\bcourt.*case\b", r"\bsuit\b.*\bfiled\b",
        r"\bcnr\b", r"\border.*cpc\b", r"\bhearing\b",
        r"\bfirst hearing\b", r"\bfiling date\b",
    ],
    "court_deadline": [
        r"\b\d{1,2}(st|nd|rd|th) (january|february|march|april|may|june|july|august|september|october|november|december)\b",
        r"\bwithin \d+ days\b", r"\bappear.*before\b",
        r"\bfailure to appear\b", r"\blast date\b.*\bfiling\b",
        r"\btime.*prescribed\b",
    ],
    "rera": [
        r"\brera\b", r"\bmaharea\b", r"\breal estate.*regulatory\b",
        r"\bregistration.*project\b",
    ],
    # ── Rental / Leave & Licence-specific ───────────────────────────────────
    "rental_terms": [
        r"\bleave and licen(c|s)e\b", r"\blicen(c|s)or\b", r"\blicen(c|s)ee\b",
        r"\blicence fee\b", r"\bsecurity deposit\b", r"\brental\b",
        r"\bmonthly.*rent\b", r"\bescalation\b", r"\brenewal\b",
        r"\bpossession\b", r"\bhabitable\b", r"\bfixtures\b",
        r"\bsublet\b", r"\bassign\b.*\bpremises\b",
        r"\bpet\b", r"\banimal\b", r"\belectricity.*charges\b",
        r"\bpolice verification\b", r"\bmaintenance charges\b",
    ],
    # ── Affidavit-specific ───────────────────────────────────────────────────
    "affidavit_sworn": [
        r"\baffidavit\b", r"\bsolemnly affirm\b", r"\bsworn\b",
        r"\bdeponent\b", r"\bnotary public\b", r"\bexecutive magistrate\b",
        r"\bon oath\b", r"\bverif(y|ication)\b.*affidavit\b",
        r"\btrue and correct\b",
    ],
    "title_encumbrance": [
        r"\bencumbrance\b", r"\bfree from\b.*\bmortgage\b",
        r"\bno.*lien\b", r"\btitle document\b", r"\bparent deed\b",
        r"\bpatta transfer\b", r"\bchitta\b", r"\bsale deed.*possession\b",
        r"\bno third party.*right\b", r"\babsolute owner\b",
        r"\bpower of attorney.*false\b", r"\bfalse statement\b",
    ],
    # ── PSA / Consulting-specific ────────────────────────────────────────────
    "scope_of_services": [
        r"\bscope of services\b", r"\bdeliverable(s)?\b",
        r"\bfeasibility (study|report)\b", r"\bpmo\b",
        r"\bprocurement\b", r"\bfinancial model\b",
        r"\bregulatory compliance\b", r"\bstakeholder\b",
        r"\bschedule [ab]\b", r"\bmilestone\b",
        r"\bmobilisation advance\b", r"\bproject manager\b",
        r"\bnavi mumbai\b", r"\bcoastal development\b",
        r"\bmmrda\b", r"\bcrz\b",
    ],
    "force_majeure": [
        r"\bforce majeure\b", r"\bact of god\b", r"\bunforeseeable\b",
        r"\bcircumstances beyond\b",
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
                # Extract regular text
                t = page.extract_text()
                if t:
                    text_parts.append(t)

                # Also extract tables and format them as readable text
                # This captures insurance benefit tables, payment schedules, etc.
                tables = page.extract_tables()
                for table in tables:
                    if not table:
                        continue
                    table_lines = []
                    for row in table:
                        if row:
                            # Filter None cells and join with " | "
                            cells = [str(c).strip() for c in row if c and str(c).strip()]
                            if cells:
                                table_lines.append(" | ".join(cells))
                    if table_lines:
                        table_text = "\n".join(table_lines)
                        # Only add if table text is not already in the page text
                        if table_text[:60] not in (t or ""):
                            text_parts.append(f"[TABLE]\n{table_text}\n[/TABLE]")

        return "\n".join(text_parts)

    def _extract_docx(self, path: str) -> str:
        doc = docx.Document(path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


# ---------------------------------------------------------------------------
# Clause Parser
# ---------------------------------------------------------------------------

class ClauseParser:
    """
    Segments document text into meaningful clauses.

    Strategy (priority order):
    1. Named section headers (KNOW ALL MEN, AUTHORITY GRANTED, etc.) — preserve
       full numbered lists WITHIN a section as one clause so context isn't lost.
    2. All-caps section titles followed by content.
    3. Double-newline paragraph breaks as fallback.

    Critical fix: numbered items like "1. to sell..." are NOT treated as section
    boundaries — they are sub-items of a section and must stay together.
    """

    # Split ONLY on major titled section breaks — all-caps words at line start
    # followed by content, OR lines that are clearly standalone section headers.
    SECTION_HEADER_RE = re.compile(
        r'\n(?=[A-Z][A-Z\s]{4,}:?\s*\n)',  # ALL CAPS HEADER on its own line
        re.MULTILINE,
    )

    # Identify blocks that look like preamble / recital / named parties
    PREAMBLE_MARKERS = [
        "KNOW ALL MEN", "GENERAL POWER OF ATTORNEY", "WHEREAS",
        "DO HEREBY NOMINATE", "DESCRIPTION OF PROPERTY", "AUTHORITY GRANTED",
        "IN WITNESS WHEREOF", "THIS POWER OF ATTORNEY",
    ]

    def parse(self, text: str) -> list[str]:
        clauses = []

        # Strategy 1: split on ALL-CAPS titled sections
        segments = self.SECTION_HEADER_RE.split(text)

        if len(segments) >= 2:
            for seg in segments:
                seg = seg.strip()
                if len(seg) > 80:
                    clauses.append(seg)
        
        # Strategy 2: if that produced too few, split on double newlines
        # but group numbered sub-items (1. ... 2. ... 10.) with their parent paragraph
        if len(clauses) < 3:
            clauses = self._paragraph_split(text)

        # Always ensure the full preamble (reason/parties) is stored as one clause
        clauses = self._ensure_preamble_intact(text, clauses)

        # Deduplicate while preserving order
        seen, unique = set(), []
        for c in clauses:
            key = c[:120]
            if key not in seen:
                seen.add(key)
                unique.append(c)

        logger.info(f"Parsed {len(unique)} clauses.")
        return unique

    def _paragraph_split(self, text: str) -> list[str]:
        """Split on blank lines, then merge lone numbered items back to prior block."""
        raw_paras = [p.strip() for p in re.split(r'\n{2,}', text) if len(p.strip()) > 60]
        merged, buffer = [], ""
        for para in raw_paras:
            # If paragraph starts with a number+dot it's a sub-item — merge with buffer
            if re.match(r'^\d+\.', para) and buffer:
                buffer += "\n" + para
            else:
                if buffer:
                    merged.append(buffer)
                buffer = para
        if buffer:
            merged.append(buffer)
        return merged

    def _ensure_preamble_intact(self, full_text: str, clauses: list[str]) -> list[str]:
        """
        Make sure the document preamble/recital block (parties + reason) is stored
        as one complete clause so retrieval can always find the 'why'.
        """
        # Find text up to first numbered authority item
        preamble_end = re.search(r'\n1\.\s+to\s+', full_text, re.IGNORECASE)
        if preamble_end:
            preamble = full_text[:preamble_end.start()].strip()
            if len(preamble) > 100:
                # Insert at front so it gets a dedicated FAISS slot
                clauses = [preamble] + [c for c in clauses if c[:100] != preamble[:100]]

        # Also ensure the "reason" paragraph is its own clause
        reason_match = re.search(
            r'(This (?:General )?Power of Attorney is granted because.*?(?:\n\n|$))',
            full_text, re.IGNORECASE | re.DOTALL
        )
        if reason_match:
            reason_text = reason_match.group(1).strip()
            if reason_text and len(reason_text) > 40:
                # Add as dedicated clause if not already covered
                already = any(reason_text[:60] in c for c in clauses)
                if not already:
                    clauses.append(reason_text)

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

    def search(self, query_embedding: np.ndarray, top_k: int = 5,
               doc_id: Optional[str] = None, doc_name: Optional[str] = None) -> list[dict]:
        """
        Return the top-k most similar clauses to a query embedding.
        
        If doc_id or doc_name is provided, results are filtered to that document only.
        This is critical when multiple documents are in the index — without filtering,
        a query about insurance grace periods could return lease agreement clauses.
        """
        if self.index.ntotal == 0:
            return []

        # Fetch more candidates than needed so we can filter by doc if required
        fetch_k = top_k * 8 if (doc_id or doc_name) else top_k
        fetch_k = min(fetch_k, self.index.ntotal)

        distances, indices = self.index.search(query_embedding, fetch_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            entry = dict(self.metadata[idx])
            entry["distance"] = float(dist)

            # Apply document filter if specified
            if doc_id and entry.get("doc_id") != doc_id:
                continue
            if doc_name and entry.get("doc_name") != doc_name:
                continue

            results.append(entry)
            if len(results) >= top_k:
                break

        return results

    def list_documents(self) -> list[dict]:
        """Return a list of all ingested documents with their doc_id, doc_name, source."""
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

        Enhancement: always stores the FULL document text as record[0] so that
        queries requiring holistic context (parties, reason, overview) can always
        retrieve the complete document in a single FAISS hit.
        """
        doc_id = doc_id or str(uuid.uuid4())
        doc_name = Path(file_path).stem  # e.g. "general_power_of_attorney"
        logger.info(f"Ingesting document: {file_path}  (doc_id={doc_id})")

        # 1. Extract text (including tables)
        raw_text = self.extractor.extract(file_path)
        logger.info(f"Extracted {len(raw_text)} characters of text.")

        # 2. Parse clauses
        clauses = self.parser.parse(raw_text)

        # 3. Build records — clause[0] is the full document text for holistic queries
        records = []

        def make_record(clause_text: str, clause_index: int, extra_types: list = None) -> dict:
            labels = self.classifier.classify(clause_text)
            if extra_types:
                labels = list(set(labels + extra_types))
            return {
                "id": str(uuid.uuid4()),
                "doc_id": doc_id,
                "doc_name": doc_name,
                "source": file_path,
                "clause_index": clause_index,
                "clause_text": clause_text,
                "legal_types": labels,
            }

        # Full-document record (always first — highest-value for holistic queries)
        full_doc_labels = self.classifier.classify(raw_text[:2000])
        full_doc_labels = list(set(full_doc_labels + ["full_document", "recital_preamble"]))
        records.append({
            "id": str(uuid.uuid4()),
            "doc_id": doc_id,
            "doc_name": doc_name,
            "source": file_path,
            "clause_index": -1,        # sentinel: -1 = full document
            "clause_text": raw_text,   # no truncation
            "legal_types": full_doc_labels,
        })

        # Individual clause records
        for i, clause_text in enumerate(clauses):
            records.append(make_record(clause_text, i))

        # Extra: store table sections as standalone searchable records
        # Tables contain critical structured data (payment schedules, benefit tables, etc.)
        table_matches = re.findall(r'\[TABLE\](.*?)\[/TABLE\]', raw_text, re.DOTALL)
        for j, table_text in enumerate(table_matches):
            table_text = table_text.strip()
            if len(table_text) > 40:
                records.append(make_record(table_text, -(j + 2), ["table_data"]))

        # 4. Embed — use first 1024 chars for FAISS embedding (semantic signal)
        texts = [r["clause_text"][:1024] for r in records]
        embeddings = self.embedder.embed(texts)

        # 5. Store in FAISS
        self.kb.add(embeddings, records)

        logger.info(f"Ingestion complete for {file_path}. {len(records)} records stored.")
        return records

    def query(self, query_text: str, top_k: int = 5,
              doc_id: Optional[str] = None, doc_name: Optional[str] = None) -> list[dict]:
        """
        Semantic search over the knowledge base.
        Optionally filter to a specific document by doc_id or doc_name.
        """
        logger.info(f"Querying knowledge base: '{query_text[:80]}...'")
        embedding = self.embedder.embed([query_text])
        results = self.kb.search(embedding, top_k=top_k, doc_id=doc_id, doc_name=doc_name)
        return results

    def list_documents(self) -> list[dict]:
        """Return metadata for all ingested documents."""
        return self.kb.list_documents()

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