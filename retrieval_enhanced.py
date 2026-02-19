"""
NCERT Physics RAG v4.0 — Enhanced Retrieval + Smart Query Classification
=========================================================================
Key Improvements over v3.4:
  • Small LLM (llama-3.1-8b-instant) for query intent classification
    — replaces brittle regex; understands "summary of properties of waves"
      vs "give chapter summary"; handles nuanced phrasing
  • Direct display for summary / points_to_ponder / exercises / examples
    — no LLM call when the full answer is already in the chunk
  • Similarity threshold (configurable) — LLM only called when top-k
    chunks clear the threshold; below threshold → "not found" message
  • Enhanced GTE-small embedding:
    — Query-type aware prefixes (passage vs query encoding)
    — Hybrid scoring: semantic + BM25-style keyword overlap
    — Two-stage retrieval: broad FAISS → keyword re-rank
  • Beautiful math/LaTeX display: KaTeX via CDN, centred display blocks
  • LLM knowledge injection control (per intent type)
  • Robust streaming display with progressive rendering
"""

import json, os, re, pickle, html
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
from collections import Counter
import math

import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE, Groq = False, None


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & CONFIG
# ══════════════════════════════════════════════════════════════════════════════

SIMILARITY_THRESHOLD = 0.30   # GTE-small IP scores typically 0.25–0.65; 0.30 is a safe floor
CLASSIFIER_MODEL     = "llama-3.1-8b-instant"   # fast, cheap, good at classification
ANSWER_MODEL         = "llama-3.3-70b-versatile" # best quality for answers

# GTE-small prefix strategy (following paper recommendations)
# For asymmetric retrieval: queries get a prefix, passages do not
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Chunk types that have COMPLETE answers → skip LLM, display directly
DIRECT_DISPLAY_TYPES = {"summary", "points_to_ponder", "exercises", "chapter_overview"}

# Chunk types that benefit from LLM but don't strictly need it
SEMI_DIRECT_TYPES = {"example", "table", "section", "subsection"}

# ══════════════════════════════════════════════════════════════════════════════
# QUERY INTENT (via LLM classifier)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class QueryIntent:
    """Parsed intent from the small LLM classifier."""
    raw_type:       str            # classifier output: one of INTENT_TYPES
    class_num:      Optional[str]  # "11", "12", or None
    chapter_num:    Optional[str]  # "3", "14", etc.
    content_type:   Optional[str]  # "summary", "exercises", "example", "section", etc.
    identifier:     Optional[str]  # "5.3" for Example 5.3
    semantic_query: str            # cleaned query for embedding
    allow_llm_knowledge: bool      # whether LLM can go beyond chunks
    is_book_level:  bool = False   # chapter list / topic list for whole class
    book_target:    Optional[str] = None  # "chapters" | "topics"
    confidence:     float = 1.0


# Canonical intent types the classifier outputs
INTENT_TYPES = [
    "direct_summary",          # "give summary of chapter 3 class 11"
    "direct_ptp",              # "points to ponder chapter 5 class 12"
    "direct_exercises",        # "exercise questions chapter 6 class 11"
    "direct_example",          # "example 5.3 class 11"
    "direct_table",            # "table 7.1 class 12"
    "list_sections",           # "list sections/topics/subsections of chapter X"
    "count_stat",              # "how many examples in chapter 3"
    "semantic_concept",        # "explain Newton's second law" → RAG + LLM
    "semantic_derivation",     # "derive equation of motion" → RAG + LLM
    "semantic_formula",        # "formula for kinetic energy" → RAG + LLM
    "semantic_definition",     # "what is Ohm's law" → RAG + LLM
    "semantic_compare",        # "difference between speed and velocity"
    "book_chapters",           # "list all chapters of class 11"
    "book_topics",             # "list all topics of class 12"
    "out_of_domain",           # chemistry, biology, math, etc.
]

CLASSIFIER_SYSTEM = """You are a query intent classifier for an NCERT Physics Class 11 & 12 Q&A system.

Classify the user's query into EXACTLY ONE of these intent types:

DIRECT intents (user wants a specific stored section — chapter/class REQUIRED):
- direct_summary    → user wants the CHAPTER SUMMARY section (e.g. "summary of chapter 3 class 11")
                      ⚠️ "summary of Newton's laws" or "summarize wave motion" → NOT this, use semantic_concept
- direct_ptp        → user wants Points to Ponder section of a specific chapter
- direct_exercises  → user wants the exercise questions list of a specific chapter
- direct_example    → user wants a specific numbered example like "example 5.3" or "example 9.1"
- direct_table      → user wants a specific table like "table 7.1"
- list_sections     → user wants a list of sections/subsections/topics of a specific chapter
- count_stat        → user wants a count: "how many examples in chapter 3 class 11"

SEMANTIC intents (conceptual/explanatory — search entire database, NO chapter/class filtering):
- semantic_concept     → explain a concept, describe how something works, why something happens
- semantic_derivation  → derive, prove, show derivation of a formula/equation
- semantic_formula     → what is the formula/equation for X
- semantic_definition  → define X, what is X, meaning of X
- semantic_compare     → compare, difference between, contrast two things

BOOK intents:
- book_chapters  → list all chapters of class 11 or class 12
- book_topics    → list all topics/sections of class 11 or class 12

OTHER:
- out_of_domain  → not about NCERT Physics Class 11 or 12 at all

EXTRACTION RULES — read carefully:
- class_num:   ONLY extract for direct_* / list_sections / count_stat / book_* intents.
               For ALL semantic_* intents → always return null (search whole database, no filtering).
- chapter_num: ONLY extract for direct_* / list_sections / count_stat intents.
               For ALL semantic_* intents → always return null.
- identifier:  For direct_example/direct_table only — the dot-number like "5.3", "9.1"
- semantic_query: The core concept or topic being asked about, stripped of metadata noise
                  (remove "class 11", "chapter 3", "in NCERT", etc.)
                  For direct_example: write "example 5.3" so we know what was asked.

Return ONLY valid JSON, no preamble:
{
  "intent": "<intent type>",
  "class_num": "11" | "12" | null,
  "chapter_num": "3" | null,
  "identifier": "5.3" | null,
  "semantic_query": "<core topic>",
  "allow_llm_knowledge": true | false
}

allow_llm_knowledge:
  true  → semantic_* intents (LLM may use its knowledge to supplement retrieved chunks)
  false → direct_* / list_sections / count_stat / book_* (answer from book only)"""


# ══════════════════════════════════════════════════════════════════════════════
# GROQ API MANAGER (persistent key rotation)
# ══════════════════════════════════════════════════════════════════════════════

class GroqAPIManager:
    def __init__(self):
        self.api_keys = self._load_keys()
        self.idx      = 0
        self.clients  = [Groq(api_key=k) for k in self.api_keys] if self.api_keys else []
        self.total_tokens = self.total_prompt = self.total_completion = 0
        self._classifier_tokens = 0

    def _load_keys(self):
        keys = []
        for i in range(1, 4):
            k = os.getenv(f"GROQ_API_KEY_{i}")
            if k: keys.append(k)
        d = os.getenv("GROQ_API_KEY")
        if d and d not in keys: keys.insert(0, d)
        return keys

    def get_client(self):
        return self.clients[self.idx] if self.clients and self.idx < len(self.clients) else None

    def _advance_key(self) -> bool:
        if self.idx < len(self.clients) - 1:
            self.idx += 1
            return True
        return False

    def generate(self, messages, temperature=0.1, max_tokens=2048, model=None):
        model    = model or ANSWER_MODEL
        attempts = len(self.clients) - self.idx
        for _ in range(attempts):
            client = self.get_client()
            if not client: break
            try:
                r = client.chat.completions.create(
                    model=model, messages=messages,
                    temperature=temperature, max_tokens=max_tokens,
                )
                u = r.usage
                self.total_prompt     += u.prompt_tokens
                self.total_completion += u.completion_tokens
                self.total_tokens     += u.total_tokens
                if model == CLASSIFIER_MODEL:
                    self._classifier_tokens += u.total_tokens
                return r.choices[0].message.content.strip(), {
                    "prompt_tokens": u.prompt_tokens,
                    "completion_tokens": u.completion_tokens,
                    "total_tokens": u.total_tokens,
                }
            except Exception as e:
                err = str(e).lower()
                if "rate" in err or "429" in err:
                    if self._advance_key():
                        st.warning(f"⚠️ Rate limit — switching to key {self.idx + 1}")
                        continue
                    else:
                        st.error("❌ All API keys rate-limited.")
                        return None, {}
                st.error(f"❌ API error: {e}")
                return None, {}
        return None, {}

    def classify_query(self, query: str) -> Optional[Dict]:
        """Use small model for query classification."""
        if not self.clients:
            return None
        messages = [
            {"role": "system", "content": CLASSIFIER_SYSTEM},
            {"role": "user",   "content": f"Query: {query}"},
        ]
        raw, _ = self.generate(messages, temperature=0.0, max_tokens=200, model=CLASSIFIER_MODEL)
        if not raw:
            return None
        # Strip markdown fences if present
        raw = re.sub(r"```(?:json)?|```", "", raw).strip()
        try:
            return json.loads(raw)
        except Exception:
            # Try to extract JSON from response
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if m:
                try: return json.loads(m.group())
                except: pass
        return None

    def usage(self):
        return {
            "total": self.total_tokens,
            "prompt": self.total_prompt,
            "completion": self.total_completion,
            "classifier": self._classifier_tokens,
        }

    def current_key_label(self) -> str:
        return f"Key {self.idx + 1}/{len(self.clients)}"


# ══════════════════════════════════════════════════════════════════════════════
# BM25-STYLE KEYWORD SCORER
# ══════════════════════════════════════════════════════════════════════════════

def _tokenize(text: str) -> List[str]:
    return re.findall(r'\b[a-z]{2,}\b', text.lower())

def _bm25_score(query_tokens: List[str], doc_tokens: List[str],
                k1: float = 1.5, b: float = 0.75, avg_dl: float = 200) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    dl   = len(doc_tokens)
    tf   = Counter(doc_tokens)
    score = 0.0
    for t in set(query_tokens):
        if t not in tf: continue
        f   = tf[t]
        idf = math.log(1 + 1)  # simplified; single doc
        score += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avg_dl))
    return score

def hybrid_score(semantic_score: float, query: str, doc_content: str,
                 alpha: float = 0.75) -> float:
    """Blend semantic score (alpha) with BM25 keyword score (1-alpha)."""
    q_tok = _tokenize(query)
    d_tok = _tokenize(doc_content)
    bm25  = _bm25_score(q_tok, d_tok)
    # Normalise bm25 roughly to [0, 1]
    bm25_norm = min(bm25 / 15.0, 1.0)
    return alpha * semantic_score + (1 - alpha) * bm25_norm


# ══════════════════════════════════════════════════════════════════════════════
# NCERT RETRIEVER — MAIN CLASS
# ══════════════════════════════════════════════════════════════════════════════

class NCERTRetriever:
    def __init__(self, index_dir: Path):
        self.index_dir = index_dir
        self.model = self.index = self.chunks = self.meta_idx = self.groq = None
        self._load()

    def _load(self):
        """
        Load pre-built index files created by build_embeddings.py.
        The model is loaded for QUERY encoding only — no re-embedding happens here.
        If index files are missing, prints a helpful error with the build command.
        """
        config_file = self.index_dir / "ncert_physics_config_gte.json"
        faiss_file  = self.index_dir / "ncert_physics_gte.faiss"
        chunks_file = self.index_dir / "ncert_physics_chunks_indexed_gte.json"
        meta_file   = self.index_dir / "ncert_physics_metadata_gte.pkl"

        missing = [f.name for f in [config_file, faiss_file, chunks_file, meta_file] if not f.exists()]
        if missing:
            st.error(
                f"❌ Index files not found in `{self.index_dir}`: {missing}\n\n"
                f"**Run the builder first:**\n"
                f"```\npython build_embeddings.py "
                f"--chunks ncert_physics_chunks.json "
                f"--output-dir {self.index_dir}\n```"
            )
            st.stop()

        try:
            cfg = json.loads(config_file.read_text())
            st.info(
                f"📦 Index built: {cfg.get('built_at','?')} · "
                f"{cfg.get('num_chunks','?')} chunks · model: `{cfg.get('model_name','?')}`"
            )

            # Load model for query encoding only (no passage encoding done here)
            self.model = SentenceTransformer(cfg["model_name"])

            # Load pre-built FAISS index
            self.index = faiss.read_index(str(faiss_file))

            # Load chunks (same order as FAISS rows)
            self.chunks = json.loads(chunks_file.read_text(encoding="utf-8"))

            # Load fast lookup dicts
            with open(meta_file, "rb") as f:
                self.meta_idx = pickle.load(f)

            if GROQ_AVAILABLE:
                self.groq = GroqAPIManager()
                if not self.groq.api_keys:
                    st.warning("⚠️ No Groq API keys — AI disabled.")

        except Exception as e:
            st.error(f"❌ Failed to load index: {e}")
            st.stop()

    # ── Embedding helpers ─────────────────────────────────────────────────────

    def _encode_query(self, query: str) -> np.ndarray:
        """
        GTE-small asymmetric encoding:
        Queries get a task prefix; stored passage vectors do NOT have the prefix
        (they were encoded without prefix during indexing).
        This matches the GTE paper's recommended asymmetric retrieval setup.
        """
        prefixed = QUERY_PREFIX + query
        emb = self.model.encode([prefixed], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(emb)
        return emb

    # ── Index lookups ─────────────────────────────────────────────────────────

    def _get_overview(self, cls: str, chapter: str) -> Optional[Dict]:
        key = f"{cls}-{chapter}"
        idx = self.meta_idx.get("chapter_overviews", {}).get(key)
        return self.chunks[idx] if idx is not None else None

    def _find_chunk(self, cls: str, chapter: str, chunk_type: str,
                    identifier: Optional[str] = None) -> Optional[Dict]:
        """Direct index lookup by metadata."""
        for chunk in self.chunks:
            m = chunk["metadata"]
            if (m.get("class") == cls and m.get("chapter") == chapter
                    and m.get("type") == chunk_type):
                if identifier is None or m.get("identifier") == identifier:
                    return chunk
        return None

    def _find_example(self, cls: str, identifier: str) -> Optional[Dict]:
        chapter = identifier.split(".")[0]
        key     = f"{cls}-{chapter}-example-{identifier}"
        idx     = self.meta_idx.get("examples", {}).get(key)
        return self.chunks[idx] if idx is not None else None

    def _find_table(self, cls: str, identifier: str) -> Optional[Dict]:
        chapter = identifier.split(".")[0]
        key     = f"{cls}-{chapter}-table-{identifier}"
        idx     = self.meta_idx.get("tables", {}).get(key)
        return self.chunks[idx] if idx is not None else None

    # ── Semantic retrieval with hybrid scoring ────────────────────────────────

    def semantic_search(self, query: str, k: int = 8,
                        filters: Optional[Dict] = None,
                        use_hybrid: bool = True) -> List[Dict]:
        emb = self._encode_query(query)
        # Fetch extra candidates for re-ranking
        scores, indices = self.index.search(emb, k * 5)
        results = []
        for sc, idx in zip(scores[0], indices[0]):
            if idx == -1: continue
            chunk = self.chunks[idx]
            m     = chunk["metadata"]
            if filters and not all(m.get(fk) == fv for fk, fv in filters.items()):
                continue
            final_score = (
                hybrid_score(float(sc), query, chunk["content"])
                if use_hybrid else float(sc)
            )
            results.append({
                "chunk":      chunk,
                "score":      float(sc),
                "hybrid":     final_score,
                "similarity": round(float(sc) * 100, 1),
            })

        # Re-rank by hybrid score
        results.sort(key=lambda x: x["hybrid"], reverse=True)
        return results[:k]

    # ── Intent classification ─────────────────────────────────────────────────

    def classify(self, query: str) -> QueryIntent:
        """
        Classify query intent.
        - class_num: ONLY set when user EXPLICITLY wrote 'class 11/12' in query (regex).
        - chapter_num: extracted from query text via regex (reliable) OR classifier fallback.
        - identifier: extracted via regex for examples/tables.
        We never trust the classifier's class guess — it hallucinates class too often.
        """
        fallback = QueryIntent(
            raw_type="semantic_concept", class_num=None, chapter_num=None,
            content_type=None, identifier=None,
            semantic_query=query, allow_llm_knowledge=True,
        )
        if not self.groq:
            return fallback

        clf = self.groq.classify_query(query)
        if not clf:
            return fallback

        intent_str  = clf.get("intent", "semantic_concept")
        is_semantic = intent_str.startswith("semantic_")

        # ── Extract class, chapter, identifier directly from query text ──────
        # NEVER trust classifier for class — it hallucinates too often.
        explicit_class   = self._regex_class(query)
        explicit_chapter = self._regex_chapter(query)
        explicit_ident   = self._regex_identifier(query)

        # For semantic intents: class and chapter always null (search whole DB)
        cls     = None if is_semantic else explicit_class
        # chapter: prefer regex extraction; fall back to classifier if regex missed
        chapter = None if is_semantic else (explicit_chapter or clf.get("chapter_num"))
        # identifier: prefer regex; fall back to classifier
        ident   = explicit_ident or clf.get("identifier")

        sem_q     = clf.get("semantic_query") or query
        allow_llm = bool(clf.get("allow_llm_knowledge", is_semantic))

        content_map = {
            "direct_summary":   "summary",
            "direct_ptp":       "points_to_ponder",
            "direct_exercises": "exercises",
            "direct_example":   "example",
            "direct_table":     "table",
        }
        content_type = content_map.get(intent_str)

        is_book  = intent_str in ("book_chapters", "book_topics")
        book_tgt = "chapters" if intent_str == "book_chapters" else (
                   "topics"   if intent_str == "book_topics"   else None)
        if is_book:
            cls = explicit_class  # book level: class only if explicitly stated

        return QueryIntent(
            raw_type=intent_str,
            class_num=cls,
            chapter_num=chapter,
            content_type=content_type,
            identifier=ident,
            semantic_query=sem_q,
            allow_llm_knowledge=allow_llm,
            is_book_level=is_book,
            book_target=book_tgt,
        )

    @staticmethod
    def _regex_class(query: str) -> Optional[str]:
        """Return '11' or '12' ONLY if explicitly written by user."""
        q = query.lower()
        if re.search(r'\bclass\s*12\b|\bxii\b|\b12th\b', q):
            return "12"
        if re.search(r'\bclass\s*11\b|\bxi\b|\b11th\b', q):
            return "11"
        return None

    @staticmethod
    def _regex_chapter(query: str) -> Optional[str]:
        """Extract chapter number from query text reliably."""
        m = re.search(r'\bchapter\s*(\d{1,2})\b', query, re.IGNORECASE)
        if m:
            return m.group(1)
        return None

    @staticmethod
    def _regex_identifier(query: str) -> Optional[str]:
        """Extract example/table identifier like '5.3' or '8.1' from query."""
        # Match patterns like "example 5.3", "ex 8.1", "table 7.1"
        m = re.search(
            r'\b(?:example|ex|table)\s*(\d{1,2}\.\d+)\b',
            query, re.IGNORECASE
        )
        if m:
            return m.group(1)
        return None

    # ── Ambiguity resolution ──────────────────────────────────────────────────

    def check_ambiguity(self, intent: QueryIntent) -> Optional[List[Dict]]:
        """
        For direct/list intents: if chapter given but no class → always ask.
        Chapters 1-14 exist in BOTH Class 11 and 12, so we must always ask.
        """
        if intent.class_num or not intent.chapter_num:
            return None
        # Both classes have chapters 1–14, so always offer selection
        options = []
        for cls in ["11", "12"]:
            ov = self._get_overview(cls, intent.chapter_num)
            if ov:
                options.append({
                    "class":         cls,
                    "chapter":       intent.chapter_num,
                    "chapter_title": ov["metadata"].get("chapter_title", ""),
                })
        return options if len(options) >= 1 else None  # even 1 option = was ambiguous

    # ── Stats helpers ─────────────────────────────────────────────────────────

    def get_stats(self, cls: str, chapter: str) -> Dict:
        ov = self._get_overview(cls, chapter)
        if not ov:
            return {"error": f"Chapter {chapter} Class {cls} not found."}
        m = ov["metadata"]
        return {
            "class": cls, "chapter": chapter,
            "chapter_title": m.get("chapter_title", ""),
            "stats": m.get("stats", {}),
            "sections": m.get("sections", []),
            "subsections": m.get("subsections", []),
        }

    def get_sections_list(self, cls: str, chapter: str) -> Dict:
        ov = self._get_overview(cls, chapter)
        if not ov:
            return {"error": "Chapter not found."}
        m = ov["metadata"]
        return {
            "class": cls, "chapter": chapter,
            "chapter_title": m.get("chapter_title", ""),
            "sections": m.get("sections", []),
            "subsections": m.get("subsections", []),
            "topics": m.get("topics", []),
        }

    # ── LLM answer generation ─────────────────────────────────────────────────

    def generate_answer(self, query: str, chunks: List[Dict],
                        allow_knowledge: bool = True) -> Tuple[Optional[str], Dict]:
        if not self.groq or not chunks:
            return None, {}

        context = "\n\n".join(
            f"━━━ Class {c['metadata']['class']} | Ch {c['metadata']['chapter']}"
            f" | {c['metadata'].get('chapter_title','')} | type={c['metadata']['type']} ━━━\n"
            f"{c['content']}"
            for c in chunks
        )

        knowledge_instruction = (
            "You may supplement context with your knowledge of NCERT Physics 11/12 when helpful."
            if allow_knowledge else
            "Answer ONLY from the provided context. Do NOT use outside knowledge."
        )

        system = f"""You are an NCERT Class 11 & 12 Physics expert.

DOMAIN RULE: ONLY answer NCERT Physics Class 11/12 questions.
If off-domain: "I can only answer questions related to NCERT Physics for Class 11 and 12."

{knowledge_instruction}

MATH FORMATTING — CRITICAL:
• ALL math MUST use LaTeX
• Inline math: $expression$
• Display/block math (equations, derivations): $$expression$$ on its OWN line
• Multi-line derivations: use $$\\begin{{align*}} ... \\end{{align*}}$$
• Use: \\vec{{F}}, \\frac{{a}}{{b}}, \\sqrt{{x}}, \\Delta, \\omega, \\theta, \\times, \\cdot, \\hat{{r}}
• NEVER write raw math without LaTeX (no plain "v = u + at", always $v = u + at$)

TEXT FORMATTING:
• **Given:** **To Find:** **Solution:** **Step N:** as bold inline labels, NEVER as headings
• Section headings: use **bold** only, no ## markdown headings inside answers
• Cite source: *From Class X Chapter Y: Title*
• Keep explanations clear and step-by-step for derivations"""

        prompt = f"""Context from NCERT Physics:
{context}

Question: {query}

Answer (proper LaTeX for all math, bold inline section labels):"""

        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ]
        return self.groq.generate(messages, max_tokens=2500)

    def generate_book_answer(self, intent: QueryIntent) -> Tuple[Optional[str], Dict]:
        """Book-level queries (chapter lists) from LLM knowledge."""
        if not self.groq:
            return None, {}
        cls_label = (
            "Class 11 and Class 12" if intent.class_num in ("both", None)
            else f"Class {intent.class_num}"
        )
        if intent.book_target == "chapters":
            q = (f"List ALL chapters of NCERT Physics {cls_label} in a markdown table "
                 f"| Chapter No. | Chapter Title |. Be complete and accurate.")
        else:
            q = (f"List all major topics/sections of NCERT Physics {cls_label}, "
                 f"grouped by chapter. Format: **Chapter N: Title** then bullet topics.")

        messages = [
            {"role": "system", "content": (
                "You are an NCERT Physics expert. Answer ONLY about NCERT Physics Class 11 & 12. "
                "Refuse anything else.")},
            {"role": "user",   "content": q},
        ]
        return self.groq.generate(messages, max_tokens=3000)

    # ── Main retrieve orchestrator ────────────────────────────────────────────

    def generate_direct_answer(self, query: str, chunk: Dict, intent_type: str) -> Tuple[Optional[str], Dict]:
        """
        Use LLM to render summary / points_to_ponder with proper LaTeX.
        The raw NCERT text is given as context; LLM formats it beautifully.
        """
        if not self.groq:
            return None, {}
        m       = chunk["metadata"]
        content = chunk["content"]
        label   = {"direct_summary": "Chapter Summary", "direct_ptp": "Points to Ponder"}.get(intent_type, "Content")

        system = """You are an NCERT Physics formatter. Your job is to reproduce the given textbook content
with PERFECT LaTeX math formatting, preserving every word and every point exactly.

RULES — follow strictly:
1. Reproduce ALL content — do NOT skip, summarize or paraphrase anything.
2. Every equation, formula, expression, or quantity with a unit → LaTeX.
   • Inline: $F = ma$,  $v = 5\\ \\text{m/s}$,  $G = 6.67 \\times 10^{-11}$
   • Display (standalone equations): $$F = \\frac{Gm_1 m_2}{r^2}$$
3. Numbered items (1. 2. 3.) → keep as markdown numbered list.
4. Bold any key terms or labels that appear in the original.
5. DO NOT add headings like ## or ###.  DO NOT add commentary or explanations.
6. Output clean markdown + LaTeX only."""

        prompt = f"""Reproduce this NCERT {label} with proper LaTeX math formatting.
Every point must be kept exactly. Format equations with $...$ or $$...$$.

Source content:
{content}

Formatted output (preserve all points, add LaTeX to all math):"""

        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ]
        return self.groq.generate(messages, max_tokens=3000)

    def generate_example_answer(self, query: str, chunk: Dict) -> Tuple[Optional[str], Dict]:
        """Call LLM to beautifully format a solved example with proper LaTeX."""
        if not self.groq:
            return None, {}
        content = chunk["content"]
        m = chunk["metadata"]
        system = """You are an NCERT Physics expert. Format the given solved example beautifully.

MATH FORMATTING — CRITICAL:
• Every equation, formula, number with unit MUST be in LaTeX
• Inline math: $expression$ (e.g. $F = ma$, $v = 5\\ \\text{m/s}$)
• Display/block equations on their own line: $$expression$$
• Multi-step working: $$\\begin{align*} ... \\end{align*}$$
• Use proper LaTeX: \\frac{a}{b}, \\sqrt{x}, \\vec{F}, \\Delta, \\times, \\cdot, \\theta, \\omega
• Units: always use \\text{unit} inside math, e.g. $10\\ \\text{m/s}^2$

TEXT FORMATTING:
• Start with the example number and question clearly
• Use bold inline labels: **Given:**, **To Find:**, **Solution:**, **Step 1:**, **Step 2:** etc.
• NEVER use ## or ### headings for Given/Solution/Steps
• Show all working steps clearly
• End with a boxed or highlighted final answer line"""

        prompt = f"""Format this NCERT solved example with full LaTeX math and clear steps:

{content}

Original question asked: {query}

Formatted solution (LaTeX math, bold step labels):"""

        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ]
        return self.groq.generate(messages, max_tokens=2000)

    def _check_example_ambiguity(self, identifier: str) -> List[Dict]:
        """Check both classes for example identifier. Returns list of found options."""
        options = []
        chapter = identifier.split(".")[0]
        for cls in ["11", "12"]:
            chunk = self._find_example(cls, identifier)
            if chunk:
                ov    = self._get_overview(cls, chapter)
                title = ov["metadata"].get("chapter_title", "") if ov else ""
                options.append({"class": cls, "chapter": chapter,
                                 "chapter_title": title, "identifier": identifier})
        return options

    def _check_table_ambiguity(self, identifier: str) -> List[Dict]:
        """Check both classes for table identifier. Returns list of found options."""
        options = []
        chapter = identifier.split(".")[0]
        for cls in ["11", "12"]:
            chunk = self._find_table(cls, identifier)
            if chunk:
                ov    = self._get_overview(cls, chapter)
                title = ov["metadata"].get("chapter_title", "") if ov else ""
                options.append({"class": cls, "chapter": chapter,
                                 "chapter_title": title, "identifier": identifier})
        return options

    def retrieve_with_class(self, selected_class: str, intent: QueryIntent,
                            k: int = 6, threshold: float = SIMILARITY_THRESHOLD) -> Dict:
        """
        Execute retrieval after user has selected a class from the ambiguity UI.
        Bypasses classify() entirely — uses the stored intent with class injected.
        """
        result = {
            "query":          intent.semantic_query,
            "intent":         intent,
            "display_mode":   "none",
            "chunk":          None,
            "chunks":         [],
            "stats":          None,
            "llm_answer":     None,
            "usage_stats":    {},
            "scores":         [],
            "is_ambiguous":   False,
            "ambiguity_opts": None,
            "threshold_miss": False,
            "best_score":     0.0,
        }

        cls     = selected_class
        chapter = intent.chapter_num

        if intent.raw_type in ("direct_summary", "direct_ptp"):
            chunk = self._find_chunk(cls, chapter, intent.content_type)
            if chunk:
                ans, usage = self.generate_direct_answer(intent.semantic_query, chunk, intent.raw_type)
                result["display_mode"] = "llm" if ans else "direct"
                result["chunk"]        = chunk
                result["llm_answer"]   = ans
                result["usage_stats"]  = usage
                result["chunks"]       = [chunk]
            else:
                result["display_mode"] = "not_found"
            return result

        if intent.raw_type == "direct_exercises":
            chunk = self._find_chunk(cls, chapter, "exercises")
            if chunk:
                result["display_mode"] = "direct"
                result["chunk"]        = chunk
            else:
                result["display_mode"] = "not_found"
            return result

        if intent.raw_type == "direct_example" and intent.identifier:
            chunk = self._find_example(cls, intent.identifier)
            if chunk:
                ans, usage = self.generate_example_answer(intent.semantic_query, chunk)
                result["display_mode"] = "llm" if ans else "direct"
                result["chunk"]        = chunk
                result["llm_answer"]   = ans
                result["usage_stats"]  = usage
                result["chunks"]       = [chunk]
            else:
                result["display_mode"] = "not_found"
            return result

        if intent.raw_type == "direct_table" and intent.identifier:
            chunk = self._find_table(cls, intent.identifier)
            if chunk:
                result["display_mode"] = "direct"
                result["chunk"]        = chunk
            else:
                result["display_mode"] = "not_found"
            return result

        if intent.raw_type == "list_sections" and chapter:
            result["display_mode"] = "sections_list"
            result["stats"]        = self.get_sections_list(cls, chapter)
            return result

        if intent.raw_type == "count_stat" and chapter:
            result["display_mode"] = "stats"
            result["stats"]        = self.get_stats(cls, chapter)
            return result

        result["display_mode"] = "not_found"
        return result

    def retrieve(self, query: str, k: int = 6,
                 threshold: float = SIMILARITY_THRESHOLD) -> Dict:
        result = {
            "query":          query,
            "intent":         None,
            "display_mode":   "none",
            "chunk":          None,
            "chunks":         [],
            "stats":          None,
            "llm_answer":     None,
            "usage_stats":    {},
            "scores":         [],
            "is_ambiguous":   False,
            "ambiguity_opts": None,
            "threshold_miss": False,
            "best_score":     0.0,
        }

        intent = self.classify(query)
        result["intent"] = intent

        # ── BOOK LEVEL ────────────────────────────────────────────────────────
        if intent.is_book_level:
            ans, usage = self.generate_book_answer(intent)
            result["display_mode"] = "book"
            result["llm_answer"]   = ans
            result["usage_stats"]  = usage
            return result

        # ── OUT OF DOMAIN ─────────────────────────────────────────────────────
        if intent.raw_type == "out_of_domain":
            result["display_mode"] = "out_of_domain"
            return result

        # ── AMBIGUITY: chapter-level for direct intents with no class ────────────
        # Must come BEFORE any direct lookup so we can ask user first.
        # Applies to: summary, ptp, exercises, list_sections, count_stat, example, table
        CHAPTER_DIRECT_INTENTS = {
            "direct_summary", "direct_ptp", "direct_exercises",
            "list_sections", "count_stat",
        }
        if intent.raw_type in CHAPTER_DIRECT_INTENTS and intent.chapter_num and not intent.class_num:
            amb = self.check_ambiguity(intent)
            if amb:
                result["is_ambiguous"]   = True
                result["ambiguity_opts"] = amb
                return result

        # ── EXAMPLES: if class already given use it; else always ask ─────────
        if intent.raw_type == "direct_example" and intent.identifier:
            if intent.class_num:
                # Class explicitly given (user selected from ambiguity UI) → fetch directly
                chunk = self._find_example(intent.class_num, intent.identifier)
                if chunk:
                    ans, usage = self.generate_example_answer(query, chunk)
                    result["display_mode"] = "llm" if ans else "direct"
                    result["chunk"]        = chunk
                    result["llm_answer"]   = ans
                    result["usage_stats"]  = usage
                    result["chunks"]       = [chunk]
                    return result
            else:
                # No class given → check both and ask user
                opts = self._check_example_ambiguity(intent.identifier)
                if len(opts) > 1:
                    result["is_ambiguous"]   = True
                    result["ambiguity_opts"] = opts
                    return result
                elif len(opts) == 1:
                    chunk = self._find_example(opts[0]["class"], intent.identifier)
                    if chunk:
                        ans, usage = self.generate_example_answer(query, chunk)
                        result["display_mode"] = "llm" if ans else "direct"
                        result["chunk"]        = chunk
                        result["llm_answer"]   = ans
                        result["usage_stats"]  = usage
                        result["chunks"]       = [chunk]
                        return result

        # ── TABLES: same pattern ──────────────────────────────────────────────
        if intent.raw_type == "direct_table" and intent.identifier:
            if intent.class_num:
                chunk = self._find_table(intent.class_num, intent.identifier)
                if chunk:
                    result["display_mode"] = "direct"
                    result["chunk"]        = chunk
                    return result
            else:
                opts = self._check_table_ambiguity(intent.identifier)
                if len(opts) > 1:
                    result["is_ambiguous"]   = True
                    result["ambiguity_opts"] = opts
                    return result
                elif len(opts) == 1:
                    chunk = self._find_table(opts[0]["class"], intent.identifier)
                    if chunk:
                        result["display_mode"] = "direct"
                        result["chunk"]        = chunk
                        return result

        cls     = intent.class_num or "11"
        chapter = intent.chapter_num

        # ── DIRECT: summary, ptp — use LLM for math/latex formatting ─────────
        if intent.raw_type in ("direct_summary", "direct_ptp"):
            if chapter:
                chunk = self._find_chunk(cls, chapter, intent.content_type)
                if chunk:
                    ans, usage = self.generate_direct_answer(query, chunk, intent.raw_type)
                    result["display_mode"] = "llm" if ans else "direct"
                    result["chunk"]        = chunk
                    result["llm_answer"]   = ans
                    result["usage_stats"]  = usage
                    result["chunks"]       = [chunk]
                    return result

        # ── DIRECT: exercises — NO LLM, display verbatim as numbered list ─────
        if intent.raw_type == "direct_exercises":
            if chapter:
                chunk = self._find_chunk(cls, chapter, "exercises")
                if chunk:
                    result["display_mode"] = "direct"
                    result["chunk"]        = chunk
                    return result

        # ── SECTIONS / COUNT — NO LLM ─────────────────────────────────────────
        if intent.raw_type == "list_sections" and chapter:
            result["display_mode"] = "sections_list"
            result["stats"]        = self.get_sections_list(cls, chapter)
            return result

        if intent.raw_type == "count_stat" and chapter:
            result["display_mode"] = "stats"
            result["stats"]        = self.get_stats(cls, chapter)
            return result

        # ── SEMANTIC SEARCH + LLM ─────────────────────────────────────────────
        # For semantic intents: NO class/chapter filter — search entire database
        filters = {}
        if not intent.raw_type.startswith("semantic_"):
            if intent.class_num:   filters["class"]   = intent.class_num
            if intent.chapter_num: filters["chapter"] = intent.chapter_num

        sem_results = self.semantic_search(
            intent.semantic_query, k=k,
            filters=filters if filters else None,
        )

        if not sem_results:
            result["display_mode"] = "not_found"
            return result

        best_score = sem_results[0]["score"]
        result["best_score"] = best_score
        result["scores"] = [{
            "class":   r["chunk"]["metadata"]["class"],
            "chapter": r["chunk"]["metadata"]["chapter"],
            "type":    r["chunk"]["metadata"]["type"],
            "title":   r["chunk"]["metadata"].get("chapter_title", ""),
            "score":   r["score"],
            "hybrid":  r["hybrid"],
            "sim":     r["similarity"],
        } for r in sem_results]
        result["chunks"] = [r["chunk"] for r in sem_results]

        if best_score < threshold:
            result["display_mode"]   = "threshold_miss"
            result["threshold_miss"] = True
            return result

        if self.groq:
            ans, usage = self.generate_answer(
                query, result["chunks"], allow_knowledge=intent.allow_llm_knowledge
            )
            result["display_mode"] = "llm"
            result["llm_answer"]   = ans
            result["usage_stats"]  = usage
        else:
            result["display_mode"] = "raw_chunks"

        return result


# ══════════════════════════════════════════════════════════════════════════════
# CONTENT FORMATTERS — direct display without LLM
# ══════════════════════════════════════════════════════════════════════════════

def _chunk_header(chunk: Dict, icon: str = "📄") -> str:
    m = chunk["metadata"]
    return (
        f'<div class="chunk-header">'
        f'{icon} <span class="ch-class">Class {m["class"]}</span> '
        f'<span class="ch-sep">·</span> '
        f'<span class="ch-chapter">Chapter {m["chapter"]}: {m.get("chapter_title","")}</span>'
        f'</div>'
    )

def format_text_with_math(text: str) -> str:
    """
    Convert raw NCERT text to display-ready HTML+KaTeX.
    Handles plain equations like "v = u + at" → wraps in $ $
    but only if the line looks mathematical.
    """
    # First escape HTML special chars (except in formula patterns)
    # We work line by line
    lines = text.split("\n")
    out   = []
    in_numbered = False

    for line in lines:
        stripped = line.strip()

        # Empty line
        if not stripped:
            out.append('<br>')
            in_numbered = False
            continue

        # Detect numbered list items like "1. some text"
        num_match = re.match(r'^(\d+)\.\s+(.+)$', stripped)
        if num_match:
            n, content = num_match.group(1), num_match.group(2)
            # Wrap inline equations
            content = _wrap_inline_math(content)
            out.append(
                f'<div class="ncert-list-item">'
                f'<span class="list-num">{n}.</span>'
                f'<span class="list-text">{content}</span>'
                f'</div>'
            )
            in_numbered = True
            continue

        # Detect ALL-CAPS section titles like "SUMMARY", "POINTS TO PONDER"
        if re.match(r'^[A-Z][A-Z\s]+$', stripped) and len(stripped) < 60:
            out.append(f'<div class="ncert-section-title">{stripped}</div>')
            continue

        # Detect display equations (lines that are mostly math)
        if _looks_like_display_eq(stripped):
            eq = stripped.replace("...", r"\ldots")
            out.append(f'<div class="math-display">$${eq}$$</div>')
            continue

        # Regular paragraph text
        content = _wrap_inline_math(stripped)
        out.append(f'<p class="ncert-para">{content}</p>')

    return "\n".join(out)


def _looks_like_display_eq(line: str) -> bool:
    """Heuristic: does this line look like a standalone equation?"""
    math_chars = set("=+−-*/^{}()[]|∫∑∂√αβγδεζηθικλμνξοπρστυφχψωΩ")
    stripped = line.strip()
    if len(stripped) < 3 or len(stripped) > 120:
        return False
    # Already has LaTeX markers
    if stripped.startswith("$$") or stripped.startswith("$"):
        return False
    # Count math indicators
    math_count = sum(1 for c in stripped if c in math_chars)
    ratio      = math_count / len(stripped)
    # Has equals sign and is short
    if "=" in stripped and len(stripped) < 60 and ratio > 0.1:
        return True
    return False


def _wrap_inline_math(text: str) -> str:
    """
    Wrap recognized short physics equations in $ $ for KaTeX rendering.
    Does NOT html.escape — caller handles escaping as needed.
    """
    if "$" in text:
        return text  # already has LaTeX markers, leave alone

    def replacer(m):
        expr = m.group(0)
        # Skip if it reads like a sentence fragment
        if any(w in expr.lower() for w in [" is ", " are ", " the ", " and ", " but ", " for "]):
            return expr
        return f"${expr}$"

    result = re.sub(
        r'(?<![a-zA-Z$])'
        r'([A-Za-z_]\w*\s*[=<>≤≥]\s*[A-Za-z0-9_\+\-\*/\.\^()²³\s]{1,40})'
        r'(?![a-zA-Z$])',
        replacer, text
    )
    return result


def render_direct_chunk(chunk: Dict) -> str:
    """
    Render a direct-display chunk (summary, ptp, exercises, example, table)
    as formatted HTML without LLM involvement.
    """
    m    = chunk["metadata"]
    ctype = m.get("type", "")
    content = chunk["content"]

    icons = {
        "summary":          "📋",
        "points_to_ponder": "💡",
        "exercises":        "✏️",
        "example":          "🧮",
        "table":            "📊",
        "section":          "📖",
    }
    icon = icons.get(ctype, "📄")

    header = _chunk_header(chunk, icon)

    # Format the content
    # Strip leading type header lines (e.g. "SUMMARY\n" "POINTS TO PONDER\n")
    clean = re.sub(r'^(SUMMARY|POINTS TO PONDER|EXERCISES?)\s*\n', '', content, flags=re.IGNORECASE).strip()

    formatted = format_text_with_math(clean)

    type_label = {
        "summary": "Chapter Summary",
        "points_to_ponder": "Points to Ponder",
        "exercises": "Exercise Questions",
        "example": f"Solved Example {m.get('identifier', '')}",
        "table": f"Table {m.get('identifier', '')}",
    }.get(ctype, ctype.replace("_", " ").title())

    return f"""
<div class="direct-card">
  {header}
  <div class="direct-type-label">{icon} {type_label}</div>
  <div class="direct-content">{formatted}</div>
</div>
"""


def render_sections_list(data: Dict) -> str:
    if "error" in data:
        return f'<div class="error-msg">⚠️ {data["error"]}</div>'

    cls, ch, title = data["class"], data["chapter"], data["chapter_title"]
    sections    = data.get("sections", [])
    subsections = data.get("subsections", [])

    rows = []
    for s in sections:
        sid   = s.get("id", "") if isinstance(s, dict) else ""
        stxt  = s.get("title", s) if isinstance(s, dict) else s
        rows.append(f'<tr><td class="sec-id">{sid}</td><td class="sec-title">{stxt}</td><td class="sec-type">Section</td></tr>')
    for s in subsections:
        sid   = s.get("id", "") if isinstance(s, dict) else ""
        stxt  = s.get("title", s) if isinstance(s, dict) else s
        rows.append(f'<tr><td class="sec-id">{sid}</td><td class="sec-title">↳ {stxt}</td><td class="sec-type sub">Sub</td></tr>')

    table_html = f"""
<table class="ncert-table">
  <thead><tr>
    <th>ID</th><th>Title</th><th>Level</th>
  </tr></thead>
  <tbody>{"".join(rows)}</tbody>
</table>""" if rows else "<p>No section data available.</p>"

    return f"""
<div class="stats-card">
  <div class="stats-header">📂 Sections &amp; Subsections — Class {cls} Chapter {ch}</div>
  <div class="stats-subtitle">{title}</div>
  {table_html}
</div>"""


def render_stats(data: Dict) -> str:
    if "error" in data:
        return f'<div class="error-msg">⚠️ {data["error"]}</div>'

    cls, ch, title = data["class"], data["chapter"], data["chapter_title"]
    stats = data.get("stats", {})

    stat_map = [
        ("total_solved_examples",    "🧮", "Solved Examples"),
        ("total_exercise_questions", "✏️", "Exercise Questions"),
        ("total_sections",           "📂", "Sections"),
        ("total_subsections",        "📁", "Subsections"),
        ("total_tables",             "📋", "Tables"),
        ("total_figures",            "🖼️", "Figures"),
        ("total_topics",             "🏷️", "Topics"),
    ]

    cards = "".join(
        f'<div class="stat-pill"><span class="stat-icon">{icon}</span>'
        f'<span class="stat-val">{stats.get(key, "—")}</span>'
        f'<span class="stat-name">{name}</span></div>'
        for key, icon, name in stat_map
    )

    return f"""
<div class="stats-card">
  <div class="stats-header">📊 Chapter Statistics — Class {cls} Chapter {ch}</div>
  <div class="stats-subtitle">{title}</div>
  <div class="stat-pills">{cards}</div>
</div>"""


# ══════════════════════════════════════════════════════════════════════════════
# LLM ANSWER FORMATTER (for math + layout)
# ══════════════════════════════════════════════════════════════════════════════

def format_llm_answer(text: str) -> str:
    """
    Post-process LLM output for clean markdown rendering:
    - Protect $$ blocks from newline mangling
    - Ensure single \n becomes markdown line-break
    - Don't touch LaTeX
    """
    parts  = re.split(r'(\$\$.*?\$\$)', text, flags=re.DOTALL)
    result = []
    for part in parts:
        if part.startswith("$$") and part.endswith("$$"):
            result.append(part)
        else:
            lines = part.split("\n")
            fixed = []
            for i, line in enumerate(lines):
                s = line.rstrip()
                is_block = bool(re.match(
                    r'^\s*$|^\s*#{1,6}\s|^\s*[-*+]\s|^\s*\d+\.\s|^\s*>\s|^\s*\|', s
                ))
                next_empty = (i + 1 < len(lines) and not lines[i+1].strip())
                prev_empty = (i == 0 or not lines[i-1].strip())
                if is_block or next_empty or prev_empty or not s:
                    fixed.append(line)
                else:
                    fixed.append(s + "  ")
            result.append("\n".join(fixed))
    return "".join(result)


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════════

KATEX_CDN = """
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body, {
    delimiters: [
      {left:'$$', right:'$$', display:true},
      {left:'$',  right:'$',  display:false}
    ],
    throwOnError: false
  });"></script>
"""

APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Sora:wght@300;400;600;700&family=Lora:ital,wght@0,400;0,600;1,400&display=swap');

:root {
  --bg:      #0b0e1a;
  --surface: #111524;
  --border:  #1e2340;
  --accent:  #4fc3f7;
  --accent2: #81d4fa;
  --gold:    #ffd54f;
  --green:   #a5d6a7;
  --purple:  #ce93d8;
  --red:     #ef9a9a;
  --text:    #dce3f0;
  --muted:   #6b7498;
  --mono:    'IBM Plex Mono', monospace;
  --sans:    'Sora', sans-serif;
  --serif:   'Lora', serif;
}

.stApp { background: var(--bg) !important; font-family: var(--sans); color: var(--text); }
section[data-testid="stSidebar"] { background: #0d1022 !important; border-right: 1px solid var(--border); }
.stChatMessage { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 14px !important; padding: 1.4rem !important; }
.stChatInput textarea {
  background: var(--surface) !important; color: var(--text) !important;
  border: 1.5px solid var(--border) !important; border-radius: 10px !important;
  font-family: var(--sans) !important; font-size: 15px !important; min-height: 70px !important;
}
.stChatInput textarea:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 3px rgba(79,195,247,.12) !important; }
h1 { font-family: var(--sans) !important; color: var(--accent) !important; font-weight: 700; letter-spacing: -0.5px; }
h4 { font-size: 1rem !important; }
.stButton button {
  background: var(--surface); color: var(--text); border: 1px solid var(--border);
  border-radius: 8px; font-family: var(--sans); transition: all .2s;
}
.stButton button:hover { border-color: var(--accent); color: var(--accent); }
code { background: #1a1e30 !important; color: var(--green) !important; border-radius: 4px; font-family: var(--mono); }
div[data-testid="stMetricValue"] { color: var(--green); font-family: var(--mono); }

/* ── Direct display cards ──────────────────────────────────────────────── */
.direct-card {
  background: #0f1225; border: 1px solid #1e2d4a;
  border-radius: 14px; overflow: hidden; margin: 4px 0;
  width: 100%; box-sizing: border-box;
}
.chunk-header {
  background: #0a1535; padding: 10px 18px; font-family: var(--mono);
  font-size: 0.82em; border-bottom: 1px solid #1e2d4a; color: var(--muted);
  width: 100%; box-sizing: border-box;
}
.ch-class   { color: var(--accent); font-weight: 600; }
.ch-sep     { color: var(--border); margin: 0 6px; }
.ch-chapter { color: var(--gold); }
.direct-type-label {
  padding: 10px 18px 4px; font-size: 0.88em; color: var(--purple);
  font-weight: 600; letter-spacing: .5px;
}
.direct-content {
  padding: 10px 18px 20px; width: 100%; box-sizing: border-box;
  display: block;
}

/* ── NCERT text formatting ─────────────────────────────────────────────── */
.ncert-section-title {
  font-family: var(--sans); font-weight: 700; font-size: 1.0em;
  color: var(--accent); margin: 16px 0 6px; letter-spacing: .3px;
  border-bottom: 1px solid var(--border); padding-bottom: 4px;
  display: block; width: 100%;
}
.ncert-para {
  font-family: var(--serif); font-size: 0.96em; line-height: 1.85;
  color: var(--text); margin: 6px 0; display: block; width: 100%;
}
.ncert-list-item {
  display: flex; gap: 10px; margin: 7px 0; align-items: flex-start;
  font-family: var(--serif); font-size: 0.96em; line-height: 1.8;
  color: var(--text); width: 100%;
}
.list-num {
  font-family: var(--mono); font-weight: 600; color: var(--gold);
  min-width: 26px; flex-shrink: 0; padding-top: 1px;
}
.list-text { flex: 1; min-width: 0; word-break: break-word; }

/* ── Math display ──────────────────────────────────────────────────────── */
.math-display {
  text-align: center; margin: 20px auto; padding: 16px 24px;
  background: #0a1020; border: 1px solid #1e2a45;
  border-left: 3px solid var(--accent); border-radius: 8px;
  overflow-x: auto; max-width: 700px;
  font-size: 1.1em;
}
/* KaTeX display blocks inside Streamlit markdown */
.stChatMessage .katex-display {
  text-align: center !important; margin: 18px auto !important;
  padding: 14px 20px !important; background: #0a1020 !important;
  border: 1px solid #1e2a45 !important; border-left: 3px solid var(--accent) !important;
  border-radius: 8px !important; overflow-x: auto !important;
  max-width: 720px !important;
}
.stChatMessage .katex { font-size: 1.08em !important; }
/* Bold labels inside LLM answer */
.stChatMessage strong { color: var(--gold) !important; }
/* Smaller headings inside chat */
.stChatMessage h3 { font-size: 1.0rem !important; color: var(--purple) !important; border-bottom: 1px solid var(--border); padding-bottom: 3px; }
.stChatMessage h4 { font-size: 0.92rem !important; color: var(--accent2) !important; }

/* ── Stats cards ───────────────────────────────────────────────────────── */
.stats-card {
  background: #0f1225; border: 1px solid var(--border);
  border-radius: 14px; padding: 18px 22px; margin: 4px 0;
}
.stats-header { font-size: 1.05em; font-weight: 700; color: var(--accent); margin-bottom: 4px; }
.stats-subtitle { font-size: 0.88em; color: var(--gold); margin-bottom: 16px; }
.stat-pills { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 10px; }
.stat-pill {
  background: #1a1e30; border: 1px solid var(--border); border-radius: 10px;
  padding: 10px 16px; display: flex; flex-direction: column; align-items: center;
  min-width: 90px;
}
.stat-icon { font-size: 1.3em; }
.stat-val  { font-family: var(--mono); font-size: 1.6em; font-weight: 700; color: var(--green); }
.stat-name { font-size: 0.7em; color: var(--muted); text-align: center; margin-top: 2px; }

/* ── Tables ────────────────────────────────────────────────────────────── */
.ncert-table {
  width: 100%; border-collapse: collapse; font-size: 0.9em; margin-top: 12px;
}
.ncert-table th {
  background: #0a1535; color: var(--accent); padding: 8px 12px;
  text-align: left; border-bottom: 2px solid var(--border);
  font-family: var(--mono); font-size: 0.85em; letter-spacing: .5px;
}
.ncert-table td { padding: 7px 12px; border-bottom: 1px solid var(--border); color: var(--text); }
.ncert-table tr:hover td { background: #151930; }
.sec-id   { font-family: var(--mono); color: var(--accent); font-weight: 600; width: 70px; }
.sec-type { font-size: 0.8em; color: var(--muted); width: 60px; }
.sec-type.sub { color: var(--purple); }

/* ── Threshold miss ────────────────────────────────────────────────────── */
.threshold-card {
  background: #1a1020; border: 1px solid #3a2040; border-radius: 12px;
  padding: 16px 20px; margin: 4px 0;
}
.threshold-title { color: var(--red); font-weight: 600; margin-bottom: 8px; }
.raw-chunk {
  background: #0f1225; border: 1px solid var(--border); border-radius: 8px;
  padding: 12px 16px; margin: 8px 0; font-family: var(--serif);
  font-size: 0.92em; line-height: 1.75; color: var(--text);
}
.raw-chunk-meta { font-family: var(--mono); font-size: 0.78em; color: var(--muted); margin-bottom: 8px; }

/* ── Intent badge ──────────────────────────────────────────────────────── */
.intent-badge {
  display: inline-block; background: #0a1535; border: 1px solid #1e2d4a;
  border-radius: 20px; padding: 2px 12px; font-family: var(--mono);
  font-size: 0.75em; color: var(--accent2); margin-bottom: 10px;
}
.score-badge {
  display: inline-block; background: #0f1a10; border: 1px solid #1e3020;
  border-radius: 20px; padding: 2px 12px; font-family: var(--mono);
  font-size: 0.75em; color: var(--green); margin-left: 8px;
}
.raw-chunk-text {
  font-family: var(--serif); font-size: 0.9em; line-height: 1.7;
  color: var(--text); white-space: pre-wrap; margin-top: 6px;
}
</style>
"""


def init_state():
    defaults = {
        "messages":   [],
        "retriever":  None,
        "_last_dir":  None,
        "pending_clarification": None,
        "show_debug": False,
        "threshold":  SIMILARITY_THRESHOLD,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def show_message(role: str, content: str, metadata: Optional[Dict] = None,
                 show_debug_inline: bool = False):
    """
    Render a chat message.
    show_debug_inline=True only for the CURRENT live response, never for history replay.
    """
    with st.chat_message(role):
        if role == "assistant" and not content.strip().startswith("<"):
            content = format_llm_answer(content)
        st.markdown(content, unsafe_allow_html=True)
        # Debug info ONLY for live response, never replayed from history
        if show_debug_inline and metadata and st.session_state.show_debug:
            _show_debug(metadata)


def _show_debug(res: Dict):
    """Always-visible debug panel: intent classification + retrieved chunks + token usage."""
    intent = res.get("intent")
    scores = res.get("scores", [])
    chunks = res.get("chunks", [])
    chunk  = res.get("chunk")   # single chunk for direct display
    usage  = res.get("usage_stats", {})

    # ── Intent Classification ─────────────────────────────────────────────────
    if intent:
        with st.expander("🎯 Intent Classification", expanded=True):
            cols = st.columns([1, 1, 1, 1])
            cols[0].markdown(f"**Type**\n\n`{intent.raw_type}`")
            cols[1].markdown(f"**Class**\n\n`{intent.class_num or '—'}`")
            cols[2].markdown(f"**Chapter**\n\n`{intent.chapter_num or '—'}`")
            cols[3].markdown(f"**Identifier**\n\n`{intent.identifier or '—'}`")
            st.markdown(f"**Semantic query:** `{intent.semantic_query}`")
            st.markdown(
                f"**LLM knowledge allowed:** `{intent.allow_llm_knowledge}` &nbsp;&nbsp; "
                f"**Best score:** `{res.get('best_score', 0):.3f}`"
            )

    # ── Retrieved Chunks ──────────────────────────────────────────────────────
    all_chunks = chunks if chunks else ([chunk] if chunk else [])
    if all_chunks:
        label = f"📄 Retrieved Chunks ({len(all_chunks)})"
        with st.expander(label, expanded=True):
            for i, c in enumerate(all_chunks):
                m = c["metadata"]
                score_info = ""
                if i < len(scores):
                    s = scores[i]
                    score_info = f" · sim `{s['score']:.3f}` · hybrid `{s['hybrid']:.3f}`"
                st.markdown(
                    f"**Chunk {i+1}:** Class `{m['class']}` · Ch `{m['chapter']}` "
                    f"· *{m.get('chapter_title','')}* · `{m['type']}`{score_info}"
                )
                st.code(c["content"][:600] + ("…" if len(c["content"]) > 600 else ""), language="text")
                if i < len(all_chunks) - 1:
                    st.divider()

    # ── Token Usage ───────────────────────────────────────────────────────────
    if usage.get("total_tokens"):
        with st.expander("📊 Token Usage", expanded=False):
            c1, c2, c3 = st.columns(3)
            c1.metric("Prompt",     usage.get("prompt_tokens", 0))
            c2.metric("Completion", usage.get("completion_tokens", 0))
            c3.metric("Total",      usage.get("total_tokens", 0))


def render_result(res: Dict) -> Tuple[str, bool]:
    """
    Returns (content_string, needs_llm_format).
    content_string: HTML/markdown to display
    needs_llm_format: True when content is raw LLM markdown needing format_llm_answer()
    """
    mode = res.get("display_mode", "none")

    if mode == "direct":
        return render_direct_chunk(res["chunk"]), False

    if mode == "stats":
        return render_stats(res["stats"]), False

    if mode == "sections_list":
        return render_sections_list(res["stats"]), False

    if mode == "llm":
        return res.get("llm_answer") or "", True

    if mode == "book":
        return res.get("llm_answer") or "", True

    if mode == "threshold_miss":
        best   = res.get("best_score", 0)
        chunks = res.get("chunks", [])[:3]
        chunk_html = "".join(
            f'<div class="raw-chunk">'
            f'<div class="raw-chunk-meta">Class {c["metadata"]["class"]} · '
            f'Ch {c["metadata"]["chapter"]} · {c["metadata"].get("chapter_title","")} · '
            f'{c["metadata"]["type"]}</div>'
            f'<div class="raw-chunk-text">{html.escape(c["content"][:700])}'
            f'{"…" if len(c["content"])>700 else ""}</div>'
            f'</div>'
            for c in chunks
        )
        return (
            f'<div class="threshold-card">'
            f'<div class="threshold-title">⚠️ Low relevance (score: {best:.2f})</div>'
            f'<p style="color:#888;font-size:0.85em">Showing closest chunks. Try rephrasing or specify class/chapter.</p>'
            f'{chunk_html}</div>'
        ), False

    if mode == "out_of_domain":
        return (
            '<div class="threshold-card">'
            '<div class="threshold-title">🚫 Out of scope</div>'
            '<p>I can only answer questions related to NCERT Physics for Class 11 and 12.</p>'
            '</div>'
        ), False

    if mode == "not_found":
        return (
            '<div class="threshold-card">'
            '<div class="threshold-title">❓ Not found</div>'
            '<p>No relevant content found. Try rephrasing or specify class and chapter.</p>'
            '</div>'
        ), False

    return '<div class="error-msg">⚠️ Could not process query.</div>', False


# ══════════════════════════════════════════════════════════════════════════════
# APP MAIN
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="NCERT Physics RAG v4",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject CSS + KaTeX
st.markdown(KATEX_CDN + APP_CSS, unsafe_allow_html=True)
init_state()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# ⚛️ NCERT Physics RAG")
    st.markdown("### ⚙️ Settings")

    idx_dir_str = st.text_input("📁 Index folder", value=".", help="Folder with .faiss / .json / .pkl files")
    idx_dir     = Path(idx_dir_str).resolve()
    k_val     = st.slider("📊 Chunks (k)", 2, 12, 6, 1)
    threshold = st.slider("🎯 Similarity threshold", 0.10, 0.70, SIMILARITY_THRESHOLD, 0.01,
                           help="Below this → show raw chunks without LLM call. GTE-small typical range: 0.25–0.60")

    st.markdown("---")
    st.markdown("### 🔑 API Status")
    n_keys = sum(1 for i in list(range(1,4)) + [None]
                 if os.getenv(f"GROQ_API_KEY_{i}" if i else "GROQ_API_KEY"))
    if n_keys:
        st.success(f"✅ {n_keys} key(s) loaded")
        if st.session_state.retriever and st.session_state.retriever.groq:
            g = st.session_state.retriever.groq
            u = g.usage()
            st.info(f"🔑 Active: {g.current_key_label()}")
            if u["total"]:
                st.caption(f"Session: {u['total']:,} tokens ({u['classifier']:,} classifier)")
    else:
        st.error("❌ No Groq API keys found")

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_clarification = None
        st.rerun()

    st.markdown("---")
    st.markdown("""### 💡 Query Examples

**📋 Direct (no LLM needed):**
- Summary of chapter 3 class 11
- Points to ponder chapter 7 class 12
- Exercise questions chapter 6 class 11
- Example 5.3 class 11
- Table 7.1 class 12

**📊 Stats & Lists:**
- How many examples in chapter 3 class 12?
- List sections of chapter 4 class 11
- Topics in chapter 8 class 12

**🧠 Semantic (LLM answers):**
- Explain Newton's second law
- Derive equation of motion
- What is Ohm's law?
- Difference between speed and velocity
- Summary of properties of waves ← (NOT chapter summary!)

**📚 Book-level:**
- List all chapters of class 11
- Topics of class 12
""")

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("⚛️ NCERT Physics — Smart RAG v4")
st.caption("GTE-small embeddings + Hybrid retrieval + LLM classifier + Direct display · v4.0")

# Load retriever
if st.session_state.retriever is None or st.session_state._last_dir != str(idx_dir):
    with st.spinner("🔄 Loading index…"):
        try:
            st.session_state.retriever = NCERTRetriever(idx_dir)
            st.session_state._last_dir = str(idx_dir)
            st.success("✅ Index loaded!")
        except Exception as e:
            st.error(f"❌ {e}")
            st.stop()

retriever: NCERTRetriever = st.session_state.retriever

# Replay history — NO debug info shown for past messages
for msg in st.session_state.messages:
    show_message(msg["role"], msg["content"], show_debug_inline=False)

# ── Ambiguity resolution UI ───────────────────────────────────────────────────
if st.session_state.pending_clarification:
    clarification = st.session_state.pending_clarification
    orig_q   = clarification["original_query"]
    options  = clarification["options"]
    intent   = clarification["intent"]          # stored QueryIntent object
    ident    = options[0].get("identifier") if options else None

    st.markdown(f"""
<div style="background:#111524;border:2px solid #ce93d8;border-radius:14px;padding:1.4rem;margin:1rem 0">
  <h4 style="color:#ce93d8;margin:0 0 .6rem">🤔 Which class do you mean?</h4>
  <p style="color:#ffd54f;margin:0"><b>Query:</b> {html.escape(orig_q)}</p>
</div>""", unsafe_allow_html=True)

    cols = st.columns(len(options))
    for i, opt in enumerate(options):
        with cols[i]:
            ch_title = opt.get("chapter_title", "")
            ex_part  = f"  ·  Example {opt['identifier']}" if opt.get("identifier") else ""
            btn_label = f"📘 Class {opt['class']}\nChapter {opt['chapter']}{ex_part}\n{ch_title}"
            if st.button(btn_label, key=f"amb_{i}", use_container_width=True):
                selected_cls = opt["class"]
                st.session_state.pending_clarification = None
                user_display = orig_q + f"\n\n*→ Class {selected_cls} selected*"
                st.session_state.messages.append({"role": "user", "content": user_display})

                with st.spinner(f"📖 Loading Class {selected_cls} content…"):
                    # Bypass classify() — go straight to content fetch
                    res = retriever.retrieve_with_class(selected_cls, intent, k=k_val, threshold=threshold)

                rc, needs_fmt = render_result(res)
                display_content = format_llm_answer(rc) if needs_fmt else rc
                st.session_state.messages.append({"role": "assistant", "content": display_content, "metadata": None})
                st.rerun()

    if st.button("❌ Cancel", use_container_width=True):
        st.session_state.pending_clarification = None
        st.rerun()

# ── Chat input ─────────────────────────────────────────────────────────────────
query = st.chat_input("💬 Ask about NCERT Physics Class 11 or 12…")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    show_message("user", query)

    with st.chat_message("assistant"):
        with st.spinner("🧠 Classifying & retrieving…"):
            res = retriever.retrieve(query.strip(), k=k_val, threshold=threshold)

        if res["is_ambiguous"]:
            st.session_state.pending_clarification = {
                "original_query": query,
                "options":        res["ambiguity_opts"],
                "intent":         res["intent"],   # store intent for retrieve_with_class
            }
            opts_txt = "\n".join(
                f"- **Class {o['class']}** · Ch {o['chapter']}: {o['chapter_title']}"
                + (f" — Example {o['identifier']}" if o.get('identifier') else "")
                for o in res["ambiguity_opts"]
            )
            rc = f"🤔 **Ambiguous — found in multiple classes. Please select above.**\n\n{opts_txt}"
            st.markdown(rc)
            st.session_state.messages.append({"role": "assistant", "content": rc, "metadata": None})
            st.rerun()

        # Render the result
        rc, needs_fmt = render_result(res)

        if needs_fmt:
            # LLM markdown — apply line-break fix for display only
            display_content = format_llm_answer(rc)
        else:
            display_content = rc

        st.markdown(display_content, unsafe_allow_html=True)

        # ── Debug panel — always shown after every response ───────────────────
        _show_debug(res)

    # Store formatted content for history replay (no debug on replay)
    st.session_state.messages.append({
        "role":     "assistant",
        "content":  display_content,
        "metadata": None,
    })