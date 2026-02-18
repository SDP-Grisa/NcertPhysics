"""
NCERT Physics Retrieval System v3.4 — Book-Level Queries + Domain Guard
=========================================================================
Changes over v3.3:
  • NEW: Book-level queries answered from LLM knowledge (no RAG needed):
      - "give all chapters of class 11/12"
      - "list all topics of class 11 and 12"
      - "what chapters are in class 12?"
      - "list all chapters of both classes"
  • NEW: Strict domain guardrail — LLM refuses non-NCERT Physics 11/12 questions
  • NEW: BOOK_LEVEL QueryType routes directly to knowledge-base prompt
  • Persistent key rotation, compact headings, line-break fixes (from v3.2/v3.3)
"""

import json, os, re, pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field

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


# ─── Color helpers ────────────────────────────────────────────────────────────
def clr(text, color): return f'<span style="color:{color};font-weight:600">{text}</span>'
def cyan(t):   return clr(t, "#00e5ff")
def green(t):  return clr(t, "#69ff47")
def yellow(t): return clr(t, "#ffe066")
def orange(t): return clr(t, "#ffb347")
def pink(t):   return clr(t, "#ff79c6")
def purple(t): return clr(t, "#bd93f9")
def red(t):    return clr(t, "#ff5555")
def white(t):  return clr(t, "#f8f8f2")


# ─── Prompts ──────────────────────────────────────────────────────────────────

# RAG prompt — used when context chunks are retrieved from FAISS
PROMPT_TEMPLATE = """You are an NCERT Class 11 & 12 Physics expert.

**DOMAIN RULE (STRICT):**
You ONLY answer questions related to NCERT Physics textbooks for Class 11 and Class 12.
If the question is about any other subject, class, board, or topic outside NCERT Physics 11/12,
respond with: "I can only answer questions related to NCERT Physics for Class 11 and 12."

**FORMATTING RULES:**
1. Use LaTeX for all math: $$ $$ for display equations, $ $ for inline.
2. Use \\begin{{align*}} ... \\end{{align*}} inside $$ $$ for multi-line derivations.
3. For derivations: start with **Derivation:** in bold, show every step.
4. For examples: show **Given:**, **To Find:**, **Solution:** sections using bold inline labels
   (do NOT use large headings like ### for Given/To Find/Solution — use **bold** only).
5. Use proper LaTeX: \\times, \\frac{{}}{{}}, \\sqrt{{}}, \\vec{{}}, \\Delta, \\theta, \\omega, etc.
6. Use markdown tables where applicable.
7. Keep section labels (Given, To Find, Solution, Step 1, Step 2 …) as **bold inline text**,
   never as markdown headings (##, ###, ####).

**ANSWER RULES:**
- Use the provided context as the primary source.
- You may supplement with your knowledge of NCERT Physics 11/12 if clearly relevant.
- If the answer is not in context and not part of NCERT Physics 11/12, say:
  "This information is not available in the provided context."
- Always cite: **From Class [X] Chapter [Y]: [Title]**

Context from NCERT:
{context}

Question:
{question}

Answer (with proper LaTeX and formatting):
"""

# Book-level / knowledge-base prompt — used WITHOUT RAG context
# for queries like "list all chapters of class 11", "topics of class 12"
BOOK_PROMPT = """You are an NCERT Class 11 & 12 Physics expert with complete knowledge of
both NCERT Physics textbooks (Class 11 and Class 12).

**DOMAIN RULE (STRICT):**
You ONLY answer questions related to NCERT Physics textbooks for Class 11 and Class 12.
If the question is about any other subject, class, board, or topic outside NCERT Physics 11/12,
respond ONLY with: "I can only answer questions related to NCERT Physics for Class 11 and 12."
Do NOT answer general knowledge, chemistry, biology, math, or any non-physics questions.

**FORMATTING RULES:**
1. Use markdown tables for structured data (chapter lists, topic lists).
2. Use proper chapter numbering as it appears in NCERT (e.g. Chapter 1: Physical World).
3. For topic/section lists: use numbered lists grouped by chapter.
4. Use LaTeX for any math: $ $ for inline, $$ $$ for display.
5. Keep headings compact — use **bold** or small headings, not large ## headings for each item.

**ANSWER RULES:**
- Answer from your training knowledge of NCERT Physics Class 11 and 12 textbooks.
- Be accurate and complete — list ALL chapters or topics as requested.
- Do NOT mention or reference context chunks; answer directly from your knowledge.
- Format chapter lists as a markdown table: | No. | Chapter Title |
- Format topic/section lists grouped under each chapter heading.

Question:
{question}

Answer:
"""


# ─── Query intent ─────────────────────────────────────────────────────────────
class QueryType(Enum):
    SEMANTIC        = "semantic"
    EXACT_MATCH     = "exact_match"
    METADATA        = "metadata"
    CHAPTER_CONTENT = "chapter_content"
    HYBRID          = "hybrid"
    BOOK_LEVEL      = "book_level"   # class-wide queries answered from LLM knowledge


@dataclass
class QueryIntent:
    query_type:     QueryType
    original_query: str
    class_num:      Optional[str]  = None
    chapter_num:    Optional[str]  = None
    content_type:   Optional[str]  = None
    identifier:     Optional[str]  = None
    semantic_query: Optional[str]  = None
    filters:        Dict[str, Any] = field(default_factory=dict)
    is_ambiguous:   bool           = False
    stat_target:    Optional[str]  = None
    is_list_query:  bool           = False
    book_level_target: Optional[str] = None  # "chapters" | "topics" | "both_classes" etc.


# ─── Stat configuration ────────────────────────────────────────────────────────
STAT_CONFIG: Dict[str, Dict] = {
    "examples":    {"stats_key": "total_solved_examples",    "chunk_type": "example",    "friendly": "Solved Examples"},
    "tables":      {"stats_key": "total_tables",             "chunk_type": "table",      "friendly": "Tables"},
    "sections":    {"stats_key": "total_sections",           "chunk_type": "section",    "friendly": "Sections"},
    "subsections": {"stats_key": "total_subsections",        "chunk_type": "subsection", "friendly": "Subsections"},
    "figures":     {"stats_key": "total_figures",            "chunk_type": None,         "friendly": "Figures"},
    "topics":      {"stats_key": "total_topics",             "chunk_type": None,         "friendly": "Topics"},
    "exercises":   {"stats_key": "total_exercise_questions", "chunk_type": "exercises",  "friendly": "Exercise Questions"},
}

LIST_TRIGGERS = [
    r"\blist\b", r"\bshow\b", r"\bgive me all\b", r"\bwhat are all\b",
    r"\bgive all\b", r"\bshow all\b", r"\ball examples?\b",
    r"\ball questions?\b", r"\ball exercises?\b", r"\ball sections?\b",
    r"\ball topics?\b", r"\bgive.*topics?\b", r"\bshow.*topics?\b",
    r"\blist.*topics?\b", r"\bwhat.*topics?\b",
]


# ─── Exercise question parser ─────────────────────────────────────────────────
def _best_exercise_chunk(exercise_chunks: List[Dict], chapter: str) -> Optional[Dict]:
    if not exercise_chunks:
        return None
    if len(exercise_chunks) == 1:
        return exercise_chunks[0]
    pat = re.compile(rf"^\s*{re.escape(chapter)}\.\d+\s", re.MULTILINE)
    return max(exercise_chunks, key=lambda c: len(pat.findall(c["content"])))


def parse_exercise_questions(all_chunks: List[Dict], cls: str, chapter: str) -> List[Dict]:
    ex_chunks = [
        c for c in all_chunks
        if c["metadata"]["class"] == cls
        and c["metadata"]["chapter"] == chapter
        and c["metadata"]["type"] == "exercises"
    ]
    best = _best_exercise_chunk(ex_chunks, chapter)
    if not best:
        return []

    content = best["content"]
    m = re.search(r"EXERCISES?\s*\n", content, re.IGNORECASE)
    if m:
        content = content[m.end():]

    ch_esc  = re.escape(chapter)
    pattern = rf"^\s*({ch_esc}\.(\d+))\s+(.+?)(?=^\s*{ch_esc}\.\d+\s|\Z)"
    matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)

    questions = []
    for full_num, num_str, text in matches:
        clean = re.sub(r"\s+", " ", text.strip())
        questions.append({"num": full_num, "n": int(num_str), "text": clean})

    questions.sort(key=lambda x: x["n"])
    return questions


def parse_examples_list(all_chunks: List[Dict], cls: str, chapter: str) -> List[Dict]:
    eg_chunks = [
        c for c in all_chunks
        if c["metadata"]["class"] == cls
        and c["metadata"]["chapter"] == chapter
        and c["metadata"]["type"] == "example"
    ]
    results = []
    for c in eg_chunks:
        content = c["content"]
        q_part = re.split(r"\nAnswer\b", content, maxsplit=1, flags=re.IGNORECASE)[0]
        q_clean = re.sub(r"\s+", " ", q_part.strip())
        results.append({
            "num":      c["metadata"]["identifier"],
            "question": q_clean[:300],
        })
    results.sort(key=lambda x: [int(p) for p in x["num"].split(".")])
    return results


# ─── Query parser ─────────────────────────────────────────────────────────────
class QueryParser:
    ITEM_PATTERNS = {
        "example": [r"example\s+(\d+)\.(\d+)", r"ex\.?\s+(\d+)\.(\d+)"],
        "table":   [r"table\s+(\d+)\.(\d+)"],
    }
    SECTION_PATTERNS = {
        "summary":          [r"summar(?:y|ies)\s+(?:of\s+)?(?:chapter\s+)?(\d+)",
                             r"chapter\s+(\d+)\s+summar(?:y|ies)"],
        "exercises":        [r"exercises?\s+(?:of\s+)?(?:chapter\s+)?(\d+)",
                             r"chapter\s+(\d+)\s+exercises?"],
        "points_to_ponder": [r"points?\s+to\s+ponder\s+(?:of\s+|in\s+|from\s+)?(?:chapter\s+)?(\d+)",
                             r"chapter\s+(\d+)\s+points?\s+to\s+ponder"],
    }
    STAT_ALIASES = {
        "examples":    ["solved example", "solved examples", "worked example",
                        "numerical", "numericals", "example", "examples"],
        "tables":      ["table", "tables"],
        "sections":    ["section", "sections"],
        "subsections": ["subsection", "subsections", "sub-section", "sub-sections"],
        "figures":     ["figure", "figures", "diagram", "diagrams", "fig", "figs"],
        "topics":      ["topic", "topics"],
        "exercises":   ["exercise question", "exercise questions", "exercise",
                        "exercises", "problem", "problems", "question", "questions"],
    }
    STAT_TRIGGERS = [
        r"how many", r"count(?: of)?", r"number of",
        r"total(?: number of| count of)?",
        r"list(?: all| of)?", r"what are(?: all| the)?",
        r"give(?: me)?(?: all| the)?", r"show(?: all| me)?",
        r"tell me(?: the)?(?: number of| how many)?",
    ]

    @classmethod
    def _extract_class_and_chapter(cls, q: str):
        cl = re.search(r"class\s+(\d+)", q)
        ch = re.search(r"chapter\s+(\d+)", q)
        return (cl.group(1) if cl else None), (ch.group(1) if ch else None)

    @classmethod
    def _detect_stat_target(cls, q: str) -> Optional[str]:
        for trigger in cls.STAT_TRIGGERS:
            if re.search(trigger, q):
                for target, aliases in cls.STAT_ALIASES.items():
                    for alias in sorted(aliases, key=len, reverse=True):
                        if re.search(r"\b" + re.escape(alias) + r"\b", q):
                            return target
        return None

    @classmethod
    def _is_list_query(cls, q: str) -> bool:
        return any(re.search(t, q) for t in LIST_TRIGGERS)

    # Book-level patterns (class-wide, no specific chapter)
    # Only accept class 11 or 12 for book-level queries
    VALID_CLASSES = {"11", "12"}

    BOOK_LEVEL_PATTERNS = [
        # chapters of a class
        (r"(?:all\s+)?chapters?\s+(?:of\s+|in\s+|for\s+)?class\s+(\d+)",                "chapters"),
        (r"(?:list|give|show|what(?:\s+are)?(?:\s+the)?)\s+(?:all\s+)?chapters?\s+(?:of\s+|in\s+)?class\s+(\d+)", "chapters"),
        (r"(?:list|give|show)\s+(?:all\s+)?chapters?\s+(?:of\s+)?(?:class\s+)?(?:11|12|both|11\s+and\s+12|12\s+and\s+11)", "chapters"),
        # what chapters are in class X
        (r"what\s+chapters?\s+(?:are\s+)?(?:in|of)\s+class\s+(\d+)",                     "chapters"),
        (r"what\s+(?:are\s+(?:the\s+)?)?chapters?\s+(?:in|of)\s+class\s+(\d+)",         "chapters"),
        # topics/sections of a class
        (r"(?:all\s+)?topics?\s+(?:of\s+|in\s+|for\s+)?class\s+(\d+)",                  "topics"),
        (r"(?:list|give|show|what(?:\s+are)?(?:\s+the)?)\s+(?:all\s+)?topics?\s+(?:of\s+|in\s+)?class\s+(\d+)", "topics"),
        # both classes
        (r"chapters?\s+(?:of\s+)?(?:both\s+)?(?:class\s+)?11\s+and\s+(?:class\s+)?12",  "chapters"),
        (r"chapters?\s+(?:of\s+)?(?:both\s+)?(?:class\s+)?12\s+and\s+(?:class\s+)?11",  "chapters"),
        (r"topics?\s+(?:of\s+)?(?:both\s+)?(?:class\s+)?11\s+and\s+(?:class\s+)?12",    "topics"),
        (r"topics?\s+(?:of\s+)?(?:both\s+)?(?:class\s+)?12\s+and\s+(?:class\s+)?11",    "topics"),
        # plain "list all chapters / topics" with no class mentioned
        (r"(?:list|give|show)\s+all\s+chapters?$",                                            "chapters"),
        (r"(?:list|give|show)\s+all\s+topics?$",                                              "topics"),
    ]

    @classmethod
    def _detect_book_level(cls, q: str) -> Optional[Tuple[str, Optional[str]]]:
        """
        Returns (target, class_num_or_None) if this is a book-level query.
        target is 'chapters' or 'topics'.
        class_num is '11', '12', 'both', or None (= both implied).
        """
        # Detect "both classes" signals
        both = bool(re.search(
            r"(?:both|11\s+and\s+(?:class\s+)?12|12\s+and\s+(?:class\s+)?11"
            r"|class\s+11\s+and\s+12|class\s+12\s+and\s+11|11\s*&\s*12)",
            q
        ))
        for pat, target in cls.BOOK_LEVEL_PATTERNS:
            m = re.search(pat, q)
            if m:
                try:
                    cls_found = m.group(1)
                except IndexError:
                    cls_found = None
                # Guard: only treat as book-level for class 11 or 12
                # (class 9, 10, etc. should fall through to semantic/other paths)
                if cls_found and cls_found not in cls.VALID_CLASSES:
                    continue
                if both:
                    cls_found = "both"
                return target, cls_found
        return None

    @classmethod
    def parse_query(cls, query: str) -> QueryIntent:
        q = query.lower().strip()
        explicit_class, chapter_num = cls._extract_class_and_chapter(q)

        # ── Book-level query (must be checked BEFORE chapter-specific patterns) ──
        book = cls._detect_book_level(q)
        if book:
            target, bl_cls = book
            # Only treat as book-level if NO specific chapter mentioned
            if not chapter_num:
                return QueryIntent(
                    QueryType.BOOK_LEVEL, query,
                    class_num=bl_cls or explicit_class,
                    book_level_target=target,
                )

        for cat, patterns in cls.ITEM_PATTERNS.items():
            for pat in patterns:
                m = re.search(pat, q)
                if m:
                    ch = m.group(1)
                    return QueryIntent(
                        QueryType.EXACT_MATCH, query,
                        class_num=explicit_class, chapter_num=ch,
                        content_type=cat, identifier=f"{m.group(1)}.{m.group(2)}",
                        is_ambiguous=(explicit_class is None),
                    )

        for cat, patterns in cls.SECTION_PATTERNS.items():
            for pat in patterns:
                m = re.search(pat, q)
                if m:
                    ch = m.group(1)
                    if cat == "exercises" and cls._is_list_query(q):
                        break
                    return QueryIntent(
                        QueryType.CHAPTER_CONTENT, query,
                        class_num=explicit_class, chapter_num=ch,
                        content_type=cat, is_ambiguous=(explicit_class is None),
                    )

        stat_target = cls._detect_stat_target(q)
        if stat_target and chapter_num:
            return QueryIntent(
                QueryType.METADATA, query,
                class_num=explicit_class, chapter_num=chapter_num,
                stat_target=stat_target,
                is_ambiguous=(explicit_class is None),
                is_list_query=cls._is_list_query(q),
            )

        if chapter_num or explicit_class:
            return QueryIntent(
                QueryType.HYBRID, query,
                class_num=explicit_class, chapter_num=chapter_num,
                semantic_query=query,
                is_ambiguous=(chapter_num is not None and explicit_class is None),
            )

        return QueryIntent(QueryType.SEMANTIC, query, semantic_query=query)


# ─── Groq API manager ─────────────────────────────────────────────────────────
class GroqAPIManager:
    """
    v3.2: Persistent key rotation.
    - self.idx is NEVER reset to 0 after a successful call.
    - Only advances forward when a rate-limit/429 error is hit on the current key.
    - If all keys are exhausted, returns None (no silent retry on key 0).
    """

    def __init__(self):
        self.api_keys = self._load_keys()
        self.idx = 0          # current key index — persists across calls
        self.clients = [Groq(api_key=k) for k in self.api_keys] if self.api_keys else []
        self.total_tokens = self.total_prompt = self.total_completion = 0

    def _load_keys(self):
        keys = []
        for i in range(1, 4):
            k = os.getenv(f"GROQ_API_KEY_{i}")
            if k:
                keys.append(k)
        d = os.getenv("GROQ_API_KEY")
        if d and d not in keys:
            keys.insert(0, d)
        return keys

    def get_client(self):
        return self.clients[self.idx] if self.clients and self.idx < len(self.clients) else None

    def _advance_key(self) -> bool:
        """Try to move to the next available key. Returns True if successful."""
        if self.idx < len(self.clients) - 1:
            self.idx += 1
            return True
        return False

    def generate(self, messages, temperature=0.1, max_tokens=2048):
        """
        Attempt generation starting from the CURRENT key (self.idx).
        On rate-limit: advance key and retry — do NOT wrap back to 0.
        """
        attempts = len(self.clients) - self.idx   # only try remaining keys
        for attempt in range(attempts):
            client = self.get_client()
            if client is None:
                break
            try:
                r = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                u = r.usage
                self.total_prompt     += u.prompt_tokens
                self.total_completion += u.completion_tokens
                self.total_tokens     += u.total_tokens
                # ✅ Do NOT reset self.idx here — key stays current
                return r.choices[0].message.content.strip(), {
                    "prompt_tokens":     u.prompt_tokens,
                    "completion_tokens": u.completion_tokens,
                    "total_tokens":      u.total_tokens,
                }
            except Exception as e:
                err_str = str(e).lower()
                if "rate" in err_str or "429" in err_str:
                    if self._advance_key():
                        st.warning(
                            f"⚠️ Key {self.idx} rate-limited — switching to key {self.idx + 1}"
                        )
                        continue
                    else:
                        st.error("❌ All API keys are rate-limited. Please wait and retry.")
                        return None, {}
                st.error(f"❌ API error: {e}")
                return None, {}
        st.error("❌ No available API keys.")
        return None, {}

    def usage(self):
        return {
            "total":      self.total_tokens,
            "prompt":     self.total_prompt,
            "completion": self.total_completion,
        }

    def current_key_label(self) -> str:
        return f"Key {self.idx + 1} of {len(self.clients)}"


# ─── Main retriever ───────────────────────────────────────────────────────────
class NCERTRetriever:
    def __init__(self, index_dir: Path):
        self.index_dir = index_dir
        self.model = self.index = self.chunks = self.meta_idx = self.groq = None
        self._load()

    def _load(self):
        try:
            cfg = json.loads((self.index_dir / "ncert_physics_config_gte.json").read_text())
            self.model  = SentenceTransformer(cfg["model_name"])
            self.index  = faiss.read_index(str(self.index_dir / "ncert_physics_gte.faiss"))
            self.chunks = json.loads((self.index_dir / "ncert_physics_chunks_indexed_gte.json").read_text(encoding="utf-8"))
            with open(self.index_dir / "ncert_physics_metadata_gte.pkl", "rb") as f:
                self.meta_idx = pickle.load(f)
            if GROQ_AVAILABLE:
                self.groq = GroqAPIManager()
                if not self.groq.api_keys:
                    st.warning("⚠️ No Groq API keys — AI disabled.")
        except Exception as e:
            st.error(f"❌ Failed to load index: {e}")
            st.stop()

    def _get_overview(self, cls: str, chapter: str) -> Optional[Dict]:
        key = f"{cls}-{chapter}"
        idx = self.meta_idx.get("chapter_overviews", {}).get(key)
        return self.chunks[idx] if idx is not None else None

    def _count_chunks_by_type(self, cls: str, chapter: str, chunk_type: str) -> List[Dict]:
        return [
            c for c in self.chunks
            if c["metadata"].get("class") == cls
            and c["metadata"].get("chapter") == chapter
            and c["metadata"].get("type") == chunk_type
        ]

    def handle_metadata_query(self, intent: QueryIntent) -> Dict:
        cls     = intent.class_num or "11"
        chapter = intent.chapter_num

        overview = self._get_overview(cls, chapter)
        if overview is None:
            return {"error": f"Chapter {chapter} (Class {cls}) not found in index."}

        meta    = overview["metadata"]
        stats   = meta.get("stats", {})
        title   = meta.get("chapter_title", "")
        target  = intent.stat_target
        is_list = intent.is_list_query

        result: Dict[str, Any] = {
            "class":         cls,
            "chapter":       chapter,
            "chapter_title": title,
            "stat_target":   target,
            "overview_meta": meta,
            "stats":         stats,
            "is_list":       is_list,
            "source":        "overview_stats",
            "items":         [],
            "sections":      meta.get("sections", []),
            "subsections":   meta.get("subsections", []),
            "tables":        meta.get("tables", []),
            "figures":       meta.get("figures", []),
            "topics":        meta.get("topics", []),
        }

        if target and target in STAT_CONFIG:
            cfg       = STAT_CONFIG[target]
            stats_key = cfg["stats_key"]
            friendly  = cfg["friendly"]

            count = stats.get(stats_key)
            if count is None:
                chunk_type = cfg["chunk_type"]
                count = len(self._count_chunks_by_type(cls, chapter, chunk_type)) if chunk_type else 0
                result["source"] = "chunk_count_fallback"

            result["friendly_name"] = friendly
            result["count"]         = count

            if is_list:
                if target == "exercises":
                    result["items"] = parse_exercise_questions(self.chunks, cls, chapter)
                elif target == "examples":
                    result["items"] = parse_examples_list(self.chunks, cls, chapter)
                elif target == "sections":
                    result["items"] = [
                        {"num": s["id"], "text": s["title"]}
                        for s in meta.get("sections", [])
                    ]
                elif target == "subsections":
                    result["items"] = [
                        {"num": s["id"], "text": s["title"]}
                        for s in meta.get("subsections", [])
                    ]
                elif target == "tables":
                    result["items"] = [{"num": tid, "text": ""} for tid in meta.get("tables", [])]
                elif target == "figures":
                    result["items"] = [{"num": fid, "text": ""} for fid in meta.get("figures", [])]
                elif target == "topics":
                    raw_topics = meta.get("topics", [])

                    def _normalise(raw):
                        out = []
                        for i, t in enumerate(raw, 1):
                            if isinstance(t, dict):
                                out.append({
                                    "num":  t.get("id", str(i)),
                                    "text": t.get("title", t.get("name", str(t))),
                                })
                            else:
                                out.append({"num": str(i), "text": str(t)})
                        return out

                    if raw_topics:
                        result["items"] = _normalise(raw_topics)
                    else:
                        # Topics not stored under 'topics' key in this index —
                        # fall back to sections + subsections (same content hierarchy).
                        sections    = meta.get("sections", [])
                        subsections = meta.get("subsections", [])
                        items_out   = []
                        if sections:
                            for s in sections:
                                if isinstance(s, dict):
                                    items_out.append({"num": s["id"], "text": s["title"]})
                                else:
                                    items_out.append({"num": "", "text": str(s)})
                            for s in subsections:
                                if isinstance(s, dict):
                                    items_out.append({"num": s["id"], "text": "↳ " + s["title"]})
                                else:
                                    items_out.append({"num": "", "text": "↳ " + str(s)})
                            result["items"] = items_out
                            result["topics_source"] = "sections_fallback"
                        elif subsections:
                            result["items"] = _normalise(subsections)
                            result["topics_source"] = "subsections_fallback"
                        else:
                            result["topics_source"] = "not_found"
        else:
            result["friendly_name"] = "All Statistics"
            result["count"]         = None

        return result

    def exact_match_search(self, intent: QueryIntent) -> Optional[Dict]:
        cls = intent.class_num or "11"
        if intent.content_type == "example":
            key = f"{cls}-{intent.chapter_num}-example-{intent.identifier}"
            idx = self.meta_idx.get("examples", {}).get(key)
        elif intent.content_type == "table":
            key = f"{cls}-{intent.chapter_num}-table-{intent.identifier}"
            idx = self.meta_idx.get("tables", {}).get(key)
        else:
            return None
        return self.chunks[idx] if idx is not None else None

    def chapter_content_search(self, intent: QueryIntent) -> Optional[Dict]:
        cls = intent.class_num or "11"
        ch  = f"{cls}-{intent.chapter_num}"
        key_map = {
            "summary":          ("summaries",       f"{ch}-summary"),
            "exercises":        ("exercises",        f"{ch}-exercises"),
            "points_to_ponder": ("points_to_ponder", f"{ch}-points"),
        }
        cat, key = key_map.get(intent.content_type, (None, None))
        if not cat:
            return None
        idx = self.meta_idx.get(cat, {}).get(key)
        return self.chunks[idx] if idx is not None else None

    def semantic_search(self, query: str, k: int = 6,
                        filters: Optional[Dict] = None) -> List[Dict]:
        emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(emb)
        scores, indices = self.index.search(emb, k * 3)
        results = []
        for sc, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            if filters and not all(
                chunk["metadata"].get(fk) == fv for fk, fv in filters.items()
            ):
                continue
            results.append({
                "chunk": chunk,
                "score": float(sc),
                "similarity_percentage": round(float(sc) * 100, 2),
            })
            if len(results) >= k:
                break
        return results

    def check_ambiguity(self, intent: QueryIntent) -> Optional[List[Dict]]:
        if not intent.is_ambiguous:
            return None
        options = []
        for cls in ["11", "12"]:
            overview = self._get_overview(cls, intent.chapter_num)
            if overview:
                options.append({
                    "class":         cls,
                    "chapter":       intent.chapter_num,
                    "chapter_title": overview["metadata"].get("chapter_title", ""),
                })
        return options if len(options) > 1 else None

    def generate_answer(self, query: str, chunks: List[Dict]) -> Tuple[Optional[str], Dict]:
        if not self.groq or not chunks:
            return None, {}
        context = "\n\n".join(
            f"**Class {c['metadata']['class']} | Ch {c['metadata']['chapter']} | "
            f"{c['metadata'].get('chapter_title', '')}\n"
            f"*Type: {c['metadata']['type']}*\n\n{c['content']}\n\n{'─' * 60}"
            for c in chunks
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are NCERT Physics Class 11 & 12 expert. Use LaTeX for all math. "
                    "ONLY answer questions about NCERT Physics Class 11 and 12. "
                    "Refuse anything outside this domain politely. "
                    "For examples/solutions use **bold inline labels** like **Given:**, "
                    "**To Find:**, **Solution:**, **Step 1:** — NEVER use markdown headings "
                    "(##, ###) for these labels."
                ),
            },
            {"role": "user", "content": PROMPT_TEMPLATE.format(context=context, question=query)},
        ]
        return self.groq.generate(messages)

    def generate_book_answer(self, intent: QueryIntent) -> Tuple[Optional[str], Dict]:
        """
        Answer book-level / class-wide queries directly from LLM knowledge.
        No RAG context injected — LLM uses its training knowledge of NCERT Physics.
        """
        if not self.groq:
            return None, {}
        cls_label = intent.class_num
        target    = intent.book_level_target or "chapters"
        q         = intent.original_query

        # Build an explicit, well-scoped question if the original is vague
        if cls_label == "both" or cls_label is None:
            scope = "Class 11 and Class 12"
        elif cls_label in ("11", "12"):
            scope = f"Class {cls_label}"
        else:
            scope = f"Class {cls_label}"

        if target == "chapters":
            explicit_q = (
                f"{q}\n\n"
                f"Please list ALL chapters of NCERT Physics {scope} in a markdown table "
                f"with columns: Chapter No. | Chapter Title. "
                f"Be complete and accurate as per the actual NCERT textbook."
            )
        else:  # topics
            explicit_q = (
                f"{q}\n\n"
                f"Please list all major topics/sections of NCERT Physics {scope}, "
                f"grouped by chapter. Use the format:\n"
                f"**Chapter N: Title**\n- Topic 1\n- Topic 2\n...\n"
                f"Be complete and accurate as per the actual NCERT textbook."
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert on NCERT Physics textbooks for Class 11 and Class 12. "
                    "You have complete and accurate knowledge of the chapter list and topics "
                    "in both NCERT Physics Part 1 and Part 2 for Class 11, and "
                    "NCERT Physics Part 1 and Part 2 for Class 12. "
                    "ONLY answer questions about NCERT Physics Class 11 and 12. "
                    "Refuse anything outside this domain with: "
                    "'I can only answer questions related to NCERT Physics for Class 11 and 12.' "
                    "Never answer chemistry, biology, math, history, or any other subject questions."
                ),
            },
            {"role": "user", "content": BOOK_PROMPT.format(question=explicit_q)},
        ]
        return self.groq.generate(messages, max_tokens=3000)

    def retrieve(self, query: str, k: int = 6) -> Dict:
        intent = QueryParser.parse_query(query)
        result = {
            "query":             query,
            "intent_type":       intent.query_type.value,
            "raw_chunks":        [],
            "statistics":        None,
            "llm_answer":        None,
            "is_ambiguous":      False,
            "ambiguity_options": None,
            "usage_stats":       {},
            "retrieval_scores":  [],
        }

        amb = self.check_ambiguity(intent)
        if amb:
            result["is_ambiguous"]      = True
            result["ambiguity_options"] = amb
            return result

        # Book-level queries are handled after retrieve() returns (see end of method)
        if intent.query_type == QueryType.BOOK_LEVEL:
            pass   # handled below after raw_chunks block

        if intent.query_type == QueryType.EXACT_MATCH:
            chunk = self.exact_match_search(intent)
            if chunk:
                result["raw_chunks"] = [chunk]

        elif intent.query_type == QueryType.CHAPTER_CONTENT:
            chunk = self.chapter_content_search(intent)
            if chunk:
                result["raw_chunks"] = [chunk]

        elif intent.query_type == QueryType.METADATA:
            result["statistics"] = self.handle_metadata_query(intent)

        else:
            filters = {}
            if intent.class_num:   filters["class"]   = intent.class_num
            if intent.chapter_num: filters["chapter"] = intent.chapter_num
            sr = self.semantic_search(intent.semantic_query or query, k=k, filters=filters)
            result["raw_chunks"] = [r["chunk"] for r in sr]
            result["retrieval_scores"] = [
                {
                    "class":      r["chunk"]["metadata"]["class"],
                    "chapter":    r["chunk"]["metadata"]["chapter"],
                    "type":       r["chunk"]["metadata"]["type"],
                    "score":      r["score"],
                    "similarity": r["similarity_percentage"],
                }
                for r in sr
            ]

        # ── Book-level: answered from LLM knowledge, no RAG context needed ──
        if intent.query_type == QueryType.BOOK_LEVEL:
            ans, usage = self.generate_book_answer(intent)
            result["llm_answer"]   = ans
            result["usage_stats"]  = usage
            result["intent_type"]  = "book_level"
            return result

        if result["raw_chunks"] and self.groq:
            ans, usage = self.generate_answer(query, result["raw_chunks"])
            result["llm_answer"] = ans
            result["usage_stats"] = usage

        return result


# ─── LLM answer formatter ─────────────────────────────────────────────────────
def _format_llm_answer(text: str) -> str:
    """
    Fix the 'everything on one line' problem.

    Streamlit's st.markdown renders `\n` as a space unless there are two
    newlines (paragraph break) or a trailing double-space before `\n`.
    LaTeX blocks ($$ ... $$) must NOT be touched.

    Strategy:
      1. Split the text into LaTeX-block segments and plain-text segments.
      2. In plain-text segments:
         - Double-newlines stay as-is (paragraph breaks work fine).
         - Single newlines → two spaces + newline  (markdown line-break)
           BUT only when the line isn't already a markdown block element
           (list item, heading, horizontal rule, empty line).
      3. Reassemble and return.
    """
    # Split around display-math blocks preserving them intact
    parts = re.split(r'(\$\$.*?\$\$)', text, flags=re.DOTALL)

    result_parts = []
    for part in parts:
        if part.startswith('$$') and part.endswith('$$'):
            # LaTeX block — leave completely untouched
            result_parts.append(part)
        else:
            # Process plain-text / inline-math segment
            lines = part.split('\n')
            new_lines = []
            for i, line in enumerate(lines):
                stripped = line.rstrip()
                # Detect lines that are already markdown block elements
                is_block = bool(re.match(
                    r'^\s*$'           # empty
                    r'|^\s{0,3}#{1,6}\s'  # heading
                    r'|^\s*[-*+]\s'    # unordered list
                    r'|^\s*\d+\.\s'    # ordered list
                    r'|^\s*[-*_]{3,}\s*$'  # hr
                    r'|^\s*>\s'        # blockquote
                    r'|^\s*\|',        # table row
                    stripped
                ))
                # Next line empty? current line ends a paragraph — no trailing spaces needed
                next_empty = (i + 1 < len(lines) and lines[i + 1].strip() == '')
                prev_empty = (i == 0 or lines[i - 1].strip() == '')

                if is_block or next_empty or prev_empty or stripped == '':
                    new_lines.append(line)
                else:
                    # Single newline in the middle of prose/steps → force line break
                    new_lines.append(stripped + '  ')
            result_parts.append('\n'.join(new_lines))

    return ''.join(result_parts)


# ─── Renderer ─────────────────────────────────────────────────────────────────

# Shared compact styles used across list renders
_HEADER_STYLE  = "font-size:1.05em;color:#00e5ff;font-weight:700;margin:0 0 4px 0"
_BADGE_STYLE   = (
    "display:inline-block;background:#1e2040;border:1px solid #2d2f5a;"
    "border-radius:6px;padding:1px 8px;font-size:0.82em;"
    "color:#ffb347;font-weight:700;margin-right:6px;font-family:'JetBrains Mono',monospace"
)
_ITEM_STYLE    = "color:#f8f8f2;margin:6px 0;line-height:1.6;font-size:0.95em"
_SUBHEAD_STYLE = "font-size:0.88em;color:#bd93f9;font-weight:600;margin:12px 0 2px 0"


def render_statistics(stats: Dict) -> str:
    """
    Three render modes:
      LIST  : full numbered/bulleted list of items  (compact, LaTeX-friendly)
      COUNT : big number badge + tip to ask for list
      FULL  : all-stats table (no specific target)
    """
    if "error" in stats:
        return f'<span style="color:#ff5555;font-weight:600">⚠️ {stats["error"]}</span>'

    cls       = stats["class"]
    chapter   = stats["chapter"]
    title     = stats["chapter_title"]
    target    = stats.get("stat_target")
    count     = stats.get("count")
    friendly  = stats.get("friendly_name", "Statistics")
    all_stats = stats.get("stats", {})
    is_list   = stats.get("is_list", False)
    items     = stats.get("items", [])
    source    = stats.get("source", "overview_stats")

    lines = []

    # ── Compact header (h4 instead of h3, tighter margin) ──────────────────
    mode_label = "List" if (is_list and items) else "Statistics"
    lines.append(
        f'<h4 style="color:#00e5ff;margin:0 0 2px 0;font-size:1.1em">'
        f'📊 {mode_label} — Class {cls} &nbsp;|&nbsp; Chapter {chapter}</h4>'
    )
    lines.append(
        f'<p style="color:#ffe066;font-size:0.95em;margin:0 0 6px 0"><b>📖 {title}</b></p>'
    )
    if source == "chunk_count_fallback":
        lines.append(
            '<p style="color:#ffb347;font-size:0.78em;margin:0 0 6px 0">ℹ️ '
            '<i>Count derived from index chunks (stats key absent in overview)</i></p>'
        )
    lines.append('<hr style="border-color:#333;margin:6px 0">')

    # ── LIST MODE ─────────────────────────────────────────────────────────────
    if is_list and items:
        n = len(items)

        if target == "exercises":
            lines.append(
                f'<p style="{_HEADER_STYLE}">✏️ Exercise Questions &nbsp;'
                f'<span style="color:#69ff47;font-size:0.9em">({n} total)</span></p>'
            )
            lines.append('<ol style="padding-left:20px;margin:4px 0">')
            for q in items:
                lines.append(
                    f'  <li style="{_ITEM_STYLE}">'
                    f'<span style="{_BADGE_STYLE}">{q["num"]}</span>'
                    f'{q["text"]}</li>'
                )
            lines.append('</ol>')

        elif target == "examples":
            lines.append(
                f'<p style="{_HEADER_STYLE}">🧮 Solved Examples &nbsp;'
                f'<span style="color:#69ff47;font-size:0.9em">({n} total)</span></p>'
            )
            lines.append('<ol style="padding-left:20px;margin:4px 0">')
            for e in items:
                lines.append(
                    f'  <li style="{_ITEM_STYLE}">'
                    f'<span style="{_BADGE_STYLE}">Ex {e["num"]}</span>'
                    f'{e["question"]}</li>'
                )
            lines.append('</ol>')

        elif target == "sections":
            lines.append(
                f'<p style="{_HEADER_STYLE}">📂 Sections &nbsp;'
                f'<span style="color:#69ff47;font-size:0.9em">({n} total)</span></p>'
            )
            lines.append('<ul style="padding-left:18px;margin:4px 0">')
            for s in items:
                lines.append(
                    f'  <li style="{_ITEM_STYLE}">'
                    f'<span style="{_BADGE_STYLE}">{s["num"]}</span>'
                    f'{s["text"]}</li>'
                )
            lines.append('</ul>')

        elif target == "subsections":
            lines.append(
                f'<p style="{_HEADER_STYLE}">📁 Subsections &nbsp;'
                f'<span style="color:#69ff47;font-size:0.9em">({n} total)</span></p>'
            )
            lines.append('<ul style="padding-left:18px;margin:4px 0">')
            for s in items:
                lines.append(
                    f'  <li style="{_ITEM_STYLE}">'
                    f'<span style="{_BADGE_STYLE}">{s["num"]}</span>'
                    f'{s["text"]}</li>'
                )
            lines.append('</ul>')

        elif target in ("tables", "figures"):
            icon  = "📋" if target == "tables" else "🖼️"
            label = "Table" if target == "tables" else "Figure"
            lines.append(
                f'<p style="{_HEADER_STYLE}">{icon} {friendly} &nbsp;'
                f'<span style="color:#69ff47;font-size:0.9em">({n} total)</span></p>'
            )
            lines.append('<ul style="padding-left:18px;margin:4px 0">')
            for item in items:
                lines.append(
                    f'  <li style="{_ITEM_STYLE}">'
                    f'<span style="{_BADGE_STYLE}">{label} {item["num"]}</span></li>'
                )
            lines.append('</ul>')

        elif target == "topics":
            topics_src = stats.get("topics_source", "")
            if topics_src == "not_found":
                lines.append(
                    '<p style="color:#ffb347">⚠️ No topics data found for this chapter.</p>'
                )
            else:
                src_note = ""
                if topics_src == "sections_fallback":
                    src_note = (
                        '<p style="color:#888;font-size:0.78em;margin:0 0 4px 0">'
                        'ℹ️ <i>Showing sections &amp; subsections as topics '
                        '(topics stored as sections in this chapter)</i></p>'
                    )
                lines.append(
                    f'<p style="{_HEADER_STYLE}">🏷️ Topics &nbsp;'
                    f'<span style="color:#69ff47;font-size:0.9em">({n} total)</span></p>'
                )
                if src_note:
                    lines.append(src_note)
                lines.append('<ul style="padding-left:18px;margin:4px 0">')
                for t in items:
                    num  = t.get("num", "")
                    text = t.get("text", "")
                    is_sub = text.startswith("\u21b3 ")   # ↳
                    indent = "padding-left:28px" if is_sub else "padding-left:0"
                    badge  = (
                        f'<span style="{_BADGE_STYLE}">{num}</span>'
                        if num and not str(num).isdigit() else ""
                    )
                    lines.append(
                        f'  <li style="{_ITEM_STYLE};list-style:none;{indent}">' +
                        badge + text + '</li>'
                    )
                lines.append('</ul>')

    # ── COUNT MODE ────────────────────────────────────────────────────────────
    elif target and count is not None:
        lines.append(
            f'<p style="font-size:1.05em;margin:8px 0">'
            f'{pink("📌 " + friendly + ":")} '
            f'<span style="color:#69ff47;font-size:1.5em;font-weight:700">{count}</span>'
            f'</p>'
        )
        listable = {"exercises", "examples", "sections", "subsections", "tables", "figures", "topics"}
        if target in listable:
            lines.append(
                f'<p style="color:#888;font-size:0.82em;margin:4px 0">💡 Tip: '
                f'"<i>list all {target} in chapter {chapter} class {cls}</i>" '
                f'shows the full list.</p>'
            )

    # ── FULL STATS TABLE ──────────────────────────────────────────────────────
    else:
        lines.append(f'<p style="{_SUBHEAD_STYLE}">📈 Full Chapter Statistics:</p>')
        stat_labels = {
            "total_solved_examples":    ("🧮 Solved Examples",    "#69ff47"),
            "total_tables":             ("📋 Tables",             "#ffb347"),
            "total_figures":            ("🖼️ Figures",            "#ff79c6"),
            "total_sections":           ("📂 Sections",           "#00e5ff"),
            "total_subsections":        ("📁 Subsections",        "#bd93f9"),
            "total_topics":             ("🏷️ Topics",             "#ffe066"),
            "total_exercise_questions": ("✏️ Exercise Questions", "#ff5555"),
        }
        lines.append('<table style="width:100%;border-collapse:collapse;margin-top:6px;font-size:0.92em">')
        lines.append(
            '<tr style="background:#1e1e2e">'
            '<th style="color:#00e5ff;padding:6px 8px;text-align:left">Statistic</th>'
            '<th style="color:#00e5ff;padding:6px 8px;text-align:center">Count</th>'
            '</tr>'
        )
        for key, (label, color) in stat_labels.items():
            val = all_stats.get(key, "—")
            lines.append(
                f'<tr style="border-bottom:1px solid #2a2a3a">'
                f'<td style="padding:5px 8px;color:{color}">{label}</td>'
                f'<td style="padding:5px 8px;text-align:center;color:#f8f8f2;font-weight:700">{val}</td>'
                f'</tr>'
            )
        lines.append('</table>')
        sections = stats.get("sections", [])
        if sections:
            lines.append(f'<p style="{_SUBHEAD_STYLE}">📂 Sections:</p>'
                         '<ul style="color:#f8f8f2;padding-left:18px;margin:2px 0">')
            for s in sections:
                lines.append(
                    f'  <li style="{_ITEM_STYLE}">'
                    f'<span style="{_BADGE_STYLE}">{s["id"]}</span>{s["title"]}</li>'
                )
            lines.append('</ul>')

    return "\n".join(lines)


# ─── Streamlit UI ─────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "messages":              [],
        "retriever":             None,
        "_last_dir":             None,
        "pending_clarification": None,
        "show_debug":            False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def show_message(role: str, content: str, metadata: Optional[Dict] = None):
    with st.chat_message(role):
        # For assistant messages that look like LLM prose (not HTML stat renders),
        # apply the line-break formatter so history replays display correctly too.
        display_content = content
        if role == "assistant" and not content.strip().startswith('<'):
            display_content = _format_llm_answer(content)
        st.markdown(display_content, unsafe_allow_html=True)
        if metadata and st.session_state.show_debug:
            chunks = metadata.get("raw_chunks", [])
            if chunks:
                with st.expander("🔍 Retrieved Chunks", expanded=False):
                    for i, ch in enumerate(chunks, 1):
                        m = ch["metadata"]
                        st.markdown(
                            f"**Chunk {i}:** Class {m['class']} · Ch {m['chapter']} · "
                            f"{m.get('chapter_title', '')} · `{m['type']}`"
                        )
                        scores = metadata.get("retrieval_scores", [])
                        if i - 1 < len(scores):
                            s = scores[i - 1]
                            st.caption(f"Similarity: {s['similarity']}%  (score {s['score']:.4f})")
                        st.code(ch["content"][:800] + ("…" if len(ch["content"]) > 800 else ""), language="text")
                        st.divider()
            if metadata.get("usage_stats"):
                with st.expander("📊 Token Usage", expanded=False):
                    u = metadata["usage_stats"]
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Prompt",     u.get("prompt_tokens", 0))
                    c2.metric("Completion", u.get("completion_tokens", 0))
                    c3.metric("Total",      u.get("total_tokens", 0))


st.set_page_config(
    page_title="NCERT Physics RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Outfit:wght@400;600;700&display=swap');
.stApp { background-color: #0d0f18; font-family: 'Outfit', sans-serif; }
.stChatMessage { padding:1.2rem; border-radius:12px; background:#161824; border:1px solid #252840; }
.stChatInput textarea {
    min-height:80px !important; font-size:15px !important; color:#e0e0ff !important;
    background:#161824 !important; border:2px solid #2d2f5a !important;
    border-radius:10px !important; font-family:'Outfit',sans-serif !important;
}
.stChatInput textarea:focus { border-color:#00e5ff !important; box-shadow:0 0 0 2px rgba(0,229,255,.15) !important; }
.stChatInput textarea::placeholder { color:#4a4d7a !important; }
section[data-testid="stSidebar"] { background:#0d0f18; border-right:1px solid #1e2040; }
h1,h2,h3,h4 { font-family:'Outfit',sans-serif !important; }
h1 { color:#00e5ff !important; letter-spacing:-0.5px; }
/* Keep h4 (used in stat renderer) reasonably sized */
h4 { font-size:1.05rem !important; margin-bottom:4px !important; }
div[data-testid="stMetricValue"] { color:#69ff47; font-family:'JetBrains Mono',monospace; }
.stButton button {
    background:#161824; color:#c0c0e0; border:1px solid #2d2f5a;
    border-radius:8px; font-family:'Outfit',sans-serif; transition:all .2s ease;
}
.stButton button:hover { background:#1e2040; border-color:#00e5ff; color:#00e5ff; transform:translateY(-1px); }
code { background:#1a1d30 !important; color:#69ff47 !important; border-radius:4px; font-family:'JetBrains Mono',monospace; }
div[data-testid="stExpander"] { background:#161824; border:1px solid #252840; border-radius:8px; }
.stAlert { background:#161824; border:1px solid #2d2f5a; }

/* ── LLM answer typography fixes ─────────────────────────────────────────── */
/* Prevent h3/h4 inside chat messages from being huge */
.stChatMessage h3 {
    font-size: 1.0rem !important;
    color: #ff79c6 !important;
    margin: 10px 0 4px 0 !important;
    border-bottom: 1px solid #2d2f5a;
    padding-bottom: 2px;
}
.stChatMessage h4 {
    font-size: 0.92rem !important;
    color: #bd93f9 !important;
    margin: 8px 0 2px 0 !important;
}
/* Style bold labels (Given, To Find, Solution) nicely */
.stChatMessage strong {
    color: #ffb347;
}
/* Make LaTeX display blocks breathe */
.stChatMessage .katex-display {
    margin: 10px 0 !important;
    padding: 8px 0 !important;
}
</style>
""", unsafe_allow_html=True)

init_state()

with st.sidebar:
    st.markdown("# ⚙️ Settings")
    idx_dir_str = st.text_input("📁 Index folder", value=".", help="Folder with .faiss, .json, .pkl files")
    idx_dir     = Path(idx_dir_str).resolve()
    k_val       = st.slider("📊 Chunks (k)", 2, 12, 6, 1)
    st.session_state.show_debug = st.checkbox("🐛 Debug Mode", st.session_state.show_debug)
    st.markdown("---")
    st.subheader("🔑 API Status")
    n_keys = sum(1 for i in list(range(1, 4)) + [None] if os.getenv(f"GROQ_API_KEY_{i}" if i else "GROQ_API_KEY"))
    if n_keys:
        st.success(f"✅ {n_keys} key(s) loaded")
        if st.session_state.retriever and st.session_state.retriever.groq:
            g = st.session_state.retriever.groq
            u = g.usage()
            st.info(f"🔑 Active: {g.current_key_label()}")
            if u["total"]:
                st.info(f"📈 Session: {u['total']:,} tokens")
    else:
        st.error("❌ No Groq API keys found")
    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_clarification = None
        st.rerun()
    st.markdown("---")
    st.subheader("💡 Query Examples")
    st.markdown("""
**📚 Book-level (from LLM knowledge):**
- Give all chapters of class 11
- List all chapters of class 12
- What chapters are in class 11 and 12?
- List all topics of class 11
- Give all topics of class 12

**📊 Count queries:**
- How many examples in chapter 6 class 11?
- How many exercise questions in chapter 3 class 12?
- Total figures in chapter 7 class 11

**📋 List queries (from index):**
- List exercise questions of chapter 6 class 11
- List all examples in chapter 3 class 11
- Show all sections in chapter 1 class 11
- Give me all subsections of chapter 5 class 12
- List all topics in chapter 12 class 11

**🔍 Specific item retrieval:**
- Example 5.3 class 11
- Table 7.1 class 12

**📖 Chapter content:**
- Summary of chapter 3 class 11
- Points to ponder chapter 5 class 12

**🧠 Semantic search:**
- Explain Newton's second law class 11
- What is Ohm's law?
    """)

st.title("📚 NCERT Physics — Smart RAG")
st.caption("🔬 Sentence Transformers + FAISS + Groq (Llama-3.3-70B) · v3.4")

if st.session_state.retriever is None or st.session_state._last_dir != str(idx_dir):
    with st.spinner("🔄 Loading FAISS index…"):
        try:
            st.session_state.retriever = NCERTRetriever(idx_dir)
            st.session_state._last_dir = str(idx_dir)
            st.success("✅ Index loaded successfully!")
        except Exception as e:
            st.error(f"❌ Load failed: {e}")
            st.stop()

retriever: NCERTRetriever = st.session_state.retriever

for msg in st.session_state.messages:
    show_message(msg["role"], msg["content"], msg.get("metadata"))

if st.session_state.pending_clarification:
    clarification = st.session_state.pending_clarification
    st.markdown(f"""
<div style="background:linear-gradient(135deg,#161824,#1a1d35);
            border:2px solid #bd93f9;border-radius:14px;padding:1.5rem;margin:1rem 0">
  <h4 style="color:#bd93f9;margin:0 0 .5rem">🤔 Ambiguous Query</h4>
  <p style="color:#c0c0e0">Found in multiple classes — please select:</p>
  <p style="color:#ffe066"><b>Your question:</b> {clarification['original_query']}</p>
</div>""", unsafe_allow_html=True)

    cols = st.columns(len(clarification["options"]))
    for i, opt in enumerate(clarification["options"]):
        with cols[i]:
            if st.button(
                f"Class {opt['class']}\nChapter {opt['chapter']}\n{opt['chapter_title']}",
                key=f"opt_{i}", use_container_width=True
            ):
                mod_query = f"class {opt['class']} {clarification['original_query']}"
                st.session_state.pending_clarification = None
                st.session_state.messages.append({
                    "role": "user",
                    "content": (
                        f"{clarification['original_query']}\n\n"
                        f"*→ Selected: Class {opt['class']} Ch {opt['chapter']}: {opt['chapter_title']}*"
                    ),
                })
                with st.spinner("🤔 Processing…"):
                    res = retriever.retrieve(mod_query.strip(), k=k_val)
                rc = (
                    render_statistics(res["statistics"]) if res.get("statistics")
                    else res.get("llm_answer") or
                    "📖 " + "\n\n".join(
                        f"**Class {c['metadata']['class']} Ch {c['metadata']['chapter']}**\n\n{c['content']}"
                        for c in res.get("raw_chunks", [])
                    ) or '<span style="color:#ff5555">⚠️ No content found.</span>'
                )
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": rc,
                    "metadata": {
                        "raw_chunks":       res.get("raw_chunks", []),
                        "usage_stats":      res.get("usage_stats", {}),
                        "retrieval_scores": res.get("retrieval_scores", []),
                    },
                })
                st.rerun()

    if st.button("❌ Cancel", use_container_width=True):
        st.session_state.pending_clarification = None
        st.rerun()

query = st.chat_input("💬 Ask about NCERT Physics Class 11 or 12…")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    show_message("user", query)

    with st.chat_message("assistant"):
        with st.spinner("🧠 Thinking…"):
            res = retriever.retrieve(query.strip(), k=k_val)

        if res["is_ambiguous"]:
            st.session_state.pending_clarification = {
                "original_query": query,
                "options":        res["ambiguity_options"],
            }
            rc = (
                "🤔 **Ambiguous query** — found in multiple classes. Please select above.\n\n"
                + "\n".join(
                    f"- **Class {o['class']}** — Ch {o['chapter']}: {o['chapter_title']}"
                    for o in res["ambiguity_options"]
                )
            )
            st.markdown(rc, unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": rc, "metadata": None})
            st.rerun()

        elif res.get("statistics"):
            rc = render_statistics(res["statistics"])
            st.markdown(rc, unsafe_allow_html=True)

        elif res.get("llm_answer"):
            rc = res["llm_answer"]
            # Preserve newlines: convert double-newlines to paragraph breaks,
            # and single newlines within a paragraph to <br> so LaTeX blocks
            # and step-by-step answers don't collapse into one line.
            # We do this ONLY for the chat display; the stored rc stays as-is
            # so re-renders from history also work correctly.
            rc_display = _format_llm_answer(rc)
            st.markdown(rc_display, unsafe_allow_html=True)

        elif res.get("raw_chunks"):
            rc = "📖 **Relevant NCERT content:**\n\n"
            for ch in res["raw_chunks"]:
                m = ch["metadata"]
                rc += (
                    f'<p><span style="color:#00e5ff"><b>Class {m["class"]} Ch {m["chapter"]}: '
                    f'{m.get("chapter_title", "")}</b></span></p>\n\n{ch["content"]}\n\n---\n\n'
                )
            st.markdown(rc, unsafe_allow_html=True)

        else:
            rc = '<span style="color:#ff5555">⚠️ No relevant content found. Try rephrasing.</span>'
            st.markdown(rc, unsafe_allow_html=True)

    st.session_state.messages.append({
        "role":     "assistant",
        "content":  rc,
        "metadata": {
            "raw_chunks":       res.get("raw_chunks", []),
            "usage_stats":      res.get("usage_stats", {}),
            "retrieval_scores": res.get("retrieval_scores", []),
        },
    })