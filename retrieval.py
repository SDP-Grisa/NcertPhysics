"""
NCERT Physics Retrieval + Llama-3.3-70B-Versatile (via Groq)
Intelligent retrieval system for NCERT Physics chunks with natural language answer generation.
"""

import json
import numpy as np
import pickle
import re
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError:
    print("Missing core dependencies!")
    print("Run: pip install sentence-transformers faiss-cpu numpy")
    exit(1)

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    Groq = None

class QueryType(Enum):
    SEMANTIC = "semantic"
    EXACT_MATCH = "exact_match"
    METADATA = "metadata"
    CHAPTER_CONTENT = "chapter_content"
    HYBRID = "hybrid"

@dataclass
class QueryIntent:
    query_type: QueryType
    original_query: str
    class_num: Optional[str] = None
    chapter_num: Optional[str] = None
    content_type: Optional[str] = None
    identifier: Optional[str] = None
    semantic_query: Optional[str] = None
    filters: Dict[str, Any] = None

    def __post_init__(self):
        if self.filters is None:
            self.filters = {}

class QueryParser:
    PATTERNS = {
        'example': [
            r'example\s+(\d+)\.(\d+)',
            r'ex\s+(\d+)\.(\d+)',
            r'show\s+example\s+(\d+)\.(\d+)',
        ],
        'table': [
            r'table\s+(\d+)\.(\d+)',
            r'show\s+table\s+(\d+)\.(\d+)',
        ],
        'summary': [
            r'summary\s+of\s+chapter\s+(\d+)',
            r'chapter\s+(\d+)\s+summary',
            r'summarize\s+chapter\s+(\d+)',
        ],
        'exercises': [
            r'exercises?\s+of\s+chapter\s+(\d+)',
            r'chapter\s+(\d+)\s+exercises?',
            r'questions?\s+(?:in|from)\s+chapter\s+(\d+)',
        ],
        'points_to_ponder': [
            r'points?\s+to\s+ponder\s+(?:of|in|from)?\s*chapter\s+(\d+)',
            r'chapter\s+(\d+)\s+points?\s+to\s+ponder',
        ],
        'chapter_stats': [
            r'how\s+many\s+(topics|sections|examples|tables|questions|exercises)\s+(?:in|from)?\s*chapter\s+(\d+)',
            r'count\s+(topics|sections|examples)\s+(?:in|from)?\s*chapter\s+(\d+)',
        ],
    }

    @staticmethod
    def parse_query(query: str) -> QueryIntent:
        q = query.lower().strip()

        for cat, patterns in QueryParser.PATTERNS.items():
            for pat in patterns:
                m = re.search(pat, q)
                if m:
                    if cat == 'example':
                        return QueryIntent(QueryType.EXACT_MATCH, query, chapter_num=m.group(1), content_type='example', identifier=f"{m.group(1)}.{m.group(2)}")
                    if cat == 'table':
                        return QueryIntent(QueryType.EXACT_MATCH, query, chapter_num=m.group(1), content_type='table', identifier=f"{m.group(1)}.{m.group(2)}")
                    if cat == 'summary':
                        return QueryIntent(QueryType.CHAPTER_CONTENT, query, chapter_num=m.group(1), content_type='summary')
                    if cat == 'exercises':
                        return QueryIntent(QueryType.CHAPTER_CONTENT, query, chapter_num=m.group(1), content_type='exercises')
                    if cat == 'points_to_ponder':
                        return QueryIntent(QueryType.CHAPTER_CONTENT, query, chapter_num=m.group(1), content_type='points_to_ponder')
                    if cat == 'chapter_stats':
                        return QueryIntent(QueryType.METADATA, query, chapter_num=m.group(2), filters={'stat_type': m.group(1)})

        ch_match = re.search(r'chapter\s+(\d+)', q)
        cl_match = re.search(r'class\s+(\d+)', q)

        chapter = ch_match.group(1) if ch_match else None
        cls    = cl_match.group(1) if cl_match else None

        if chapter or cls:
            return QueryIntent(QueryType.HYBRID, query, class_num=cls, chapter_num=chapter, semantic_query=query)

        return QueryIntent(QueryType.SEMANTIC, query, semantic_query=query)

class NCERTRetriever:
    def __init__(self, index_dir: str | Path = "."):
        self.index_dir = Path(index_dir).resolve()
        print(f"Loading index from: {self.index_dir}")

        # Load config
        with open(self.index_dir / "ncert_physics_config.json", "r") as f:
            self.config = json.load(f)

        # Load model
        self.model = SentenceTransformer(self.config["model_name"])

        # Load FAISS
        self.index = faiss.read_index(str(self.index_dir / "ncert_physics.faiss"))

        # Load chunks
        with open(self.index_dir / "ncert_physics_chunks_indexed.json", "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        # Load metadata index
        with open(self.index_dir / "ncert_physics_metadata.pkl", "rb") as f:
            self.metadata_index = pickle.load(f)

        # Groq client
        self.groq = None
        if GROQ_AVAILABLE:
            api_key = os.getenv("GROQ_API_KEY")
            if api_key:
                self.groq = Groq(api_key=api_key)
                print("Groq + Llama-3.3-70B-Versatile enabled")
            else:
                print("GROQ_API_KEY not found → LLM answers disabled")
        else:
            print("groq package not installed → LLM answers disabled")

    def semantic_search(self, query: str, k: int = 5, filters: Optional[Dict] = None) -> List[Tuple[int, float, Dict]]:
        emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(emb)
        scores, indices = self.index.search(emb, k * 2)

        results = []
        for sc, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            if filters and not self._apply_filters(chunk["metadata"], filters):
                continue
            results.append((idx, float(sc), chunk))
            if len(results) >= k:
                break
        return results

    def exact_match_search(self, intent: QueryIntent) -> Optional[Dict]:
        cls = intent.class_num or "11"
        if intent.content_type == "example":
            key = f"{cls}-{intent.chapter_num}-example-{intent.identifier}"
            idx = self.metadata_index.get("examples", {}).get(key)
        elif intent.content_type == "table":
            key = f"{cls}-{intent.chapter_num}-table-{intent.identifier}"
            idx = self.metadata_index.get("tables", {}).get(key)
        else:
            return None
        return self.chunks[idx] if idx is not None else None

    def chapter_content_search(self, intent: QueryIntent) -> Optional[Dict]:
        cls = intent.class_num or "11"
        ch_key = f"{cls}-{intent.chapter_num}"
        key_map = {
            "summary": ("summaries", f"{ch_key}-summary"),
            "exercises": ("exercises", f"{ch_key}-exercises"),
            "points_to_ponder": ("points_to_ponder", f"{ch_key}-points"),
        }
        cat, key = key_map.get(intent.content_type, (None, None))
        if not cat:
            return None
        idx = self.metadata_index.get(cat, {}).get(key)
        return self.chunks[idx] if idx is not None else None

    def metadata_query(self, intent: QueryIntent) -> Dict:
        cls = intent.class_num or "11"
        ch_key = f"{cls}-{intent.chapter_num}"
        overview_idx = self.metadata_index.get("chapter_overviews", {}).get(ch_key)
        if overview_idx is None:
            return {"error": f"Chapter {intent.chapter_num} not found"}

        overview = self.chunks[overview_idx]
        stats = overview["metadata"].get("stats", {})
        stat_type = intent.filters.get("stat_type", "").lower()

        result = {
            "class": cls,
            "chapter": intent.chapter_num,
            "title": overview["metadata"].get("chapter_title", ""),
        }

        if "topic" in stat_type or "section" in stat_type:
            result["type"] = "topics/sections"
            result["count"] = stats.get("total_topics", 0)
            result["details"] = {
                "sections": stats.get("total_sections", 0),
                "subsections": stats.get("total_subsections", 0),
            }
        elif "example" in stat_type:
            ex_chunks = [c for c in self.chunks if c["metadata"].get("type") == "example" and
                         c["metadata"]["class"] == cls and c["metadata"]["chapter"] == intent.chapter_num]
            result["type"] = "examples"
            result["count"] = len(ex_chunks)
        elif "table" in stat_type:
            result["type"] = "tables"
            result["count"] = stats.get("total_tables", 0)
        else:
            result["stats"] = stats

        return result

    def _apply_filters(self, meta: Dict, filters: Dict) -> bool:
        for k, v in filters.items():
            if meta.get(k) != v:
                return False
        return True

    def generate_answer(self, query: str, chunks: List[Dict]) -> str:
        if not self.groq or not chunks:
            return None

        context = "\n\n".join(
            f"Class {c['metadata']['class']} | Chapter {c['metadata']['chapter']} | {c['metadata'].get('chapter_title','')}\n"
            f"Type: {c['metadata']['type']}\n"
            f"{c['content']}\n{'─'*80}"
            for c in chunks
        )

        system = """You are an excellent NCERT Physics tutor.
Use only the provided textbook excerpts.
Answer clearly, accurately, step-by-step.
Use bullet points, numbered lists, equations (use LaTeX like E = mc^2).
Cite chapter and section when relevant.
Never add information not present in the excerpts."""

        user = f"""Question: {query}

NCERT excerpts:
{context}

Write a helpful, well-structured answer:"""

        try:
            resp = self.groq.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=0.3,
                max_tokens=2048,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"Groq error: {e}")
            return None

    def retrieve(self, query: str, k: int = 6) -> Dict:
        intent = QueryParser.parse_query(query)
        result = {
            "query": query,
            "intent_type": intent.query_type.value,
            "class": intent.class_num,
            "chapter": intent.chapter_num,
            "content_type": intent.content_type,
            "success": False,
        }

        if intent.query_type == QueryType.EXACT_MATCH:
            chunk = self.exact_match_search(intent)
            result["raw_chunks"] = [chunk] if chunk else []
        elif intent.query_type == QueryType.CHAPTER_CONTENT:
            chunk = self.chapter_content_search(intent)
            result["raw_chunks"] = [chunk] if chunk else []
        elif intent.query_type == QueryType.METADATA:
            result["statistics"] = self.metadata_query(intent)
        else:  # semantic or hybrid
            filters = {}
            if intent.class_num:   filters["class"]   = intent.class_num
            if intent.chapter_num: filters["chapter"] = intent.chapter_num
            sr = self.semantic_search(intent.semantic_query or query, k=k, filters=filters)
            result["raw_chunks"] = [chunk for _, _, chunk in sr]

        result["success"] = bool(result.get("raw_chunks") or result.get("statistics"))

        if result.get("raw_chunks") and self.groq:
            result["llm_answer"] = self.generate_answer(query, result["raw_chunks"])

        return result

    def format_result(self, result: Dict, show_content: bool = True) -> str:
        lines = ["═"*88]

        lines.append(f" Query : {result['query']}")
        lines.append(f" Type  : {result['intent_type']}")
        if result.get("class"):   lines.append(f" Class : {result['class']}")
        if result.get("chapter"): lines.append(f" Chapter : {result['chapter']}")
        lines.append("═"*88)

        if "statistics" in result:
            st = result["statistics"]
            lines.append(" STATISTICS")
            lines.append(f"  Class {st.get('class')} - Chapter {st.get('chapter')} : {st.get('title')}")
            if "count" in st:
                lines.append(f"  {st.get('type','count').title()}: {st['count']}")
                if "details" in st:
                    for k, v in st["details"].items():
                        lines.append(f"    • {k.title()}: {v}")
            elif "stats" in st:
                for k, v in st["stats"].items():
                    lines.append(f"  {k.replace('_',' ').title()}: {v}")
            if "error" in st:
                lines.append(f"  Error: {st['error']}")

        elif "llm_answer" in result:
            lines.append(" AI ANSWER (Llama-3.3-70B-Versatile)")
            lines.append("─"*88)
            lines.append(result["llm_answer"])
            lines.append("─"*88)

            if show_content and result["raw_chunks"]:
                lines.append(" Retrieved chunks (reference):")
                for i, chunk in enumerate(result["raw_chunks"], 1):
                    m = chunk["metadata"]
                    lines.append(f" [{i}] Class {m['class']} Ch {m['chapter']} | {m.get('chapter_title','')}")
                    lines.append(f"     Type: {m['type']} | ID: {m.get('identifier','')}")
                    if show_content:
                        text = chunk["content"].strip()
                        preview = (text[:600] + " …") if len(text) > 600 else text
                        lines.append(preview)
                    lines.append("")

        elif result.get("raw_chunks"):
            lines.append(" RAW MATCHES (no LLM available)")
            lines.append("─"*88)
            for i, c in enumerate(result["raw_chunks"], 1):
                m = c["metadata"]
                lines.append(f" [{i}] {m.get('chapter_title','')} — {m['type']}")
                if show_content:
                    lines.append(c["content"][:800] + ("…" if len(c["content"]) > 800 else ""))
                lines.append("")

        else:
            lines.append(" No relevant content found.")

        lines.append("═"*88)
        return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", default=".", help="index folder")
    parser.add_argument("--query", help="single query (non-interactive)")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--no-content", action="store_true", help="hide long content")
    args = parser.parse_args()

    try:
        ret = NCERTRetriever(args.index_dir)
    except Exception as e:
        print("Initialization failed:", e)
        return

    if args.query:
        res = ret.retrieve(args.query, k=args.k)
        print(ret.format_result(res, show_content=not args.no_content))
        return

    print("\nInteractive mode. Type 'exit' or 'quit' to stop.\n")
    while True:
        try:
            q = input("Query > ").strip()
            if q.lower() in {"exit", "quit", "q"}:
                print("Bye!")
                break
            if not q:
                continue
            res = ret.retrieve(q, k=args.k)
            print(ret.format_result(res, show_content=not args.no_content))
            print()
        except KeyboardInterrupt:
            print("\nBye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    main()