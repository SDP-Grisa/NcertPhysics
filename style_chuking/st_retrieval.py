"""
NCERT Physics Retrieval System + Llama-3.3-70B-Versatile  
Enhanced with: Ambiguity Resolution, Better UI, Debug Mode, Hallucination Prevention
"""

import json, os, re, pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

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

# Enhanced Prompt with strict formatting rules and no-hallucination guidelines
PROMPT_TEMPLATE = """You are an NCERT Class 11 & 12 Physics expert. Your task is to provide answers exactly as they appear in the NCERT textbook.

**CRITICAL FORMATTING RULES:**

1. **For Mathematical Equations:**
   - ALL equations must be in LaTeX format
   - Use $$ $$ for display equations (equations on their own line)
   - Use $ $ for inline math (equations within text)
   
2. **For Derivations:**
   - Start with "**Derivation:**" in bold
   - Show each step clearly
   - Use \\begin{{align*}} ... \\end{{align*}} inside $$ $$ blocks for multi-line derivations
   - Each line should use \\\\ for line breaks
   - Use & for alignment at = signs
   - Example format:
   $$
   \\begin{{align*}}
   F &= ma \\\\
   &= m \\frac{{dv}}{{dt}} \\\\
   &= m \\frac{{d^2x}}{{dt^2}}
   \\end{{align*}}
   $$

3. **For Examples/Problems:**
   - Start with "**Example X.Y:**" in bold
   - Show the **Given:** section with all known values
   - Show the **To Find:** section
   - Show the **Solution:** with step-by-step calculation
   - Use proper LaTeX symbols: \\times, \\div, ^{{}}, _{{}}, \\Delta, \\theta, etc.
   - Show final answer clearly with units

4. **Proper LaTeX Symbols:**
   - Subscripts: use _{{}} → $v_{{0}}$, $F_{{net}}$
   - Superscripts: use ^{{}} → $10^{{11}}$, $m^{{2}}$
   - Greek letters: $\\Delta$, $\\theta$, $\\omega$, $\\alpha$, $\\beta$, etc.
   - Fractions: $\\frac{{numerator}}{{denominator}}$
   - Multiplication: $\\times$
   - Square root: $\\sqrt{{x}}$
   - Vectors: $\\vec{{v}}$, $\\vec{{F}}$

5. **For Tables:**
   - Use markdown table format
   - Keep formatting clean and aligned

6. **MANDATORY Anti-Hallucination Rules:**
   - ONLY use information from the provided context below
   - Do NOT add information from general knowledge
   - If answer not in context, say: "This information is not available in the provided context."
   - Always cite: "**From Class [X] Chapter [Y]: [Chapter Title]**"
   - Reference figures naturally: "As shown in Fig. X.Y ..."

Context from NCERT:
{context}

Question:
{question}

Answer (with proper LaTeX formatting):
"""

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
    is_ambiguous: bool = False
    def __post_init__(self):
        if self.filters is None: self.filters = {}

class QueryParser:
    PATTERNS = {
        'example': [r'example\s+(\d+)\.(\d+)', r'ex\s+(\d+)\.(\d+)'],
        'table': [r'table\s+(\d+)\.(\d+)'],
        'summary': [r'summary\s+of\s+chapter\s+(\d+)', r'chapter\s+(\d+)\s+summary'],
        'exercises': [r'exercises?\s+of\s+chapter\s+(\d+)', r'chapter\s+(\d+)\s+exercises?'],
        'points_to_ponder': [r'points?\s+to\s+ponder\s+(?:of|in|from)?\s*chapter\s+(\d+)'],
        'chapter_stats': [r'how\s+many\s+(topics|sections|examples|tables|questions|exercises)\s+(?:in|from)?\s*chapter\s+(\d+)'],
    }
    
    @staticmethod
    def parse_query(query: str) -> QueryIntent:
        q = query.lower().strip()
        cl_match = re.search(r'class\s+(\d+)', q)
        explicit_class = cl_match.group(1) if cl_match else None
        
        for cat, patterns in QueryParser.PATTERNS.items():
            for pat in patterns:
                m = re.search(pat, q)
                if m:
                    if cat == 'example':
                        return QueryIntent(QueryType.EXACT_MATCH, query, class_num=explicit_class,
                            chapter_num=m.group(1), content_type='example', 
                            identifier=f"{m.group(1)}.{m.group(2)}", is_ambiguous=(explicit_class is None))
                    if cat == 'table':
                        return QueryIntent(QueryType.EXACT_MATCH, query, class_num=explicit_class,
                            chapter_num=m.group(1), content_type='table', 
                            identifier=f"{m.group(1)}.{m.group(2)}", is_ambiguous=(explicit_class is None))
                    if cat in ('summary', 'exercises', 'points_to_ponder'):
                        content_type = cat if cat != 'points_to_ponder' else 'points_to_ponder'
                        return QueryIntent(QueryType.CHAPTER_CONTENT, query, class_num=explicit_class,
                            chapter_num=m.group(1), content_type=content_type, is_ambiguous=(explicit_class is None))
                    if cat == 'chapter_stats':
                        return QueryIntent(QueryType.METADATA, query, class_num=explicit_class,
                            chapter_num=m.group(2), filters={'stat_type': m.group(1)}, is_ambiguous=(explicit_class is None))
        
        ch_match = re.search(r'chapter\s+(\d+)', q)
        chapter = ch_match.group(1) if ch_match else None
        if chapter or explicit_class:
            return QueryIntent(QueryType.HYBRID, query, class_num=explicit_class, chapter_num=chapter,
                semantic_query=query, is_ambiguous=(chapter is not None and explicit_class is None))
        return QueryIntent(QueryType.SEMANTIC, query, semantic_query=query)

class GroqAPIManager:
    def __init__(self):
        self.api_keys = self._load_api_keys()
        self.current_key_index = 0
        self.clients = [Groq(api_key=k) for k in self.api_keys] if self.api_keys else []
        self.total_tokens = self.total_prompt = self.total_completion = 0
    
    def _load_api_keys(self) -> List[str]:
        keys = []
        for i in range(1, 4):
            k = os.getenv(f"GROQ_API_KEY_{i}")
            if k: keys.append(k)
        default = os.getenv("GROQ_API_KEY")
        if default and default not in keys: keys.insert(0, default)
        return keys
    
    def get_client(self): return self.clients[self.current_key_index] if self.clients else None
    def switch_to_next_key(self): 
        if self.current_key_index < len(self.clients) - 1:
            self.current_key_index += 1
            return True
        return False
    def reset_key_index(self): self.current_key_index = 0
    
    def generate_completion(self, messages: List[Dict], temperature: float = 0.1, max_tokens: int = 2048) -> Optional[Tuple[str, Dict]]:
        for attempt in range(len(self.clients)):
            try:
                client = self.get_client()
                if not client: return None, {}
                resp = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=messages,
                    temperature=temperature, max_tokens=max_tokens)
                usage = resp.usage
                stats = {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens, "total_tokens": usage.total_tokens}
                self.total_prompt += usage.prompt_tokens
                self.total_completion += usage.completion_tokens
                self.total_tokens += usage.total_tokens
                self.reset_key_index()
                return resp.choices[0].message.content.strip(), stats
            except Exception as e:
                if "rate" in str(e).lower() or "limit" in str(e).lower() or "429" in str(e):
                    if self.switch_to_next_key():
                        st.warning(f"⚠️ Rate limit. Switching to key {self.current_key_index + 1}")
                        continue
                    else:
                        st.error("❌ All keys rate limited")
                        return None, {}
                else:
                    st.error(f"❌ API Error: {e}")
                    return None, {}
        return None, {}
    
    def get_total_usage(self): return {"total_tokens": self.total_tokens, "prompt_tokens": self.total_prompt, "completion_tokens": self.total_completion}

class NCERTRetriever:
    def __init__(self, index_dir: Path):
        self.index_dir, self.model, self.index, self.chunks, self.metadata_index, self.groq_manager = index_dir, None, None, None, None, None
        self._load()
    
    def _load(self):
        try:
            cfg = json.loads((self.index_dir / "ncert_physics_config_gte.json").read_text())
            self.model = SentenceTransformer(cfg["model_name"])
            self.index = faiss.read_index(str(self.index_dir / "ncert_physics_gte.faiss"))
            self.chunks = json.loads((self.index_dir / "ncert_physics_chunks_indexed_gte.json").read_text(encoding="utf-8"))
            with open(self.index_dir / "ncert_physics_metadata_gte.pkl", "rb") as f:
                self.metadata_index = pickle.load(f)
            if GROQ_AVAILABLE:
                self.groq_manager = GroqAPIManager()
                if not self.groq_manager.api_keys: st.warning("⚠️ No Groq keys. AI disabled.")
        except Exception as e:
            st.error(f"Failed to load: {e}")
            st.stop()
    
    def check_ambiguity(self, intent: QueryIntent) -> Optional[List[Dict]]:
        if not intent.is_ambiguous: return None
        options = []
        if intent.content_type == "example":
            for cls in ["11", "12"]:
                key = f"{cls}-{intent.chapter_num}-example-{intent.identifier}"
                idx = self.metadata_index.get("examples", {}).get(key)
                if idx is not None:
                    chunk = self.chunks[idx]
                    options.append({"class": cls, "chapter": intent.chapter_num, 
                        "chapter_title": chunk["metadata"].get("chapter_title", ""), "type": "example", "identifier": intent.identifier})
        elif intent.content_type == "table":
            for cls in ["11", "12"]:
                key = f"{cls}-{intent.chapter_num}-table-{intent.identifier}"
                idx = self.metadata_index.get("tables", {}).get(key)
                if idx is not None:
                    chunk = self.chunks[idx]
                    options.append({"class": cls, "chapter": intent.chapter_num,
                        "chapter_title": chunk["metadata"].get("chapter_title", ""), "type": "table", "identifier": intent.identifier})
        elif intent.content_type in ("summary", "exercises", "points_to_ponder"):
            cat_key = {"summary": "summaries", "exercises": "exercises", "points_to_ponder": "points_to_ponder"}[intent.content_type]
            for cls in ["11", "12"]:
                ch_key = f"{cls}-{intent.chapter_num}"
                key = f"{ch_key}-{intent.content_type.split('_')[0]}" if intent.content_type != "points_to_ponder" else f"{ch_key}-points"
                idx = self.metadata_index.get(cat_key, {}).get(key)
                if idx is not None:
                    chunk = self.chunks[idx]
                    options.append({"class": cls, "chapter": intent.chapter_num,
                        "chapter_title": chunk["metadata"].get("chapter_title", ""), "type": intent.content_type, "identifier": None})
        elif intent.query_type == QueryType.HYBRID and intent.chapter_num:
            for cls in ["11", "12"]:
                ch_key = f"{cls}-{intent.chapter_num}"
                idx = self.metadata_index.get("chapter_overviews", {}).get(ch_key)
                if idx is not None:
                    chunk = self.chunks[idx]
                    options.append({"class": cls, "chapter": intent.chapter_num,
                        "chapter_title": chunk["metadata"].get("chapter_title", ""), "type": "chapter", "identifier": None})
        return options if len(options) > 1 else None
    
    def semantic_search(self, query: str, k: int = 6, filters: Optional[Dict] = None) -> List[Dict]:
        emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(emb)
        scores, indices = self.index.search(emb, k * 2)
        results = []
        for sc, idx in zip(scores[0], indices[0]):
            if idx == -1: continue
            chunk = self.chunks[idx]
            if filters and not self._apply_filters(chunk["metadata"], filters): continue
            results.append({"chunk": chunk, "score": float(sc), "similarity_percentage": round(float(sc) * 100, 2)})
            if len(results) >= k: break
        return results
    
    def exact_match_search(self, intent: QueryIntent) -> Optional[Dict]:
        cls = intent.class_num or "11"
        if intent.content_type == "example":
            key = f"{cls}-{intent.chapter_num}-example-{intent.identifier}"
            idx = self.metadata_index.get("examples", {}).get(key)
        elif intent.content_type == "table":
            key = f"{cls}-{intent.chapter_num}-table-{intent.identifier}"
            idx = self.metadata_index.get("tables", {}).get(key)
        else: return None
        return self.chunks[idx] if idx is not None else None
    
    def chapter_content_search(self, intent: QueryIntent) -> Optional[Dict]:
        cls = intent.class_num or "11"
        ch_key = f"{cls}-{intent.chapter_num}"
        key_map = {"summary": ("summaries", f"{ch_key}-summary"), "exercises": ("exercises", f"{ch_key}-exercises"),
            "points_to_ponder": ("points_to_ponder", f"{ch_key}-points")}
        cat, key = key_map.get(intent.content_type, (None, None))
        if not cat: return None
        idx = self.metadata_index.get(cat, {}).get(key)
        return self.chunks[idx] if idx is not None else None
    
    def metadata_query(self, intent: QueryIntent) -> Dict:
        cls = intent.class_num or "11"
        ch_key = f"{cls}-{intent.chapter_num}"
        overview_idx = self.metadata_index.get("chapter_overviews", {}).get(ch_key)
        if overview_idx is None: return {"error": f"Chapter {intent.chapter_num} not found"}
        overview = self.chunks[overview_idx]
        stats = overview["metadata"].get("stats", {})
        stat_type = intent.filters.get("stat_type", "").lower()
        result = {"class": cls, "chapter": intent.chapter_num, "title": overview["metadata"].get("chapter_title", "")}
        if "topic" in stat_type or "section" in stat_type:
            result["type"], result["count"] = "topics/sections", stats.get("total_topics", 0)
            result["details"] = {"sections": stats.get("total_sections", 0), "subsections": stats.get("total_subsections", 0)}
        elif "example" in stat_type:
            exs = [c for c in self.chunks if c["metadata"].get("type") == "example" and c["metadata"]["class"] == cls and c["metadata"]["chapter"] == intent.chapter_num]
            result["type"], result["count"] = "examples", len(exs)
        elif "table" in stat_type:
            result["type"], result["count"] = "tables", stats.get("total_tables", 0)
        else: result["stats"] = stats
        return result
    
    @staticmethod
    def _apply_filters(meta: Dict, filters: Dict) -> bool:
        for k, v in filters.items():
            if meta.get(k) != v: return False
        return True
    
    def generate_answer(self, query: str, chunks: List[Dict]) -> Tuple[Optional[str], Dict]:
        if not self.groq_manager or not chunks: return None, {}
        context = "\n\n".join(
            f"**Class {c['metadata']['class']} | Ch {c['metadata']['chapter']} | {c['metadata'].get('chapter_title', '')}\n"
            f"*Type: {c['metadata']['type']}*\n\n{c['content']}\n\n{'─'*70}"
            for c in chunks
        )
        messages = [
            {"role": "system", "content": "You are NCERT Physics expert. Use LaTeX. ONLY use provided context. NO general knowledge."}, 
            {"role": "user", "content": PROMPT_TEMPLATE.format(context=context, question=query)}
        ]
        result = self.groq_manager.generate_completion(messages, temperature=0.1, max_tokens=2048)
        return result if result else (None, {})
    
    def retrieve(self, query: str, k: int = 6) -> Dict:
        intent = QueryParser.parse_query(query)
        result = {"query": query, "intent_type": intent.query_type.value, "class": intent.class_num, "chapter": intent.chapter_num,
            "content_type": intent.content_type, "raw_chunks": [], "statistics": None, "llm_answer": None,
            "is_ambiguous": False, "ambiguity_options": None, "usage_stats": {}, "retrieval_scores": []}
        
        ambiguity_options = self.check_ambiguity(intent)
        if ambiguity_options:
            result["is_ambiguous"], result["ambiguity_options"] = True, ambiguity_options
            return result
        
        if intent.query_type == QueryType.EXACT_MATCH:
            chunk = self.exact_match_search(intent)
            if chunk: result["raw_chunks"] = [chunk]
        elif intent.query_type == QueryType.CHAPTER_CONTENT:
            chunk = self.chapter_content_search(intent)
            if chunk: result["raw_chunks"] = [chunk]
        elif intent.query_type == QueryType.METADATA:
            result["statistics"] = self.metadata_query(intent)
        else:
            filters = {}
            if intent.class_num: filters["class"] = intent.class_num
            if intent.chapter_num: filters["chapter"] = intent.chapter_num
            sr = self.semantic_search(intent.semantic_query or query, k=k, filters=filters)
            result["raw_chunks"] = [r["chunk"] for r in sr]
            result["retrieval_scores"] = [{"class": r["chunk"]["metadata"]["class"], "chapter": r["chunk"]["metadata"]["chapter"],
                "type": r["chunk"]["metadata"]["type"], "score": r["score"], "similarity": r["similarity_percentage"]} for r in sr]
        
        if result["raw_chunks"] and self.groq_manager:
            answer, usage_stats = self.generate_answer(query, result["raw_chunks"])
            result["llm_answer"], result["usage_stats"] = answer, usage_stats
        return result

def initialize_session_state():
    for key, val in [("messages", []), ("retriever", None), ("_last_index_dir", None), ("pending_clarification", None), ("show_debug", False)]:
        if key not in st.session_state: st.session_state[key] = val

def display_chat_message(role: str, content: str, metadata: Optional[Dict] = None):
    with st.chat_message(role):
        st.markdown(content)
        if metadata:
            if metadata.get("raw_chunks") and st.session_state.show_debug:
                with st.expander("🔍 Retrieved Chunks (Debug)", expanded=False):
                    for i, chunk in enumerate(metadata["raw_chunks"], 1):
                        meta = chunk["metadata"]
                        st.markdown(f"**Chunk {i}:** Class {meta['class']} • Ch {meta['chapter']} • {meta.get('chapter_title','')}" )
                        st.caption(f"Type: {meta['type']}" )
                        if metadata.get("retrieval_scores") and i-1 < len(metadata["retrieval_scores"]):
                            score_info = metadata["retrieval_scores"][i-1]
                            st.caption(f"Similarity: {score_info['similarity']}% (Score: {score_info['score']:.4f})" )
                        with st.container():
                            preview = chunk["content"][:1000]
                            if len(chunk["content"]) > 1000: preview += "\n\n... (truncated)"
                            st.code(preview, language="text")
                        if i < len(metadata["raw_chunks"]): st.divider()
            if metadata.get("usage_stats") and st.session_state.show_debug:
                with st.expander("📊 Token Usage", expanded=False):
                    stats = metadata["usage_stats"]
                    c1, c2, c3 = st.columns(3)
                    with c1: st.metric("Prompt", stats.get("prompt_tokens", 0))
                    with c2: st.metric("Completion", stats.get("completion_tokens", 0))
                    with c3: st.metric("Total", stats.get("total_tokens", 0))

st.set_page_config(page_title="NCERT Physics RAG", page_icon="📚", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* Main background and text colors */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Chat messages */
    .stChatMessage {
        padding: 1.5rem;
        border-radius: 0.75rem;
        background-color: #1e1e1e;
        color: #e0e0e0;
    }
    
    /* Expander styling */
    div[data-testid="stExpander"] {
        background-color: #262730;
        border-radius: 0.5rem;
        padding: 0.75rem;
        border: 1px solid #3d3d3d;
    }
    
    /* Make text input larger and VISIBLE */
    .stChatInput textarea {
        min-height: 100px !important;
        font-size: 16px !important;
        color: #e0e0e0 !important;
        background-color: #1e1e1e !important;
        border: 2px solid #3d3d3d !important;
    }
    
    /* Input placeholder text */
    .stChatInput textarea::placeholder {
        color: #64748b !important;
    }
    
    /* Input focus state */
    .stChatInput textarea:focus {
        border-color: #00ff88 !important;
        box-shadow: 0 0 0 1px #00ff88 !important;
    }
    
    /* Code blocks */
    code {
        background-color: #1e1e1e;
        color: #00ff00;
        padding: 2px 6px;
        border-radius: 4px;
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        color: #00ff88;
    }
    
    /* Buttons */
    .stButton button {
        background-color: #262730;
        color: #e0e0e0;
        border: 1px solid #3d3d3d;
        transition: all 0.3s ease;
        font-weight: 500;
    }
    
    .stButton button:hover {
        background-color: #3d3d3d;
        border-color: #00ff88;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 255, 136, 0.2);
    }
    
    /* Primary button for selections */
    button[data-baseweb="button"][kind="primary"] {
        background-color: #00ff88 !important;
        color: #0e1117 !important;
        font-weight: 600 !important;
        border: none !important;
    }
    
    button[data-baseweb="button"][kind="primary"]:hover {
        background-color: #00cc6f !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 255, 136, 0.4) !important;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #00ff88 !important;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0e1117;
    }
    
    /* Better contrast for text */
    p, li, span, label {
        color: #e0e0e0;
    }
    
    /* Warning and info boxes */
    .stAlert {
        background-color: #262730;
        border: 1px solid #3d3d3d;
        color: #e0e0e0;
    }
    
    /* Info box text */
    .stAlert p {
        color: #e0e0e0 !important;
    }
    
    /* Ambiguity selection styling */
    .ambiguity-container {
        background: linear-gradient(135deg, #1e1e1e 0%, #262730 100%);
        border: 2px solid #00ff88;
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 8px 24px rgba(0, 255, 136, 0.15);
        animation: slideIn 0.3s ease-out;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(-10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Make info boxes more visible in ambiguity context */
    .ambiguity-container .stAlert {
        background-color: rgba(0, 255, 136, 0.1) !important;
        border-left: 4px solid #00ff88 !important;
        color: #e0e0e0 !important;
    }
</style>
""", unsafe_allow_html=True)

initialize_session_state()

with st.sidebar:
    st.title("⚙️ Settings")
    index_dir_str = st.text_input("📁 Index folder", value=".", help="Folder with .faiss, .json, .pkl files")
    index_dir = Path(index_dir_str).resolve()
    k_value = st.slider("📊 Chunks (k)", 2, 12, 6, 1, help="Chunks to retrieve")
    st.session_state.show_debug = st.checkbox("🐛 Debug Mode", st.session_state.show_debug, help="Show chunks, tokens, scores")
    st.markdown("---")
    st.subheader("🔑 API Status")
    api_keys_count = len([k for k in [os.getenv(f"GROQ_API_KEY_{i}") for i in range(1,4)] + [os.getenv("GROQ_API_KEY")] if k])
    if api_keys_count > 0:
        st.success(f"✅ {api_keys_count} key(s) loaded")
        if st.session_state.retriever and st.session_state.retriever.groq_manager:
            usage = st.session_state.retriever.groq_manager.get_total_usage()
            if usage["total_tokens"] > 0: st.info(f"📈 Session: {usage['total_tokens']:,} tokens (P:{usage['prompt_tokens']:,}, C:{usage['completion_tokens']:,})")
    else:
        st.error("❌ No API keys")
        st.info("Add to .env:\n")
    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages, st.session_state.pending_clarification = [], None
        st.rerun()
    st.markdown("---")
    st.subheader("💡 Examples")
    st.markdown("""
**With class (no ambiguity):**
- Class 11 example 5.3
- Class 12 table 7.1

**Ambiguous (asks for clarification):**
- Example 3.3
- Summary of chapter 4

**Search:**
- Explain Newton's laws class 11
- What is kinetic energy?
    """)

st.title("📚 NCERT Physics RAG")
st.caption("🔬 Sentence Transformers + FAISS + Groq (Llama-3.3-70B)")

if st.session_state.retriever is None or st.session_state._last_index_dir != str(index_dir):
    with st.spinner("🔄 Loading index..."):
        try:
            st.session_state.retriever = NCERTRetriever(index_dir)
            st.session_state._last_index_dir = str(index_dir)
            st.success("✅ Index loaded!")
        except Exception as e:
            st.error(f"❌ Load failed: {e}")
            st.stop()

retriever = st.session_state.retriever

# Display chat history FIRST
for msg in st.session_state.messages:
    display_chat_message(msg["role"], msg["content"], msg.get("metadata"))

# Show ambiguity selection AFTER chat history (inline with conversation)
if st.session_state.pending_clarification:
    clarification = st.session_state.pending_clarification
    
    # Use custom container for better styling
    st.markdown('<div class="ambiguity-container">', unsafe_allow_html=True)
    st.info("🤔 Your query is ambiguous. Please select which class/chapter you meant:")
    
    # Show original query
    st.markdown(f"**Your Question:** {clarification['original_query']}")
    st.markdown("---")
    
    cols = st.columns(len(clarification["options"]))
    for i, opt in enumerate(clarification["options"]):
        with cols[i]:
            label = f"**Class {opt['class']}**\n\nChapter {opt['chapter']}\n\n*{opt['chapter_title']}*"
            if st.button(label, key=f"opt_{i}", use_container_width=True, type="primary"):
                # Store selection and clear clarification
                modified_query = f"class {opt['class']} {clarification['original_query']}"
                original_query = clarification['original_query']
                selected_class = opt['class']
                selected_chapter = opt['chapter']
                selected_title = opt['chapter_title']
                
                # Clear clarification
                st.session_state.pending_clarification = None
                
                # Add user message showing selection
                user_msg = f"{original_query}\n\n*→ Selected: Class {selected_class}, Chapter {selected_chapter}: {selected_title}*"
                st.session_state.messages.append({"role": "user", "content": user_msg})
                
                # Process with spinner
                with st.spinner("🤔 Processing your selection..."):
                    result = retriever.retrieve(modified_query.strip(), k=k_value)
                
                # Build response content properly
                response_content = ""
                if result.get("statistics"):
                    stats = result["statistics"]
                    response_content = f"**Statistics for Class {stats.get('class', 'N/A')} Chapter {stats.get('chapter', 'N/A')}:**\n\n"
                    response_content += f"**Title:** {stats.get('title', 'N/A')}\n\n"
                    if 'count' in stats:
                        response_content += f"**{stats.get('type', 'Items')}:** {stats['count']}\n\n"
                    if 'details' in stats:
                        response_content += "**Details:**\n" + "\n".join(f"- {k}: {v}" for k, v in stats['details'].items()) + "\n"
                    if 'stats' in stats:
                        response_content += "\n**Full Statistics:**\n" + "\n".join(f"- {k}: {v}" for k, v in stats['stats'].items())
                
                elif result.get("llm_answer"):
                    response_content = result["llm_answer"]
                
                elif result.get("raw_chunks"):
                    response_content = "📖 Here is the relevant content from NCERT:\n\n"
                    for chunk in result["raw_chunks"]:
                        meta = chunk["metadata"]
                        response_content += f"**From Class {meta['class']} Chapter {meta['chapter']}: {meta.get('chapter_title', '')}**\n\n"
                        response_content += f"{chunk['content']}\n\n---\n\n"
                else:
                    response_content = "⚠️ No content found for this query."
                
                # Add assistant message with metadata
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response_content,
                    "metadata": {
                        "raw_chunks": result.get("raw_chunks", []), 
                        "usage_stats": result.get("usage_stats", {}), 
                        "retrieval_scores": result.get("retrieval_scores", [])
                    }
                })
                st.rerun()
    
    # Cancel button
    if st.button("❌ Cancel", use_container_width=True):
        st.session_state.pending_clarification = None
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# Chat input at bottom
query = st.chat_input("💬 Ask about NCERT Physics (Class 11 or 12)...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    display_chat_message("user", query)
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            result = retriever.retrieve(query.strip(), k=k_value)
        if result["is_ambiguous"]:
            st.session_state.pending_clarification = {"original_query": query, "options": result["ambiguity_options"]}
            response_content = "🤔 Found in multiple classes. Select:\n\n" + "\n".join(
                f"- **Class {o['class']}** - Ch {o['chapter']}: {o['chapter_title']}" for o in result["ambiguity_options"])
            st.markdown(response_content)
            st.session_state.messages.append({"role": "assistant", "content": response_content, "metadata": None})
            st.rerun()
        
        if result["statistics"]:
            stats = result["statistics"]
            response_content = f"**Stats for Class {stats.get('class','N/A')} Ch {stats.get('chapter','N/A')}:**\n\n**Title:** {stats.get('title','N/A')}\n\n"
            if 'count' in stats: response_content += f"**{stats.get('type','Items')}:** {stats['count']}\n\n"
            if 'details' in stats:
                response_content += "**Details:**\n" + "\n".join(f"- {k}: {v}" for k,v in stats['details'].items())
            if 'stats' in stats:
                response_content += "\n**Full Stats:**\n" + "\n".join(f"- {k}: {v}" for k,v in stats['stats'].items())
            st.markdown(response_content)
        elif result["llm_answer"]:
            response_content = result["llm_answer"]
            st.markdown(response_content)
        else:
            if result["raw_chunks"]:
                response_content = "📖 Relevant NCERT content:\n\n" + "\n\n---\n\n".join(
                    f"**From Class {c['metadata']['class']} Ch {c['metadata']['chapter']}: {c['metadata'].get('chapter_title','')}**\n\n{c['content']}" 
                    for c in result["raw_chunks"])
                st.markdown(response_content)
            else:
                response_content = "⚠️ No relevant content. Try rephrasing."
                st.warning(response_content)
    st.session_state.messages.append({"role": "assistant", "content": response_content,
        "metadata": {"raw_chunks": result.get("raw_chunks", []), "usage_stats": result.get("usage_stats", {}), "retrieval_scores": result.get("retrieval_scores", [])}})