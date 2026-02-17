"""
NCERT Physics Retrieval System + Llama-3.3-70B-Versatile
Enhanced Streamlit Web App with Chat Interface
"""

import json
import os
import re
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional

import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    Groq = None

# ────────────────────────────────────────────────
#   Enhanced Prompt Template
# ────────────────────────────────────────────────

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

6. **General Rules:**
   - Be accurate and match textbook style exactly
   - Reference figures naturally: "As shown in Fig. X.Y ..."
   - If no relevant info found → say "Not found in this chapter."
   - Do not add extra explanations beyond what's in the textbook
   - Always cite the chapter and section when providing information

**Context from NCERT:**
{context}

**Question:**
{question}

**Answer (with proper LaTeX formatting):**"""

# ────────────────────────────────────────────────
#   Query Types & Parser
# ────────────────────────────────────────────────

from enum import Enum
from dataclasses import dataclass

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
            r'example\s+(\d+)\.(\d+)', r'ex\s+(\d+)\.(\d+)', 
            r'show\s+example\s+(\d+)\.(\d+)',
        ],
        'table': [
            r'table\s+(\d+)\.(\d+)', r'show\s+table\s+(\d+)\.(\d+)',
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
                        return QueryIntent(
                            QueryType.EXACT_MATCH, query, 
                            chapter_num=m.group(1), 
                            content_type='example', 
                            identifier=f"{m.group(1)}.{m.group(2)}"
                        )
                    if cat == 'table':
                        return QueryIntent(
                            QueryType.EXACT_MATCH, query, 
                            chapter_num=m.group(1), 
                            content_type='table', 
                            identifier=f"{m.group(1)}.{m.group(2)}"
                        )
                    if cat in ('summary', 'exercises', 'points_to_ponder'):
                        content_type = cat if cat != 'points_to_ponder' else 'points_to_ponder'
                        return QueryIntent(
                            QueryType.CHAPTER_CONTENT, query, 
                            chapter_num=m.group(1), 
                            content_type=content_type
                        )
                    if cat == 'chapter_stats':
                        return QueryIntent(
                            QueryType.METADATA, query, 
                            chapter_num=m.group(2), 
                            filters={'stat_type': m.group(1)}
                        )

        ch_match = re.search(r'chapter\s+(\d+)', q)
        cl_match = re.search(r'class\s+(\d+)', q)

        chapter = ch_match.group(1) if ch_match else None
        cls = cl_match.group(1) if cl_match else None

        if chapter or cls:
            return QueryIntent(
                QueryType.HYBRID, query, 
                class_num=cls, 
                chapter_num=chapter, 
                semantic_query=query
            )

        return QueryIntent(QueryType.SEMANTIC, query, semantic_query=query)

# ────────────────────────────────────────────────
#   Groq API Manager with Fallback
# ────────────────────────────────────────────────

class GroqAPIManager:
    """Manages multiple Groq API keys with automatic fallback"""
    
    def __init__(self):
        self.api_keys = self._load_api_keys()
        self.current_key_index = 0
        self.clients = [Groq(api_key=key) for key in self.api_keys] if self.api_keys else []
    
    def _load_api_keys(self) -> List[str]:
        """Load API keys from environment variables"""
        keys = []
        
        # Load GROQ_API_KEY_1, GROQ_API_KEY_2, GROQ_API_KEY_3
        for i in range(1, 4):
            key = os.getenv(f"GROQ_API_KEY_{i}")
            if key:
                keys.append(key)
        
        # Also check for default GROQ_API_KEY
        default_key = os.getenv("GROQ_API_KEY")
        if default_key and default_key not in keys:
            keys.insert(0, default_key)
        
        return keys
    
    def get_client(self) -> Optional[Groq]:
        """Get current Groq client"""
        if not self.clients:
            return None
        return self.clients[self.current_key_index]
    
    def switch_to_next_key(self) -> bool:
        """Switch to next API key. Returns True if switched, False if no more keys"""
        if self.current_key_index < len(self.clients) - 1:
            self.current_key_index += 1
            return True
        return False
    
    def reset_key_index(self):
        """Reset to first API key"""
        self.current_key_index = 0
    
    def generate_completion(
        self, 
        messages: List[Dict], 
        temperature: float = 0.25, 
        max_tokens: int = 2048
    ) -> Optional[str]:
        """Generate completion with automatic fallback"""
        for attempt in range(len(self.clients)):
            try:
                client = self.get_client()
                if not client:
                    return None
                
                resp = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                
                # Reset to first key on success
                self.reset_key_index()
                return resp.choices[0].message.content.strip()
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Check if it's a rate limit error
                if "rate" in error_msg or "limit" in error_msg or "429" in error_msg:
                    if self.switch_to_next_key():
                        st.warning(
                            f"⚠️ Rate limit reached. Switching to backup API key "
                            f"{self.current_key_index + 1}..."
                        )
                        continue
                    else:
                        st.error(
                            "❌ All API keys have reached their rate limits. "
                            "Please try again later."
                        )
                        return None
                else:
                    # For other errors, don't switch keys
                    st.error(f"❌ API Error: {e}")
                    return None
        
        return None

# ────────────────────────────────────────────────
#   Retriever Class
# ────────────────────────────────────────────────

class NCERTRetriever:
    def __init__(self, index_dir: Path):
        self.index_dir = index_dir
        self.model = None
        self.index = None
        self.chunks = None
        self.metadata_index = None
        self.groq_manager = None

        self._load()

    def _load(self):
        try:
            config = json.loads(
                (self.index_dir / "ncert_physics_config_gte.json").read_text()
            )
            self.model = SentenceTransformer(config["model_name"])
            self.index = faiss.read_index(
                str(self.index_dir / "ncert_physics_gte.faiss")
            )
            self.chunks = json.loads(
                (self.index_dir / "ncert_physics_chunks_indexed_gte.json")
                .read_text(encoding="utf-8")
            )
            with open(self.index_dir / "ncert_physics_metadata_gte.pkl", "rb") as f:
                self.metadata_index = pickle.load(f)

            if GROQ_AVAILABLE:
                self.groq_manager = GroqAPIManager()
                if not self.groq_manager.api_keys:
                    st.warning(
                        "⚠️ No Groq API keys found in .env file. "
                        "AI answers will be disabled."
                    )
        except Exception as e:
            st.error(f"Failed to load index: {e}")
            st.stop()

    def semantic_search(
        self, 
        query: str, 
        k: int = 6, 
        filters: Optional[Dict] = None
    ) -> List[Dict]:
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
            results.append({"chunk": chunk, "score": float(sc)})
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
            "title": overview["metadata"].get("chapter_title", "")
        }

        if "topic" in stat_type or "section" in stat_type:
            result["type"] = "topics/sections"
            result["count"] = stats.get("total_topics", 0)
            result["details"] = {
                "sections": stats.get("total_sections", 0),
                "subsections": stats.get("total_subsections", 0)
            }
        elif "example" in stat_type:
            exs = [
                c for c in self.chunks
                if c["metadata"].get("type") == "example"
                and c["metadata"]["class"] == cls
                and c["metadata"]["chapter"] == intent.chapter_num
            ]
            result["type"] = "examples"
            result["count"] = len(exs)
        elif "table" in stat_type:
            result["type"] = "tables"
            result["count"] = stats.get("total_tables", 0)
        else:
            result["stats"] = stats

        return result

    @staticmethod
    def _apply_filters(meta: Dict, filters: Dict) -> bool:
        for k, v in filters.items():
            if meta.get(k) != v:
                return False
        return True

    def generate_answer(self, query: str, chunks: List[Dict]) -> str:
        if not self.groq_manager or not chunks:
            return None

        context = "\n\n".join(
            f"**Class {c['metadata']['class']} | Chapter {c['metadata']['chapter']} "
            f"| {c['metadata'].get('chapter_title','')}**\n"
            f"*Type: {c['metadata']['type']}*\n\n{c['content']}\n\n{'─'*70}"
            for c in chunks
        )

        user_prompt = PROMPT_TEMPLATE.format(context=context, question=query)

        messages = [
            {
                "role": "system",
                "content": "You are an NCERT Physics expert. Format all mathematical "
                          "content in LaTeX."
            },
            {"role": "user", "content": user_prompt}
        ]

        return self.groq_manager.generate_completion(
            messages, 
            temperature=0.25, 
            max_tokens=2048
        )

    def retrieve(self, query: str, k: int = 6) -> Dict:
        intent = QueryParser.parse_query(query)
        result = {
            "query": query,
            "intent_type": intent.query_type.value,
            "class": intent.class_num,
            "chapter": intent.chapter_num,
            "content_type": intent.content_type,
            "raw_chunks": [],
            "statistics": None,
            "llm_answer": None,
        }

        if intent.query_type == QueryType.EXACT_MATCH:
            chunk = self.exact_match_search(intent)
            if chunk:
                result["raw_chunks"] = [chunk]
        elif intent.query_type == QueryType.CHAPTER_CONTENT:
            chunk = self.chapter_content_search(intent)
            if chunk:
                result["raw_chunks"] = [chunk]
        elif intent.query_type == QueryType.METADATA:
            result["statistics"] = self.metadata_query(intent)
        else:  # HYBRID or SEMANTIC
            filters = {}
            if intent.class_num:
                filters["class"] = intent.class_num
            if intent.chapter_num:
                filters["chapter"] = intent.chapter_num
            sr = self.semantic_search(
                intent.semantic_query or query, 
                k=k, 
                filters=filters
            )
            result["raw_chunks"] = [r["chunk"] for r in sr]

        if result["raw_chunks"] and self.groq_manager:
            result["llm_answer"] = self.generate_answer(query, result["raw_chunks"])

        return result

# ────────────────────────────────────────────────
#   Streamlit App Functions
# ────────────────────────────────────────────────

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "_last_index_dir" not in st.session_state:
        st.session_state._last_index_dir = None

def display_chat_message(
    role: str, 
    content: str, 
    metadata: Optional[Dict] = None
):
    """Display a chat message with proper formatting"""
    with st.chat_message(role):
        st.markdown(content)

        if metadata and metadata.get("raw_chunks"):
            with st.expander("📚 Source Context (click to view)", expanded=False):
                for i, chunk in enumerate(metadata["raw_chunks"], 1):
                    meta = chunk["metadata"]
                    st.markdown(
                        f"**Source {i}:** Class {meta['class']} • "
                        f"Chapter {meta['chapter']} • "
                        f"{meta.get('chapter_title','')}"
                    )
                    st.caption(f"Type: {meta['type']}")
                    with st.container():
                        content_preview = chunk["content"][:800]
                        if len(chunk["content"]) > 800:
                            content_preview += " ..."
                        st.text(content_preview)
                    if i < len(metadata["raw_chunks"]):
                        st.divider()

# ────────────────────────────────────────────────
#   Main App
# ────────────────────────────────────────────────

st.set_page_config(
    page_title="NCERT Physics RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better chat interface
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    div[data-testid="stExpander"] {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 0.5rem;
    }
    .stChatInput {
        position: fixed;
        bottom: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
initialize_session_state()

# ── Sidebar ───────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")

    # Index directory
    index_dir_str = st.text_input(
        "📁 Index folder path",
        value=".",
        help="Folder containing .faiss, .json, .pkl files"
    )
    index_dir = Path(index_dir_str).resolve()

    # Number of chunks
    k_value = st.slider(
        "📊 Number of chunks (k)",
        min_value=2,
        max_value=12,
        value=6,
        step=1,
        help="Number of relevant chunks to retrieve"
    )

    st.markdown("---")

    # API Key status
    st.subheader("🔑 API Status")
    api_keys = [
        os.getenv(f"GROQ_API_KEY_{i}") 
        for i in range(1, 4)
    ] + [os.getenv("GROQ_API_KEY")]
    api_keys_count = len([k for k in api_keys if k])

    if api_keys_count > 0:
        st.success(f"✅ {api_keys_count} API key(s) loaded")
    else:
        st.error("❌ No API keys found")
        st.info(
            "Add API keys to your .env file:\n"
            "```\n"
            "GROQ_API_KEY_1=your_key_1\n"
            "GROQ_API_KEY_2=your_key_2\n"
            "GROQ_API_KEY_3=your_key_3\n"
            "```"
        )

    st.markdown("---")

    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")

    # Example queries
    st.subheader("💡 Example Queries")
    st.markdown("""
    **Exact matches:**
    - Example 5.3
    - Table 7.1
    - Summary of chapter 9

    **Search queries:**
    - Explain Newton's laws
    - What is kinetic energy?
    - Derive equations of motion

    **Statistics:**
    - How many examples in chapter 4?
    - Exercises of chapter 6
    """)

# ── Main area ─────────────────────────────────────

st.title("📚 NCERT Physics RAG System")
st.caption("Powered by Sentence Transformers + FAISS + Groq (Llama-3.3-70B)")

# Load retriever if needed
if (st.session_state.retriever is None or 
    st.session_state._last_index_dir != str(index_dir)):
    with st.spinner("🔄 Loading NCERT index... (may take 10–30 seconds)"):
        try:
            st.session_state.retriever = NCERTRetriever(index_dir)
            st.session_state._last_index_dir = str(index_dir)
            st.success("✅ Index loaded successfully!", icon="✅")
        except Exception as e:
            st.error(f"❌ Cannot load index:\n{e}")
            st.stop()

retriever = st.session_state.retriever

# Display chat history
for message in st.session_state.messages:
    display_chat_message(
        message["role"],
        message["content"],
        message.get("metadata")
    )

# Chat input at bottom
query = st.chat_input("💬 Ask about NCERT Physics...")

if query:
    # Add user message to chat
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    # Display user message
    display_chat_message("user", query)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            result = retriever.retrieve(query.strip(), k=k_value)

        # Prepare response content
        response_content = ""

        if result["statistics"]:
            # For statistics queries
            stats = result["statistics"]
            response_content = (
                f"**Statistics for Chapter {stats.get('chapter', 'N/A')}:**\n\n"
            )
            response_content += f"**Title:** {stats.get('title', 'N/A')}\n\n"

            if 'count' in stats:
                response_content += (
                    f"**{stats.get('type', 'Items')}:** {stats['count']}\n\n"
                )

            if 'details' in stats:
                response_content += "**Details:**\n"
                for key, value in stats['details'].items():
                    response_content += f"- {key}: {value}\n"

            if 'stats' in stats:
                response_content += "\n**Full Statistics:**\n"
                for key, value in stats['stats'].items():
                    response_content += f"- {key}: {value}\n"

            st.markdown(response_content)

        elif result["llm_answer"]:
            # For AI-generated answers
            response_content = result["llm_answer"]
            st.markdown(response_content)

        else:
            # No answer generated
            response_content = (
                "⚠️ No AI answer could be generated. "
                "Please check the source context below for relevant information."
            )
            st.warning(response_content)

        # Display source context
        if result["raw_chunks"]:
            with st.expander("📚 Source Context (click to view)", expanded=False):
                for i, chunk in enumerate(result["raw_chunks"], 1):
                    meta = chunk["metadata"]
                    st.markdown(
                        f"**Source {i}:** Class {meta['class']} • "
                        f"Chapter {meta['chapter']} • "
                        f"{meta.get('chapter_title','')}"
                    )
                    st.caption(f"Type: {meta['type']}")
                    with st.container():
                        content_preview = chunk["content"][:800]
                        if len(chunk["content"]) > 800:
                            content_preview += " ..."
                        st.text(content_preview)
                    if i < len(result["raw_chunks"]):
                        st.divider()

        # Show info if no content found
        if not result["raw_chunks"] and not result["statistics"]:
            response_content = (
                "ℹ️ No relevant content found for this query. "
                "Please try rephrasing your question."
            )
            st.info(response_content)

    # Add assistant message to chat history
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_content,
        "metadata": (
            {"raw_chunks": result.get("raw_chunks", [])}
            if result.get("raw_chunks")
            else None
        )
    })


# ## Setup Instructions

# ### 1. Create `.env` file:
# ```
# GROQ_API_KEY_1=gsk_your_first_key_here
# GROQ_API_KEY_2=gsk_your_second_key_here
# GROQ_API_KEY_3=gsk_your_third_key_here