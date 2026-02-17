import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
import os
import re
from dotenv import load_dotenv

load_dotenv()

# ──── CONFIG ────────────────────────────────────────────────────────────────
TEXT_INDEX_FILE     = "gte_small_full.index"
TEXT_METADATA_FILE  = "metadata_full.pkl"
IMAGE_INDEX_FILE    = "gte_small_images_11.index"
IMAGE_METADATA_FILE = "image_metadata_11.pkl"
EMBEDDING_MODEL     = "thenlper/gte-small"

# Multiple API keys for fallback when rate limits are hit
GROQ_API_KEYS = [
    os.getenv("api_key"),
    os.getenv("api_key_2"),
    os.getenv("api_key_3")
]

# Filter out None values and keep only valid keys
GROQ_API_KEYS = [key for key in GROQ_API_KEYS if key]

if not GROQ_API_KEYS:
    st.error("No GROQ_API_KEY found. Please add at least one to your .env file (api_key, api_key_2, or api_key_3).")
    st.stop()

# Track current API key index
if 'current_api_key_index' not in st.session_state:
    st.session_state.current_api_key_index = 0

# ──── TOKEN OPTIMIZATION CONFIG ─────────────────────────────────────────────
TOKEN_OPTIMIZATION = {
    'max_context_chunks': {
        'example': 4,      # Reduced from 8 for specific examples
        'derivation': 5,   # Reduced from 8 for derivations
        'summary': 3,      # Reduced from 5 for summaries
        'ptp': 3,          # Reduced from 5 for points to ponder
        'general': 6       # Reduced from 8 for general queries
    },
    'max_chunk_length': 800,  # Truncate very long chunks
    'max_tokens': 2048,        # Increased from 1024 for better quality
    'temperature': 0.1,        # Lower temperature for more focused responses
}

# ──── STREAMLIT CONFIG ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="NCERT Physics AI Tutor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──── ENHANCED UI STYLING WITH BETTER MATH DISPLAY ──────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .main { background: #f8fafc; padding: 1.5rem 1rem; }

    .block-container { padding: 1rem; max-width: 1200px; }

    .header-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3);
        text-align: center;
        margin-bottom: 1.5rem;
        color: white;
    }

    .header-title { 
        font-size: 2.3rem; 
        font-weight: 700; 
        color: white;
        margin-bottom: 0.5rem;
    }

    .header-subtitle {
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.9);
        font-weight: 400;
    }

    .chat-container {
        background: transparent;
        padding: 0;
        margin: 0;
    }

    .stChatMessage {
        border-radius: 16px;
        margin-bottom: 1.2rem;
        padding: 0.2rem !important;
    }

    div[data-testid="stChatMessage"]:has(div[aria-label="user"]) {
        background: #e0e7ff !important;
        border-left: 4px solid #6366f1;
    }

    div[data-testid="stChatMessage"]:has(div[aria-label="assistant"]) {
        background: white !important;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }

    /* ═══ ENHANCED MATHEMATICAL DISPLAY ═══ */
    
    .math-container {
        background: #fafbff;
        border-radius: 12px;
        padding: 2rem 1.5rem;
        margin: 1.5rem 0;
        border: 2px solid #e0e7ff;
        box-shadow: 0 2px 8px rgba(99, 102, 241, 0.1);
    }

    .derivation-container {
        background: linear-gradient(to right, #fefcff, #faf5ff);
        border-radius: 12px;
        padding: 2rem;
        margin: 1.5rem 0;
        border-left: 5px solid #a855f7;
        box-shadow: 0 2px 12px rgba(168, 85, 247, 0.15);
    }

    .example-container {
        background: linear-gradient(to right, #fefefe, #fafffe);
        border-radius: 12px;
        padding: 2rem;
        margin: 1.5rem 0;
        border-left: 5px solid #10b981;
        box-shadow: 0 2px 12px rgba(16, 185, 129, 0.15);
    }

    .step-box {
        background: white;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }

    .step-header {
        font-size: 1.05rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .step-number {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
        background: #3b82f6;
        color: white;
        border-radius: 50%;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .equation-block {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1.2rem 0;
        border: 2px dashed #cbd5e1;
        text-align: center;
        overflow-x: auto;
    }

    .final-answer {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        text-align: center;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
        font-size: 1.1rem;
        font-weight: 600;
    }

    .given-section, .find-section {
        background: #f1f5f9;
        border-radius: 10px;
        padding: 1.2rem;
        margin: 1rem 0;
        border-left: 4px solid #64748b;
    }

    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .latex-block {
        background: transparent !important;
        border: none !important;
        padding: 1.5rem 0 !important;
        margin: 1.5rem 0 !important;
        font-size: 1.1rem !important;
    }

    /* Inline math styling */
    .stMarkdown code {
        background: #f1f5f9;
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.95rem;
    }

    .filter-badge {
        display: inline-flex;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1.2rem;
        border-radius: 9999px;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }

    .image-container {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
    }

    .image-container img {
        width: 100% !important;
        height: 300px !important;
        object-fit: contain !important;
        border-radius: 8px;
        background: #f8fafc;
    }

    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1e293b;
        margin: 2rem 0 1rem 0;
        padding-left: 1rem;
        border-left: 5px solid #3b82f6;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .token-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        background: #fef3c7;
        color: #92400e;
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }

    .optimization-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        background: #dcfce7;
        color: #166534;
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* Debug styling improvements */
    .debug-card {
        background: #f8fafc;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
    }

    .debug-header {
        font-size: 1rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }

    .debug-stat {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }

    .debug-stat-label {
        font-size: 0.85rem;
        color: #64748b;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }

    .debug-stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e293b;
    }

    .ambiguity-card {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border: 2px solid #3b82f6;
    }

    .ambiguity-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }

    .ambiguity-question {
        font-size: 1rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }

    .footer-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 2rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }

    .footer-title {
        font-size: 1rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.25rem;
    }

    .footer-subtitle {
        font-size: 0.85rem;
        color: #64748b;
    }

    footer, #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ──── LOAD RESOURCES ────────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    
    text_index = faiss.read_index(TEXT_INDEX_FILE)
    with open(TEXT_METADATA_FILE, "rb") as f:
        text_chunks = pickle.load(f)
    
    image_index = faiss.read_index(IMAGE_INDEX_FILE)
    with open(IMAGE_METADATA_FILE, "rb") as f:
        image_entries = pickle.load(f)
    
    documents = []
    for chunk in text_chunks:
        doc = Document(
            page_content=chunk['content'],
            metadata=chunk.get('metadata', {})
        )
        documents.append(doc)
    
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 5
    
    return embed_model, text_index, text_chunks, image_index, image_entries, documents, bm25_retriever

def get_llm(query_type='general'):
    """Get LLM instance with current API key and optimized token settings"""
    current_key = GROQ_API_KEYS[st.session_state.current_api_key_index]
    
    # Adjust max_tokens based on query type
    max_tokens = TOKEN_OPTIMIZATION['max_tokens']
    if query_type in ['example', 'derivation']:
        max_tokens = min(max_tokens, 2048)  # More tokens for complex math
    elif query_type in ['summary', 'ptp']:
        max_tokens = min(max_tokens, 1536)  # Medium for summaries
    else:
        max_tokens = min(max_tokens, 1024)  # Less for general queries
    
    return ChatGroq(
        model_name="llama-3.3-70b-versatile",
        groq_api_key=current_key,
        temperature=TOKEN_OPTIMIZATION['temperature'],
        max_tokens=max_tokens,
        top_p=0.92,
    )

embed_model, text_index, text_chunks, image_index, image_entries, documents, bm25_retriever = load_resources()

# ──── HELPER FUNCTIONS ──────────────────────────────────────────────────────

def truncate_chunk(content, max_length=800):
    """Truncate chunk content to save tokens while preserving important info"""
    if len(content) <= max_length:
        return content
    
    # Try to truncate at sentence boundary
    truncated = content[:max_length]
    last_period = truncated.rfind('.')
    last_newline = truncated.rfind('\n')
    
    cut_point = max(last_period, last_newline)
    if cut_point > max_length * 0.7:  # Only use if we're not cutting too much
        return content[:cut_point + 1]
    
    return truncated + "..."

def optimize_context(docs, query_type='general', max_chunks=None):
    """Optimize context by limiting chunks and truncating content"""
    if max_chunks is None:
        max_chunks = TOKEN_OPTIMIZATION['max_context_chunks'].get(query_type, 6)
    
    # Limit number of chunks
    docs = docs[:max_chunks]
    
    # Truncate each chunk
    max_length = TOKEN_OPTIMIZATION['max_chunk_length']
    optimized_docs = []
    
    for doc in docs:
        if isinstance(doc, Document):
            truncated_content = truncate_chunk(doc.page_content, max_length)
            optimized_doc = Document(
                page_content=truncated_content,
                metadata=doc.metadata
            )
            optimized_docs.append(optimized_doc)
        else:
            optimized_docs.append(doc)
    
    return optimized_docs

def embed_query(query: str):
    prefixed = f"Represent this sentence for searching relevant passages: {query}"
    emb = embed_model.encode([prefixed], normalize_embeddings=True)[0]
    return emb.astype(np.float32).reshape(1, -1)

def is_specific_example(query):
    pattern = r"(example|sum|problem|ex|eg|ques|question|solved example|exercise)\s*[\.:]?\s*(\d+\.\d+)"
    match = re.search(pattern, query.lower())
    return match.group(2) if match else None

def is_points_to_ponder(query):
    patterns = [
        r"points?\s+to\s+ponder",
        r"ptp",
        r"point\s+to\s+ponder\s+(\d+)",
    ]
    for pattern in patterns:
        if re.search(pattern, query.lower()):
            return True
    return False

def is_summary_query(query):
    summary_keywords = [
        "summary", "summarize", "summarise", "brief overview",
        "chapter summary", "give me summary"
    ]
    return any(keyword in query.lower() for keyword in summary_keywords)

def is_derivation_query(query):
    derivation_keywords = [
        "derive", "derivation", "proof", "prove", "show that",
        "establish", "obtain the equation", "obtain the expression"
    ]
    return any(keyword in query.lower() for keyword in derivation_keywords)

def extract_class_and_chapter(query):
    class_num = None
    chapter_num = None
    
    class_patterns = [
        r"class\s*(\d+)",
        r"std\s*(\d+)",
        r"(\d+)th\s+std",
        r"(\d+)th\s+class",
        r"grade\s*(\d+)"
    ]
    for pattern in class_patterns:
        match = re.search(pattern, query.lower())
        if match:
            num = match.group(1)
            if num in ['11', '12']:
                class_num = num
                break
    
    chapter_patterns = [
        r"chapter\s*(\d+)",
        r"ch\s*(\d+)",
        r"ch\.\s*(\d+)"
    ]
    for pattern in chapter_patterns:
        match = re.search(pattern, query.lower())
        if match:
            chapter_num = match.group(1)
            break
    
    example_num = is_specific_example(query)
    if example_num and not chapter_num:
        chapter_num = example_num.split('.')[0]
    
    return class_num, chapter_num

def check_ambiguity(query, text_chunks):
    example_num = is_specific_example(query)
    is_ptp = is_points_to_ponder(query)
    is_summary = is_summary_query(query)
    
    if not (example_num or is_ptp or is_summary):
        return False, []
    
    specified_class, specified_chapter = extract_class_and_chapter(query)
    
    if specified_class and specified_chapter:
        return False, []
    
    matches = {}
    
    for chunk in text_chunks:
        metadata = chunk.get('metadata', {})
        content = chunk.get('content', '').lower()
        
        is_match = False
        
        if example_num:
            example_patterns = [
                f"example {example_num}",
                f"example{example_num}",
                f"ex {example_num}",
                f"ex. {example_num}",
                f"e {example_num}",
            ]
            is_match = any(pattern in content for pattern in example_patterns)
        elif is_ptp:
            is_match = "points to ponder" in content or "point to ponder" in content
        elif is_summary:
            is_match = ("summary" in content and 
                       (content.startswith("summary") or 
                        "chapter summary" in content or
                        "\nsummary" in content))
        
        if is_match:
            class_num = str(metadata.get('class', '')).strip()
            chapter_num = str(metadata.get('chapter_number', '')).strip()
            chapter_title = metadata.get('chapter_title', 'Unknown Chapter')
            
            if specified_class and class_num != str(specified_class).strip():
                continue
            if specified_chapter and chapter_num != str(specified_chapter).strip():
                continue
            
            if not class_num or not chapter_num:
                continue
            
            key = f"{class_num}_{chapter_num}"
            if key not in matches:
                matches[key] = {
                    'class': class_num,
                    'chapter_number': chapter_num,
                    'chapter_title': chapter_title
                }
    
    if len(matches) > 1:
        return True, list(matches.values())
    
    return False, []

def retrieve_summary(class_num, chapter_num, max_chunks=3):
    """Retrieve summary with token optimization"""
    summary_chunks = []
    
    for chunk in text_chunks:
        metadata = chunk.get('metadata', {})
        content = chunk.get('content', '').lower()
        
        chunk_class = str(metadata.get('class', '')).strip()
        chunk_chapter = str(metadata.get('chapter_number', '')).strip()
        
        if chunk_class != str(class_num).strip():
            continue
        if chunk_chapter != str(chapter_num).strip():
            continue
        
        summary_markers = [
            content.startswith("summary"),
            "\nsummary\n" in content,
            "chapter summary" in content,
            content.startswith("chapter summary"),
            ("summary" in content and len(content) > 200)
        ]
        
        if any(summary_markers):
            doc = Document(
                page_content=chunk['content'],
                metadata=metadata
            )
            summary_chunks.append(doc)
    
    return summary_chunks[:max_chunks]

def retrieve_points_to_ponder(class_num, chapter_num, max_chunks=3):
    """Retrieve points to ponder with token optimization"""
    ptp_chunks = []
    
    for chunk in text_chunks:
        metadata = chunk.get('metadata', {})
        content = chunk.get('content', '').lower()
        
        chunk_class = str(metadata.get('class', '')).strip()
        chunk_chapter = str(metadata.get('chapter_number', '')).strip()
        
        if chunk_class != str(class_num).strip():
            continue
        if chunk_chapter != str(chapter_num).strip():
            continue
        
        ptp_markers = [
            "points to ponder" in content,
            "point to ponder" in content,
            content.startswith("points to ponder"),
            "\npoints to ponder\n" in content,
        ]
        
        if any(ptp_markers):
            doc = Document(
                page_content=chunk['content'],
                metadata=metadata
            )
            ptp_chunks.append(doc)
    
    return ptp_chunks[:max_chunks]

def filter_chunks_by_metadata(chunks, class_num=None, chapter_num=None):
    filtered = []
    
    for chunk in chunks:
        metadata = chunk.get('metadata', {}) if isinstance(chunk, dict) else chunk.metadata
        
        chunk_class = str(metadata.get('class', '')).strip()
        chunk_chapter = str(metadata.get('chapter_number', '')).strip()
        
        if class_num:
            if chunk_class != str(class_num).strip():
                continue
        
        if chapter_num:
            if chunk_chapter != str(chapter_num).strip():
                continue
        
        filtered.append(chunk)
    
    return filtered

def retrieve_text_vector(query_emb, k=5, class_num=None, chapter_num=None):
    k_retrieve = k * 10 if (class_num or chapter_num) else k
    
    D, I = text_index.search(query_emb, k_retrieve)
    retrieved = []
    
    for dist, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        
        chunk = text_chunks[idx]
        metadata = chunk.get('metadata', {})
        
        chunk_class = str(metadata.get('class', '')).strip()
        chunk_chapter = str(metadata.get('chapter_number', '')).strip()
        
        if class_num:
            if chunk_class != str(class_num).strip():
                continue
        
        if chapter_num:
            if chunk_chapter != str(chapter_num).strip():
                continue
        
        doc = Document(
            page_content=chunk['content'],
            metadata=metadata
        )
        retrieved.append(doc)
        
        if len(retrieved) >= k:
            break
    
    return retrieved

def ensemble_retrieval(query: str, query_emb, max_docs=6, class_num=None, chapter_num=None):
    """Ensemble retrieval with token optimization"""
    try:
        docs_keyword = bm25_retriever.invoke(query) or []
        docs_vector = retrieve_text_vector(query_emb, k=4, class_num=class_num, chapter_num=chapter_num)
        
        if class_num or chapter_num:
            docs_keyword = filter_chunks_by_metadata(docs_keyword, class_num, chapter_num)
        
        all_docs = docs_keyword + docs_vector
        
        seen = set()
        merged = []
        for doc in all_docs:
            if not isinstance(doc, Document):
                continue
            content = doc.page_content
            if content not in seen:
                seen.add(content)
                merged.append(doc)
        
        return merged[:max_docs]
    
    except Exception as e:
        st.error(f"Retrieval failed: {str(e)}")
        return []

def format_docs(items):
    return "\n\n".join(
        item.page_content if isinstance(item, Document) else str(item)
        for item in items
    )

def find_relevant_images(query: str, retrieved_docs, max_images=2):
    """Only show images that are explicitly referenced in the retrieved content"""
    fig_refs = set()
    for doc in retrieved_docs:
        content = doc.page_content if isinstance(doc, Document) else doc.get('content', '')
        matches = re.findall(r"Fig\.?\s*(\d+\.\d+)", content, re.IGNORECASE)
        fig_refs.update(matches)
    
    relevant = [entry for entry in image_entries if entry['metadata'].get('fig_id') in fig_refs]
    
    return relevant[:max_images]

# ──── ENHANCED PROMPT TEMPLATES WITH BETTER FORMATTING ──────────────────────

PROMPT_TEMPLATE = """
You are an NCERT Physics expert. Provide answers EXACTLY as in the textbook with ENHANCED visual formatting.

**CRITICAL FORMATTING RULES:**

1. **Mathematical Equations:**
   - Use $$ $$ for display equations (centered, own line)
   - Use $ $ for inline math
   - Always use proper LaTeX symbols

2. **For Regular Content:**
   - Use clear paragraphs
   - Bold important terms using **term**
   - Use proper spacing

3. **LaTeX Symbols:**
   - Subscripts: $v_0$, $F_{{net}}$
   - Superscripts: $10^{{{{11}}}}$, $m^2$
   - Greek: $\\Delta$, $\\theta$, $\\omega$
   - Fractions: $\\frac{{{{a}}}}{{{{b}}}}$
   - Vectors: $\\vec{{{{v}}}}$

Context:
{context}

Question:
{question}

Answer:
"""

EXAMPLE_PROMPT_TEMPLATE = """
You are reproducing Example {num} from NCERT with ENHANCED formatting for visual clarity.

**STRICT FORMAT:**

**Example {num}:** [Title if any]

**Problem:**
[State the problem clearly]

**Given:**
• $parameter_1 = value_1$
• $parameter_2 = value_2$
[List all given values with LaTeX]

**To Find:**
[What needs to be calculated]

**Solution:**

**Step 1: [Description]**
$$
equation_or_formula
$$
[Brief explanation if needed]

**Step 2: [Description]**
$$
\\begin{{{{align*}}}}
step &= intermediate_result \\\\
&= next_step
\\end{{{{align*}}}}
$$

[Continue for all steps...]

**Final Answer:**
$$\\boxed{{{{final\\_result = value \\, \\text{{{{units}}}}}}}}$$

**REQUIREMENTS:**
- Each step in its own clearly marked section
- Use \\boxed{{{{}}}} for final answer
- Show ALL intermediate calculations
- Use proper units throughout
- Align multi-line equations with &

Context:
{{context}}

Reproduce Example {num}:
"""

DERIVATION_PROMPT_TEMPLATE = """
You are presenting a DERIVATION from NCERT with step-by-step clarity.

**FORMAT:**

**Derivation: [Title]**

**Given/Assumptions:**
• [List assumptions]

**Derivation:**

**Step 1: [Starting point]**
$$
initial\\_equation
$$

**Step 2: [Transformation]**
$$
\\begin{{{{align*}}}}
expression &= transformation_1 \\\\
&= transformation_2 \\\\
&= result
\\end{{{{align*}}}}
$$

[Continue with clear steps...]

**Conclusion:**
$$\\boxed{{{{final\\_equation}}}}$$

**KEY REQUIREMENTS:**
- Number each major step
- Explain the reasoning for each transformation
- Use \\begin{{{{align*}}}} for multi-line derivations
- Align at = signs using &
- Box the final result

Context:
{{context}}

Question:
{{question}}

Provide derivation:
"""

SUMMARY_PROMPT_TEMPLATE = """
Present the CHAPTER SUMMARY from NCERT textbook exactly as it appears.

**FORMAT:**

**Summary**

**Key Concepts:**
• [Concept 1 with formula if any: $formula$]
• [Concept 2]
• [Concept 3]

**Important Formulas:**
$$
formula_1
$$
$$
formula_2
$$

**Main Points:**
1. [Point 1]
2. [Point 2]

Keep it concise and organized. Use LaTeX for all mathematical expressions.

Context:
{{context}}

Present summary:
"""

POINTS_TO_PONDER_PROMPT_TEMPLATE = """
Present POINTS TO PONDER from NCERT exactly as they appear.

**FORMAT:**

**Points to Ponder**

1. [First point/question]

2. [Second point/question]

3. [Third point/question]

[Continue...]

Preserve the reflective tone. Use LaTeX for any math: $equation$

Context:
{{context}}

Present Points to Ponder:
"""

def render_enhanced_content(text, content_type='general'):
    """Enhanced rendering with special containers for different content types"""
    
    # Detect content type markers
    is_example = '**Example' in text or 'Example' in text[:50]
    is_derivation = '**Derivation' in text or 'Derivation:' in text[:50]
    is_final_answer = '\\boxed{' in text or '**Final Answer' in text
    
    # Choose container based on content type
    if is_derivation or content_type == 'derivation':
        st.markdown('<div class="derivation-container">', unsafe_allow_html=True)
    elif is_example or content_type == 'example':
        st.markdown('<div class="example-container">', unsafe_allow_html=True)
    else:
        st.markdown('<div class="math-container">', unsafe_allow_html=True)
    
    # Split into sections
    sections = re.split(r'\*\*Step \d+:', text)
    
    if len(sections) > 1:
        # Render initial part
        render_text_with_latex(sections[0])
        
        # Render each step
        for i, section in enumerate(sections[1:], 1):
            st.markdown(f'<div class="step-box">', unsafe_allow_html=True)
            st.markdown(f'<div class="step-header"><span class="step-number">{i}</span> Step {i}</div>', unsafe_allow_html=True)
            render_text_with_latex(section)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        # No steps, render normally
        render_text_with_latex(text)
    
    # Close container
    st.markdown('</div>', unsafe_allow_html=True)

def render_text_with_latex(text):
    """Render text with proper LaTeX handling"""
    # Split by display math
    parts = re.split(r'(\$\$.*?\$\$)', text, flags=re.DOTALL)
    
    for part in parts:
        if part.startswith("$$") and part.endswith("$$"):
            latex_content = part[2:-2].strip()
            
            # Check if it's a final answer
            if '\\boxed{' in latex_content:
                st.markdown('<div class="final-answer">📝 Final Answer</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="equation-block">', unsafe_allow_html=True)
            st.latex(latex_content)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Handle inline math and regular text
            if '$' in part:
                # Process inline math
                inline_parts = re.split(r'(\$[^\$]+?\$)', part)
                rendered_text = ""
                
                for inline_part in inline_parts:
                    if inline_part.startswith('$') and inline_part.endswith('$') and len(inline_part) > 2:
                        rendered_text += inline_part
                    else:
                        rendered_text += inline_part
                
                if rendered_text.strip():
                    # Check for special sections
                    if '**Given:**' in rendered_text:
                        st.markdown('<div class="given-section">', unsafe_allow_html=True)
                        st.markdown(rendered_text)
                        st.markdown('</div>', unsafe_allow_html=True)
                    elif '**To Find:**' in rendered_text or '**Find:**' in rendered_text:
                        st.markdown('<div class="find-section">', unsafe_allow_html=True)
                        st.markdown(rendered_text)
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(rendered_text)
            else:
                if part.strip():
                    st.markdown(part)

def generate_answer(context_docs, question, example_num=None, is_derivation=False, 
                   is_summary=False, is_ptp=False, query_type='general'):
    """Generate answer with token optimization"""
    
    # Optimize context before generating
    context_docs = optimize_context(context_docs, query_type)
    context_str = format_docs(context_docs)
    
    if example_num:
        prompt = ChatPromptTemplate.from_template(EXAMPLE_PROMPT_TEMPLATE)
        inputs = {"num": example_num, "context": context_str}
        query_type = 'example'
    elif is_derivation:
        prompt = ChatPromptTemplate.from_template(DERIVATION_PROMPT_TEMPLATE)
        inputs = {"context": context_str, "question": question}
        query_type = 'derivation'
    elif is_summary:
        prompt = ChatPromptTemplate.from_template(SUMMARY_PROMPT_TEMPLATE)
        inputs = {"context": context_str}
        query_type = 'summary'
    elif is_ptp:
        prompt = ChatPromptTemplate.from_template(POINTS_TO_PONDER_PROMPT_TEMPLATE)
        inputs = {"context": context_str}
        query_type = 'ptp'
    else:
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        inputs = {"context": context_str, "question": question}
    
    # Try each API key in sequence if rate limit is hit
    for attempt in range(len(GROQ_API_KEYS)):
        try:
            llm = get_llm(query_type)
            chain = prompt | llm
            response = chain.invoke(inputs)
            answer = StrOutputParser().invoke(response)
            
            tokens = {
                'input_tokens': response.usage_metadata.get('input_tokens', 0),
                'output_tokens': response.usage_metadata.get('output_tokens', 0),
                'chunks_used': len(context_docs),
                'optimization_applied': True
            }
            
            return answer, tokens
            
        except Exception as e:
            error_msg = str(e).lower()
            
            if 'rate limit' in error_msg or 'rate_limit' in error_msg or '429' in error_msg:
                st.session_state.current_api_key_index = (st.session_state.current_api_key_index + 1) % len(GROQ_API_KEYS)
                
                if attempt < len(GROQ_API_KEYS) - 1:
                    st.warning(f"⚠️ Rate limit hit. Switching to backup key...")
                    continue
                else:
                    raise Exception("All API keys have hit their rate limits. Please try again later.")
            else:
                raise e
    
    raise Exception("Failed to generate answer with all available API keys.")

def process_query(query, selected_class=None, selected_chapter=None, selected_title=None):
    try:
        query_emb = embed_query(query)
        
        example_num = is_specific_example(query)
        is_deriv = is_derivation_query(query)
        is_sum = is_summary_query(query)
        is_ptp_query = is_points_to_ponder(query)
        
        class_num, chapter_num = extract_class_and_chapter(query)
        if selected_class:
            class_num = selected_class
        if selected_chapter:
            chapter_num = selected_chapter
        
        # Determine query type for optimization
        if is_sum:
            query_type = 'summary'
            docs_retrieved = retrieve_summary(class_num, chapter_num)
        elif is_ptp_query:
            query_type = 'ptp'
            docs_retrieved = retrieve_points_to_ponder(class_num, chapter_num)
        elif example_num:
            query_type = 'example'
            keyword_q = f"Example {example_num}"
            max_docs = TOKEN_OPTIMIZATION['max_context_chunks']['example']
            docs_retrieved = ensemble_retrieval(keyword_q, query_emb, max_docs=max_docs,
                                               class_num=class_num, chapter_num=chapter_num)
        elif is_deriv:
            query_type = 'derivation'
            max_docs = TOKEN_OPTIMIZATION['max_context_chunks']['derivation']
            docs_retrieved = ensemble_retrieval(query, query_emb, max_docs=max_docs,
                                               class_num=class_num, chapter_num=chapter_num)
        else:
            query_type = 'general'
            max_docs = TOKEN_OPTIMIZATION['max_context_chunks']['general']
            docs_retrieved = ensemble_retrieval(query, query_emb, max_docs=max_docs,
                                               class_num=class_num, chapter_num=chapter_num)
        
        images = find_relevant_images(query, docs_retrieved, max_images=2)
        answer, tokens = generate_answer(docs_retrieved, query, example_num, is_deriv, 
                                        is_sum, is_ptp_query, query_type)
        
        answer_type = query_type
        
        filter_badge = None
        if class_num or chapter_num:
            filter_parts = []
            if class_num:
                filter_parts.append(f"Class {class_num}")
            if chapter_num:
                filter_parts.append(f"Chapter {chapter_num}")
            if selected_title:
                filter_parts.append(selected_title)
            filter_badge = " • ".join(filter_parts)
        
        return {
            'content': answer,
            'images': images,
            'type': answer_type,
            'debug_docs': docs_retrieved,
            'tokens': tokens,
            'filter_badge': filter_badge,
            'class_num': class_num,
            'chapter_num': chapter_num
        }
    
    except Exception as e:
        return {
            'content': f"❌ Error: {str(e)}",
            'images': [],
            'type': 'general',
            'debug_docs': [],
            'tokens': {'input_tokens': 0, 'output_tokens': 0, 'chunks_used': 0},
            'filter_badge': None,
            'class_num': None,
            'chapter_num': None
        }

# ──── MAIN UI ───────────────────────────────────────────────────────────────

st.markdown("""
<div class="header-card">
    <div class="header-title">📚 NCERT Physics AI Tutor</div>
    <div class="header-subtitle">Class 11 & 12 • Token-Optimized • Enhanced Math Display</div>
</div>
""", unsafe_allow_html=True)

st.success(f"✅ System Ready: {len(GROQ_API_KEYS)} API key(s) • Token Optimization Enabled • Enhanced Math Rendering")

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'pending_queries' not in st.session_state:
    st.session_state.pending_queries = []
if 'awaiting_selection' not in st.session_state:
    st.session_state.awaiting_selection = False
if 'options' not in st.session_state:
    st.session_state.options = []
if 'current_query' not in st.session_state:
    st.session_state.current_query = ""
if 'show_debug' not in st.session_state:
    st.session_state.show_debug = False
if 'total_tokens_used' not in st.session_state:
    st.session_state.total_tokens_used = 0

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.session_state.show_debug = st.checkbox("🔍 Debug Info", value=False)
    
    # Token usage stats
    st.markdown("---")
    st.markdown("## 📊 Token Usage")
    st.metric("Total Tokens Used", f"{st.session_state.total_tokens_used:,}")
    
    if st.button("🔄 Reset Counter"):
        st.session_state.total_tokens_used = 0
        st.rerun()
    
    st.markdown("---")
    st.markdown("## 💡 Optimization Info")
    st.info("""
    **Active Optimizations:**
    • Reduced context chunks
    • Smart content truncation
    • Adaptive token limits
    • Efficient retrieval
    """)
    
    st.markdown("---")
    st.markdown("## 📖 Quick Guide")
    st.markdown("""
    **Ask about:**
    - 📝 Examples (e.g., "Example 8.3")
    - 📐 Derivations
    - 📋 Summaries
    - 💡 Points to ponder
    """)

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        if msg['role'] == 'user':
            st.markdown(msg['content'])
        else:
            # Filter badge
            if msg.get('filter_badge'):
                st.markdown(f'<div class="filter-badge">🎯 {msg["filter_badge"]}</div>', 
                          unsafe_allow_html=True)
            
            # Token optimization badge
            tokens = msg.get('tokens', {})
            if tokens.get('optimization_applied'):
                total = tokens.get('input_tokens', 0) + tokens.get('output_tokens', 0)
                chunks = tokens.get('chunks_used', 0)
                st.markdown(f'<span class="optimization-badge">⚡ Optimized • {chunks} chunks • {total:,} tokens</span>', 
                          unsafe_allow_html=True)
            
            # Enhanced content rendering
            content_type = msg.get('type', 'general')
            render_enhanced_content(msg['content'], content_type)
            
            # Images
            images = msg.get('images', [])
            if images:
                st.markdown('<div class="section-header">📊 Referenced Diagrams</div>', 
                          unsafe_allow_html=True)
                cols = st.columns(min(len(images), 2))
                
                for idx, img_entry in enumerate(images):
                    path = img_entry['metadata']['path']
                    if os.path.exists(path):
                        with cols[idx % len(cols)]:
                            st.markdown('<div class="image-container">', unsafe_allow_html=True)
                            st.image(path, use_container_width=True,
                                   caption=f"Fig. {img_entry['metadata'].get('fig_id', '—')}")
                            st.markdown('</div>', unsafe_allow_html=True)
            
            # Debug info
            if st.session_state.show_debug:
                with st.expander("🔍 Debug Information"):
                    tokens = msg.get('tokens', {})
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Input", f"{tokens.get('input_tokens', 0):,}")
                    with col2:
                        st.metric("Output", f"{tokens.get('output_tokens', 0):,}")
                    with col3:
                        st.metric("Total", f"{tokens.get('input_tokens', 0) + tokens.get('output_tokens', 0):,}")
                    with col4:
                        st.metric("Chunks", tokens.get('chunks_used', 0))

# Handle ambiguity selection
if st.session_state.awaiting_selection and st.session_state.options:
    st.markdown('<div class="ambiguity-card">', unsafe_allow_html=True)
    st.markdown('<div class="ambiguity-header">🤔 Multiple Topics Found</div>', unsafe_allow_html=True)
    
    selected_option = st.radio(
        "📚 Select chapter:",
        options=range(len(st.session_state.options)),
        format_func=lambda i: f"Class {st.session_state.options[i]['class']} • Ch {st.session_state.options[i]['chapter_number']} • {st.session_state.options[i]['chapter_title']}"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Confirm", use_container_width=True, type="primary"):
            selected = st.session_state.options[selected_option]
            st.session_state.messages.append({'role': 'user', 'content': st.session_state.current_query})
            
            with st.spinner("🔄 Processing..."):
                result = process_query(st.session_state.current_query,
                                     selected_class=selected['class'],
                                     selected_chapter=selected['chapter_number'],
                                     selected_title=selected['chapter_title'])
                st.session_state.messages.append({'role': 'assistant', **result})
                
                # Update token counter
                tokens = result.get('tokens', {})
                st.session_state.total_tokens_used += tokens.get('input_tokens', 0) + tokens.get('output_tokens', 0)
            
            st.session_state.awaiting_selection = False
            st.session_state.options = []
            st.rerun()
    
    with col2:
        if st.button("❌ Cancel", use_container_width=True):
            st.session_state.awaiting_selection = False
            st.session_state.options = []
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# Process pending queries
elif st.session_state.pending_queries:
    current_query = st.session_state.pending_queries.pop(0)
    st.session_state.current_query = current_query
    
    with st.spinner("🔄 Analyzing..."):
        is_ambiguous, options = check_ambiguity(current_query, text_chunks)
        
        if is_ambiguous:
            st.session_state.awaiting_selection = True
            st.session_state.options = options
            st.rerun()
        else:
            st.session_state.messages.append({'role': 'user', 'content': current_query})
            with st.spinner("🔄 Generating answer..."):
                result = process_query(current_query)
                st.session_state.messages.append({'role': 'assistant', **result})
                
                # Update token counter
                tokens = result.get('tokens', {})
                st.session_state.total_tokens_used += tokens.get('input_tokens', 0) + tokens.get('output_tokens', 0)
            
            st.rerun()

# Chat input
query = st.chat_input("🔍 Ask your physics question...")

if query:
    questions = [q.strip() for q in query.split('\n') if q.strip()]
    st.session_state.pending_queries.extend(questions)
    st.rerun()

# Footer
st.markdown("""
<div class="footer-card">
    <div class="footer-title">📚 NCERT Physics AI Tutor • Enhanced Edition</div>
    <div class="footer-subtitle">Token-Optimized Retrieval • Enhanced Math Display • Multi-API Fallback</div>
</div>
""", unsafe_allow_html=True)