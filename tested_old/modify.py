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

# ──── STREAMLIT CONFIG ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="NCERT Physics AI Tutor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──── MODERN UI STYLING ─────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .main { background: #f8fafc; padding: 1.5rem 1rem; }

    .block-container { padding: 1rem; max-width: 1200px; }

    .header-card {
        background: white;
        border-radius: 16px;
        padding: 1.8rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
        margin-bottom: 1.5rem;
    }

    .header-title { font-size: 2.1rem; font-weight: 700; color: #1e293b; }

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
        background: #dbeafe !important;
    }

    div[data-testid="stChatMessage"]:has(div[aria-label="assistant"]) {
        background: #f8fafc !important;
        border: 1px solid #e2e8f0;
    }

    .answer-content {
        background: white;
        border-radius: 12px;
        padding: 1.4rem;
        border-left: 4px solid #3b82f6;
        margin: 0.8rem 0;
        line-height: 1.65;
    }

    .latex-block {
        background: transparent !important;
        border: none !important;
        padding: 1.1rem 0 !important;
        margin: 1.3rem 0 !important;
    }

    .filter-badge {
        display: inline-flex;
        background: #eff6ff;
        color: #1d4ed8;
        padding: 0.4rem 0.9rem;
        border-radius: 9999px;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }

    .stButton > button {
        background: #3b82f6;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 1.5rem;
        font-weight: 600;
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
        font-size: 1.1rem;
        font-weight: 600;
        color: #1e293b;
        margin: 1.5rem 0 1rem 0;
        padding-left: 0.5rem;
        border-left: 4px solid #3b82f6;
    }

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

def get_llm():
    """Get LLM instance with current API key"""
    current_key = GROQ_API_KEYS[st.session_state.current_api_key_index]
    return ChatGroq(
        model_name="llama-3.3-70b-versatile",
        groq_api_key=current_key,
        temperature=0.2,
        max_tokens=1024,
        top_p=0.92,
    )

embed_model, text_index, text_chunks, image_index, image_entries, documents, bm25_retriever = load_resources()

# ──── HELPER FUNCTIONS (keeping all your original functions) ────────────────
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

def retrieve_summary(class_num, chapter_num, max_chunks=5):
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

def retrieve_points_to_ponder(class_num, chapter_num, max_chunks=5):
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

def ensemble_retrieval(query: str, query_emb, max_docs=8, class_num=None, chapter_num=None):
    try:
        docs_keyword = bm25_retriever.invoke(query) or []
        docs_vector = retrieve_text_vector(query_emb, k=5, class_num=class_num, chapter_num=chapter_num)
        
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
    
    # Only return images that are explicitly referenced
    relevant = [entry for entry in image_entries if entry['metadata'].get('fig_id') in fig_refs]
    
    return relevant[:max_images]

# Prompt templates (keeping all your original prompts)
PROMPT_TEMPLATE = """
You are an NCERT Class 11 Physics expert. Your task is to provide answers exactly as they appear in the NCERT textbook.

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
   - Subscripts: use _{{}} → v_{{0}}, F_{{net}}
   - Superscripts: use ^{{}} → 10^{{11}}, m^{{2}}
   - Greek letters: \\Delta, \\theta, \\omega, \\alpha, \\beta, etc.
   - Fractions: \\frac{{numerator}}{{denominator}}
   - Multiplication: \\times
   - Square root: \\sqrt{{x}}
   - Vectors: \\vec{{v}}, \\vec{{F}}

5. **For Tables:**
   - Use markdown table format
   - Keep formatting clean and aligned

6. **General Rules:**
   - Be accurate and match textbook style exactly
   - Reference figures naturally: "As shown in Fig. X.Y ..."
   - If no relevant info found → say "Not found in this chapter."
   - Do not add extra explanations beyond what's in the textbook

Context from NCERT:
{context}

Question:
{question}

Answer (with proper LaTeX formatting):
"""

EXAMPLE_PROMPT_TEMPLATE = """
You are reproducing **Example {num}** from the NCERT Class 11 Physics textbook EXACTLY as it appears.

**FORMAT REQUIREMENTS:**

1. Title: **Example {num}:** [Problem title if any]

2. **Problem Statement:** 
   Write the exact problem/question

3. **Given:**
   List all given values with proper units and LaTeX
   Example: 
   - Length: $L = 2.5 \\, \\text{{m}}$
   - Force: $F = 100 \\, \\text{{N}}$

4. **To Find:**
   State what needs to be calculated

5. **Solution:**
   Show complete step-by-step solution with LaTeX
   
   For multi-step calculations, use:
   $$
   \\begin{{align*}}
   \\text{{Step 1: }} & \\text{{Description}} \\\\
   & \\text{{Equation}} \\\\
   \\text{{Step 2: }} & \\text{{Next step}} \\\\
   & \\text{{Result}}
   \\end{{align*}}
   $$

6. **Final Answer:**
   Clearly state the final answer with proper units in LaTeX
   Example: $\\Delta L = 2.5 \\times 10^{{-5}} \\, \\text{{m}}$

**CRITICAL:** Use proper LaTeX formatting throughout:
- All numbers with units: $value \\, \\text{{unit}}$
- All equations: use $$ $$ blocks
- Subscripts/superscripts correctly formatted
- Use \\times for multiplication
- Use \\frac{{}}{{}} for fractions

Context from textbook:
{context}

Reproduce Example {num} exactly:
"""

DERIVATION_PROMPT_TEMPLATE = """
You are reproducing a DERIVATION from the NCERT Class 11 Physics textbook.

**DERIVATION FORMAT:**

1. **Title:** State what is being derived
   Example: "**Derivation of the equation of motion: $v = u + at$**"

2. **Given/Assumptions:**
   List any initial conditions or assumptions

3. **Derivation Steps:**
   Use this exact format:
   
   $$
   \\begin{{align*}}
   \\text{{Starting with: }} & \\text{{Initial equation}} \\\\
   & = \\text{{step 1}} \\\\
   & = \\text{{step 2}} \\\\
   \\text{{Therefore, }} & \\text{{final result}}
   \\end{{align*}}
   $$

4. **Conclusion:**
   State the final derived equation clearly

**LaTeX Requirements:**
- Use & for alignment at = or other operators
- Use \\\\ for line breaks between steps
- Add text descriptions using \\text{{description}}
- Show intermediate steps clearly
- Use proper mathematical symbols

Context:
{context}

Question:
{question}

Provide the complete derivation:
"""

SUMMARY_PROMPT_TEMPLATE = """
You are presenting the CHAPTER SUMMARY from the NCERT Physics textbook EXACTLY as it appears.

**FORMAT REQUIREMENTS:**

1. **Title:** Start with "## Summary" or "## Chapter Summary"

2. **Content Organization:**
   - Present key concepts in the order they appear in the textbook
   - Use bullet points or numbered lists if the textbook uses them
   - Keep the original structure and flow
   - Include all important formulas with proper LaTeX formatting

3. **Mathematical Content:**
   - All equations must use proper LaTeX: $equation$ for inline, $$equation$$ for display
   - Use \\begin{{align*}}...\\end{{align*}} for multi-line equations
   - Include all important formulas mentioned in the summary

4. **Key Points to Cover:**
   - Main concepts introduced in the chapter
   - Important definitions
   - Key formulas and equations
   - Important principles or laws
   - Practical applications if mentioned

5. **Style:**
   - Match the textbook's concise, clear style
   - Use simple language as in NCERT
   - Don't add extra explanations not in the original summary

Context from textbook summary section:
{context}

Present the complete chapter summary:
"""

POINTS_TO_PONDER_PROMPT_TEMPLATE = """
You are presenting the POINTS TO PONDER section from the NCERT Physics textbook EXACTLY as it appears.

**FORMAT REQUIREMENTS:**

1. **Title:** Start with "## Points to Ponder" 

2. **Format:**
   - Present as a numbered list or bullet points (match the textbook)
   - Each point should be a complete thought or question
   - Preserve the exact wording from the textbook

3. **Content:**
   - These are thought-provoking questions or observations
   - Often relate to common misconceptions
   - May include interesting facts or applications
   - Sometimes pose questions for students to think about

4. **Style:**
   - Keep the reflective, questioning tone
   - Maintain the educational purpose of making students think deeper
   - Don't answer the questions - present them as the textbook does

Context from textbook Points to Ponder section:
{context}

Present all the Points to Ponder:
"""

def render_mixed_latex(text):
    parts = re.split(r'(\$\$.*?\$\$)', text, flags=re.DOTALL)
    
    for part in parts:
        if part.startswith("$$") and part.endswith("$$"):
            latex_content = part[2:-2].strip()
            st.markdown('<div class="latex-block">', unsafe_allow_html=True)
            st.latex(latex_content)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            if '$' in part:
                inline_parts = re.split(r'(\$[^\$]+?\$)', part)
                rendered_text = ""
                
                for inline_part in inline_parts:
                    if inline_part.startswith('$') and inline_part.endswith('$') and len(inline_part) > 2:
                        rendered_text += inline_part
                    else:
                        rendered_text += inline_part
                
                if rendered_text.strip():
                    st.markdown(rendered_text)
            else:
                if part.strip():
                    st.markdown(part)

def generate_answer(context_docs, question, example_num=None, is_derivation=False, is_summary=False, is_ptp=False):
    context_str = format_docs(context_docs)
    
    if example_num:
        prompt = ChatPromptTemplate.from_template(EXAMPLE_PROMPT_TEMPLATE)
        inputs = {
            "num": example_num,
            "context": context_str
        }
    elif is_derivation:
        prompt = ChatPromptTemplate.from_template(DERIVATION_PROMPT_TEMPLATE)
        inputs = {
            "context": context_str,
            "question": question
        }
    elif is_summary:
        prompt = ChatPromptTemplate.from_template(SUMMARY_PROMPT_TEMPLATE)
        inputs = {
            "context": context_str
        }
    elif is_ptp:
        prompt = ChatPromptTemplate.from_template(POINTS_TO_PONDER_PROMPT_TEMPLATE)
        inputs = {
            "context": context_str
        }
    else:
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        inputs = {
            "context": context_str,
            "question": question
        }
    
    # Try each API key in sequence if rate limit is hit
    for attempt in range(len(GROQ_API_KEYS)):
        try:
            llm = get_llm()
            chain = prompt | llm
            response = chain.invoke(inputs)
            answer = StrOutputParser().invoke(response)
            
            tokens = {
                'input_tokens': response.usage_metadata.get('input_tokens', 0),
                'output_tokens': response.usage_metadata.get('output_tokens', 0)
            }
            
            return answer, tokens
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check if it's a rate limit error
            if 'rate limit' in error_msg or 'rate_limit' in error_msg or '429' in error_msg:
                # Move to next API key
                st.session_state.current_api_key_index = (st.session_state.current_api_key_index + 1) % len(GROQ_API_KEYS)
                
                if attempt < len(GROQ_API_KEYS) - 1:
                    st.warning(f"⚠️ Rate limit hit on API key {attempt + 1}. Switching to backup key {st.session_state.current_api_key_index + 1}...")
                    continue
                else:
                    # All keys exhausted
                    raise Exception("All API keys have hit their rate limits. Please try again later.")
            else:
                # Non-rate-limit error, raise immediately
                raise e
    
    # Should never reach here, but just in case
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
        
        if is_sum and class_num and chapter_num:
            docs_retrieved = retrieve_summary(class_num, chapter_num, max_chunks=5)
        elif is_ptp_query and class_num and chapter_num:
            docs_retrieved = retrieve_points_to_ponder(class_num, chapter_num, max_chunks=5)
        elif example_num:
            keyword_q = f"Example {example_num}"
            docs_retrieved = ensemble_retrieval(
                keyword_q, query_emb, max_docs=8,
                class_num=class_num, chapter_num=chapter_num
            )
        else:
            docs_retrieved = ensemble_retrieval(
                query, query_emb, max_docs=8,
                class_num=class_num, chapter_num=chapter_num
            )
        
        images = find_relevant_images(query, docs_retrieved, max_images=2)
        answer, tokens = generate_answer(docs_retrieved, query, example_num, is_deriv, is_sum, is_ptp_query)
        
        answer_type = 'general'
        if is_sum:
            answer_type = 'summary'
        elif is_ptp_query:
            answer_type = 'ptp'
        elif example_num:
            answer_type = 'example'
        elif is_deriv:
            answer_type = 'derivation'
        
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
            'content': f"❌ Something went wrong: {str(e)}",
            'images': [],
            'type': 'general',
            'debug_docs': [],
            'tokens': {'input_tokens': 0, 'output_tokens': 0},
            'filter_badge': None,
            'class_num': None,
            'chapter_num': None
        }

# ──── MAIN UI ───────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="header-card">
    <div class="header-title">📚 NCERT Physics AI Tutor</div>
    <div class="header-subtitle">Class 11 & 12 • Powered by Hybrid RAG • Llama 3.3 70B</div>
</div>
""", unsafe_allow_html=True)

# Success message
st.success(f"✅ System Ready: gte-small embeddings • FAISS vector search • BM25 keyword search • Groq LLM ({len(GROQ_API_KEYS)} API key{'s' if len(GROQ_API_KEYS) > 1 else ''} loaded)")

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

# Sidebar for settings
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.session_state.show_debug = st.checkbox("🔍 Show Debug Info", value=False, help="Display token usage and retrieved chunks")
    
    st.markdown("---")
    st.markdown("## 📖 Quick Guide")
    st.markdown("""
    **Ask me about:**
    - 📝 Specific examples (e.g., "Example 8.3")
    - 📐 Derivations & proofs
    - 📋 Chapter summaries
    - 💡 Points to ponder
    - ❓ Any physics concept
    
    **Pro Tips:**
    - Mention class & chapter for precise results
    - Ask for diagrams or figures
    - Request step-by-step solutions
    """)
    
    st.markdown("---")
    st.markdown("### 🎯 Example Queries")
    st.code("Show me Example 9.4 from Class 11")
    st.code("Derive the equation for work-energy theorem")
    st.code("Summarize Chapter 8 on gravitation")
    st.code("What are the points to ponder in Chapter 5?")

# Chat container
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        if msg['role'] == 'user':
            st.markdown(msg['content'])
        else:
            # Filter badge
            if msg.get('filter_badge'):
                st.markdown(f'<div class="filter-badge">🎯 {msg["filter_badge"]}</div>', unsafe_allow_html=True)
            
            # Answer box
            st.markdown('<div class="answer-content">', unsafe_allow_html=True)
            render_mixed_latex(msg['content'])
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Images
            images = msg.get('images', [])
            if images:
                st.markdown('<div class="section-header">📊 Relevant Diagrams</div>', unsafe_allow_html=True)
                cols = st.columns(min(len(images), 2))
                
                for idx, img_entry in enumerate(images):
                    path = img_entry['metadata']['path']
                    if os.path.exists(path):
                        with cols[idx % len(cols)]:
                            st.markdown('<div class="image-container">', unsafe_allow_html=True)
                            st.image(
                                path,
                                caption=img_entry['text'][:150] + "..." if len(img_entry['text']) > 150 else img_entry['text'],
                                use_container_width=True
                            )
                            fig_id = img_entry['metadata'].get('fig_id', '—')
                            page = img_entry['metadata'].get('page', '—')
                            st.caption(f"**Fig. {fig_id}** | Page {page}")
                            st.markdown('</div>', unsafe_allow_html=True)
            
            # Debug info
            if st.session_state.show_debug:
                with st.expander("🔍 Debug Information"):
                    st.markdown('<div class="debug-card">', unsafe_allow_html=True)
                    st.markdown('<div class="debug-header">Performance Metrics</div>', unsafe_allow_html=True)
                    
                    tokens = msg.get('tokens', {'input_tokens': 0, 'output_tokens': 0})
                    docs_retrieved = msg.get('debug_docs', [])
                    
                    # Token stats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown('<div class="debug-stat">', unsafe_allow_html=True)
                        st.markdown('<div class="debug-stat-label">Input Tokens</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="debug-stat-value">{tokens["input_tokens"]:,}</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="debug-stat">', unsafe_allow_html=True)
                        st.markdown('<div class="debug-stat-label">Output Tokens</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="debug-stat-value">{tokens["output_tokens"]:,}</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown('<div class="debug-stat">', unsafe_allow_html=True)
                        st.markdown('<div class="debug-stat-label">Total Tokens</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="debug-stat-value">{tokens["input_tokens"] + tokens["output_tokens"]:,}</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Retrieval info
                    st.markdown('<div class="debug-header" style="margin-top: 1.5rem;">Retrieval Information</div>', unsafe_allow_html=True)
                    
                    class_num = msg.get('class_num')
                    chapter_num = msg.get('chapter_num')
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown('<div class="debug-stat">', unsafe_allow_html=True)
                        st.markdown('<div class="debug-stat-label">Chunks Retrieved</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="debug-stat-value">{len(docs_retrieved)}</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="debug-stat">', unsafe_allow_html=True)
                        st.markdown('<div class="debug-stat-label">Retrieval Method</div>', unsafe_allow_html=True)
                        method = "Filtered Ensemble" if (class_num or chapter_num) else "Hybrid Ensemble"
                        st.markdown(f'<div class="debug-stat-value" style="font-size: 1rem;">{method}</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Filters applied
                    if class_num or chapter_num:
                        st.info(f"**Filters:** Class {class_num or 'Any'} • Chapter {chapter_num or 'Any'}")
                    
                    # Unique class-chapter combinations
                    class_chapter_combos = set()
                    for doc in docs_retrieved:
                        if isinstance(doc, Document):
                            meta = doc.metadata
                            combo = f"Class {meta.get('class', 'N/A')} - Ch {meta.get('chapter_number', 'N/A')} - {meta.get('chapter_title', 'N/A')}"
                            class_chapter_combos.add(combo)
                    
                    if class_chapter_combos:
                        st.markdown("**Sources:**")
                        for combo in sorted(class_chapter_combos):
                            st.markdown(f"  • {combo}")
                    
                    # Retrieved chunks
                    st.markdown('<div class="debug-header" style="margin-top: 1.5rem;">Retrieved Chunks</div>', unsafe_allow_html=True)
                    for i, doc in enumerate(docs_retrieved):
                        if isinstance(doc, Document):
                            with st.expander(f"📄 Chunk {i+1}"):
                                st.text(doc.page_content[:600] + ("..." if len(doc.page_content) > 600 else ""))
                                meta = doc.metadata
                                st.caption(f"Class: {meta.get('class', 'N/A')} | Chapter: {meta.get('chapter_number', 'N/A')} | Page: {meta.get('page', 'N/A')}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Handle ambiguity selection
if st.session_state.awaiting_selection and st.session_state.options:
    st.markdown('<div class="ambiguity-card">', unsafe_allow_html=True)
    
    st.markdown('<div class="ambiguity-header">🤔 Multiple Topics Found</div>', unsafe_allow_html=True)
    st.markdown('<div class="ambiguity-question">Your question could refer to different chapters. Please select the one you need:</div>', unsafe_allow_html=True)
    
    st.info(f"**Your Question:** {st.session_state.current_query}")
    
    selected_option = st.radio(
        "📚 Select your chapter:",
        options=range(len(st.session_state.options)),
        format_func=lambda i: f"Class {st.session_state.options[i]['class']} • Chapter {st.session_state.options[i]['chapter_number']} • {st.session_state.options[i]['chapter_title']}",
        key="ambiguity_selector"
    )
    
    col1, col2, col3 = st.columns([1.5, 1.5, 3])
    
    with col1:
        confirm_button = st.button("✅ Confirm Selection", use_container_width=True, type="primary")
    
    with col2:
        cancel_button = st.button("❌ Cancel", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if confirm_button:
        selected = st.session_state.options[selected_option]
        st.session_state.messages.append({'role': 'user', 'content': st.session_state.current_query})
        
        with st.spinner("🔄 Searching textbook & generating answer..."):
            assistant_msg = process_query(
                st.session_state.current_query,
                selected_class=selected['class'],
                selected_chapter=selected['chapter_number'],
                selected_title=selected['chapter_title']
            )
            st.session_state.messages.append({'role': 'assistant', **assistant_msg})
        
        st.session_state.awaiting_selection = False
        st.session_state.options = []
        st.session_state.current_query = ""
        st.rerun()
    
    if cancel_button:
        if st.session_state.pending_queries and st.session_state.current_query == st.session_state.pending_queries[0]:
            st.session_state.pending_queries.pop(0)
        st.session_state.awaiting_selection = False
        st.session_state.options = []
        st.session_state.current_query = ""
        st.rerun()

# Process pending queries
elif st.session_state.pending_queries:
    current_query = st.session_state.pending_queries.pop(0)
    st.session_state.current_query = current_query
    
    with st.spinner("🔄 Analyzing query..."):
        is_ambiguous, options = check_ambiguity(current_query, text_chunks)
        
        if is_ambiguous:
            st.session_state.awaiting_selection = True
            st.session_state.options = options
            st.rerun()
        else:
            st.session_state.messages.append({'role': 'user', 'content': current_query})
            with st.spinner("🔄 Searching textbook & generating answer..."):
                assistant_msg = process_query(current_query)
                st.session_state.messages.append({'role': 'assistant', **assistant_msg})
            st.session_state.current_query = ""
            st.rerun()

# Chat input
query = st.chat_input("🔍 Ask your physics question here...", key="query_input")

if query:
    questions = [q.strip() for q in query.split('\n') if q.strip()]
    st.session_state.pending_queries.extend(questions)
    st.rerun()

# Footer
st.markdown("""
<div class="footer-card">
    <div class="footer-title">📚 NCERT Physics AI Tutor | Class 11 & 12</div>
    <div class="footer-subtitle">Powered by Hybrid RAG (BM25 + FAISS) • GTE-Small Embeddings • Groq Llama 3.3 70B</div>
</div>
""", unsafe_allow_html=True)