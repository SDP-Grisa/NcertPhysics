import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
import re
import json
import hashlib
from pathlib import Path
from langchain_core.documents import Document

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    st.warning("⚠️ PyMuPDF not installed. Image extraction will be disabled. Install with: pip install PyMuPDF")

# ──── CONFIG ────────────────────────────────────────────────────
CHUNKS_FILE = "chunks_ncert_physics_images.json"
IMAGES_DIR = "extracted_images"
IMAGE_METADATA_FILE = "image_metadata.json"

# Known chapters
CHAPTER_LIST1 = [
    "1 - Physical World",
    "2 - Units and Measurements",
    "3 - Motion in a Straight Line",
    "4 - Motion in a Plane",
    "5 - Laws of Motion",
    "6 - Work, Energy and Power",
    "7 - Systems of Particles and Rotational Motion",
    "8 - Mechanical Properties of Solids",
    "9 - Mechanical Properties of Fluids",
    "10 - Thermal Properties of Matter",
    "11 - Thermodynamics",
    "12 - Kinetic Theory",
    "13 - Oscillations",
    "14 - Waves",
]

CHAPTER_LIST = [
    "1 - Electric Charges and Fields",
    "2 - Electrostatic Potential and Capacitance",
    "3 - Current Electricity",
    "4 - Moving Charges and Magnetism",
    "5 - Magnetism and Matter",
    "6 - Electromagnetic Induction",
    "7 - Alternating Current",
    "8 - Electromagnetic Waves",
    "9 - Ray Optics and Optical Instruments",
    "10 - Wave Optics",
    "11 - Dual Nature of Radiation and Matter",
    "12 - Atoms",
    "13 - Nuclei",
    "14 - Semiconductor Electronics: Materials, Devices and Simple Circuits",
]

# ──── ENVIRONMENT ───────────────────────────────────────────────
load_dotenv()
# GROQ_API_KEY = os.getenv("api_key")
# HF_TOKEN = os.getenv("hf_token")

# ──── STREAMLIT CONFIG ──────────────────────────────────────────
st.set_page_config(page_title="NCERT Physics RAG Solver", layout="wide")
st.title("📘 NCERT Class 11 & 12 Physics RAG Solver with Diagrams")

# Add custom CSS
st.markdown("""
<style>
    .stMarkdown {
        font-size: 16px;
        line-height: 1.6;
    }
    .latex-block {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        border-left: 4px solid #4CAF50;
    }
    .derivation-box {
        background-color: #fff3cd;
        padding: 20px;
        border-radius: 8px;
        margin: 15px 0;
        border: 2px solid #ffc107;
    }
    .example-box {
        background-color: #e7f3ff;
        padding: 20px;
        border-radius: 8px;
        margin: 15px 0;
        border: 2px solid #2196F3;
    }
    .diagram-box {
        background-color: #f0f0f0;
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
        border: 2px solid #9C27B0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class_std = st.selectbox("Select Class", ["11", "12"])
selected_chapter = st.selectbox("Select Chapter", CHAPTER_LIST)
uploaded_file = st.file_uploader(f"Upload PDF for {selected_chapter} (Class {class_std})", type=["pdf"])

def render_mixed_latex(text):
    """Enhanced LaTeX rendering with better formatting."""
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

# ──── IMAGE EXTRACTION FUNCTIONS ────────────────────────────────
def get_diagram_bbox(page, caption_block, split_x):
    """Finds diagrams strictly within the same column as the caption."""
    caption_rect = fitz.Rect(caption_block[:4])
    is_left_col = caption_rect.x0 < split_x
    
    # Define Column Boundaries
    col_min_x = 0 if is_left_col else split_x
    col_max_x = split_x if is_left_col else page.rect.width
    
    # Define Vertical Search Area (Above the caption)
    search_area = fitz.Rect(col_min_x, caption_rect.y0 - 300, col_max_x, caption_rect.y0)
    
    # Filter drawings within column
    drawings = [
        d["rect"] for d in page.get_drawings() 
        if d["rect"].intersects(search_area) and 
           d["rect"].x0 >= col_min_x and 
           d["rect"].x1 <= col_max_x
    ]
    
    if not drawings:
        return None
        
    diagram_box = drawings[0]
    for d_rect in drawings[1:]:
        diagram_box |= d_rect
        
    return diagram_box + (-5, -5, 5, 5)

def extract_images_from_pdf(pdf_path, chapter_num, chapter_title, class_std):
    """Extract all figures/diagrams from PDF with their captions."""
    if not PYMUPDF_AVAILABLE:
        return []
    
    # Create images directory
    os.makedirs(IMAGES_DIR, exist_ok=True)
    
    # Chapter-specific subdirectory
    chapter_dir = os.path.join(IMAGES_DIR, f"class_{class_std}_ch_{chapter_num}")
    os.makedirs(chapter_dir, exist_ok=True)
    
    doc = fitz.open(pdf_path)
    extracted_images = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        split_x = page.rect.width / 2
        
        # Get all text blocks
        blocks = page.get_text("blocks")
        
        for b in blocks:
            text = b[4].strip()
            
            # Match figure captions like "Fig. 8.1", "Figure 8.2", etc.
            match = re.search(r"Fig\.?\s+(\d+\.\d+)", text, re.IGNORECASE)
            
            if match:
                area = get_diagram_bbox(page, b, split_x)
                if area:
                    # Extract figure ID (e.g., '8.1') and format as 'fig_8_1.png'
                    fig_id = match.group(1).replace('.', '_')
                    img_filename = f"fig_{fig_id}.png"
                    img_path = os.path.join(chapter_dir, img_filename)
                    
                    # Extract image with high resolution
                    pix = page.get_pixmap(clip=area, matrix=fitz.Matrix(3, 3))
                    pix.save(img_path)
                    
                    # Store metadata
                    image_info = {
                        "fig_id": match.group(1),
                        "path": img_path,
                        "caption": text,
                        "chapter": chapter_num,
                        "chapter_title": chapter_title,
                        "class": class_std,
                        "page": page_num + 1,
                        "is_left": area.x0 < split_x
                    }
                    extracted_images.append(image_info)
    
    doc.close()
    return extracted_images

def load_image_metadata(filepath=IMAGE_METADATA_FILE):
    """Load image metadata from JSON file."""
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Failed to load image metadata: {str(e)}")
        return []

def save_image_metadata(metadata, filepath=IMAGE_METADATA_FILE):
    """Save image metadata to JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

def find_relevant_images(query, chapter_num, class_std, all_image_metadata, retrieved_docs=None, max_images=2):
    """
    Find images based on:
    1. Explicit figure reference in query (e.g., "Fig 2.3")
    2. Figure references found in retrieved text chunks
    3. Simple caption overlap with query terms
    """
    
    # Filter by chapter and class
    chapter_images = [
        img for img in all_image_metadata 
        if img["chapter"] == chapter_num and img["class"] == class_std
    ]
    
    if not chapter_images:
        return []
    
    query_lower = query.lower()
    
    # Priority 1: Check for explicit figure reference in query
    fig_match = re.search(r"fig\.?\s*(\d+\.\d+)", query_lower)
    if fig_match:
        fig_id = fig_match.group(1)
        exact_match = [img for img in chapter_images if img["fig_id"] == fig_id]
        if exact_match:
            return exact_match[:1]
    
    # Priority 2: Extract figure references from retrieved documents
    figure_refs_from_docs = set()
    if retrieved_docs:
        for doc in retrieved_docs:
            doc_text = doc.page_content if isinstance(doc, Document) else str(doc)
            # Find all Fig references in retrieved chunks
            fig_matches = re.findall(r"Fig\.?\s+(\d+\.\d+)", doc_text, re.IGNORECASE)
            figure_refs_from_docs.update(fig_matches)
    
    # If we found figure references in retrieved docs, use those
    if figure_refs_from_docs:
        referenced_images = [
            img for img in chapter_images 
            if img["fig_id"] in figure_refs_from_docs
        ]
        if referenced_images:
            # Return images in the order they appear in figure references
            return referenced_images[:max_images]
    
    # Priority 3: Simple fallback - check if query words appear in caption
    # (Only if no figure references were found)
    # This is a minimal fallback to catch cases where the text doesn't explicitly mention figure numbers
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                  'of', 'with', 'is', 'are', 'was', 'were', 'what', 'how', 'explain',
                  'show', 'me', 'display', 'give', 'tell', 'about', 'describe', 'between'}
    
    query_terms = [word for word in query_lower.split() if word not in stop_words and len(word) > 3]
    
    if not query_terms:
        return []
    
    # Count how many query terms appear in each caption
    caption_matches = []
    for img in chapter_images:
        caption_lower = img["caption"].lower()
        match_count = sum(1 for term in query_terms if term in caption_lower)
        
        if match_count > 0:
            caption_matches.append((match_count, img))
    
    # Sort by number of matching terms (descending)
    caption_matches.sort(reverse=True, key=lambda x: x[0])
    
    # Only return if at least 2 terms match (or 1 term if query is short)
    min_matches = 1 if len(query_terms) <= 2 else 2
    relevant_images = [img for count, img in caption_matches if count >= min_matches]
    
    return relevant_images[:max_images]

# ──── SAFE FORMAT DOCS ──────────────────────────────────────────
def format_docs(items):
    return "\n\n".join(
        item.page_content if isinstance(item, Document) else str(item)
        for item in items
    )

# ──── CUSTOM ENSEMBLE ───────────────────────────────────────────
def ensemble_fn(query: str, bm25_ret, vec_ret, max_docs=8):
    try:
        docs_keyword = bm25_ret.invoke(query) or []
        docs_vector   = vec_ret.invoke(query)   or []
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

# ──── CHUNK PERSISTENCE ─────────────────────────────────────────
def load_all_chunks(filepath=CHUNKS_FILE):
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        docs = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in data]
        st.info(f"Loaded {len(docs)} existing chunks from disk")
        return docs
    except Exception as e:
        st.warning(f"Failed to load chunks: {str(e)}")
        return []

def save_all_chunks(chunks, filepath=CHUNKS_FILE):
    data = [{"page_content": d.page_content, "metadata": d.metadata} for d in chunks]
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    st.success(f"Saved {len(chunks)} chunks to {filepath}")

def content_hash(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()

# ──── PROCESS & APPEND NEW CHAPTER ──────────────────────────────
def process_chapter(pdf_path, class_std, chapter_title):
    existing_chunks = load_all_chunks()
    existing_images = load_image_metadata()

    # Extract chapter number
    chapter_num = re.match(r"(\d+)", chapter_title).group(1) if re.match(r"(\d+)", chapter_title) else "?"
    
    # Extract images first
    if PYMUPDF_AVAILABLE:
        st.info("📸 Extracting diagrams and figures...")
        new_images = extract_images_from_pdf(pdf_path, chapter_num, chapter_title, class_std)
        
        # Merge with existing images (avoid duplicates)
        existing_img_paths = {img["path"] for img in existing_images}
        unique_images = [img for img in new_images if img["path"] not in existing_img_paths]
        
        if unique_images:
            all_images = existing_images + unique_images
            save_image_metadata(all_images)
            st.success(f"✅ Extracted {len(unique_images)} new diagrams")
        else:
            st.info("No new diagrams found (already extracted)")
    
    # Process text chunks
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    for doc in docs:
        doc.metadata["class"] = class_std
        doc.metadata["chapter_number"] = chapter_num
        doc.metadata["chapter_title"] = chapter_title

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=[
            "\n\n", "\n",
            r"^\d+\.\d+\s",
            r"► \*\*Example \d+\.\d+",
            r"Table \d+\.\d+",
            r"Fig\. \d+\.\d+",
            "SUMMARY", "POINTS TO PONDER", "EXERCISES",
            r"^\d+\.\d+\s",
            "Answer", "Solution",
            r"\.\s+\n", r"!\s+\n", r"\?\s+\n",
            " ", ""
        ],
        keep_separator=True,
    )

    new_chunks = splitter.split_documents(docs)

    # Merge special sections
    def merge_special(chunks):
        merged = []
        buffer = ""
        in_special = False
        markers = ["Example", "Table", "Fig.", "Answer", "Solution"]
        calc_pat = re.compile(r"ΔL| Y =|10\^|\d+\.\d+ \times|∫|align")

        for chunk in chunks:
            txt = chunk.page_content.strip()
            if any(m in txt for m in markers) or calc_pat.search(txt):
                if not in_special:
                    if buffer:
                        merged.append(Document(page_content=buffer.strip(), metadata=chunk.metadata))
                        buffer = ""
                    in_special = True
                buffer += "\n" + txt
            else:
                if in_special:
                    merged.append(Document(page_content=buffer.strip(), metadata=chunk.metadata))
                    buffer = ""
                    in_special = False
                merged.append(chunk)
        if buffer:
            merged.append(Document(page_content=buffer.strip(), metadata=chunk.metadata))
        return merged

    new_merged = merge_special(new_chunks)

    # Deduplicate chunks
    existing_hashes = {content_hash(d.page_content) for d in existing_chunks}
    unique_new = [c for c in new_merged if content_hash(c.page_content) not in existing_hashes]

    if unique_new:
        all_chunks = existing_chunks + unique_new
        save_all_chunks(all_chunks)
        st.success(f"Appended {len(unique_new)} new chunks for {chapter_title}")
    else:
        all_chunks = existing_chunks
        st.info("No new unique chunks — already processed?")

    return all_chunks

# ──── BUILD RETRIEVERS FROM CHUNKS ──────────────────────────────
@st.cache_resource
def get_retrievers(_chunks):
    """Build retrievers from chunks."""
    if not _chunks:
        return None, None, None
    
    embeddings = HuggingFaceEndpointEmbeddings(
        model="BAAI/bge-large-en-v1.5",
        huggingfacehub_api_token=HF_TOKEN,
        task="feature-extraction",
    )

    vectordb = FAISS.from_documents(_chunks, embeddings)
    vector_ret = vectordb.as_retriever(search_kwargs={"k": 5})

    bm25_ret = BM25Retriever.from_documents(_chunks)
    bm25_ret.k = 5

    return vector_ret, bm25_ret, vectordb

# ──── ENHANCED PROMPTS ──────────────────────────────────────────
PROMPT_TEMPLATE = """
You are an NCERT Class {class_std} Physics expert. Your task is to provide answers exactly as they appear in the NCERT textbook.

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
   
3. **For Examples/Problems:**
   - Start with "**Example X.Y:**" in bold
   - Show the **Given:** section with all known values
   - Show the **To Find:** section
   - Show the **Solution:** with step-by-step calculation
   - Use proper LaTeX symbols
   - Show final answer clearly with units

4. **When diagrams are mentioned:**
   - Reference them naturally (e.g., "As shown in Fig. 8.1")
   - Don't try to describe the diagram in detail - the actual image will be displayed
   
5. **Proper LaTeX Symbols:**
   - Subscripts: _{{}} → v_{{0}}, F_{{net}}
   - Superscripts: ^{{}} → 10^{{11}}, m^{{2}}
   - Greek letters: \\Delta, \\theta, \\omega, \\alpha, etc.
   - Fractions: \\frac{{num}}{{denom}}
   - Vectors: \\vec{{v}}

Context from NCERT:
{context}

Question:
{question}

Answer (with proper LaTeX formatting):
"""

EXAMPLE_PROMPT_TEMPLATE = """
You are reproducing **Example {num}** from the NCERT Class {class_std} Physics textbook EXACTLY as it appears.

Follow the exact format with Given, To Find, Solution, and Final Answer sections.
Use proper LaTeX formatting throughout.

Context from textbook:
{context}

Reproduce Example {num} exactly:
"""

DERIVATION_PROMPT_TEMPLATE = """
You are reproducing a DERIVATION from the NCERT Class {class_std} Physics textbook.

Use proper alignment with \\begin{{align*}}...\\end{{align*}} inside $$ $$ blocks.
Show all intermediate steps clearly.

Context:
{context}

Question:
{question}

Provide the complete derivation:
"""

# ──── LLM ───────────────────────────────────────────────────────
# llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2, api_key=GROQ_API_KEY)

# ──── HELPERS ───────────────────────────────────────────────────
def is_specific_example(query):
    pattern = r"(example|sum|problem|ex|eg|ques|question|solved example|exercise)\s*[\.:]?\s*(\d+\.\d+)"
    match = re.search(pattern, query.lower())
    return match.group(2) if match else None

def is_derivation_query(query):
    """Check if query is asking for a derivation."""
    derivation_keywords = [
        "derive", "derivation", "proof", "prove", "show that",
        "establish", "obtain the equation", "obtain the expression"
    ]
    return any(keyword in query.lower() for keyword in derivation_keywords)

def get_metadata_filter(query, class_std):
    match = re.search(r"chapter\s*(\d+)", query.lower())
    if match:
        ch_num = match.group(1)
        return {"class": class_std, "chapter_number": ch_num}
    return None

def extract_chapter_from_query(query):
    """Extract chapter number from query."""
    match = re.search(r"chapter\s*(\d+)", query.lower())
    if match:
        return match.group(1)
    return None

# ──── MAIN LOGIC ────────────────────────────────────────────────
all_chunks = load_all_chunks()
all_images = load_image_metadata()

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    st.success(f"Processing {selected_chapter} (Class {class_std})...")
    all_chunks = process_chapter(pdf_path, class_std, selected_chapter)
    all_images = load_image_metadata()  # Reload after processing
    
    # Clean up temp file
    try:
        os.unlink(pdf_path)
    except:
        pass

# Check if we have chunks
if not all_chunks:
    st.warning("⚠️ No chunks found. Please upload at least one chapter PDF to get started.")
    st.info("👆 Use the file uploader above to upload an NCERT Physics chapter PDF.")
    st.stop()

# Build retrievers
vector_retriever, bm25_retriever, vectordb = get_retrievers(all_chunks)

if vector_retriever is None:
    st.error("Failed to initialize retrievers. Please check your chunks.")
    st.stop()

st.success(f"✅ Index ready with {len(all_chunks)} chunks and {len(all_images)} diagrams! Ask any question below.")

# Example questions
with st.expander("💡 Example Questions"):
    st.markdown("""
    **For Examples/Problems:**
    - "Show me Example 8.3"
    - "Solve Example 5.2"
    
    **For Derivations:**
    - "Derive the equation v² = u² + 2as"
    - "Show the derivation of Young's modulus"
    
    **For Concepts with Diagrams:**
    - "Explain stress and strain curve"
    - "Show me the stress-strain diagram"
    - "Explain Fig. 8.1"
    
    **For Chapter Summaries:**
    - "Summarize chapter 8"
    - "Key points from chapter on motion"
    """)

question = st.text_input("Ask a question (diagrams will be displayed automatically)")

if question:
    with st.spinner("Thinking..."):
        try:
            # Determine chapter context
            md_filter = get_metadata_filter(question, class_std)
            query_chapter = extract_chapter_from_query(question)
            
            if not query_chapter:
                # Try to infer from selected chapter
                chapter_match = re.match(r"(\d+)", selected_chapter)
                if chapter_match:
                    query_chapter = chapter_match.group(1)
            
            # Retrieve text chunks FIRST
            current_vector_retriever = vector_retriever
            if md_filter:
                current_vector_retriever = vectordb.as_retriever(
                    search_kwargs={"k": 6, "filter": md_filter}
                )

            example_num = is_specific_example(question)
            is_derivation = is_derivation_query(question)

            # Get relevant documents based on query type
            if example_num:
                keyword_q = f"Example {example_num}"
                docs = bm25_retriever.invoke(keyword_q)
                if not docs:
                    docs = ensemble_fn(keyword_q, bm25_retriever, current_vector_retriever)
            else:
                docs = ensemble_fn(question, bm25_retriever, current_vector_retriever, max_docs=10 if is_derivation else 8)
            
            # NOW find relevant images based on both query AND retrieved documents
            relevant_images = []
            if query_chapter:
                relevant_images = find_relevant_images(
                    question, 
                    query_chapter, 
                    class_std, 
                    all_images, 
                    retrieved_docs=docs,  # Pass retrieved docs
                    max_images=2
                )
            
            # Display images FIRST if found
            if relevant_images:
                st.markdown("## 📊 Relevant Diagram(s)")
                for idx, img_info in enumerate(relevant_images, 1):
                    if os.path.exists(img_info["path"]):
                        st.markdown('<div class="diagram-box">', unsafe_allow_html=True)
                        st.image(img_info["path"], caption=img_info["caption"], use_container_width=True)
                        st.markdown(f"**Figure {img_info['fig_id']}** | Page {img_info['page']}")
                        st.markdown('</div>', unsafe_allow_html=True)
            
            # Generate text answer
            if example_num:
                exact_prompt = ChatPromptTemplate.from_template(EXAMPLE_PROMPT_TEMPLATE)
                chain = exact_prompt | llm | StrOutputParser()
                answer = chain.invoke({
                    "num": example_num,
                    "class_std": class_std,
                    "context": format_docs(docs)
                })
                
                st.markdown('<div class="example-box">', unsafe_allow_html=True)
                st.markdown("## 📝 Example Solution")
                render_mixed_latex(answer)
                st.markdown('</div>', unsafe_allow_html=True)
                
            elif is_derivation:
                prompt = ChatPromptTemplate.from_template(DERIVATION_PROMPT_TEMPLATE)
                chain = prompt | llm | StrOutputParser()
                answer = chain.invoke({
                    "class_std": class_std,
                    "context": format_docs(docs),
                    "question": question
                })
                
                st.markdown('<div class="derivation-box">', unsafe_allow_html=True)
                st.markdown("## 🔬 Derivation")
                render_mixed_latex(answer)
                st.markdown('</div>', unsafe_allow_html=True)
                
            else:
                prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
                chain = prompt | llm | StrOutputParser()
                answer = chain.invoke({
                    "class_std": class_std,
                    "context": format_docs(docs),
                    "question": question
                })
                
                st.markdown("## 📖 Answer")
                render_mixed_latex(answer)

            # Show retrieved chunks
            with st.expander("📄 Retrieved Chunks & Metadata"):
                if docs:
                    for i, doc in enumerate(docs):
                        meta = doc.metadata
                        st.write(f"**Chunk {i+1}**  |  Class {meta.get('class')} | Chapter {meta.get('chapter_number')} - {meta.get('chapter_title','')}")
                        st.write(doc.page_content[:600] + ("..." if len(doc.page_content) > 600 else ""))
                        st.markdown("---")
                else:
                    st.write("No relevant chunks found.")
                    
            # Show image metadata
            if relevant_images:
                with st.expander("🖼️ Image Metadata"):
                    for img in relevant_images:
                        st.json(img)

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            with st.expander("🐛 Full Error Traceback"):
                st.code(traceback.format_exc(), language="python")