# NCERT Physics AI Tutor & RAG Solver – System Report

**Project Name**: NCERT Physics AI Tutor (Hybrid RAG + Diagram Extraction)  

## 1. Overview

This is a **dual-mode educational RAG application** for NCERT Class 11 & 12 Physics:

- **Mode 1** (pre-indexed, fast chat): Uses pre-computed FAISS indices (`gte_small_full.index` & `gte_small_images_11.index`) with **gte-small** embeddings + BM25 keyword index → chat-style Q&A with strict figure grounding.
- **Mode 2** (on-demand PDF processing): User uploads chapter PDFs → extracts diagrams (PyMuPDF) + chunks text → builds FAISS/BM25 retrievers incrementally → supports new chapter addition.

Both modes share:
- Strict NCERT fidelity
- Type-specific prompting (examples, derivations, summaries, PTP)
- Perfect LaTeX rendering
- Diagrams shown **only** when explicitly referenced in context

## 2. Tech Stack & Libraries

| Category                | Library / Tool                                      | Purpose / Notes                                      |
|-------------------------|-----------------------------------------------------|------------------------------------------------------|
| Web UI                  | `streamlit`                                         | Chat, sidebar, inputs, CSS styling                   |
| Vector Search           | `faiss`                                             | Fast similarity search (pre-built & runtime)         |
| Embeddings              | `sentence-transformers`                             | Model: `thenlper/gte-small` (384-dim)                |
| Serialization           | `pickle`                                            | Load pre-computed FAISS indices & metadata           |
| LLM                     | `langchain_groq.ChatGroq`                           | Model: `llama-3.3-70b-versatile` (multi-key fallback)|
| Prompting               | `langchain_core.prompts.ChatPromptTemplate`         | Type-specific NCERT prompts                          |
| Output Parsing          | `langchain_core.output_parsers.StrOutputParser`     | Clean string output                                  |
| Document Handling       | `langchain_core.documents.Document`                 | Text + metadata structure                            |
| Keyword Retrieval       | `langchain_community.retrievers.BM25Retriever`      | BM25 keyword search                                  |
| PDF Text Loading        | `langchain_community.document_loaders.PyPDFLoader`  | Raw text extraction                                  |
| Text Splitting          | `langchain_text_splitters.RecursiveCharacterTextSplitter` | Chunking (1000 chars, 200 overlap)            |
| Diagram Extraction      | `fitz` (PyMuPDF)                                    | Drawings detection, high-res PNG export              |
| Persistence             | `json`, `hashlib` (MD5), `tempfile`, `pathlib`      | Chunks/images, deduplication, temp files             |
| Regex & Utils           | `re`, `os`, `dotenv`                                | Patterns, file ops, env vars                         |

## 3. Chunking Strategy (Mode 2 – PDF Upload)

**Chunk Size & Overlap**  
- **Chunk size**: 1000 characters  
- **Chunk overlap**: 200 characters  
- Implemented via `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, keep_separator=True)`

**Custom Separators** for answer, points to ponder, summary

## 3. Core Functionalities & Flow

### 3.1 Pre-indexed Mode (First Code – Fast Chat)
- **Initialization**:
  - Loads pre-built FAISS indices (`get_small_full.index`) for text + images
  - Loads metadata (`metadata_full.pkl`)
  - Builds BM25 retriever from chunks
  - Initializes `gte-small` embedder

- **Query Processing** (`process_query`):
  1. Embed query with prefix → `embed_query`
  2. Classify type → `is_specific_example`, `is_derivation_query`, etc.
  3. Extract/apply class & chapter → `extract_class_and_chapter`
  4. Check ambiguity → `check_ambiguity` → user selection UI if needed
  5. Retrieve:
     - Summary/PTP: `retrieve_summary` / `retrieve_points_to_ponder`
     - Example: BM25 on "Example X.Y"
     - General: `ensemble_retrieval` (BM25 + vector + dedup + filter)
  6. Find images → `find_relevant_images` (only explicit "Fig. X.Y" refs)
  7. Generate → `generate_answer` (type-specific prompt + LLM rotation on rate limit)
  8. Render → `render_mixed_latex` + chat UI + debug expander

### 3.2 PDF Upload & Processing Mode (Second Code – Incremental)
- **User Flow**:
  1. Select class/chapter → upload PDF
  2. Temp save → `process_chapter`
  3. Extract diagrams → `extract_images_from_pdf` + `get_diagram_bbox`
  4. Load text → `PyPDFLoader`
  5. Chunk → `RecursiveCharacterTextSplitter` + custom merge (`merge_special`)
  6. Deduplicate → `content_hash` (MD5)
  7. Persist → `save_all_chunks` & `save_image_metadata` (JSON)
  8. Rebuild retrievers → `get_retrievers` (FAISS + BM25)

- **Diagram Extraction Details**:
  - Assumes 2-column layout → splits page at midpoint
  - Detects captions → searches 300 units above in same column
  - Filters drawings → merges bounding boxes → extracts 3× resolution PNG
  - Saves metadata (fig_id, path, caption, page, class, chapter)

- **Image Matching** (`find_relevant_images`):
  - Priority 1: Explicit fig in query
  - Priority 2: Figs referenced in retrieved chunks
  - Fallback: Keyword overlap in captions (min 2 terms unless short query)

## 4. Strengths & Design Choices
- **Hybrid Retrieval**: Combines semantic (gte-small/FAISS) + keyword (BM25) → robust for exact terms and concepts
- **Explicit Image Grounding**: Only shows figures mentioned in context → avoids hallucinations
- **Rate Limit Resilience**: Rotates multiple Groq API keys
- **Incremental Learning**: New PDFs append to existing index
- **Specialized Prompting**: Strict NCERT style + perfect LaTeX for examples/derivations
- **Persistence**: JSON + hashing → no re-processing of same content

## 5. Limitations & Possible Improvements
- No automatic header/footer removal (still present in chunks)
- Diagram extraction assumes 2-column layout + caption above figure
- gte-small is lightweight but less powerful than bge-large (trade-off for speed)
- No OCR fallback for scanned PDFs
- Future: Add vector DB persistence (FAISS save/load), better ambiguity UI, dark mode

**Prepared by Grok** – February 16, 2026