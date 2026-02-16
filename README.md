# NcertPhysics

# NCERT Physics AI Tutor App Flow  
**Class 11 & 12 – Hybrid RAG Powered**

## Preprocessing & Indexing

- Pre-processed NCERT textbook pages → cleaned (headers/footers removed during chunking)
- Text extracted → split using `RecursiveCharacterTextSplitter` (~800–1000 chars + overlap)
- Chunks embedded using **gte-small** (`thenlper/gte-small`) → stored in **two FAISS indices**:
  - One for text chunks
  - One for figure/image captions & descriptions
- Same chunks also indexed with **BM25** (via `langchain_community.retrievers.BM25Retriever`) for keyword search

## Real-time Query Flow

1. **User Query**  
   Examples:  
   - "Example 5.3"  
   - "derive Bernoulli's equation"  
   - "summary chapter 9"  
   - general concept question

2. **Query Classification & Metadata Extraction** (regex + rules)  
   - Detect specific example → "Example X.Y"  
   - Derivation / proof request  
   - Summary request  
   - Points to Ponder (PTP)  
   - Extract mentioned class & chapter → used for filtering

3. **Ambiguity Resolution**  
   For summary / example / PTP queries without clear class+chapter:  
   → Scan metadata → if multiple class/chapter matches → prompt user to select

4. **Hybrid Retrieval (Ensemble)**  
   - Semantic: query → gte-small embedding → FAISS vector search → top-k (~5–10)  
   - Keyword: BM25 search → top-k  
   - Merge + deduplicate → apply class/chapter filter (when specified)  
   → Final: usually ~8 best chunks

5. **Figure / Image Support**  
   - Extract `Fig. X.Y` references from retrieved chunks  
   - If none found → fallback to semantic search on image FAISS index using query embedding  
   → Return 1–2 most relevant figure images + captions

6. **Generation with LLM**  
   - Retrieved chunks + metadata → context  
   - Select specialized prompt based on query type:  
     - General question → strict NCERT-style prompt  
     - Example → structured "reproduce Example X.Y" format  
     - Derivation → step-by-step `align*` LaTeX format  
     - Summary → `## Summary` style  
     - PTP → `## Points to Ponder` list style  
   - Model: **Groq Llama-3.3-70B**  
   - Emphasis: exact textbook wording, perfect LaTeX, no hallucination

7. **Rendering in Streamlit UI**  
   - Clean light theme (white cards + subtle blue accent)  
   - User message → light blue background  
   - Assistant answer → white card with left blue border  
   - LaTeX rendered properly (inline `$...$` + display `$$...$$`)  
   - Relevant figures displayed below answer (with caption + Fig. number)  
   - Optional debug expander: tokens, retrieved chunks, sources

## One-line Summary

**Physics**  
Textbook → cleaned → recursive split → gte-small embeddings → FAISS (text + images) + BM25  
Query → classify → optional metadata filter → hybrid retrieval (vector + BM25) → dedup  
→ figure refs / fallback image search → type-specific prompt → Groq Llama-3.3-70B  
→ clean LaTeX answer + images in modern Streamlit UI
