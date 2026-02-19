# NCERT Physics RAG System (Class 11 & 12)

## 1. Project Overview

This project implements a **production-grade Retrieval-Augmented Generation (RAG) system** for the NCERT Physics textbooks of **Class 11 and Class 12**.

The system enables:

* Accurate semantic question answering
* Direct retrieval of summaries, examples, tables, and exercises
* Structured mathematical rendering using KaTeX
* Hybrid semantic + keyword retrieval
* Intelligent LLM usage with hallucination prevention

Unlike basic RAG demos, this system is designed with **real educational reliability, explainability, and performance constraints** in mind.

---

## 2. System Architecture

### End-to-End Flow

```
NCERT PDFs
   ↓
Advanced Chunking + Metadata Extraction
   ↓
Semantic Embeddings
   ↓
FAISS Vector Index + Metadata Lookup
   ↓
Hybrid Retrieval + Intent Classification
   ↓
Direct Answer OR LLM Response
   ↓
Interactive Streamlit UI
```

---

## 3. Functional Strategies, Libraries, and Tools Used

This section maps **each core functionality** to the **strategy implemented** and the **libraries/tools used**, giving a clear engineering picture of the system.

### 3.1 Text Extraction & Chunking

**Strategy**

* Structure-aware parsing instead of fixed-size splitting
* Detection of chapters, sections, examples, tables, summaries, and exercises
* Metadata-rich chunks for deterministic retrieval
* Synthetic chapter overview generation

**Libraries / Tools**

* `PyMuPDF / pdfplumber` → PDF text extraction
* `re (regex)` → Pattern detection for academic structures
* `json` → Structured chunk storage

---

### 3.2 Metadata Design & Indexing

**Strategy**

* O(1) lookup dictionaries for:

  * examples
  * tables
  * summaries
  * exercises
  * chapter overviews
* Chapter-wise and type-wise grouping
* Alignment between **chunk index and FAISS row index**

**Libraries / Tools**

* `pickle` → Fast metadata serialization
* Python dictionaries → Constant-time retrieval

---

### 3.3 Semantic Embeddings

**Strategy**

* Sentence-level dense embeddings
* Asymmetric retrieval:

  * passages stored normally
  * queries prefixed for semantic search
* Header-enriched embedding text:

  ```
  Class X Chapter Y Title [Type]: Content
  ```

**Libraries / Tools**

* `sentence-transformers` → Embedding generation
* GTE-small model → Lightweight, high-quality semantic vectors
* `NumPy` → Vector handling and normalization

---

### 3.4 Vector Search Index

**Strategy**

* Cosine similarity via **inner product on normalized vectors**
* Flat index for **maximum accuracy**
* Persistent on-disk storage

**Libraries / Tools**

* `FAISS` → High-performance vector similarity search

---

### 3.5 Hybrid Retrieval & Ranking

**Strategy**

* Combine:

  * semantic similarity
  * keyword/BM25-style relevance
* Score fusion reranking
* Similarity threshold gating to avoid hallucinations

**Libraries / Tools**

* `FAISS` → Semantic retrieval
* `rank-bm25 / rapidfuzz / custom scoring` → Keyword relevance
* `NumPy` → Score fusion

---

### 3.6 LLM-Based Query Understanding

**Strategy**

* Intent classification using LLM instead of regex
* Detect:

  * question type
  * class & chapter
  * identifiers (example/table)
  * permission for external knowledge

**Libraries / Tools**

* LLM API (OpenAI-compatible endpoint) → Intent parsing & reasoning
* Prompt engineering → Structured JSON output

---

### 3.7 Hallucination Prevention

**Strategy**

* Similarity threshold guard
* Direct chunk return when confidence is low
* Bypass LLM when textbook already contains answer
* Domain restriction to NCERT Physics

**Libraries / Tools**

* Custom retrieval logic
* Score comparison using `NumPy`

---

### 3.8 Direct Answer Modes (No LLM)

**Strategy**

Deterministic retrieval for:

* summaries
* exercises
* examples
* tables
* points to ponder
* statistics

**Benefits**

* Zero hallucination
* Lower latency
* Reduced API cost

**Libraries / Tools**

* Metadata lookup dictionaries
* FAISS index alignment

---

### 3.9 Mathematical Formatting & Rendering

**Strategy**

* Detect physics equations automatically
* Convert to LaTeX-safe output
* Render using KaTeX in UI

**Libraries / Tools**

* Regex-based math detection
* KaTeX (frontend rendering)
* Streamlit markdown components

---

### 3.10 Interactive User Interface

**Strategy**

* Conversational QA interface
* Ambiguity resolution between Class 11 & 12
* Debug transparency:

  * intent
  * retrieved chunks
  * similarity scores
  * token usage

**Libraries / Tools**

* `Streamlit` → Web UI framework
* Custom CSS/markdown → Styling & math rendering

---

## 4. Engineering Strengths

This project demonstrates:

* Real **production-style RAG architecture**
* Hybrid retrieval with **controlled LLM reasoning**
* **Hallucination-safe educational QA**
* Transparent debugging and observability
* Cost-aware and latency-aware design

Overall, it reflects **industry-level applied AI engineering**, not a basic tutorial implementation.

---

## 5. Conclusion

The NCERT Physics RAG system is a **robust, safe, and educationally aligned AI assistant** for textbook learning.

It integrates:

* Structure-aware preprocessing
* Efficient semantic retrieval
* Controlled LLM reasoning
* Mathematical rendering
* Transparent interaction

This represents a **complete end-to-end applied AI solution suitable for real academic deployment**.
