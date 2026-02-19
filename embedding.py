"""
build_embeddings.py — NCERT Physics Embedding Builder
======================================================
Run this ONCE to create a persistent FAISS index from your JSON chunks file.
Outputs (saved to --output-dir):
  • ncert_physics_gte.faiss             — FAISS flat IP index (normalized vectors)
  • ncert_physics_chunks_indexed_gte.json — original chunks (preserves order = FAISS row index)
  • ncert_physics_metadata_gte.pkl      — fast lookup dicts (examples, tables, summaries, etc.)
  • ncert_physics_config_gte.json       — model name + build metadata

Usage:
  python build_embeddings.py --chunks ncert_physics_chunks.json
  python build_embeddings.py --chunks ncert_physics_chunks.json --output-dir ./index --model thenlper/gte-small
  python build_embeddings.py --chunks ncert_physics_chunks.json --batch-size 64
"""

import json
import pickle
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ── GTE-small passage prefix (asymmetric retrieval) ──────────────────────────
# Passages are stored WITHOUT prefix; queries use the prefix at query time.
# This matches the GTE paper's recommended asymmetric setup.
PASSAGE_PREFIX = ""          # intentionally empty — queries carry the prefix
DEFAULT_MODEL  = "thenlper/gte-small"
DEFAULT_BATCH  = 64


# ── Text preparation ──────────────────────────────────────────────────────────

def prepare_text(chunk: Dict) -> str:
    """
    Build the text that will be embedded for a chunk.

    Strategy:
      - Prepend a short structured header so the embedding encodes both
        the topic label AND the content. This improves retrieval precision
        for chapter/class-specific queries.
      - Header: "Class {X} Chapter {Y} {chapter_title} [{type}]: "
      - Content: raw chunk content (trimmed to 512 tokens worth)
    """
    m      = chunk["metadata"]
    cls    = m.get("class", "")
    ch     = m.get("chapter", "")
    title  = m.get("chapter_title", "")
    ctype  = m.get("type", "")
    ident  = m.get("identifier", "")

    # Build a compact prefix that helps the model know what this chunk IS
    parts = [f"Class {cls} Chapter {ch} {title}"]
    if ctype not in ("chapter_overview", "chapter_title"):
        parts.append(f"[{ctype}]")
    if ident and ident not in ("overview", ctype):
        parts.append(ident)

    header  = " ".join(parts) + ": "
    content = chunk["content"].strip()

    # GTE-small has a 512-token limit; leave ~50 tokens for header
    # Rough heuristic: 4 chars ≈ 1 token → keep ~1850 chars of content
    MAX_CONTENT_CHARS = 1850
    if len(content) > MAX_CONTENT_CHARS:
        content = content[:MAX_CONTENT_CHARS]

    return PASSAGE_PREFIX + header + content


# ── Metadata index builders ───────────────────────────────────────────────────

def build_metadata_index(chunks: List[Dict]) -> Dict[str, Any]:
    """
    Build fast O(1) lookup dictionaries keyed by chunk position (FAISS index = list index).

    Keys in returned dict:
      chapter_overviews  → "cls-chapter"        → list_index
      examples           → "cls-ch-example-id"  → list_index
      tables             → "cls-ch-table-id"    → list_index
      summaries          → "cls-ch-summary"      → list_index
      exercises          → "cls-ch-exercises"    → list_index
      points_to_ponder   → "cls-ch-points"       → list_index
      by_type            → type → [list_indices]
      by_class_chapter   → "cls-ch" → [list_indices]
    """
    idx = {
        "chapter_overviews": {},
        "examples":          {},
        "tables":            {},
        "summaries":         {},
        "exercises":         {},
        "points_to_ponder":  {},
        "by_type":           {},
        "by_class_chapter":  {},
    }

    for i, chunk in enumerate(chunks):
        m     = chunk["metadata"]
        cls   = m.get("class", "")
        ch    = m.get("chapter", "")
        ctype = m.get("type", "")
        ident = m.get("identifier", "")

        # by_type
        idx["by_type"].setdefault(ctype, []).append(i)

        # by_class_chapter
        ck = f"{cls}-{ch}"
        idx["by_class_chapter"].setdefault(ck, []).append(i)

        if ctype == "chapter_overview":
            idx["chapter_overviews"][ck] = i

        elif ctype == "example":
            key = f"{cls}-{ch}-example-{ident}"
            idx["examples"][key] = i

        elif ctype == "table":
            key = f"{cls}-{ch}-table-{ident}"
            idx["tables"][key] = i

        elif ctype == "summary":
            idx["summaries"][f"{ck}-summary"] = i

        elif ctype == "exercises":
            idx["exercises"][f"{ck}-exercises"] = i

        elif ctype == "points_to_ponder":
            idx["points_to_ponder"][f"{ck}-points"] = i

    return idx


# ── Embedding + FAISS builder ─────────────────────────────────────────────────

def build_index(
    chunks_path: Path,
    output_dir:  Path,
    model_name:  str  = DEFAULT_MODEL,
    batch_size:  int  = DEFAULT_BATCH,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load chunks ──────────────────────────────────────────────────────────
    print(f"📂 Loading chunks from: {chunks_path}")
    with open(chunks_path, encoding="utf-8") as f:
        chunks: List[Dict] = json.load(f)
    print(f"   ✅ {len(chunks)} chunks loaded")

    # ── Type distribution ────────────────────────────────────────────────────
    type_counts: Dict[str, int] = {}
    for c in chunks:
        t = c["metadata"].get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    print("   Chunk types:")
    for t, n in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"     {t:25s}: {n}")

    # ── Load model ───────────────────────────────────────────────────────────
    print(f"\n🤖 Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    dim   = model.get_sentence_embedding_dimension()
    print(f"   ✅ Embedding dimension: {dim}")

    # ── Prepare texts ────────────────────────────────────────────────────────
    print("\n📝 Preparing texts for embedding…")
    texts = [prepare_text(c) for c in chunks]

    # Show a few samples
    print("   Sample prepared texts:")
    for i in [0, 50, 200]:
        if i < len(texts):
            print(f"   [{i}] {texts[i][:120]}…")

    # ── Encode in batches ─────────────────────────────────────────────────────
    print(f"\n⚙️  Encoding {len(texts)} chunks (batch_size={batch_size})…")
    t0 = time.time()

    all_embeddings = []
    batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]

    for batch in tqdm(batches, desc="Embedding", unit="batch"):
        emb = model.encode(
            batch,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,   # L2-normalize here for IP = cosine
        )
        all_embeddings.append(emb)

    embeddings = np.vstack(all_embeddings).astype("float32")
    elapsed    = time.time() - t0
    print(f"   ✅ Done in {elapsed:.1f}s  |  Shape: {embeddings.shape}")

    # ── Verify normalization ─────────────────────────────────────────────────
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"   Norm check — min: {norms.min():.4f}  max: {norms.max():.4f}  mean: {norms.mean():.4f}")
    # Re-normalize to be safe (handles any floating point drift)
    faiss.normalize_L2(embeddings)

    # ── Build FAISS index ────────────────────────────────────────────────────
    print(f"\n🗄️  Building FAISS IndexFlatIP (inner product = cosine on normalized vectors)…")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"   ✅ FAISS index: {index.ntotal} vectors, dim={dim}")

    # ── Build metadata lookup ─────────────────────────────────────────────────
    print("\n🗂️  Building metadata lookup index…")
    meta_idx = build_metadata_index(chunks)
    print(f"   chapter_overviews : {len(meta_idx['chapter_overviews'])}")
    print(f"   examples          : {len(meta_idx['examples'])}")
    print(f"   tables            : {len(meta_idx['tables'])}")
    print(f"   summaries         : {len(meta_idx['summaries'])}")
    print(f"   exercises         : {len(meta_idx['exercises'])}")
    print(f"   points_to_ponder  : {len(meta_idx['points_to_ponder'])}")

    # ── Save everything ───────────────────────────────────────────────────────
    print(f"\n💾 Saving to: {output_dir}/")

    faiss_path  = output_dir / "ncert_physics_gte.faiss"
    chunks_path_out = output_dir / "ncert_physics_chunks_indexed_gte.json"
    meta_path   = output_dir / "ncert_physics_metadata_gte.pkl"
    config_path = output_dir / "ncert_physics_config_gte.json"

    faiss.write_index(index, str(faiss_path))
    print(f"   ✅ FAISS index  → {faiss_path.name}  ({faiss_path.stat().st_size/1024:.0f} KB)")

    with open(chunks_path_out, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    print(f"   ✅ Chunks JSON  → {chunks_path_out.name}  ({chunks_path_out.stat().st_size/1024:.0f} KB)")

    with open(meta_path, "wb") as f:
        pickle.dump(meta_idx, f)
    print(f"   ✅ Metadata pkl → {meta_path.name}  ({meta_path.stat().st_size/1024:.0f} KB)")

    config = {
        "model_name":     model_name,
        "embedding_dim":  int(dim),
        "num_chunks":     len(chunks),
        "passage_prefix": PASSAGE_PREFIX,
        "built_at":       time.strftime("%Y-%m-%d %H:%M:%S"),
        "chunk_types":    type_counts,
        "faiss_file":     faiss_path.name,
        "chunks_file":    chunks_path_out.name,
        "metadata_file":  meta_path.name,
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"   ✅ Config JSON  → {config_path.name}")

    # ── Quick sanity search ───────────────────────────────────────────────────
    print("\n🔍 Sanity check — searching 'Newton second law'…")
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
    test_emb = model.encode(
        [QUERY_PREFIX + "Newton second law force mass acceleration"],
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype("float32")
    scores, idxs = index.search(test_emb, 3)
    for sc, i in zip(scores[0], idxs[0]):
        m = chunks[i]["metadata"]
        print(f"   score={sc:.3f}  Class {m['class']} Ch {m['chapter']} [{m['type']}] {m.get('chapter_title','')}")

    print("\n🎉 All done! Index is ready. Run ncert_rag_v4.py and point it at this folder.")
    print(f"   Output dir: {output_dir.resolve()}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build FAISS embedding index from NCERT Physics JSON chunks."
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("ncert_physics_chunks.json"),
        help="Path to the input chunks JSON file (default: ncert_physics_chunks.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory to save index files (default: current directory)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"SentenceTransformer model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH,
        help=f"Encoding batch size (default: {DEFAULT_BATCH})",
    )

    args = parser.parse_args()

    if not args.chunks.exists():
        print(f"❌ Chunks file not found: {args.chunks}")
        exit(1)

    build_index(
        chunks_path=args.chunks,
        output_dir=args.output_dir,
        model_name=args.model,
        batch_size=args.batch_size,
    )