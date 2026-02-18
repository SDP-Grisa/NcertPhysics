"""
NCERT Physics Embeddings Generator
===================================
Creates embeddings for NCERT Physics chunks using sentence-transformers
and stores them in a FAISS index for efficient retrieval.

Requirements:
    pip install sentence-transformers faiss-cpu numpy --break-system-packages
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pickle
from datetime import datetime

try:
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError:
    print("❌ Required packages not installed!")
    print("Run: pip install sentence-transformers faiss-cpu numpy --break-system-packages")
    exit(1)


class NCERTEmbeddingGenerator:
    """Generate and store embeddings for NCERT Physics chunks"""
    
    def __init__(self, model_name: str = "thenlper/gte-small"):
        """
        Initialize the embedding generator
        
        Args:
            model_name: HuggingFace model name for embeddings
                       (default: gte-small - optimized for semantic search)
        """
        print(f"🔄 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"✅ Model loaded! Embedding dimension: {self.dimension}")
    
    def create_searchable_text(self, chunk: Dict[str, Any]) -> str:
        """
        Create enhanced searchable text from chunk
        
        Combines content with metadata for better semantic search
        """
        metadata = chunk['metadata']
        content = chunk['content']
        
        # Build searchable text with context
        parts = []
        
        # Add hierarchical context
        parts.append(f"Class {metadata['class']}")
        parts.append(f"Chapter {metadata['chapter']}: {metadata['chapter_title']}")
        
        # Add type-specific context
        if metadata['type'] == 'section':
            parts.append(f"Section {metadata['identifier']}: {metadata.get('title', '')}")
        elif metadata['type'] == 'subsection':
            parts.append(f"Subsection {metadata['identifier']}: {metadata.get('title', '')}")
        elif metadata['type'] == 'example':
            parts.append(f"Example {metadata['identifier']}")
        elif metadata['type'] == 'table':
            parts.append(f"Table {metadata['identifier']}")
        elif metadata['type'] == 'summary':
            parts.append("Chapter Summary")
        elif metadata['type'] == 'exercises':
            parts.append("Exercise Questions")
        elif metadata['type'] == 'points_to_ponder':
            parts.append("Points to Ponder")
        
        # Add the actual content
        parts.append(content)
        
        return " | ".join(parts)
    
    def create_metadata_index(self, chunks: List[Dict]) -> Dict:
        """
        Create metadata index for quick filtering
        
        Returns a structured index for non-semantic queries
        """
        metadata_index = {
            'by_class': {},
            'by_chapter': {},
            'by_type': {},
            'by_identifier': {},
            'sections': {},
            'examples': {},
            'tables': {},
            'exercises': {},
            'summaries': {},
            'points_to_ponder': {},
            'chapter_overviews': {}
        }
        
        for idx, chunk in enumerate(chunks):
            meta = chunk['metadata']
            
            # Index by class
            class_key = meta['class']
            if class_key not in metadata_index['by_class']:
                metadata_index['by_class'][class_key] = []
            metadata_index['by_class'][class_key].append(idx)
            
            # Index by chapter
            chapter_key = f"{meta['class']}-{meta['chapter']}"
            if chapter_key not in metadata_index['by_chapter']:
                metadata_index['by_chapter'][chapter_key] = []
            metadata_index['by_chapter'][chapter_key].append(idx)
            
            # Index by type
            type_key = meta['type']
            if type_key not in metadata_index['by_type']:
                metadata_index['by_type'][type_key] = []
            metadata_index['by_type'][type_key].append(idx)
            
            # Index by identifier
            identifier_key = f"{chapter_key}-{meta['identifier']}"
            metadata_index['by_identifier'][identifier_key] = idx
            
            # Type-specific indices
            if type_key == 'section':
                section_key = f"{chapter_key}-section-{meta['identifier']}"
                metadata_index['sections'][section_key] = idx
            elif type_key == 'example':
                example_key = f"{chapter_key}-example-{meta['identifier']}"
                metadata_index['examples'][example_key] = idx
            elif type_key == 'table':
                table_key = f"{chapter_key}-table-{meta['identifier']}"
                metadata_index['tables'][table_key] = idx
            elif type_key == 'exercises':
                exercise_key = f"{chapter_key}-exercises"
                metadata_index['exercises'][exercise_key] = idx
            elif type_key == 'summary':
                summary_key = f"{chapter_key}-summary"
                metadata_index['summaries'][summary_key] = idx
            elif type_key == 'points_to_ponder':
                ponder_key = f"{chapter_key}-points"
                metadata_index['points_to_ponder'][ponder_key] = idx
            elif type_key == 'chapter_overview':
                overview_key = chapter_key
                metadata_index['chapter_overviews'][overview_key] = idx
        
        return metadata_index
    
    def generate_embeddings(self, chunks: List[Dict], 
                          batch_size: int = 32,
                          show_progress: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Generate embeddings for all chunks
        
        Args:
            chunks: List of chunk dictionaries
            batch_size: Batch size for embedding generation
            show_progress: Show progress during generation
            
        Returns:
            Tuple of (embeddings array, metadata index)
        """
        print(f"\n📊 Processing {len(chunks)} chunks...")
        
        # Create searchable texts
        searchable_texts = []
        for i, chunk in enumerate(chunks):
            if show_progress and (i + 1) % 100 == 0:
                print(f"  Preparing text {i + 1}/{len(chunks)}...")
            searchable_texts.append(self.create_searchable_text(chunk))
        
        print(f"\n🔄 Generating embeddings (batch_size={batch_size})...")
        
        # Generate embeddings in batches
        embeddings = self.model.encode(
            searchable_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        print(f"✅ Generated embeddings: {embeddings.shape}")
        
        # Create metadata index
        print("\n🔄 Creating metadata index...")
        metadata_index = self.create_metadata_index(chunks)
        
        print(f"✅ Metadata index created")
        print(f"   - Classes: {len(metadata_index['by_class'])}")
        print(f"   - Chapters: {len(metadata_index['by_chapter'])}")
        print(f"   - Sections: {len(metadata_index['sections'])}")
        print(f"   - Examples: {len(metadata_index['examples'])}")
        print(f"   - Tables: {len(metadata_index['tables'])}")
        print(f"   - Summaries: {len(metadata_index['summaries'])}")
        print(f"   - Exercises: {len(metadata_index['exercises'])}")
        
        return embeddings, metadata_index
    
    def create_faiss_index(self, embeddings: np.ndarray, 
                          use_gpu: bool = False) -> faiss.Index:
        """
        Create FAISS index from embeddings
        
        Args:
            embeddings: Numpy array of embeddings
            use_gpu: Whether to use GPU (if available)
            
        Returns:
            FAISS index
        """
        print(f"\n🔄 Creating FAISS index...")
        print(f"   Dimension: {self.dimension}")
        print(f"   Vectors: {len(embeddings)}")
        
        # Use IndexFlatIP for cosine similarity (after normalization)
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create index
        index = faiss.IndexFlatIP(self.dimension)
        
        # Add vectors
        index.add(embeddings)
        
        print(f"✅ FAISS index created")
        print(f"   Total vectors: {index.ntotal}")
        
        return index
    
    def save_index(self, index: faiss.Index, 
                   chunks: List[Dict],
                   metadata_index: Dict,
                   output_dir: Path = Path(".")):
        """
        Save FAISS index and metadata to disk
        
        Args:
            index: FAISS index
            chunks: Original chunks
            metadata_index: Metadata index
            output_dir: Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss_path = output_dir / "ncert_physics_gte.faiss"
        print(f"\n💾 Saving FAISS index to: {faiss_path}")
        faiss.write_index(index, str(faiss_path))
        
        # Save chunks
        chunks_path = output_dir / "ncert_physics_chunks_indexed_gte.json"
        print(f"💾 Saving chunks to: {chunks_path}")
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        
        # Save metadata index
        metadata_path = output_dir / "ncert_physics_metadata_gte.pkl"
        print(f"💾 Saving metadata index to: {metadata_path}")
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata_index, f)
        
        # Save configuration
        config = {
            'model_name': self.model_name,
            'dimension': self.dimension,
            'total_chunks': len(chunks),
            'created_at': datetime.now().isoformat(),
            'index_type': 'IndexFlatIP',
            'similarity_metric': 'cosine'
        }
        
        config_path = output_dir / "ncert_physics_config_gte.json"
        print(f"💾 Saving configuration to: {config_path}")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n✅ All files saved successfully!")
        print(f"\n📦 Generated files:")
        print(f"   1. {faiss_path.name} - FAISS index")
        print(f"   2. {chunks_path.name} - Indexed chunks")
        print(f"   3. {metadata_path.name} - Metadata index")
        print(f"   4. {config_path.name} - Configuration")
        
        # Print file sizes
        print(f"\n📊 File sizes:")
        for path in [faiss_path, chunks_path, metadata_path, config_path]:
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"   {path.name}: {size_mb:.2f} MB")


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate embeddings for NCERT Physics chunks"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='ncert_physics_chunks.json',
        help='Input chunks JSON file (default: ncert_physics_chunks.json)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='.',
        help='Output directory (default: current directory)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='thenlper/gte-small',
        help='Sentence transformer model name'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for embedding generation (default: 32)'
    )
    
    args = parser.parse_args()
    
    # Check input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        print(f"\nPlease ensure you have processed PDFs using the chunker first.")
        return 1
    
    print("="*80)
    print("NCERT PHYSICS EMBEDDING GENERATOR")
    print("="*80)
    print(f"\n📁 Input: {input_path}")
    print(f"📁 Output: {args.output_dir}")
    print(f"🤖 Model: {args.model}")
    print(f"📦 Batch size: {args.batch_size}")
    print()
    
    # Load chunks
    print(f"📖 Loading chunks from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"✅ Loaded {len(chunks)} chunks")
    
    # Initialize generator
    generator = NCERTEmbeddingGenerator(model_name=args.model)
    
    # Generate embeddings
    embeddings, metadata_index = generator.generate_embeddings(
        chunks,
        batch_size=args.batch_size,
        show_progress=True
    )
    
    # Create FAISS index
    faiss_index = generator.create_faiss_index(embeddings)
    
    # Save everything
    generator.save_index(
        faiss_index,
        chunks,
        metadata_index,
        output_dir=Path(args.output_dir)
    )
    
    print("\n" + "="*80)
    print("✅ EMBEDDING GENERATION COMPLETE!")
    print("="*80)
    print(f"\nYou can now use the retrieval system to query the embeddings.")
    print(f"Run: python retrieval_strategy.py --help")
    
    return 0


if __name__ == "__main__":
    exit(main())