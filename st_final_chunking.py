"""
NCERT Physics Chunker - Streamlit App
======================================
Interactive web interface for processing NCERT Physics PDFs
"""

import streamlit as st
import json
import tempfile
import os
from pathlib import Path
from collections import Counter, defaultdict
import re

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False
    st.error("pypdf not installed. Run: pip install pypdf")

# Import the chunking functions
import sys
sys.path.append(str(Path(__file__).parent))

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

CHUNKS_FILE = Path("ncert_physics_chunks.json")

CHAPTERS_CLASS_11 = [
    "1 - Units and Measurement", "2 - Motion in a Straight Line",
    "3 - Motion in a Plane", "4 - Laws of Motion",
    "5 - Work, Energy and Power", "6 - System of Particles and Rotational Motion",
    "7 - Gravitation", "8 - Mechanical Properties of Solids",
    "9 - Mechanical Properties of Fluids", "10 - Thermal Properties of Matter",
    "11 - Thermodynamics", "12 - Kinetic Theory",
    "13 - Oscillations", "14 - Waves",
]

CHAPTERS_CLASS_12 = [
    "1 - Electric Charges and Fields", "2 - Electrostatic Potential and Capacitance",
    "3 - Current Electricity", "4 - Moving Charges and Magnetism",
    "5 - Magnetism and Matter", "6 - Electromagnetic Induction",
    "7 - Alternating Current", "8 - Electromagnetic Waves",
    "9 - Ray Optics and Optical Instruments", "10 - Wave Optics",
    "11 - Dual Nature of Radiation and Matter", "12 - Atoms",
    "13 - Nuclei", "14 - Semiconductor Electronics",
]

# ═══════════════════════════════════════════════════════════════
# CHUNKING LOGIC (Embedded from main script)
# ═══════════════════════════════════════════════════════════════

@dataclass
class ContentBoundary:
    """Represents a content boundary with multi-page support"""
    content_type: str
    identifier: str
    title: Optional[str]
    start_page: int
    start_line: int
    end_page: Optional[int] = None
    end_line: Optional[int] = None
    level: int = 0
    styling: Optional[Dict] = None


def extract_text_with_pypdf(pdf_path: str) -> List[Dict]:
    """Extract text from PDF"""
    reader = PdfReader(pdf_path)
    pages_data = []
    
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        pages_data.append({
            "page_num": page_num,
            "text": text,
            "lines": text.split('\n') if text else []
        })
    
    return pages_data


def detect_blue_box_topics(pages_data: List[Dict]) -> Optional[ContentBoundary]:
    """Detect blue box with chapter topics on first page"""
    if not pages_data:
        return None
    
    first_page = pages_data[0]
    lines = first_page['lines']
    
    topic_start = -1
    topic_count = 0
    
    for i in range(35, len(lines)):
        line = lines[i].strip()
        if not line:
            continue
        
        if topic_start == -1:
            if re.match(r'^\d+\.\d+\s+\w', line) and len(line) < 50:
                if i > 5 and any(len(lines[j].strip()) > 80 for j in range(i-5, i)):
                    topic_start = i
                    topic_count = 1
        else:
            if (re.match(r'^\d+\.\d+\s+', line) or 
                re.match(r'^(Summary|Exercises?|Points)', line, re.IGNORECASE) or
                len(line) < 30):
                topic_count += 1
            elif "Reprint" in line:
                break
            else:
                break
    
    if topic_start != -1 and topic_count >= 7:
        return ContentBoundary(
            content_type="chapter_topics",
            identifier="topics",
            title="Chapter Topics (Blue Box)",
            start_page=0,
            start_line=topic_start,
            end_page=0,
            end_line=topic_start + topic_count,
            level=0,
            styling={"color": "blue", "box": True, "location": "left_sidebar"}
        )
    
    return None


def detect_all_boundaries(pages_data: List[Dict]) -> List[ContentBoundary]:
    """Detect all content boundaries across all pages"""
    boundaries = []
    
    topics_box = detect_blue_box_topics(pages_data)
    if topics_box:
        boundaries.append(topics_box)
    
    skip_lines = set()
    if topics_box:
        skip_lines = set(range(topics_box.start_line, topics_box.end_line))
    
    for page_idx, page in enumerate(pages_data):
        lines = page['lines']
        
        for line_idx, line in enumerate(lines):
            line_stripped = line.strip()
            
            if not line_stripped or "reprint" in line_stripped.lower():
                continue
            if page_idx == 0 and line_idx in skip_lines:
                continue
            
            # Chapter title
            if re.match(r'^CHAPTER\s+\w+', line_stripped, re.IGNORECASE):
                match = re.search(r'chapter\s+(\w+)', line_stripped, re.IGNORECASE)
                if match:
                    boundaries.append(ContentBoundary(
                        content_type="chapter_title",
                        identifier=match.group(1).upper(),
                        title=line_stripped,
                        start_page=page_idx,
                        start_line=line_idx,
                        level=0
                    ))
            
            # Section
            elif re.match(r'^\d+\.\d+\s+[A-Z]{2,}', line_stripped):
                match = re.match(r'^(\d+\.\d+)\s+(.+)', line_stripped)
                if match:
                    boundaries.append(ContentBoundary(
                        content_type="section",
                        identifier=match.group(1),
                        title=match.group(2).strip(),
                        start_page=page_idx,
                        start_line=line_idx,
                        level=1
                    ))
            
            # Subsection
            elif re.match(r'^\d+\.\d+\.\d+\s+', line_stripped):
                match = re.match(r'^(\d+\.\d+\.\d+)\s+(.+)', line_stripped)
                if match:
                    boundaries.append(ContentBoundary(
                        content_type="subsection",
                        identifier=match.group(1),
                        title=match.group(2).strip(),
                        start_page=page_idx,
                        start_line=line_idx,
                        level=2
                    ))
            
            # Example
            elif line_stripped.startswith('Example') and re.search(r'Example\s+(\d+\.\d+)', line_stripped):
                match = re.search(r'Example\s+(\d+\.\d+)', line_stripped)
                if match:
                    boundaries.append(ContentBoundary(
                        content_type="example",
                        identifier=match.group(1),
                        title=line_stripped,
                        start_page=page_idx,
                        start_line=line_idx,
                        level=2
                    ))
            
            # Table
            elif line_stripped.startswith('Table') and re.search(r'Table\s+(\d+\.\d+)', line_stripped):
                match = re.search(r'Table\s+(\d+\.\d+)', line_stripped)
                if match:
                    boundaries.append(ContentBoundary(
                        content_type="table",
                        identifier=match.group(1),
                        title=line_stripped,
                        start_page=page_idx,
                        start_line=line_idx,
                        level=2,
                        styling={
                            "table_number": {"bold": True, "color": "black", "small": True},
                            "table_title": {"color": "blue"},
                            "table_content": {"background": "grey"}
                        }
                    ))
            
            # Summary
            elif re.match(r'^SUMMARY$', line_stripped, re.IGNORECASE):
                boundaries.append(ContentBoundary(
                    content_type="summary",
                    identifier="summary",
                    title="SUMMARY",
                    start_page=page_idx,
                    start_line=line_idx,
                    level=1
                ))
            
            # Exercises
            elif re.match(r'^EXERCISES?$', line_stripped, re.IGNORECASE):
                boundaries.append(ContentBoundary(
                    content_type="exercises",
                    identifier="exercises",
                    title="EXERCISES",
                    start_page=page_idx,
                    start_line=line_idx,
                    level=1,
                    styling={"bold": True, "color": "blue", "capital": True}
                ))
            
            # Points to Ponder
            elif re.match(r'^POINTS?\s+TO\s+PONDER', line_stripped, re.IGNORECASE):
                boundaries.append(ContentBoundary(
                    content_type="points_to_ponder",
                    identifier="points_to_ponder",
                    title="POINTS TO PONDER",
                    start_page=page_idx,
                    start_line=line_idx,
                    level=1
                ))
    
    return boundaries


def set_boundary_ends(boundaries: List[ContentBoundary], pages_data: List[Dict]):
    """Set end positions for all boundaries"""
    boundaries.sort(key=lambda b: (b.start_page, b.start_line))
    
    for i, boundary in enumerate(boundaries):
        if boundary.end_page is not None and boundary.end_line is not None:
            continue
        
        if i + 1 < len(boundaries):
            next_boundary = boundaries[i + 1]
            boundary.end_page = next_boundary.start_page
            boundary.end_line = next_boundary.start_line
        else:
            boundary.end_page = len(pages_data) - 1
            boundary.end_line = len(pages_data[-1]['lines'])
        
        # Limit tables to 150 lines
        if boundary.content_type == "table":
            total_lines = 0
            for p in range(boundary.start_page, boundary.end_page + 1):
                if p == boundary.start_page:
                    total_lines += len(pages_data[p]['lines']) - boundary.start_line
                elif p == boundary.end_page:
                    total_lines += boundary.end_line
                else:
                    total_lines += len(pages_data[p]['lines'])
            
            if total_lines > 150:
                lines_counted = 0
                for p in range(boundary.start_page, len(pages_data)):
                    page_lines = pages_data[p]['lines']
                    if p == boundary.start_page:
                        available = len(page_lines) - boundary.start_line
                        if lines_counted + available >= 150:
                            boundary.end_page = p
                            boundary.end_line = boundary.start_line + (150 - lines_counted)
                            break
                        lines_counted += available
                    else:
                        if lines_counted + len(page_lines) >= 150:
                            boundary.end_page = p
                            boundary.end_line = 150 - lines_counted
                            break
                        lines_counted += len(page_lines)


def extract_content_multi_page(pages_data: List[Dict], boundary: ContentBoundary) -> Tuple[str, List[str]]:
    """Extract content across multiple pages with figure detection"""
    content_lines = []
    figures = []
    
    for page_idx in range(boundary.start_page, boundary.end_page + 1):
        if page_idx >= len(pages_data):
            break
        
        page = pages_data[page_idx]
        lines = page['lines']
        
        for line_idx, line in enumerate(lines):
            include = False
            
            if page_idx == boundary.start_page and page_idx == boundary.end_page:
                include = boundary.start_line <= line_idx < boundary.end_line
            elif page_idx == boundary.start_page:
                include = line_idx >= boundary.start_line
            elif page_idx == boundary.end_page:
                include = line_idx < boundary.end_line
            else:
                include = True
            
            if include:
                line_stripped = line.strip()
                if line_stripped and "reprint" not in line_stripped.lower():
                    content_lines.append(line_stripped)
                    fig_matches = re.findall(r'Fig\.?\s*(\d+\.\d+)', line_stripped, re.IGNORECASE)
                    figures.extend(fig_matches)
    
    figures = sorted(list(set(figures)))
    return "\n".join(content_lines), figures


def create_chunks(pages_data: List[Dict], boundaries: List[ContentBoundary],
                  class_num: str, chapter_num: str, chapter_title: str) -> List[Dict]:
    """Create chunks with all metadata"""
    chunks = []
    
    all_sections = []
    all_subsections = []
    all_tables = []
    all_figures = set()
    
    for boundary in boundaries:
        content, figures = extract_content_multi_page(pages_data, boundary)
        
        if len(content.strip()) < 20:
            continue
        
        if boundary.content_type == "section":
            all_sections.append({"id": boundary.identifier, "title": boundary.title})
        elif boundary.content_type == "subsection":
            all_subsections.append({"id": boundary.identifier, "title": boundary.title})
        elif boundary.content_type == "table":
            all_tables.append(boundary.identifier)
        
        all_figures.update(figures)
        
        metadata = {
            "class": class_num,
            "chapter": chapter_num,
            "chapter_title": chapter_title,
            "type": boundary.content_type,
            "identifier": boundary.identifier,
            "title": boundary.title,
            "page_start": boundary.start_page + 1,
            "page_end": boundary.end_page + 1 if boundary.end_page else boundary.start_page + 1,
            "level": boundary.level,
            "spans_multiple_pages": boundary.end_page != boundary.start_page
        }
        
        if figures:
            metadata["figures"] = figures
            metadata["figure_count"] = len(figures)
        
        if boundary.content_type == "example":
            metadata["example_id"] = boundary.identifier
        elif boundary.content_type == "section":
            metadata["section_id"] = boundary.identifier
            metadata["is_conceptual"] = True
        elif boundary.content_type == "subsection":
            metadata["subsection_id"] = boundary.identifier
            metadata["is_conceptual"] = True
        elif boundary.content_type == "table":
            metadata["table_id"] = boundary.identifier
        
        if boundary.styling:
            metadata["styling"] = boundary.styling
        
        chunks.append({"content": content, "metadata": metadata})
    
    # Create chapter overview
    overview_content = f"CHAPTER {chapter_num}: {chapter_title}\n\n"
    overview_content += "="*80 + "\n"
    overview_content += "CHAPTER STRUCTURE OVERVIEW\n"
    overview_content += "="*80 + "\n\n"
    
    overview_content += f"SECTIONS ({len(all_sections)}):\n"
    overview_content += "-"*80 + "\n"
    for sec in all_sections:
        overview_content += f"  {sec['id']}  {sec['title']}\n"
    
    overview_content += f"\n\nSUBSECTIONS ({len(all_subsections)}):\n"
    overview_content += "-"*80 + "\n"
    for subsec in all_subsections:
        overview_content += f"  {subsec['id']}  {subsec['title']}\n"
    
    overview_content += f"\n\nTABLES ({len(all_tables)}):\n"
    overview_content += "-"*80 + "\n"
    if all_tables:
        overview_content += f"  Table " + ", Table ".join(all_tables) + "\n"
    
    overview_content += f"\n\nFIGURES ({len(all_figures)}):\n"
    overview_content += "-"*80 + "\n"
    if all_figures:
        sorted_figures = sorted(list(all_figures))
        overview_content += f"  Fig. " + ", Fig. ".join(sorted_figures) + "\n"
    
    overview_content += "\n" + "="*80 + "\n"
    overview_content += f"Total Sections: {len(all_sections)}\n"
    overview_content += f"Total Subsections: {len(all_subsections)}\n"
    overview_content += f"Total Tables: {len(all_tables)}\n"
    overview_content += f"Total Figures: {len(all_figures)}\n"
    
    overview_chunk = {
        "content": overview_content,
        "metadata": {
            "class": class_num,
            "chapter": chapter_num,
            "chapter_title": chapter_title,
            "type": "chapter_overview",
            "identifier": "overview",
            "title": f"Chapter {chapter_num} Overview",
            "page_start": 1,
            "page_end": 1,
            "level": 0,
            "sections": all_sections,
            "subsections": all_subsections,
            "tables": all_tables,
            "figures": sorted(list(all_figures)),
            "stats": {
                "total_sections": len(all_sections),
                "total_subsections": len(all_subsections),
                "total_tables": len(all_tables),
                "total_figures": len(all_figures),
                "total_topics": len(all_sections) + len(all_subsections)
            }
        }
    }
    
    chunks.insert(0, overview_chunk)
    return chunks


def process_pdf(pdf_path: str, class_num: str, chapter_num: str, 
                chapter_title: str, progress_callback=None) -> Tuple[List[Dict], Dict]:
    """Process PDF and return chunks with statistics"""
    
    if progress_callback:
        progress_callback("Extracting text from PDF...", 0.1)
    pages_data = extract_text_with_pypdf(pdf_path)
    
    if progress_callback:
        progress_callback("Detecting content boundaries...", 0.3)
    boundaries = detect_all_boundaries(pages_data)
    
    if progress_callback:
        progress_callback("Setting boundary ends...", 0.5)
    set_boundary_ends(boundaries, pages_data)
    
    if progress_callback:
        progress_callback("Creating chunks...", 0.7)
    chunks = create_chunks(pages_data, boundaries, class_num, chapter_num, chapter_title)
    
    if progress_callback:
        progress_callback("Complete!", 1.0)
    
    # Calculate statistics
    stats = {
        "total_chunks": len(chunks),
        "multi_page_chunks": len([c for c in chunks if c['metadata'].get('spans_multiple_pages')]),
        "chunks_with_figures": len([c for c in chunks if c['metadata'].get('figures')]),
        "by_type": dict(Counter(c["metadata"]["type"] for c in chunks)),
        "has_overview": any(c["metadata"]["type"] == "chapter_overview" for c in chunks),
    }
    
    return chunks, stats


# ═══════════════════════════════════════════════════════════════
# STORAGE FUNCTIONS (FIXED - Saves in current directory)
# ═══════════════════════════════════════════════════════════════

def load_chunks() -> List[Dict]:
    """Load all chunks from file in current directory"""
    try:
        # Use current working directory
        file_path = Path.cwd() / CHUNKS_FILE.name
        
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data
        return []
    except Exception as e:
        st.error(f"❌ Load error: {e}")
        return []


def save_chunks(chunks: List[Dict]) -> bool:
    """Save all chunks to file in current directory"""
    try:
        # Ensure we're saving to current working directory
        file_path = Path.cwd() / CHUNKS_FILE.name
        
        # Write to file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        
        # Verify the file was created
        if file_path.exists():
            file_size = file_path.stat().st_size
            st.success(f"✅ Saved successfully to: {file_path.absolute()}")
            st.info(f"📊 File size: {file_size / 1024:.2f} KB")
            return True
        else:
            st.error("❌ File not created")
            return False
            
    except Exception as e:
        st.error(f"❌ Save error: {e}")
        import traceback
        st.code(traceback.format_exc())
        return False


def append_chapter_chunks(new_chunks: List[Dict], class_num: str, chapter_num: str) -> int:
    """Append new chapter chunks, replacing if exists"""
    try:
        # Load existing chunks
        all_chunks = load_chunks()
        
        # Count before
        count_before = len(all_chunks)
        
        # Remove existing chunks for this chapter
        filtered = [
            c for c in all_chunks
            if not (c["metadata"]["class"] == class_num and 
                   c["metadata"]["chapter"] == chapter_num)
        ]
        
        # Count removed
        removed_count = count_before - len(filtered)
        if removed_count > 0:
            st.info(f"🔄 Replacing {removed_count} existing chunks for Class {class_num}, Chapter {chapter_num}")
        
        # Add new chunks
        filtered.extend(new_chunks)
        
        # Save
        if save_chunks(filtered):
            st.success(f"✅ Added {len(new_chunks)} new chunks")
            st.success(f"📊 Total chunks in database: {len(filtered)}")
            return len(filtered)
        else:
            return 0
            
    except Exception as e:
        st.error(f"❌ Error appending chunks: {e}")
        import traceback
        st.code(traceback.format_exc())
        return 0


def get_database_stats() -> Dict:
    """Get statistics about the entire database"""
    chunks = load_chunks()
    
    if not chunks:
        return {"total": 0}
    
    by_chapter = defaultdict(list)
    for c in chunks:
        key = (c["metadata"]["class"], c["metadata"]["chapter"])
        by_chapter[key].append(c)
    
    return {
        "total": len(chunks),
        "chapters": len(by_chapter),
        "by_chapter": {
            f"Class {k[0]}, Ch {k[1]}": len(v)
            for k, v in sorted(by_chapter.items())
        }
    }


# ═══════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="NCERT Physics Chunker",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 NCERT Physics PDF Chunker")
    st.markdown("**Process NCERT Physics textbooks into intelligent chunks**")
    
    # Initialize session state
    if 'processed_chunks' not in st.session_state:
        st.session_state.processed_chunks = None
    if 'processing_info' not in st.session_state:
        st.session_state.processing_info = None
    
    # Show save location
    save_location = Path.cwd() / CHUNKS_FILE.name
    st.info(f"📁 Save location: `{save_location.absolute()}`")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Database")
        
        # Database stats
        db_stats = get_database_stats()
        st.metric("Total Chunks", db_stats.get("total", 0))
        st.metric("Chapters Processed", db_stats.get("chapters", 0))
        
        if db_stats.get("by_chapter"):
            with st.expander("📖 Chapters in Database"):
                for chapter, count in db_stats["by_chapter"].items():
                    st.write(f"**{chapter}**: {count} chunks")
        
        st.markdown("---")
        
        # Download database
        if db_stats.get("total", 0) > 0:
            st.subheader("💾 Export")
            file_path = Path.cwd() / CHUNKS_FILE.name
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    st.download_button(
                        label="Download Database (JSON)",
                        data=f.read(),
                        file_name="ncert_physics_chunks.json",
                        mime="application/json"
                    )
        
        st.markdown("---")
        st.caption(f"📁 Storage: {save_location.absolute()}")
    
    # Main content
    tab1, tab2 = st.tabs(["🔄 Process PDF", "🔍 Browse Database"])
    
    # Tab 1: Process PDF
    with tab1:
        st.header("Process New Chapter")
        
        col1, col2 = st.columns(2)
        
        with col1:
            class_num = st.selectbox(
                "Select Class",
                ["11", "12"],
                key="class_select"
            )
        
        with col2:
            chapters = CHAPTERS_CLASS_11 if class_num == "11" else CHAPTERS_CLASS_12
            chapter_selection = st.selectbox(
                "Select Chapter",
                chapters,
                key="chapter_select"
            )
        
        # Parse chapter
        match = re.match(r"(\d+)\s*-\s*(.+)", chapter_selection)
        if match:
            chapter_num = match.group(1)
            chapter_title = match.group(2).strip()
            
            st.info(f"📖 Processing: **Class {class_num}, Chapter {chapter_num}** - {chapter_title}")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload PDF",
            type=["pdf"],
            help="Select the NCERT Physics PDF chapter"
        )
        
        if uploaded_file:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            
            # Process button
            if st.button("🚀 Process PDF", type="primary", use_container_width=True):
                
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def update_progress(msg, value):
                        status_text.text(msg)
                        progress_bar.progress(value)
                    
                    # Process PDF
                    chunks, stats = process_pdf(
                        tmp_path,
                        class_num,
                        chapter_num,
                        chapter_title,
                        progress_callback=update_progress
                    )
                    
                    # Store in session state so it persists
                    st.session_state.processed_chunks = chunks
                    st.session_state.processing_info = {
                        'class_num': class_num,
                        'chapter_num': chapter_num,
                        'chapter_title': chapter_title,
                        'stats': stats
                    }
                    
                    # Display results
                    st.success(f"✅ Processing complete!")
                    st.rerun()  # Rerun to show the save button properly
                
                except Exception as e:
                    st.error(f"❌ Error processing PDF: {str(e)}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())
                
                finally:
                    # Cleanup
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
        
        # Display processed chunks (outside the button click)
        if st.session_state.processed_chunks is not None:
            chunks = st.session_state.processed_chunks
            info = st.session_state.processing_info
            stats = info['stats']
            
            st.markdown("---")
            st.success("✅ Chunks are ready to save!")
            
            # Statistics
            st.subheader("📊 Processing Results")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Chunks", stats["total_chunks"])
            col2.metric("Multi-page", stats["multi_page_chunks"])
            col3.metric("With Figures", stats["chunks_with_figures"])
            col4.metric("Chapter Overview", "✓" if stats["has_overview"] else "✗")
            
            # Content types
            st.subheader("📋 Content Types")
            type_cols = st.columns(4)
            for i, (content_type, count) in enumerate(sorted(stats["by_type"].items())):
                type_cols[i % 4].metric(content_type, count)
            
            # Preview chunks
            st.subheader("👀 Preview")
            
            # Get overview chunk
            overview = [c for c in chunks if c['metadata']['type'] == 'chapter_overview']
            if overview:
                with st.expander("📖 Chapter Overview", expanded=True):
                    st.text(overview[0]['content'])
            
            # Show first few chunks
            st.write("**First 5 chunks:**")
            for i, chunk in enumerate(chunks[1:6]):  # Skip overview
                with st.expander(
                    f"#{i+1}: {chunk['metadata']['type']} - {chunk['metadata']['identifier']}"
                ):
                    st.json(chunk['metadata'])
                    st.text_area("Content", chunk['content'][:500] + "...", height=200, key=f"preview_{i}")
            
            # Save to database
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("💾 Save to Database", type="primary", use_container_width=True, key="save_btn"):
                    with st.spinner("Saving to current directory..."):
                        total = append_chapter_chunks(
                            st.session_state.processed_chunks,
                            info['class_num'],
                            info['chapter_num']
                        )
                        if total > 0:
                            # Clear session state after successful save
                            st.session_state.processed_chunks = None
                            st.session_state.processing_info = None
                            st.balloons()
                            st.success("✅ Saved successfully! Refreshing...")
                            # Force rerun to update sidebar
                            st.rerun()
    
    # Tab 2: Browse Database
    with tab2:
        st.header("Browse Processed Chunks")
        
        chunks = load_chunks()
        
        if not chunks:
            st.info("📭 No chunks in database yet. Process a PDF to get started!")
            return
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            classes = sorted(list(set(c['metadata']['class'] for c in chunks)))
            selected_class = st.selectbox("Filter by Class", ["All"] + classes)
        
        with col2:
            if selected_class != "All":
                filtered_chunks = [c for c in chunks if c['metadata']['class'] == selected_class]
            else:
                filtered_chunks = chunks
            
            chapters = sorted(list(set(c['metadata']['chapter'] for c in filtered_chunks)))
            selected_chapter = st.selectbox("Filter by Chapter", ["All"] + chapters)
        
        with col3:
            if selected_chapter != "All":
                filtered_chunks = [c for c in filtered_chunks if c['metadata']['chapter'] == selected_chapter]
            
            types = sorted(list(set(c['metadata']['type'] for c in filtered_chunks)))
            selected_type = st.multiselect("Filter by Type", types, default=types)
        
        # Final filter
        filtered_chunks = [c for c in filtered_chunks if c['metadata']['type'] in selected_type]
        
        st.write(f"**Showing {len(filtered_chunks)} chunks**")
        
        # Display chunks
        for i, chunk in enumerate(filtered_chunks[:50]):  # Show first 50
            with st.expander(
                f"#{i+1}: Class {chunk['metadata']['class']}, "
                f"Ch {chunk['metadata']['chapter']}.{chunk['metadata']['identifier']} - "
                f"{chunk['metadata']['type']}"
            ):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.json(chunk['metadata'])
                
                with col2:
                    st.text_area("Content", chunk['content'], height=300, key=f"content_{i}")
        
        if len(filtered_chunks) > 50:
            st.info(f"ℹ️ Showing first 50 of {len(filtered_chunks)} chunks")


if __name__ == "__main__":
    if not PYPDF_AVAILABLE:
        st.error("Please install pypdf: pip install pypdf")
    else:
        main()