"""
NCERT Physics Comprehensive Chunking System - FINAL FIXED v2.2
===============================================================

FIXES:
1. Multi-page content extraction - sections spanning multiple pages
2. Table content extraction - captures actual table data (grey box)
3. Proper boundary management across pages
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter, defaultdict
from dataclasses import dataclass

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

CHUNKS_FILE = Path("all_chunks_final_fixed2.json")

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
    
    def __repr__(self):
        return f"<{self.content_type} {self.identifier} p{self.start_page}:{self.start_line}>"


def extract_text_with_pypdf(pdf_path: str) -> List[Dict]:
    """Extract text from PDF"""
    if not PYPDF_AVAILABLE:
        raise ImportError("pypdf is not available")
    
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
    """
    Detect blue box with chapter topics on first page.
    
    Pattern (around line 40):
    1.1 Introduction
    1.2 The international system of units
    ...
    Summary
    Exercises
    """
    if not pages_data:
        return None
    
    first_page = pages_data[0]
    lines = first_page['lines']
    
    # Find consecutive short lines with section numbers after line 35
    topic_start = -1
    topic_count = 0
    
    for i in range(35, len(lines)):
        line = lines[i].strip()
        
        # Skip empty
        if not line:
            continue
        
        # Check if this could be start of topic box
        if topic_start == -1:
            # Look for first short section number line
            if re.match(r'^\d+\.\d+\s+\w', line) and len(line) < 50:
                # Check if previous lines were long (actual content)
                if i > 5 and any(len(lines[j].strip()) > 80 for j in range(i-5, i)):
                    topic_start = i
                    topic_count = 1
        else:
            # Count consecutive topic lines
            if (re.match(r'^\d+\.\d+\s+', line) or 
                re.match(r'^(Summary|Exercises?|Points)', line, re.IGNORECASE) or
                len(line) < 30):  # Continuation lines like "units", "quantities"
                topic_count += 1
            elif "Reprint" in line:
                break
            else:
                # End of topic box
                break
    
    if topic_start != -1 and topic_count >= 7:  # At least 7 lines for complete topic box
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
    """
    Detect all content boundaries across all pages.
    
    Key: Track page numbers for multi-page content!
    """
    boundaries = []
    
    # 1. Detect blue box topics
    topics_box = detect_blue_box_topics(pages_data)
    if topics_box:
        boundaries.append(topics_box)
        print(f"  ✓ Chapter topics (page {topics_box.start_page+1}, lines {topics_box.start_line}-{topics_box.end_line})")
    
    # Create set of lines to skip (blue box)
    skip_lines = set()
    if topics_box:
        skip_lines = set(range(topics_box.start_line, topics_box.end_line))
    
    # 2. Detect content across all pages
    for page_idx, page in enumerate(pages_data):
        lines = page['lines']
        
        for line_idx, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Skip empty, reprint, or blue box lines
            if not line_stripped:
                continue
            if "reprint" in line_stripped.lower():
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
            
            # Section (UPPERCASE after section number)
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
            
            # Table - with extended content capture
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
    """
    Set end positions for all boundaries.
    
    CRITICAL: Support multi-page content!
    A section starting on page 3 can end on page 4 or later.
    """
    # Sort by page and line
    boundaries.sort(key=lambda b: (b.start_page, b.start_line))
    
    for i, boundary in enumerate(boundaries):
        # Skip if already set (like blue box)
        if boundary.end_page is not None and boundary.end_line is not None:
            continue
        
        if i + 1 < len(boundaries):
            next_boundary = boundaries[i + 1]
            # End where next boundary starts
            boundary.end_page = next_boundary.start_page
            boundary.end_line = next_boundary.start_line
        else:
            # Last boundary - extends to end of document
            boundary.end_page = len(pages_data) - 1
            boundary.end_line = len(pages_data[-1]['lines'])
        
        # For tables, limit to reasonable size (max 150 lines or until next major boundary)
        if boundary.content_type == "table":
            # Calculate total lines
            total_lines = 0
            for p in range(boundary.start_page, boundary.end_page + 1):
                if p == boundary.start_page:
                    total_lines += len(pages_data[p]['lines']) - boundary.start_line
                elif p == boundary.end_page:
                    total_lines += boundary.end_line
                else:
                    total_lines += len(pages_data[p]['lines'])
            
            # If too long, limit to 150 lines from start
            if total_lines > 150:
                # Recalculate end position
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
    """
    Extract content across multiple pages.
    
    CRITICAL FIX: Properly handle content spanning multiple pages!
    
    Returns:
        Tuple of (content_text, list_of_figure_ids)
    """
    content_lines = []
    figures = []
    
    # Extract from start_page to end_page
    for page_idx in range(boundary.start_page, boundary.end_page + 1):
        if page_idx >= len(pages_data):
            break
        
        page = pages_data[page_idx]
        lines = page['lines']
        
        for line_idx, line in enumerate(lines):
            # Determine if this line should be included
            include = False
            
            if page_idx == boundary.start_page and page_idx == boundary.end_page:
                # Single page - between start and end lines
                include = boundary.start_line <= line_idx < boundary.end_line
            elif page_idx == boundary.start_page:
                # First page - from start_line to end
                include = line_idx >= boundary.start_line
            elif page_idx == boundary.end_page:
                # Last page - from beginning to end_line
                include = line_idx < boundary.end_line
            else:
                # Middle pages - include all
                include = True
            
            if include:
                line_stripped = line.strip()
                if line_stripped and "reprint" not in line_stripped.lower():
                    content_lines.append(line_stripped)
                    
                    # Extract figure references from this line
                    # Patterns: Fig. 1.1, Fig 1.2, Figure 1.3, Fig.1.4
                    fig_matches = re.findall(r'Fig\.?\s*(\d+\.\d+)', line_stripped, re.IGNORECASE)
                    figures.extend(fig_matches)
    
    # Remove duplicates and sort
    figures = sorted(list(set(figures)))
    
    return "\n".join(content_lines), figures


def create_chunks(pages_data: List[Dict], boundaries: List[ContentBoundary],
                  class_num: str, chapter_num: str, chapter_title: str) -> List[Dict]:
    """Create chunks with multi-page content extraction and figure detection"""
    chunks = []
    
    # Track all topics and figures for chapter overview
    all_sections = []
    all_subsections = []
    all_tables = []
    all_figures = set()
    
    for boundary in boundaries:
        # Extract content across pages with figure detection
        content, figures = extract_content_multi_page(pages_data, boundary)
        
        # Skip very short content
        if len(content.strip()) < 20:
            continue
        
        # Track for chapter overview
        if boundary.content_type == "section":
            all_sections.append({
                "id": boundary.identifier,
                "title": boundary.title
            })
        elif boundary.content_type == "subsection":
            all_subsections.append({
                "id": boundary.identifier,
                "title": boundary.title
            })
        elif boundary.content_type == "table":
            all_tables.append(boundary.identifier)
        
        # Add figures to global set
        all_figures.update(figures)
        
        # Build metadata
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
        
        # Add figures to metadata if found
        if figures:
            metadata["figures"] = figures
            metadata["figure_count"] = len(figures)
        
        # Type-specific metadata
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
    
    # Create chapter overview chunk
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
    else:
        overview_content += "  None\n"
    
    overview_content += f"\n\nFIGURES ({len(all_figures)}):\n"
    overview_content += "-"*80 + "\n"
    if all_figures:
        sorted_figures = sorted(list(all_figures))
        overview_content += f"  Fig. " + ", Fig. ".join(sorted_figures) + "\n"
    else:
        overview_content += "  None\n"
    
    overview_content += "\n" + "="*80 + "\n"
    overview_content += "STATISTICS\n"
    overview_content += "="*80 + "\n"
    overview_content += f"Total Sections: {len(all_sections)}\n"
    overview_content += f"Total Subsections: {len(all_subsections)}\n"
    overview_content += f"Total Tables: {len(all_tables)}\n"
    overview_content += f"Total Figures: {len(all_figures)}\n"
    overview_content += f"Total Topics: {len(all_sections) + len(all_subsections)}\n"
    
    # Add chapter overview as first chunk
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
    
    # Insert overview at the beginning
    chunks.insert(0, overview_chunk)
    
    return chunks


def load_chunks() -> List[Dict]:
    """Load chunks"""
    try:
        if CHUNKS_FILE.exists():
            with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error: {e}")
    return []


def save_chunks(chunks: List[Dict]) -> bool:
    """Save chunks"""
    try:
        with open(CHUNKS_FILE, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


def append_chapter(new_chunks: List[Dict], class_num: str, chapter_num: str) -> int:
    """Append chapter chunks"""
    all_chunks = load_chunks()
    filtered = [c for c in all_chunks
                if not (c["metadata"]["class"] == class_num and 
                       c["metadata"]["chapter"] == chapter_num)]
    filtered.extend(new_chunks)
    return len(filtered) if save_chunks(filtered) else 0


def get_stats(chunks: List[Dict]) -> Dict:
    """Get statistics"""
    multi_page = [c for c in chunks if c['metadata'].get('spans_multiple_pages', False)]
    
    return {
        "total": len(chunks),
        "by_type": dict(Counter(c["metadata"]["type"] for c in chunks)),
        "multi_page_chunks": len(multi_page),
        "has_chapter_topics": any(c["metadata"]["type"] == "chapter_topics" for c in chunks),
    }


def process_pdf(pdf_path: str, class_num: str, chapter_num: str, chapter_title: str) -> List[Dict]:
    """Process PDF with multi-page support"""
    print(f"\n{'='*80}")
    print(f"Processing: Class {class_num}, Chapter {chapter_num} - {chapter_title}")
    print(f"{'='*80}\n")
    
    print("Step 1: Extracting text...")
    pages_data = extract_text_with_pypdf(pdf_path)
    print(f"✓ {len(pages_data)} pages")
    
    print("\nStep 2: Detecting boundaries (multi-page support)...")
    boundaries = detect_all_boundaries(pages_data)
    print(f"✓ {len(boundaries)} boundaries")
    
    print("\nStep 3: Setting boundary ends (cross-page)...")
    set_boundary_ends(boundaries, pages_data)
    print(f"✓ Boundaries configured")
    
    print("\nStep 4: Extracting content (multi-page)...")
    chunks = create_chunks(pages_data, boundaries, class_num, chapter_num, chapter_title)
    print(f"✓ {len(chunks)} chunks created")
    
    stats = get_stats(chunks)
    print(f"\n{'='*80}")
    print("Statistics:")
    print(f"{'='*80}")
    print(f"Total chunks: {stats['total']}")
    print(f"Multi-page chunks: {stats['multi_page_chunks']}")
    print(f"Has chapter topics: {stats['has_chapter_topics']}")
    print(f"\nBy type:")
    for t, c in sorted(stats['by_type'].items()):
        print(f"  {t}: {c}")
    
    return chunks


def main():
    import sys
    if len(sys.argv) < 4:
        print("Usage: python script.py <pdf_path> <class_num> <chapter>")
        return
    
    pdf_path = sys.argv[1]
    class_num = sys.argv[2]
    chapter_selection = sys.argv[3]
    
    match = re.match(r"(\d+)\s*-\s*(.+)", chapter_selection)
    if not match:
        print(f"Error: Invalid format")
        return
    
    chapter_num = match.group(1)
    chapter_title = match.group(2).strip()
    
    if not Path(pdf_path).exists():
        print(f"Error: File not found")
        return
    
    chunks = process_pdf(pdf_path, class_num, chapter_num, chapter_title)
    
    print(f"\n{'='*80}")
    print("Saving to database...")
    print(f"{'='*80}")
    total = append_chapter(chunks, class_num, chapter_num)
    if total > 0:
        print(f"✓ Saved successfully!")
        print(f"✓ Total chunks in database: {total}")
        print(f"✓ File: {CHUNKS_FILE.absolute()}")
    else:
        print("✗ Save failed")


if __name__ == "__main__":
    main()