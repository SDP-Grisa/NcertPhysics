#!/usr/bin/env python3
"""
Improved NCERT Physics figure extractor - better caption detection & association

Usage:
    python extract_ncert_diagrams_improved.py "leph1210.pdf" 12 "10 - Wave Optics"
"""

import sys
import re
import fitz  # PyMuPDF
from pathlib import Path
from PIL import Image
import io

def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    class_num = sys.argv[2]
    chapter_name = sys.argv[3].strip()

    safe_chapter = re.sub(r'[^a-zA-Z0-9_-]', '_', chapter_name)
    out_dir = Path(f"ncert_class_{class_num}_ch_{safe_chapter}_figures")
    out_dir.mkdir(exist_ok=True)

    print(f"→ Processing: {pdf_path.name}")
    print(f"→ Output:     {out_dir}\n")

    doc = fitz.open(pdf_path)

    # Pattern: FIGURE 10.13 ... or Fig. 10.4(a) etc.
    fig_pattern = re.compile(
        r'(?:FIGURE|Fig\.?|Figure)\s*'
        r'([0-9]+(?:\.[0-9]+)?(?:\s*[a-zA-Z])?)'
        r'\b\s*(.*?)(?=\n{2,}|\n\s*(?:[0-9]+\.|FIGURE|\Z))',
        re.IGNORECASE | re.DOTALL
    )

    markdown_lines = [
        f"# NCERT Class {class_num} - {chapter_name}\n",
        f"Extracted figures from **{pdf_path.name}**\n\n"
    ]

    saved = 0

    for page_num, page in enumerate(doc, 1):
        text = page.get_text("text")
        matches = list(fig_pattern.finditer(text))

        if not matches:
            continue

        # Get all images, sort top → bottom
        img_list = []
        for img_idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            bbox = page.get_image_bbox(img)
            if bbox and bbox.width > 140 and bbox.height > 90:  # stricter size filter
                img_list.append((bbox.y0, xref, bbox, img_idx))  # y0 = top

        img_list.sort()  # sort by top y-coordinate

        used = set()

        for match in matches:
            fig_num = match.group(1).strip()
            caption = match.group(2).strip().replace('\n', ' ').replace('  ', ' ')
            caption = re.sub(r'\s+', ' ', caption).strip()

            # Find caption location
            instances = page.search_for(match.group(0)[:30])  # partial match for robustness
            if not instances:
                continue
            cap_rect = instances[0]
            cap_top_y = cap_rect.y0

            # Find nearest reasonable image ABOVE caption
            best = None
            best_dist = 9999

            for y_top, xref, bbox, idx in img_list:
                if idx in used:
                    continue
                img_bottom = bbox.y1
                dist = cap_top_y - img_bottom
                if 5 <= dist <= 480 and dist < best_dist:  # image ends before caption starts
                    best_dist = dist
                    best = (xref, bbox, idx)

            if best:
                xref, bbox, idx = best
                used.add(idx)

                base_img = doc.extract_image(xref)
                img_bytes = base_img["image"]
                ext = base_img["ext"]

                try:
                    pil = Image.open(io.BytesIO(img_bytes))
                    w, h = pil.size
                    if w < 180 or h < 100:
                        continue
                except:
                    continue

                # Filename: fig_10_13.png
                num_clean = fig_num.replace(".", "_").replace(" ", "")
                fname = f"fig_{num_clean}.png"
                fpath = out_dir / fname

                fpath.write_bytes(img_bytes)

                # Metadata txt
                meta_path = out_dir / f"fig_{num_clean}.txt"
                with meta_path.open("w", encoding="utf-8") as mf:
                    mf.write(f"Class     : {class_num}\n")
                    mf.write(f"Chapter   : {chapter_name}\n")
                    mf.write(f"Page      : {page_num}\n")
                    mf.write(f"Figure    : {fig_num}\n")
                    mf.write(f"Caption   : {caption}\n")
                    mf.write(f"Size      : {w} × {h} px\n")

                # Add to summary markdown
                markdown_lines.append(f"**Fig. {fig_num}**  (page {page_num})\n")
                markdown_lines.append(f"{caption}\n")
                markdown_lines.append(f"![Fig {fig_num}]({fname})\n\n")

                print(f"  Saved: {fname:18}  |  {fig_num:8}  |  {caption[:70]}{'...' if len(caption)>70 else ''}")

                saved += 1
            # else:
            #     print(f"  Skipped: {fig_num} on page {page_num} (no matching image above)")

    # Save summary markdown
    summary_md = out_dir / "_summary.md"
    summary_md.write_text("".join(markdown_lines), encoding="utf-8")

    doc.close()

    print(f"\nFinished. Extracted **{saved}** figures → {out_dir}")
    print(f"Summary with captions: {summary_md.name}")

if __name__ == "__main__":
    main()