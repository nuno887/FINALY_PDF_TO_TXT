import io
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from PIL import Image
import pytesseract

CONFIG = {
    "input_dir": "input",
    "output_dir": "output",
    "pdf_filename": "IISerie-121-2018-08-13.pdf",     # e.g., "mydoc.pdf"; if None, auto-pick
    "autopick": "newest",     # when pdf_filename is None: "newest" or "first"
    "ignore_top_percent": 0.10,
    "skip_last_page": True,
    "dpi": 300,
    "ocr_lang": "por",
}

def ensure_output_dir(pdf_path: Path, output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    sub = output_root / pdf_path.stem
    sub.mkdir(parents=True, exist_ok=True)
    return sub

def export_single_page_pdf(src_doc: fitz.Document, page_index: int, dest_path: Path) -> None:
    one = fitz.open()
    one.insert_pdf(src_doc, from_page=page_index, to_page=page_index)
    one.save(dest_path.as_posix())
    one.close()

def page_clip_rect(page: fitz.Page, page_index: int, ignore_top_fraction: float) -> fitz.Rect:
    rect = page.rect
    if page_index == 0 or ignore_top_fraction <= 0:
        return rect
    top_cut = rect.height * ignore_top_fraction
    return fitz.Rect(rect.x0, rect.y0 + top_cut, rect.x1, rect.y1)
"""
def extract_text_digital(page: fitz.Page, clip: fitz.Rect) -> str:
    txt = page.get_text("text", clip=clip) or ""
    return txt.strip()
"""

def extract_text_digital(page: fitz.Page, clip: fitz.Rect) -> str:
    # Try block-aware ordering to keep a top full-width "Sumário" before two columns.
    blocks = page.get_text("blocks", clip=clip) or []
    if blocks:
        # sort by top (y0), then left (x0)
        blocks.sort(key=lambda b: (b[1], b[0]))

        # page width in the region we’re reading
        page_w = (clip.x1 - clip.x0) if clip else page.rect.width
        wide_threshold = 0.75  # 75% of the width counts as "full-width"

        def is_wide(b):
            return (b[2] - b[0]) >= wide_threshold * page_w

        # 1) take leading full-width blocks (Sumário continuation)
        head = []
        i = 0
        while i < len(blocks) and is_wide(blocks[i]):
            head.append(blocks[i])
            i += 1

        # 2) remaining blocks -> split by column (left vs right of page mid)
        remaining = blocks[i:]
        if remaining:
            mid_x = ((clip.x0 + clip.x1) / 2) if clip else ((page.rect.x0 + page.rect.x1) / 2)

            def center_x(b):
                return (b[0] + b[2]) / 2

            left = [b for b in remaining if center_x(b) < mid_x]
            right = [b for b in remaining if center_x(b) >= mid_x]

            left.sort(key=lambda b: (b[1], b[0]))
            right.sort(key=lambda b: (b[1], b[0]))

            ordered = head + left + right
        else:
            ordered = head

        text = "\n".join((b[4] or "").strip() for b in ordered if (b[4] or "").strip())
        if text.strip():
            return text.strip()

    # Fallback to PyMuPDF's default flow if blocks are empty
    return (page.get_text("text", clip=clip) or "").strip()



def extract_text_ocr(page: fitz.Page, clip: fitz.Rect, dpi: int, lang: str) -> str:
    pix = page.get_pixmap(dpi=dpi, clip=clip, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    try:
        return pytesseract.image_to_string(img, lang=lang).strip()
    except Exception:
        return pytesseract.image_to_string(img).strip()

def choose_pdf(input_dir: Path, filename: Optional[str], autopick: str) -> Optional[Path]:
    if filename:
        path = input_dir / filename
        return path if path.exists() else None
    pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        return None
    if autopick == "first":
        return pdfs[0]
    return max(pdfs, key=lambda p: p.stat().st_mtime)

def process_pdf_file(pdf_path: Path, output_root: Path, ignore_top_fraction: float, skip_last: bool,
                     dpi: int, ocr_lang: str):
    out_dir = ensure_output_dir(pdf_path, output_root)
    doc = fitz.open(pdf_path.as_posix())
    page_count = doc.page_count
    last_index = page_count - 1
    effective_last = last_index - 1 if (skip_last and page_count >= 1) else last_index
    if effective_last < 0:
        print("Nothing to process.")
        return
    for i in range(0, effective_last + 1):
        page = doc[i]
        clip = page_clip_rect(page, i, ignore_top_fraction)
        single_pdf_path = out_dir / f"page-{i+1:03d}.pdf"
        export_single_page_pdf(doc, i, single_pdf_path)
        text = extract_text_digital(page, clip=clip)
        if len(text) < 3:
            text = extract_text_ocr(page, clip=clip, dpi=dpi, lang=ocr_lang)
        (out_dir / f"page-{i+1:03d}.txt").write_text(text, encoding="utf-8")
    print(out_dir.as_posix())

def main():
    input_dir = Path(CONFIG["input_dir"]).resolve()
    output_root = Path(CONFIG["output_dir"]).resolve()
    input_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    pdf_path = choose_pdf(input_dir, CONFIG.get("pdf_filename"), CONFIG.get("autopick", "newest"))
    if not pdf_path:
        print(f"No PDF found in {input_dir.as_posix()}")
        return
    process_pdf_file(
        pdf_path=pdf_path,
        output_root=output_root,
        ignore_top_fraction=CONFIG.get("ignore_top_percent", 0.10),
        skip_last=CONFIG.get("skip_last_page", True),
        dpi=CONFIG.get("dpi", 300),
        ocr_lang=CONFIG.get("ocr_lang", "por"),
    )

if __name__ == "__main__":
    main()
