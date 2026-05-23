import tempfile
import os
import fitz  # PyMuPDF
from main import process_image_ocr, create_easyocr_reader

def process_file(
    file_path: str,
    preset: str = "accurate",
    crop_document: bool = True,
    reader=None,
):
    """
    Process an uploaded file (PDF or Image) and extract text.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.pdf':
        return process_pdf(file_path, reader=reader, preset=preset, crop_document=crop_document)
    elif ext in ['.jpg', '.jpeg', '.png']:
        return process_image_ocr(
            file_path,
            save_to_db=False,
            reader=reader,
            preset=preset,
            crop_document=crop_document,
        )
    else:
        raise ValueError(f"Unsupported file extension {ext}. Only PDF, JPG, JPEG, and PNG are allowed.")

def process_pdf(
    pdf_path: str,
    reader=None,
    preset: str = "accurate",
    crop_document: bool = True,
):
    """
    Process PDF — FREE TIER: Only the FIRST PAGE is extracted.
    Full multi-page parsing is reserved for the full-access version.
    Reuses a single EasyOCR Reader when ``reader`` is provided.
    """
    own_reader = reader is None
    if own_reader:
        reader = create_easyocr_reader()

    try:
        doc = fitz.open(pdf_path)

        # ── FREE TIER: Process only the first page (page index 0) ────────────
        # To restore full multi-page parsing, replace the block below with the
        # commented-out loop further down.
        page_num = 0
        page = doc.load_page(page_num)

        # --- HYBRID APPROACH: Try native text layer first ---
        native_text = page.get_text("text").strip()

        # If we found significant native text, use it directly (it's 100% accurate)
        if len(native_text) > 10:
            plain    = native_text
            head_c   = 1.0
            rowwise  = f"=== Native PDF Text Layer (High Accuracy) ===\n\n{native_text}"
            raw_c    = 1.0
            table_tsv = ""
        else:
            # --- SCANNED PAGE: Fallback to EasyOCR ---
            # Render the page at high resolution (approx 300 DPI)
            zoom = 300 / 72  # 300 DPI
            mat  = fitz.Matrix(zoom, zoom)
            pix  = page.get_pixmap(matrix=mat, alpha=False)

            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                tmp_path = tmp_file.name

            try:
                pix.save(tmp_path)
                plain, head_c, rowwise, raw_c, table_tsv = process_image_ocr(
                    tmp_path,
                    save_to_db=False,
                    reader=reader,
                    preset=preset,
                    crop_document=crop_document,
                )
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        # Wrap in page header for consistency
        final_plain   = f"--- Page 1 ---\n{plain}" if plain.strip() else ""
        final_rowwise = f"--- Page 1 ---\n{rowwise}" if rowwise.strip() else ""
        final_table   = f"--- Page 1 ---\n{table_tsv.strip()}" if table_tsv.strip() else ""

        return final_plain, head_c, final_rowwise, raw_c, final_table

        # ── FULL ACCESS: Multi-page loop (commented out — free tier only) ────
        # Uncomment the block below and remove the single-page block above
        # to restore full PDF parsing for all pages.
        #
        # full_plain, full_rowwise, full_tables = [], [], []
        # total_head, total_raw, count = 0.0, 0.0, 0
        #
        # for page_num in range(len(doc)):
        #     page = doc.load_page(page_num)
        #     native_text = page.get_text("text").strip()
        #     if len(native_text) > 10:
        #         plain = native_text
        #         head_c = 1.0
        #         rowwise = f"=== Native PDF Text Layer (High Accuracy) ===\n\n{native_text}"
        #         raw_c = 1.0
        #         table_tsv = ""
        #     else:
        #         zoom = 300 / 72
        #         mat = fitz.Matrix(zoom, zoom)
        #         pix = page.get_pixmap(matrix=mat, alpha=False)
        #         with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        #             tmp_path = tmp_file.name
        #         try:
        #             pix.save(tmp_path)
        #             plain, head_c, rowwise, raw_c, table_tsv = process_image_ocr(
        #                 tmp_path, save_to_db=False, reader=reader,
        #                 preset=preset, crop_document=crop_document,
        #             )
        #         finally:
        #             if os.path.exists(tmp_path):
        #                 os.remove(tmp_path)
        #     if plain.strip() or rowwise.strip():
        #         full_plain.append(f"--- Page {page_num + 1} ---\n{plain}")
        #         full_rowwise.append(f"--- Page {page_num + 1} ---\n{rowwise}")
        #     if table_tsv.strip():
        #         full_tables.append(f"--- Page {page_num + 1} ---\n{table_tsv.strip()}")
        #     total_head += head_c
        #     total_raw += raw_c
        #     count += 1
        # final_plain   = "\n\n".join(full_plain)
        # final_rowwise = "\n\n".join(full_rowwise)
        # final_table   = "\n\n".join(full_tables)
        # avg_head = total_head / count if count > 0 else 0.0
        # avg_raw  = total_raw  / count if count > 0 else 0.0
        # return final_plain, avg_head, final_rowwise, avg_raw, final_table
        # ─────────────────────────────────────────────────────────────────────

    finally:
        if own_reader:
            del reader
