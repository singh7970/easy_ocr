"""
app.py — FastAPI web server for the OCR app.

Routes:
  GET  /           → upload page (index.html)
  POST /upload/    → process file with EasyOCR, return JSON
  POST /save/      → save OCR result to SQLite database
  GET  /diary/     → diary view (diary.html)
  GET  /health     → health check (used by run_app.py to know server is ready)
  GET  /uploads/{filename} → serve uploaded images stored in ./uploads/
"""

import os
import sys
import time
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, Depends, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

# ── Path resolution ──────────────────────────────────────────────────────────
# When run via run_app.py (or as a PyInstaller EXE) APP_BASE_DIR is set to the
# bundle root so Jinja2 / StaticFiles can find the templates/ and static/ dirs.
BASE_DIR = Path(os.environ.get("APP_BASE_DIR", os.path.dirname(os.path.abspath(__file__))))

TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR    = BASE_DIR / "static"
UPLOADS_DIR   = Path("uploads")          # keep uploads OUTSIDE the bundle
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(title="OCR App")

# Mount static files (only if the folder exists)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Mount uploads so the diary can show original images
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# ── DB dependency ────────────────────────────────────────────────────────────
from database import get_db, OCRDocument

# ── OCR imports ──────────────────────────────────────────────────────────────
from main import process_image_ocr, create_easyocr_reader
from process_document import process_file

# Create one global EasyOCR reader so it loads only once at startup
_reader = None

def get_reader():
    global _reader
    if _reader is None:
        _reader = create_easyocr_reader()
    return _reader


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload/")
async def upload_file(
    file: UploadFile = File(...),
    user_name: str = Form(""),
    preset: str = Form("accurate"),
    crop_document: bool = Form(True),
):
    """Receive an uploaded file, run OCR, return extracted text as JSON."""
    allowed = {".pdf", ".jpg", ".jpeg", ".png"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unsupported file type: {suffix}. Use PDF, JPG, or PNG."},
        )

    # Save to a temp file
    tmp_name = f"{uuid.uuid4().hex}{suffix}"
    tmp_path = UPLOADS_DIR / tmp_name
    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        reader = get_reader()
        t0 = time.perf_counter()
        plain, conf, rowwise, raw_conf, table_tsv = process_file(
            str(tmp_path),
            preset=preset,
            crop_document=crop_document,
            reader=reader,
        )
        elapsed = time.perf_counter() - t0

        return {
            "text":              plain,
            "text_plain":        plain,
            "rowwise":           rowwise,
            "table_tsv":         table_tsv,
            "confidence":        f"{conf:.1%}",
            "time_taken":        f"{elapsed:.1f}s",
            "image_path":        f"/uploads/{tmp_name}",
            "download_filename": Path(file.filename).stem + "_ocr.txt",
        }

    except Exception as exc:
        # Clean up on error
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.post("/save/")
async def save_to_db(
    filename:      str = Form(...),
    text_content:  str = Form(...),
    document_date: str = Form(""),
    image_path:    str = Form(""),
    user_name:     str = Form(""),
    db: Session = Depends(get_db),
):
    """Save an OCR result to the database."""
    doc = OCRDocument(
        filename=filename,
        text_content=text_content,
        document_date=document_date or None,
        image_path=image_path or None,
        user_name=user_name or None,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return {"status": "saved", "id": doc.id}


@app.get("/diary/", response_class=HTMLResponse)
def diary(request: Request, q: Optional[str] = None, db: Session = Depends(get_db)):
    """Diary / archive view — shows all saved OCR documents."""
    query = db.query(OCRDocument).order_by(OCRDocument.created_at.desc())

    if q:
        like = f"%{q}%"
        query = query.filter(
            OCRDocument.filename.ilike(like)
            | OCRDocument.text_content.ilike(like)
            | OCRDocument.document_date.ilike(like)
            | OCRDocument.user_name.ilike(like)
        )

    documents = query.all()

    # Build autocomplete suggestions (unique user names + dates)
    suggestions = list(
        {d.user_name for d in documents if d.user_name}
        | {d.document_date for d in documents if d.document_date}
    )

    return templates.TemplateResponse(
        "diary.html",
        {
            "request":   request,
            "documents": documents,
            "q":         q or "",
            "suggestions": suggestions,
        },
    )
