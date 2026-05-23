import os
import tempfile
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Request, Depends, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from database import OCRDocument, get_db
import datetime
import uuid
import shutil
import os
from pydantic import BaseModel

from main import READTEXT_PRESETS, configure_torch_threads, create_easyocr_reader
from process_document import process_file


@asynccontextmanager
async def lifespan(app: FastAPI):
    # One EasyOCR load for the whole server — avoids 10–60s model reload on every upload.
    configure_torch_threads()
    force_cpu = os.environ.get("OCR_FORCE_CPU", "").strip().lower() in {"1", "true", "yes", "on"}
    if force_cpu:
        print("OCR_FORCE_CPU is set — using CPU only.")
    print("⏳ Pre-loading EasyOCR (first start only; grab coffee on CPU)...")
    t0 = time.perf_counter()
    app.state.ocr_reader = create_easyocr_reader(gpu=False if force_cpu else None)
    print(f"✅ EasyOCR ready in {time.perf_counter() - t0:.1f}s — uploads will skip this step.")
    yield
    if hasattr(app.state, "ocr_reader"):
        del app.state.ocr_reader


app = FastAPI(title="OCR Document Analyzer API", lifespan=lifespan)

# ── Path resolution (works both in dev and as a PyInstaller EXE) ─────────────
# APP_BASE_DIR → where templates/ and static/ were bundled (sys._MEIPASS in EXE)
# APP_DATA_DIR → writable folder next to the .exe (for uploads & SQLite DB)
_base_dir = os.environ.get("APP_BASE_DIR", os.path.dirname(os.path.abspath(__file__)))
_data_dir = os.environ.get("APP_DATA_DIR", os.path.dirname(os.path.abspath(__file__)))

_static_dir  = os.path.join(_base_dir, "static")
_uploads_dir = os.path.join(_data_dir, "static", "uploads")
os.makedirs(_uploads_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=_static_dir), name="static")

# Setup templates
templates = Jinja2Templates(directory=os.path.join(_base_dir, "templates"))


def format_processing_error(error: Exception) -> str:
    return str(error)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/diary/", response_class=HTMLResponse)
async def read_diary(request: Request, q: str = None, db: Session = Depends(get_db)):
    all_docs = db.query(OCRDocument).order_by(OCRDocument.created_at.desc()).all()
    documents = []
    
    if q:
        q_lower = q.lower()
        search_terms = q_lower.split()
        for doc in all_docs:
            created_str = doc.created_at.strftime('%B %d, %Y').lower()
            created_str_alt = doc.created_at.strftime('%d %B %Y').lower()
            uname = (doc.user_name or "").lower()
            ddate = (doc.document_date or "").lower()
            fname = (doc.filename or "").lower()
            textc = (doc.text_content or "").lower()
            
            search_corpus = f"{uname} {ddate} {fname} {created_str} {created_str_alt} {textc}"
            
            if all(term in search_corpus for term in search_terms):
                documents.append(doc)
    else:
        documents = all_docs
    
    suggestions = set()
    for doc in all_docs:
        if doc.user_name and doc.user_name.strip():
            suggestions.add(doc.user_name.strip())
        if doc.document_date and doc.document_date.strip():
            suggestions.add(doc.document_date.strip())
        suggestions.add(doc.created_at.strftime('%d %B %Y'))
        
    return templates.TemplateResponse(request=request, name="diary.html", context={
        "documents": documents, 
        "q": q or "",
        "suggestions": sorted(list(suggestions))
    })

class SaveBody(BaseModel):
    filename: str
    text_content: str
    user_name: str | None = None
    document_date: str | None = None
    image_path: str | None = None

@app.post("/save/")
async def save_document(req: SaveBody, db: Session = Depends(get_db)):
    new_doc = OCRDocument(
        filename=req.filename,
        text_content=req.text_content,
        user_name=req.user_name,
        document_date=req.document_date,
        image_path=req.image_path,
        created_at=datetime.datetime.now()
    )
    db.add(new_doc)
    db.commit()
    return {"message": "Saved successfully"}

@app.post("/upload/")
async def upload_file(
    request: Request, 
    file: UploadFile = File(...), 
    user_name: str = Form(None), 
    preset: str = Form("accurate"), 
    db: Session = Depends(get_db)
):
    # Validate file extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".pdf", ".jpg", ".jpeg", ".png"]:
        return JSONResponse(
            status_code=400, 
            content={"error": "Invalid file type. Please upload a PDF, JPG, or PNG file."}
        )
    
    # Clear old temporary outputs if requested (keeping disk clean except for saved images)
    os.makedirs(_uploads_dir, exist_ok=True)
    if os.path.exists("outputs"):
        for f in os.listdir("outputs"):
            try: os.remove(os.path.join("outputs", f))
            except Exception: pass

    # Save uploaded file with a unique name to prevent overwriting and allow persistent diary storage
    unique_name = f"{uuid.uuid4()}{ext}"
    tmp_path = os.path.join(_uploads_dir, unique_name)
    with open(tmp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    image_path = f"/static/uploads/{unique_name}"  # URL path (always forward slashes)

    if preset not in READTEXT_PRESETS:
        preset = os.environ.get("OCR_PRESET", "accurate").strip() or "accurate"
    if preset not in READTEXT_PRESETS:
        preset = "accurate"
    reader = getattr(request.app.state, "ocr_reader", None)

    try:
        print(f"Processing uploaded file: {file.filename} (preset={preset})")
        start_time = time.perf_counter()

        text_plain, confidence, text_rowwise, confidence_raw, table_tsv = process_file(
            tmp_path,
            preset=preset,
            reader=reader,
        )

        time_taken = round(time.perf_counter() - start_time, 2)

        base_name = os.path.splitext(file.filename or "document")[0]
        safe_base = "".join(c for c in base_name if c.isalnum() or c in (" ", "-", "_")).strip() or "document"
        download_filename = f"{safe_base}.txt"

        return JSONResponse(status_code=200, content={
            "filename": file.filename,
            "image_path": image_path,
            "download_filename": download_filename,
            "text": text_plain,
            "text_detail": text_rowwise,
            "text_plain": text_plain,
            "table_tsv": table_tsv,
            "confidence": f"{confidence:.2%}",
            "confidence_raw_segments": f"{confidence_raw:.2%}",
            "confidence_hint": (
                "Main badge weights the clearest text regions. "
                "Raw is the average of all segments — closer to strict model scores. "
                "Neither number is guaranteed word-level accuracy."
            ),
            "time_taken": f"{time_taken}s",
        })
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return JSONResponse(
            status_code=500, 
            content={"error": format_processing_error(e)}
        )
    finally:
        # File is kept in static folder, do not delete
        pass

if __name__ == "__main__":
    import uvicorn
    # Make sure we start from this point correctly
    uvicorn.run(app, host="0.0.0.0", port=8000)
