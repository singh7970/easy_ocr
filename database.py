from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

import os

# ── Database path resolution ─────────────────────────────────────────────────
# In EXE mode: APP_DATA_DIR is set by run_app.py → points to folder containing
# the .exe → database is created there (persistent, next to the EXE).
# In dev mode: falls back to current working directory (normal behaviour).
_data_dir = os.environ.get("APP_DATA_DIR", "")
if _data_dir:
    # Absolute path — safe even when CWD is a temp/zip folder
    _db_path = os.path.join(_data_dir, "rawdat.db").replace("\\", "/")
    DATABASE_URL = f"sqlite:///{_db_path}"
else:
    DATABASE_URL = "sqlite:///./rawdat.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# --- PRODUCTION MODE: MySQL (uncomment below and comment out SQLite above) ---
# DATABASE_URL = "mysql+pymysql://rawdat_user:rawdata123@localhost/rawdat"
# engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class OCRDocument(Base):
    __tablename__ = "ocr_documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), index=True)
    text_content = Column(Text)
    document_date = Column(String(255), nullable=True)
    image_path = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    user_name = Column(String(255), nullable=True)

# Create tables in the database if they don't exist
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
