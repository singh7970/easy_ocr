from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

# --- DEMO MODE: SQLite (no installation needed, auto-creates rawdat.db) ---
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
