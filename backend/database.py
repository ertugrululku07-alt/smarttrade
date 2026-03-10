import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# Fallback/Default handling
if not DATABASE_URL:
    print("⚠️ DATABASE_URL not found! Using local fallback.")
    DATABASE_URL = "postgresql://user:password@localhost:5434/smarttrade"

# SQLAlchemy requires postgresql:// instead of postgres:// which Railway provides
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

print(f"📡 Attempting to initialize DB engine (URL prefix: {DATABASE_URL.split(':')[0]})")

try:
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
except Exception as e:
    print(f"🚨 CRITICAL: Engine creation failed: {e}")
    # Create a dummy engine to prevent import errors, allowing main.py to handle the failure gracefully
    engine = create_engine("sqlite:///:memory:")
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
