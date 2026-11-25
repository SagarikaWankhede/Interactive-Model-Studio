import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base

# SQLite database file stored in backend/data.db
DB_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{os.path.join(DB_DIR, 'data.db')}")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """Dependency for acquiring DB session in FastAPI routes."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Creates database tables if they do not exist and applies migrations."""
    Base.metadata.create_all(bind=engine)
    
    # Run soft column migrations for SQLite
    with engine.connect() as conn:
        try:
            conn.execute(text("ALTER TABLE runs ADD COLUMN recipe_json JSON"))
            conn.commit()
        except Exception:
            pass
        for ddl in (
            "ALTER TABLE runs ADD COLUMN model_config_json JSON",
            "ALTER TABLE runs ADD COLUMN graph_json JSON",
            "ALTER TABLE runs ADD COLUMN generated_code TEXT",
            "ALTER TABLE runs ADD COLUMN parent_run_id INTEGER",
        ):
            try:
                conn.execute(text(ddl))
                conn.commit()
            except Exception:
                conn.rollback()
