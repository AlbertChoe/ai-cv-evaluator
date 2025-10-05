from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from app.settings import settings

engine = create_engine(
    f"sqlite:///{settings.SQLITE_PATH}", echo=False, future=True)
SessionLocal = sessionmaker(
    bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    pass


def init_db():
    from infra.db.models import FileRecord, JobRecord, JobResultRecord
    Base.metadata.create_all(bind=engine)
