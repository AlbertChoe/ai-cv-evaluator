from sqlalchemy import Column, String, Float, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from infra.db.session import Base

class FileRecord(Base):
    __tablename__ = "files"
    id = Column(String, primary_key=True)
    type = Column(String, nullable=False)   # 'cv' | 'report'
    path = Column(String, nullable=False)
    name = Column(String, nullable=False)
    created_at = Column(DateTime, server_default=func.now())

class JobRecord(Base):
    __tablename__ = "jobs"
    id = Column(String, primary_key=True)
    status = Column(String, nullable=False, default="queued")
    job_title = Column(String, nullable=False)
    cv_file_id = Column(String, ForeignKey("files.id"), nullable=False)
    report_file_id = Column(String, ForeignKey("files.id"), nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    result = relationship("JobResultRecord", back_populates="job", uselist=False)

class JobResultRecord(Base):
    __tablename__ = "job_results"
    job_id = Column(String, ForeignKey("jobs.id"), primary_key=True)
    cv_match_rate = Column(Float, nullable=True)
    cv_feedback = Column(Text, nullable=True)
    project_score = Column(Float, nullable=True)
    project_feedback = Column(Text, nullable=True)
    overall_summary = Column(Text, nullable=True)
    job = relationship("JobRecord", back_populates="result")