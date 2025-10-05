import uuid
from infra.db.session import SessionLocal
from infra.db.models import FileRecord

class FilesRepository:
    def save(self, ftype: str, path: str, name: str) -> str:
        fid = f"file_{uuid.uuid4().hex}"
        with SessionLocal() as s:
            s.add(FileRecord(id=fid, type=ftype, path=path, name=name))
            s.commit()
        return fid

    def exists(self, file_id: str) -> bool:
        with SessionLocal() as s:
            return s.get(FileRecord, file_id) is not None

    def get_path(self, file_id: str) -> str:
        with SessionLocal() as s:
            rec = s.get(FileRecord, file_id)
            if not rec:
                raise KeyError("file not found")
            return rec.path