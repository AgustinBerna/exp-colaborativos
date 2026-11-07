"""Cloud Storage checkpoints for an active run on local disk.

Set PIPELINE_REMOTE_URI and PIPELINE_WORK_DIR through run_pipeline.py. Complete
files are uploaded before their DONE marker. The active SQLite DB is copied
with sqlite3.backup(), never by reading its live bytes.
"""
from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import tempfile
from pathlib import Path


def _context():
    uri = os.environ.get("PIPELINE_REMOTE_URI", "").rstrip("/")
    work = os.environ.get("PIPELINE_WORK_DIR", "")
    if not uri:
        return None
    if not uri.startswith("gs://") or not work:
        raise ValueError("Checkpoint remoto requiere gs:// y PIPELINE_WORK_DIR")
    if not shutil.which("gcloud"):
        raise RuntimeError("gcloud no está disponible para checkpoints Spot")
    return Path(work).resolve(), uri


def upload(path: Path):
    """Upload one completed file to its matching relative path in the run."""
    context = _context()
    if context is None:
        return
    work, uri = context
    relative = path.resolve().relative_to(work).as_posix()
    subprocess.run(["gcloud", "storage", "cp", str(path), f"{uri}/{relative}"], check=True)


def upload_files_then_marker(files: list[Path], marker: Path):
    """A marker on GCS can never precede the artifacts it promises."""
    for path in files:
        upload(path)
    upload(marker)


def snapshot_sqlite(path: Path):
    """Publish a consistent Optuna state after a committed trial."""
    if _context() is None or not path.exists():
        return
    with tempfile.TemporaryDirectory() as temporary:
        copy = Path(temporary) / path.name
        with sqlite3.connect(path) as source, sqlite3.connect(copy) as target:
            source.backup(target)
        work, uri = _context()
        relative = path.resolve().relative_to(work).as_posix()
        subprocess.run(["gcloud", "storage", "cp", str(copy), f"{uri}/{relative}"], check=True)


def restore(work: Path, uri: str):
    """Restore remote checkpoints into a fresh local work-dir if present."""
    if not shutil.which("gcloud"):
        raise RuntimeError("gcloud no está disponible para restaurar la corrida")
    probe = subprocess.run(["gcloud", "storage", "ls", f"{uri.rstrip('/')}/run_identity.json"],
                           capture_output=True, text=True)
    if probe.returncode and not any(message in probe.stderr.lower() for message in
                                    ("no urls matched", "matched no objects", "not found")):
        raise RuntimeError(probe.stderr.strip())
    if not probe.stdout.strip():
        return
    subprocess.run(["gcloud", "storage", "rsync", uri.rstrip("/"), str(work), "--recursive"], check=True)


def sync_stage(work: Path):
    """Publish a finished stage; markers last. Exclude local CSV cache."""
    context = _context()
    if context is None:
        return
    _, uri = context
    subprocess.run(["gcloud", "storage", "rsync", str(work), uri, "--recursive", "--exclude",
                    r"(^|/)input/|(^|/)DONE\.json$|\.DONE\.json$|FINALIZADO\.json$|\.partial$|\.tmp$|study\.sqlite3$|\.pipeline\.lock$"], check=True)
    for path in work.rglob("study.sqlite3"):
        snapshot_sqlite(path)
    markers = [p for p in work.rglob("*.json") if p.name in ("DONE.json", "FINALIZADO.json") or p.name.endswith(".DONE.json")]
    for marker in sorted(markers, key=lambda path: (-len(path.parts), str(path))):
        upload(marker)

