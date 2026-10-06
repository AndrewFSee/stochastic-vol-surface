"""Incremental backup of the data folder to another drive.

The option chains in ``data/options`` cannot be re-downloaded (Yahoo serves
no history), so they exist nowhere else.  Everything else can be rebuilt,
but backing up the whole folder is cheap (about 1.4 GB) and makes a restore a
plain copy.

Rules
-----
* **Incremental:** a file is copied only when it is missing from the backup
  or differs in size or modification time.
* **Never deletes:** files removed locally stay in the backup, so an
  accidental deletion or a bad rebuild cannot propagate.
* **Verified:** after copying, ``data/options`` is compared file-for-file
  (count and total bytes) against the backup.
* Secrets are not in ``data/`` (they live in ``.env``), so none are copied.

Restore: copy ``<backup_dir>/data`` back over the project's ``data`` folder.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

#: Folders whose loss would be permanent; verified after every backup.
IRREPLACEABLE = ("options",)
#: Folders not worth copying (scratch space).
SKIP_DIRS = {"__pycache__"}


class BackupError(RuntimeError):
    """The backup could not be completed or did not verify."""


@dataclass
class BackupResult:
    target: str
    files_copied: int = 0
    bytes_copied: int = 0
    files_total: int = 0
    seconds: float = 0.0
    verified: dict = field(default_factory=dict)
    finished_at: str = ""


def _changed(src: Path, dst: Path) -> bool:
    if not dst.exists():
        return True
    s, d = src.stat(), dst.stat()
    return s.st_size != d.st_size or abs(s.st_mtime - d.st_mtime) > 2


def _tree_summary(root: Path) -> tuple[int, int]:
    files = [p for p in root.rglob("*") if p.is_file()] if root.exists() else []
    return len(files), sum(p.stat().st_size for p in files)


def backup_data(data_dir: str | Path = "data", backup_dir: str | Path = "D:/stochastic-vol-surface-backup") -> BackupResult:
    """Copy new or changed files from *data_dir* to ``<backup_dir>/data``.

    Raises :class:`BackupError` if the backup location is unavailable or the
    irreplaceable folders do not match afterwards.
    """
    src_root = Path(data_dir).resolve()
    backup_root = Path(backup_dir)
    anchor = Path(backup_root.anchor) if backup_root.anchor else backup_root
    if not anchor.exists():
        raise BackupError(f"backup drive {anchor} is not available (unplugged?)")
    dst_root = backup_root / "data"
    dst_root.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    res = BackupResult(target=str(dst_root))
    for src in src_root.rglob("*"):
        if not src.is_file() or SKIP_DIRS.intersection(src.relative_to(src_root).parts):
            continue
        res.files_total += 1
        dst = dst_root / src.relative_to(src_root)
        if _changed(src, dst):
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            res.files_copied += 1
            res.bytes_copied += src.stat().st_size

    for name in IRREPLACEABLE:
        local, remote = _tree_summary(src_root / name), _tree_summary(dst_root / name)
        res.verified[name] = {"local_files": local[0], "local_bytes": local[1],
                              "backup_files": remote[0], "backup_bytes": remote[1]}
        # The backup never deletes, so it may hold more than local; never less.
        if remote[0] < local[0] or remote[1] < local[1]:
            raise BackupError(f"backup of data/{name} does not verify: "
                              f"{remote[0]} files / {remote[1]:,} bytes in backup vs "
                              f"{local[0]} / {local[1]:,} locally")

    res.seconds = round(time.time() - t0, 1)
    res.finished_at = datetime.now().isoformat(timespec="seconds")
    (backup_root / "last_backup.json").write_text(json.dumps(asdict(res), indent=1), encoding="utf-8")
    logger.info("Backup: %d of %d files copied (%.1f MB) to %s in %.0fs",
                res.files_copied, res.files_total, res.bytes_copied / 1e6, dst_root, res.seconds)
    return res
