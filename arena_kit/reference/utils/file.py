"""File and directory utilities."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Union


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory if it doesn't exist, return Path object."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def copy_file(src: Union[str, Path], dst: Union[str, Path]) -> Path:
    """Copy file from src to dst, creating parent directories if needed."""
    src = Path(src)
    dst = Path(dst)
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return dst