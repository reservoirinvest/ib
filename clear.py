from __future__ import annotations

from pathlib import Path
from typing import Iterable

from build import ROOT

TARGET_DIRS: tuple[Path, ...] = (ROOT / "log", ROOT / "data")


def iter_files(directory: Path, recursive: bool = True) -> Iterable[Path]:
    """Yield files under `directory`, optionally descending into subdirectories."""
    if recursive:
        yield from (p for p in directory.rglob("*") if p.is_file())
    else:
        yield from (p for p in directory.iterdir() if p.is_file())


def clear_directory(directory: Path, recursive: bool = True, dry_run: bool = False) -> list[Path]:
    """
    Delete files inside `directory`.
    Returns the list of files removed (or that would be removed when dry_run=True).
    """
    removed = []
    if not directory.exists():
        return removed

    for file_path in iter_files(directory, recursive=recursive):
        removed.append(file_path)
        if not dry_run:
            file_path.unlink()

    return removed


def delete_files(recursive: bool = True, dry_run: bool = False) -> dict[Path, list[Path]]:
    """
    Clear every folder in TARGET_DIRS.

    Args:
        recursive: whether to descend into subdirectories.
        dry_run: if True, only report what would be deleted.

    Returns:
        Mapping of directory → list of files removed (or slated for removal).
    """
    report: dict[Path, list[Path]] = {}
    for directory in TARGET_DIRS:
        report[directory] = clear_directory(directory, recursive=recursive, dry_run=dry_run)
    return report


if __name__ == "__main__":
    deleted = delete_files()
    for directory, files in deleted.items():
        print(f"{directory}: deleted {len(files)} file(s)")
        for file_path in files:
            print(f"  - {file_path.relative_to(ROOT)}")