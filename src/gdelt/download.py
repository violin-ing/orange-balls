"""
Downloads GDELT files with verification and caching.

This module accomplishes two things:
    1. Verify the MD5
    2. Cache and never re-download
"""

from __future__ import annotations

import csv
import hashlib
import io
import logging
import zipfile
from pathlib import Path

import httpx
import pandas as pd

from src.gdelt.filelist import GdeltFile
from src.gdelt.schema import EVENT_COLUMNS, USED_COLUMNS

logger = logging.getLogger(__name__)


class ChecksumMismatch(RuntimeError):
    """
    Downloaded bytes did not match the published MD5.
    """


def _md5(payload: bytes) -> str:
    return hashlib.md5(payload).hexdigest()


def download(
        file: GdeltFile,
        cache_dir: Path,
        verify: bool = True,
        timeout: float = 60.0,
    ) -> Path:
    """
    Fetch one file into the cache, returning its local path.

    Note: an already-cached file is returned untouched.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / file.filename

    if destination.exists():
        logger.debug("cache hit %s", file.filename)
        return destination

    response = httpx.get(file.url, timeout=timeout, follow_redirects=True)
    response.raise_for_status()
    payload = response.content

    if verify:
        actual = _md5(payload)
        if actual != file.md5:
            raise ChecksumMismatch(
                f"{file.filename}: expected {file.md5}, got {actual}"
            )

    # Write to a temporary name first, then move
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_bytes(payload)
    temporary.replace(destination)

    logger.info("downloaded %s (%d bytes)", file.filename, len(payload))
    return destination


def read_events(
        path: Path, 
        columns: tuple[str, ...] = USED_COLUMNS
    ) -> pd.DataFrame:
    """
    Read one zipped export file into a DataFrame.

    Things to note:
        - The file is tab-delimited (not comma-delimited)
        - The file has no header row -> names must be supplied
        - Fields are never quoted -> quoting is switched off
        - Encoding is UTF-8, so any bad byte is replaced rather than raising
    """
    with zipfile.ZipFile(path) as archive:
        members = archive.namelist()
        if len(members) != 1:
            raise ValueError(
                f"{path.name}: expected one member, found {members}"
            )
        raw = archive.read(members[0])

    # Check: 61 columns = 60 tabs on every line
    # This is in case pandas pads short rows without warning
    tab_counts = {line.count(b"\t") for line in raw.splitlines() if line}
    if tab_counts != {60}:
        logger.warning(
            "%s: unexpected tab counts %s", path.name, sorted(tab_counts)
        )

    frame = pd.read_csv(
        io.BytesIO(raw),
        sep="\t",
        header=None,
        names=EVENT_COLUMNS,
        usecols=list(columns),
        dtype={
            "EventCode": "string",
            "EventRootCode": "string",
            "ActionGeo_CountryCode": "string",
            "Day": "string",
        },
        quoting=csv.QUOTE_NONE, # GDELT never quotes fields
        keep_default_na=False, # Do not treat "NA", "NULL" etc. as missing
        na_values=[""], # Only an empty field counts as missing
        encoding="utf-8",
        encoding_errors="replace", # Bad bytes -> placeholders (don't crash)
        on_bad_lines="warn",
    )
    
    return frame


def read_day(
        paths: list[Path], 
        columns: tuple[str, ...] = USED_COLUMNS
    ) -> pd.DataFrame:
    """
    Read and concatenate a day's worth of export files.
    """
    frames = [read_events(path, columns) for path in paths]
    if not frames:
        raise ValueError("no files given")
    
    return pd.concat(frames, ignore_index=True)
