"""Offline tests for scripts/download_lichess.py (no network access)."""

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "download_lichess.py"
_spec = importlib.util.spec_from_file_location("download_lichess", _PATH)
dl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dl)


def test_filename_for_valid_month():
    assert dl.filename_for("2025-01") == "lichess_db_standard_rated_2025-01.pgn.zst"


@pytest.mark.parametrize("bad", ["2025-13", "2025-00", "25-01", "2025/01", "latest", ""])
def test_filename_for_rejects_malformed_months(bad):
    with pytest.raises(ValueError):
        dl.filename_for(bad)


def test_parse_months_newest_first_and_deduplicated():
    text = "\n".join(
        [
            f"{dl.BASE}/lichess_db_standard_rated_2024-12.pgn.zst",
            f"{dl.BASE}/lichess_db_standard_rated_2026-08.pgn.zst",
            f"{dl.BASE}/lichess_db_standard_rated_2024-12.pgn.zst",
            "not a file line",
        ]
    )
    assert dl.parse_months(text) == ["2026-08", "2024-12"]


def test_parse_checksums_reads_sha256sum_format():
    digest = "a" * 64
    text = (
        f"{digest}  lichess_db_standard_rated_2013-01.pgn.zst\n"
        f"{'b' * 64} *lichess_db_standard_rated_2013-02.pgn.zst\n"
        "garbage line\n"
    )
    sums = dl.parse_checksums(text)
    assert sums["lichess_db_standard_rated_2013-01.pgn.zst"] == digest
    assert sums["lichess_db_standard_rated_2013-02.pgn.zst"] == "b" * 64
    assert len(sums) == 2


def test_sha256_of_matches_hashlib(tmp_path):
    import hashlib

    data = b"chess" * 100_000
    path = tmp_path / "blob"
    path.write_bytes(data)
    assert dl.sha256_of(path, len(data)) == hashlib.sha256(data).hexdigest()
