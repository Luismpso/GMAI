"""Download a monthly Lichess game archive, safely and resumably.

The archives are large (about 33 GB for 2025-01), so this script:

* resumes an interrupted download instead of starting over, using HTTP range
  requests against a ``.part`` file;
* retries with back-off when the connection drops;
* checks there is enough free disk space before it starts;
* verifies the finished file against the SHA-256 checksum Lichess publishes,
  and only then gives it its final name.

Standard library only, so it runs before anything else is installed.

    python scripts/download_lichess.py                 # 2025-01 into data/raw
    python scripts/download_lichess.py --month latest
    python scripts/download_lichess.py --list          # available months

Run the same command again after an interruption: it picks up where it stopped.

You may not need the whole file at all. ``extract_lichess.py --month`` streams
straight from Lichess and stops once it has enough positions, which usually
means reading a fraction of the archive. A local copy pays off when you will
extract several times with different settings.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://database.lichess.org/standard"
LIST_URL = f"{BASE}/list.txt"
CHECKSUMS_URL = f"{BASE}/sha256sums.txt"
USER_AGENT = "gmai-chessnet-downloader/1.0"

CHUNK = 1 << 20  # 1 MiB
MAX_CONSECUTIVE_FAILURES = 30
MONTH_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
FILE_RE = re.compile(r"lichess_db_standard_rated_(\d{4}-\d{2})\.pgn\.zst")


# --------------------------------------------------------------- pure helpers
def filename_for(month: str) -> str:
    if not MONTH_RE.match(month):
        raise ValueError(f"month must look like 2025-01, got {month!r}")
    return f"lichess_db_standard_rated_{month}.pgn.zst"


def parse_months(list_text: str) -> list[str]:
    """Months present in Lichess's list.txt, newest first."""
    return sorted(set(FILE_RE.findall(list_text)), reverse=True)


def parse_checksums(text: str) -> dict[str, str]:
    """``sha256sum``-style lines into {filename: hex digest}."""
    sums = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and len(parts[0]) == 64:
            sums[parts[1].lstrip("*")] = parts[0].lower()
    return sums


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1000 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n:.0f} B"
        n /= 1000
    return f"{n:.1f} TB"


# ------------------------------------------------------------------- network
def _request(url: str, headers: dict | None = None, method: str = "GET"):
    req = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, **(headers or {})}, method=method
    )
    return urllib.request.urlopen(req, timeout=60)


def fetch_text(url: str) -> str:
    with _request(url) as resp:
        return resp.read().decode("utf-8", errors="replace")


def remote_size(url: str) -> int:
    with _request(url, method="HEAD") as resp:
        return int(resp.headers["Content-Length"])


def download(url: str, part: Path, total: int) -> None:
    """Fill ``part`` up to ``total`` bytes, resuming and retrying as needed."""
    failures = 0
    last_print = 0.0
    session_start = time.time()
    session_bytes = 0

    while True:
        have = part.stat().st_size if part.exists() else 0
        if have >= total:
            break
        try:
            headers = {"Range": f"bytes={have}-"} if have else {}
            with _request(url, headers) as resp:
                if have and resp.status != 206:
                    # The server ignored the range: start over rather than
                    # appending the beginning of the file to its middle.
                    print("\nserver does not resume; restarting from zero")
                    have = 0
                mode = "ab" if have else "wb"
                with part.open(mode) as out:
                    while True:
                        chunk = resp.read(CHUNK)
                        if not chunk:
                            break
                        out.write(chunk)
                        have += len(chunk)
                        session_bytes += len(chunk)
                        failures = 0
                        now = time.time()
                        if now - last_print >= 2 or have >= total:
                            rate = session_bytes / max(now - session_start, 1e-6)
                            left = (total - have) / max(rate, 1)
                            print(
                                f"\r  {human(have)} / {human(total)} "
                                f"({have / total:.1%}) | {human(rate)}/s "
                                f"| ~{left / 60:.0f} min left   ",
                                end="",
                                flush=True,
                            )
                            last_print = now
        except urllib.error.HTTPError as err:
            if err.code == 416:  # range starts at the end: already complete
                break
            failures += 1
            _backoff(failures, f"HTTP {err.code}")
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as err:
            failures += 1
            _backoff(failures, type(err).__name__)
    print()


def _backoff(failures: int, reason: str) -> None:
    if failures > MAX_CONSECUTIVE_FAILURES:
        sys.exit(
            f"\ngiving up after {MAX_CONSECUTIVE_FAILURES} failed attempts "
            f"({reason}). Run again later to resume."
        )
    wait = min(60, 2 ** min(failures, 6))
    print(
        f"\n  connection problem ({reason}); retrying in {wait}s "
        f"[{failures}/{MAX_CONSECUTIVE_FAILURES}]",
        flush=True,
    )
    time.sleep(wait)


def sha256_of(path: Path, total: int) -> str:
    digest = hashlib.sha256()
    done, last_print = 0, 0.0
    with path.open("rb") as f:
        while chunk := f.read(8 * CHUNK):
            digest.update(chunk)
            done += len(chunk)
            now = time.time()
            if now - last_print >= 2 or done >= total:
                print(f"\r  verifying {done / max(total, 1):.0%}   ", end="", flush=True)
                last_print = now
    print()
    return digest.hexdigest()


# ---------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--month", default="2025-01", help="e.g. 2025-01, or 'latest'")
    ap.add_argument("--out-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--list", action="store_true", help="show available months and exit")
    ap.add_argument("--no-verify", action="store_true", help="skip the SHA-256 check")
    args = ap.parse_args()

    if args.list or args.month == "latest":
        months = parse_months(fetch_text(LIST_URL))
        if not months:
            print("could not read the list of months from Lichess", file=sys.stderr)
            return 1
        if args.list:
            print(f"{len(months)} months available, newest first:")
            for i in range(0, len(months), 12):
                print("  " + "  ".join(months[i : i + 12]))
            return 0
        args.month = months[0]

    try:
        name = filename_for(args.month)
    except ValueError as err:
        print(err, file=sys.stderr)
        return 1

    url = f"{BASE}/{name}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    final = args.out_dir / name
    part = args.out_dir / (name + ".part")

    try:
        total = remote_size(url)
    except urllib.error.HTTPError as err:
        print(
            f"{url}: HTTP {err.code} (does that month exist? try --list)", file=sys.stderr
        )
        return 1

    print(f"month  : {args.month}")
    print(f"source : {url}")
    print(f"size   : {human(total)}")
    print(f"target : {final}")

    if final.exists() and final.stat().st_size == total:
        print("\nalready downloaded.")
        return 0

    have = part.stat().st_size if part.exists() else 0
    needed = total - have
    free = shutil.disk_usage(args.out_dir).free
    if free < needed + 1_000_000_000:  # keep 1 GB of headroom
        print(
            f"\nnot enough disk space: need {human(needed)}, have {human(free)} free",
            file=sys.stderr,
        )
        return 1
    if have:
        print(f"resume : {human(have)} already on disk")

    print()
    try:
        download(url, part, total)
    except KeyboardInterrupt:
        print("\ninterrupted. Run the same command again to resume.")
        return 130

    if part.stat().st_size != total:
        print(
            f"size mismatch: expected {total}, got {part.stat().st_size}", file=sys.stderr
        )
        return 1

    if not args.no_verify:
        expected = parse_checksums(fetch_text(CHECKSUMS_URL)).get(name)
        if expected is None:
            print("no published checksum for this month; skipping verification")
        else:
            actual = sha256_of(part, total)
            if actual != expected:
                bad = part.with_name(name + ".corrupt")
                part.replace(bad)
                print(
                    f"checksum MISMATCH. File moved to {bad}; delete it and "
                    "download again.",
                    file=sys.stderr,
                )
                return 2
            print("checksum OK (matches the one published by Lichess)")

    part.replace(final)
    print(f"\ndone: {final}")
    print(
        f'next : python scripts/extract_lichess.py --file "{final}" '
        "--positions 50000000 --min-elo 2000"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
