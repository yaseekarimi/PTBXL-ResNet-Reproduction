"""
Download only the PTB-XL v1.0.1 files needed for the 100 Hz reproduction:
  - ptbxl_database.csv
  - scp_statements.csv
  - records100/**/*.hea
  - records100/**/*.dat

The existing PTB-XL v1.0.3 directory is not modified.

Official source:
https://physionet.org/files/ptb-xl/1.0.1/
"""

from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
import csv
import shutil
import sys
import time

BASE_URL = "https://physionet.org/files/ptb-xl/1.0.1/"
DEST = Path("data/ptbxl/ptb-xl-1.0.1")
META = ("ptbxl_database.csv", "scp_statements.csv")
USER_AGENT = "Mozilla/5.0 PTBXL-reproduction-downloader"

def download(url: str, dest: Path, retries: int = 4):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return False

    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            req = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(req, timeout=60) as r, open(part, "wb") as f:
                shutil.copyfileobj(r, f, length=1024 * 1024)
            part.replace(dest)
            return True
        except (URLError, HTTPError, TimeoutError, OSError) as e:
            if part.exists():
                part.unlink()
            if attempt == retries:
                raise
            print(f"  Retry {attempt}/{retries} after error: {e}")
            time.sleep(2 * attempt)

def main():
    print("=" * 68)
    print("PTB-XL v1.0.1 — 100 Hz downloader")
    print(f"Destination: {DEST}")
    print("Existing PTB-XL v1.0.3 will NOT be modified.")
    print("=" * 68)

    DEST.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] Downloading metadata...")
    for name in META:
        changed = download(BASE_URL + name, DEST / name)
        print(f"  {'Downloaded' if changed else 'Already exists'}: {name}")

    print("\n[2/3] Reading official filename_lr paths...")
    db_path = DEST / "ptbxl_database.csv"
    records = []
    with open(db_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel = row["filename_lr"].strip()
            if rel:
                records.append(rel)

    print(f"  Metadata rows / 100-Hz records: {len(records)}")
    if len(records) != 21837:
        print(f"WARNING: expected 21,837 rows for PTB-XL v1.0.1, got {len(records)}.")

    print("\n[3/3] Downloading records100 waveform pairs (.hea + .dat)...")
    total_files = len(records) * 2
    completed = 0
    downloaded = 0

    for i, rel in enumerate(records, start=1):
        # filename_lr has no extension, e.g. records100/00000/00001_lr
        for ext in (".hea", ".dat"):
            target_rel = rel + ext
            changed = download(BASE_URL + target_rel, DEST / target_rel)
            downloaded += int(changed)
            completed += 1

        if i == 1 or i % 250 == 0 or i == len(records):
            print(
                f"  Records: {i:5d}/{len(records)} | "
                f"files checked: {completed:5d}/{total_files} | "
                f"new downloads: {downloaded}"
            )

    print("\nDownload complete.")
    print(f"PTB-XL v1.0.1 path: {DEST}")
    print(f"Metadata records: {len(records)}")
    print("Only records100 was downloaded; records500 was not requested.")
    print("\nNext step: point the reproduction script PTBXL_PATH to:")
    print('data/ptbxl/ptb-xl-1.0.1/')

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user. Re-run the script to resume; existing files are skipped.")
        sys.exit(130)
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        print("Re-run the script to retry. Already-downloaded files will be skipped.")
        sys.exit(1)
