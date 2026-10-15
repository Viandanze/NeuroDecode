#!/usr/bin/env python3
"""Parallel download of PhysioNet eegmmidb EDFs (S001-S010, runs 4/8/12) into MNE cache layout.

v2: AWS Open Data S3 mirror (4-5x faster than physionet.org) + strict size check
(complete EDFs are 2.5-2.6 MB; anything smaller is a partial download and gets re-fetched).
"""
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = Path(r"C:\Users\17600\NeuroDecode\data\MNE-eegbci-data\files\eegmmidb\1.0.0")
URL = "https://physionet-open.s3.amazonaws.com/eegmmidb/1.0.0/{s}/{s}R{r:02d}.edf"
MIN_COMPLETE = 2_400_000  # complete files are ~2.55-2.6 MB

todo = []
for subj in range(1, 11):
    s = f"S{subj:03d}"
    for r in (4, 8, 12):
        dest = BASE / s / f"{s}R{r:02d}.edf"
        if not (dest.exists() and dest.stat().st_size >= MIN_COMPLETE):
            todo.append((s, r, dest))

print(f"to download: {len(todo)}", flush=True)


def fetch(item):
    s, r, dest = item
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    url = URL.format(s=s, r=r)
    for attempt in range(3):
        p = subprocess.run(["curl.exe", "-s", "-f", "-o", str(tmp), "--max-time", "600", url],
                           capture_output=True)
        if p.returncode == 0 and tmp.exists() and tmp.stat().st_size >= MIN_COMPLETE:
            tmp.replace(dest)  # atomic: only complete files land at final name
            return f"{s}R{r:02d} OK {dest.stat().st_size}"
    return f"{s}R{r:02d} FAIL (attempt {attempt+1})"


with ThreadPoolExecutor(max_workers=10) as ex:
    for msg in ex.map(fetch, todo):
        print(msg, flush=True)

n_ok = 0
for subj in range(1, 11):
    for r in (4, 8, 12):
        f = BASE / f"S{subj:03d}" / f"S{subj:03d}R{r:02d}.edf"
        if f.exists() and f.stat().st_size >= MIN_COMPLETE:
            n_ok += 1
print(f"DONE. {n_ok}/30 files complete", flush=True)
