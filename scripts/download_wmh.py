#!/usr/bin/env python3
"""
Download the WMH Segmentation Challenge 2017 data (170 subjects, 5 scanners, 3 sites).

The challenge closed in 2022 and released training + test data with manual annotations
on DataverseNL, DOI 10.34894/AECRSD, licence CC-BY-NC-4.0. Each subject folder holds
orig/ and pre/ copies of FLAIR, T1 and 3DT1 plus wmh.nii.gz (the manual WMH mask).

How. Single-file access redirects to an S3 store that serves ~30 KB/s per connection
from here, so the 8.75 GB would take days. The Dataverse bundle endpoint
(/api/access/datafiles/<id,id,...>) streams a zip from the Dataverse host itself at
>1 MB/s, so files are fetched in zips of at most --batch-mb, a few in parallel, extracted
into the dataset's own directory layout and checked against the SHA-1 Dataverse lists.
Anything missing from a bundle (the server caps zip size) falls back to a per-file,
range-resumed download. Re-running skips files already verified.

  python scripts/download_wmh.py                 # -> data/plan_b_raw/wmh/
  python scripts/download_wmh.py --include-results   # also the participants' result.nii.gz
"""
import argparse
import hashlib
import io
import json
import sys
import threading
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API = "https://dataverse.nl/api"
DOI = "doi:10.34894/AECRSD"
_print_lock = threading.Lock()


def log(msg):
    with _print_lock:
        print(msg, flush=True)


def listing():
    url = f"{API}/datasets/:persistentId/?persistentId={DOI}"
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)["data"]["latestVersion"]["files"]


def safe_rel(directory_label: str, filename: str) -> Path:
    # One Amsterdam folder is named "Philips_VU .PETMR_01." (space, trailing dot), which
    # this filesystem refuses; map each path component to a safe name.
    parts = [p.strip(" .").replace(" ", "_") for p in directory_label.split("/") if p]
    return Path(*parts) / filename if parts else Path(filename)


def sha1_of(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verified(path: Path, size: int, sha: str) -> bool:
    return path.exists() and path.stat().st_size == size and (not sha or sha1_of(path) == sha)


def fetch_single(item, tries: int = 12):
    """Per-file fallback with HTTP-Range resume (the S3 store drops long transfers)."""
    fid, dest, size, sha = item["id"], item["dest"], item["size"], item["sha"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, tries + 1):
        have = tmp.stat().st_size if tmp.exists() else 0
        if have > size:
            tmp.unlink()
            have = 0
        try:
            if have < size:
                req = urllib.request.Request(f"{API}/access/datafile/{fid}",
                                             headers={"Range": f"bytes={have}-"} if have else {})
                with urllib.request.urlopen(req, timeout=120) as r:
                    if have and r.status != 206:
                        have = 0
                    with open(tmp, "ab" if have else "wb") as f:
                        for chunk in iter(lambda: r.read(1 << 20), b""):
                            f.write(chunk)
            if tmp.stat().st_size != size:
                raise IOError(f"incomplete {tmp.stat().st_size}/{size}")
            if sha and sha1_of(tmp) != sha:
                tmp.unlink()
                raise IOError("sha1 mismatch, restarting file")
            tmp.rename(dest)
            return
        except Exception as e:  # noqa: BLE001
            log(f"  retry {attempt}/{tries} for {dest.name} (have {tmp.stat().st_size if tmp.exists() else 0}): {e}")
            time.sleep(min(60, 5 * attempt))
    raise SystemExit(f"giving up on {dest}")


def fetch_bundle(batch, tries: int = 4):
    """One zip for a batch of files; returns the items the zip did not contain."""
    by_entry = {f"{it['dir']}/{it['name']}" if it["dir"] else it["name"]: it for it in batch}
    ids = ",".join(str(it["id"]) for it in batch)
    for attempt in range(1, tries + 1):
        try:
            with urllib.request.urlopen(f"{API}/access/datafiles/{ids}", timeout=300) as r:
                buf = io.BytesIO(r.read())
            with zipfile.ZipFile(buf) as z:
                got = set()
                for info in z.infolist():
                    it = by_entry.get(info.filename)
                    if it is None:
                        continue
                    it["dest"].parent.mkdir(parents=True, exist_ok=True)
                    with z.open(info) as src, open(it["dest"], "wb") as dst:
                        for chunk in iter(lambda: src.read(1 << 20), b""):
                            dst.write(chunk)
                    if verified(it["dest"], it["size"], it["sha"]):
                        got.add(info.filename)
                    else:
                        it["dest"].unlink(missing_ok=True)
            return [it for k, it in by_entry.items() if k not in got]
        except Exception as e:  # noqa: BLE001
            log(f"  bundle retry {attempt}/{tries} ({len(batch)} files): {e}")
            time.sleep(10 * attempt)
    return list(batch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=str(ROOT / "data/plan_b_raw/wmh"))
    ap.add_argument("--include-results", action="store_true",
                    help="also download result.nii.gz (challenge participants' outputs)")
    ap.add_argument("--batch-mb", type=float, default=80.0, help="max bundle size per request")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    dest = Path(args.dest)

    items = []
    for f in listing():
        df = f["dataFile"]
        if df["filename"] == "result.nii.gz" and not args.include_results:
            continue
        ck = df.get("checksum", {})
        items.append({"id": df["id"], "dir": f.get("directoryLabel", ""), "name": df["filename"],
                      "size": df["filesize"], "sha": ck.get("value", "") if ck.get("type") == "SHA-1" else "",
                      "dest": dest / safe_rel(f.get("directoryLabel", ""), df["filename"])})
    total = sum(it["size"] for it in items)
    todo = [it for it in items if not verified(it["dest"], it["size"], it["sha"])]
    log(f"{len(items)} files, {total / 1e9:.2f} GB; {len(items) - len(todo)} already verified, {len(todo)} to fetch -> {dest}")

    # Pack remaining files into bundles under the size cap, largest first.
    cap = int(args.batch_mb * 1e6)
    batches, cur, cur_size = [], [], 0
    for it in sorted(todo, key=lambda x: -x["size"]):
        if cur and cur_size + it["size"] > cap:
            batches.append(cur)
            cur, cur_size = [], 0
        cur.append(it)
        cur_size += it["size"]
    if cur:
        batches.append(cur)

    done_bytes, leftovers, t0 = 0, [], time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_bundle, b): b for b in batches}
        for i, fut in enumerate(as_completed(futs), 1):
            b, missing = futs[fut], fut.result()
            leftovers.extend(missing)
            done_bytes += sum(it["size"] for it in b if it not in missing)
            rate = done_bytes / max(1, time.time() - t0) / 1e6
            log(f"[{i}/{len(batches)}] {done_bytes / 1e9:6.2f} GB  {rate:.2f} MB/s  ({len(missing)} missing from bundle)")
    if leftovers:
        log(f"{len(leftovers)} files not delivered in bundles; fetching individually")
        for it in leftovers:
            fetch_single(it)

    bad = [it for it in items if not verified(it["dest"], it["size"], it["sha"])]
    (dest / "SOURCE.txt").write_text(
        "WMH Segmentation Challenge 2017, DOI 10.34894/AECRSD (DataverseNL), CC-BY-NC-4.0.\n"
        "Kuijf et al., IEEE TMI 2019, doi:10.1109/TMI.2019.2905770\n")
    if bad:
        raise SystemExit(f"{len(bad)} files still unverified, e.g. {bad[0]['dest']}")
    log("done: all files verified")


if __name__ == "__main__":
    main()
