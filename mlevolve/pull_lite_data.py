#!/usr/bin/env python
"""
Pull MLE-Bench Lite (TIGER-Lab HF mirror) and extract tasks into mlevolve/data/.

CPU-only. No Kaggle credentials, no mlebench package. Resumable (huggingface-cli
caches; re-running skips already-downloaded parts).

Mirror: TIGER-Lab/mle-bench = "Simple fork of MLE-Bench Lite" (112GB, 6 split parts).
Reassemble: cat data_part_* > data.zip; the zip holds one folder per competition:
<slug>/prepared/{public,private}/...  (mlebench native prepared layout).

Extracts the 4 Lite tasks not yet on PVC (spooky + mlsp-2013-bird already present),
then cross-checks the mirror's spooky train.csv md5 against the existing on-disk
spooky (which came from an official mlebench prepare) as a fidelity test.
"""
import os, sys, subprocess, zipfile, hashlib, shutil, collections
from pathlib import Path

_env_path = os.environ.get("PATH", "")
os.environ["PATH"] = f"/opt/anaconda3/bin:/usr/local/bin:/usr/bin:/bin:{_env_path}"

PARTS = Path("/workspace/mlebench_lite_parts")
ZIP = Path("/workspace/mlebench_lite.zip")
DATA = Path("/workspace/nautilus/mlevolve/data")
WANT = [
    "leaf-classification",
    "aerial-cactus-identification",
    "denoising-dirty-documents",
    "new-york-city-taxi-fare-prediction",
]


def run(cmd, **kw):
    print(f"$ {cmd if isinstance(cmd, str) else ' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, **kw)


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


# 1. ensure huggingface_hub
try:
    import huggingface_hub  # noqa: F401
except ImportError:
    run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])

PARTS.mkdir(parents=True, exist_ok=True)

# 2. download 6 parts (resumable)
print("=== [1/5] downloading 6 parts (~112GB, resumable) ===", flush=True)
run([
    "huggingface-cli", "download", "TIGER-Lab/mle-bench",
    "--repo-type", "dataset", "--include", "data_part_*",
    "--local-dir", str(PARTS),
])

# 3. assemble zip (only if not already assembled)
parts = sorted(PARTS.glob("data_part_*"))
print(f"parts on disk: {[p.name for p in parts]}", flush=True)
if not parts:
    print("ERROR: no data_part_* found after download", flush=True)
    sys.exit(1)
if not ZIP.exists():
    print("=== [2/5] cat parts -> zip ===", flush=True)
    with open(ZIP, "wb") as out:
        for p in parts:
            with open(p, "rb") as f:
                shutil.copyfileobj(f, out, 1 << 20)
else:
    print("zip already assembled, skipping cat", flush=True)

# 4. inspect + selective extract
print("=== [3/5] extracting 4 tasks into mlevolve/data/ (selective) ===", flush=True)
z = zipfile.ZipFile(ZIP)
names = z.namelist()
top = collections.Counter(n.split("/")[0] for n in names)
print(f"zip top-level folders ({len(top)}): {sorted(top.keys())}", flush=True)
DATA.mkdir(parents=True, exist_ok=True)
for s in WANT:
    members = [n for n in names if n.startswith(s + "/")]
    if not members:
        print(f"  WARNING: no zip members for {s}", flush=True)
        continue
    z.extractall(str(DATA), members=members)
    desc = DATA / s / "prepared" / "public" / "description.md"
    train = DATA / s / "prepared" / "public" / "train.csv"
    nrows = sum(1 for _ in open(train)) if train.exists() else "N/A"
    print(f"  OK  {s:42s} files={len(members):4d}  train.csv_rows={nrows}  desc={desc.exists()}", flush=True)

# 5. spooky fidelity cross-check (mirror vs existing official-prepared spooky)
print("=== [4/5] spooky fidelity cross-check (mirror vs existing) ===", flush=True)
vdir = Path("/workspace/_v_spooky")
sp_members = [n for n in names if n.startswith("spooky-author-identification/")]
z.extractall(str(vdir), members=sp_members)
try:
    m = md5(vdir / "spooky-author-identification" / "prepared" / "public" / "train.csv")
    e = md5(DATA / "spooky-author-identification" / "prepared" / "public" / "train.csv")
    print(f"  mirror  spooky train.csv md5 = {m}", flush=True)
    print(f"  existing spooky train.csv md5 = {e}", flush=True)
    print("  SPOOKY_MATCH  (mirror == existing -> faithful)" if m == e
          else "  SPOOKY_DIFFER (md5 mismatch -> investigate before citing numbers)", flush=True)
except Exception as ex:
    print(f"  cross-check skipped: {ex}", flush=True)
shutil.rmtree(vdir, ignore_errors=True)
z.close()

# 6. free assembled zip (keep the 6 parts so other Lite tasks can be extracted later w/o re-download)
ZIP.unlink(missing_ok=True)
print(f"=== [5/5] freed assembled zip; parts kept at {PARTS} for future Lite tasks ===", flush=True)
print("ALL_DONE", flush=True)
