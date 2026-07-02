#!/usr/bin/env python3
"""Upload LeWAM datasets + checkpoints to private HuggingFace repos.

Run ON L40S (or wherever /mnt/minghao_data/.stable-wm lives), in the `lewm` venv
that has huggingface_hub. RESUMABLE: re-run after any interruption and it picks up
where it stopped (upload_large_folder tracks state in a .cache dir inside each folder).

Auth: either `huggingface-cli login` first (stores the token), or `export HF_TOKEN=...`.

  export HF_HOME=/mnt/minghao_data/hf          # keep HF cache on the big disk
  python upload_to_hf.py                        # uploads everything
  python upload_to_hf.py data                   # only the datasets
  python upload_to_hf.py ckpts                  # only checkpoints + decoders
"""
import os
import sys
from huggingface_hub import HfApi

BASE = os.environ.get("STABLEWM_HOME", "/mnt/minghao_data/.stable-wm")
USER = "MinghaoFu"
WORKERS = int(os.environ.get("HF_UPLOAD_WORKERS", "8"))

what = sys.argv[1] if len(sys.argv) > 1 else "all"
api = HfApi(token=os.environ.get("HF_TOKEN"))  # None -> uses stored `huggingface-cli login`

# 1. create the private repos (idempotent — no-op if they already exist)
api.create_repo(f"{USER}/lewm-official-data",  repo_type="dataset", private=False, exist_ok=True)
api.create_repo(f"{USER}/lewm-official-ckpts", repo_type="model",   private=False, exist_ok=True)
print("[upload] repos ready:", f"{USER}/lewm-official-data (dataset),", f"{USER}/lewm-official-ckpts (model)", flush=True)

# 2. datasets (~425 GB) -> lewm-official-data  [resumable, parallel, multi-commit]
if what in ("all", "data"):
    print(f"[upload] === datasets: {BASE}/datasets -> {USER}/lewm-official-data ===", flush=True)
    api.upload_large_folder(
        repo_id=f"{USER}/lewm-official-data", repo_type="dataset",
        folder_path=f"{BASE}/datasets", num_workers=WORKERS, print_report=True)

# 3. checkpoints + decoders (~112 GB) -> lewm-official-ckpts (structure preserved: checkpoints/** + decoders/**)
if what in ("all", "ckpts"):
    print(f"[upload] === checkpoints+decoders: {BASE} -> {USER}/lewm-official-ckpts ===", flush=True)
    api.upload_large_folder(
        repo_id=f"{USER}/lewm-official-ckpts", repo_type="model",
        folder_path=BASE, allow_patterns=["checkpoints/**", "decoders/**"],
        num_workers=WORKERS, print_report=True)

print("[upload] ALL DONE — pull on the new box with:", flush=True)
print(f"  huggingface-cli download {USER}/lewm-official-data  --repo-type dataset --local-dir $DATA/.stable-wm/datasets", flush=True)
print(f"  huggingface-cli download {USER}/lewm-official-ckpts --repo-type model   --local-dir $DATA/.stable-wm", flush=True)
