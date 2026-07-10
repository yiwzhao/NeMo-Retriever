# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Download the T2-RAGBench dataset (fixed revision) with `hf download` — HTTP
snapshot, never a git clone.

Layout at this revision:
    data/<subset>/<split>/pdf/*.pdf
    data/<subset>/<split>/metadata.jsonl   # id, question, program_answer,
                                            # original_answer, file_name, ...

Modes:
    quick  — only data/FinQA/dev/** (~72 MB)     [default]
    full   — the whole dataset (2.98 GB, 7,353 PDFs)

CLI:
    python download_dataset.py quick
    python download_dataset.py full --dir ~/t2-ragbench

Attribution / licensing: see deploy/brev/THIRD_PARTY_DATA.md.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess

REPO_ID = "G4KMU/t2-ragbench"
# Pinned so the corpus never shifts under the benchmark.
REVISION = "adf7fe1541ac37351ce1142544d8e3b43010ed92"

DATASET_DIR = pathlib.Path(os.environ.get("DATASET_DIR", str(pathlib.Path.home() / "t2-ragbench")))

MODES = {
    # Quick Demo: just the FinQA dev split (PDFs + metadata.jsonl), ~72 MB.
    "quick": {"include": ["data/FinQA/dev/**"], "desc": "FinQA dev (~72 MB)"},
    # Full Benchmark: everything at the pinned revision (2.98 GB, 7,353 PDFs).
    "full": {"include": None, "desc": "full benchmark (2.98 GB, 7,353 PDFs)"},
}

# subset/split pairs to ingest for each mode
SPLITS = {
    "quick": [("FinQA", "dev")],
    "full": [(s, sp) for s in ("FinQA", "ConvFinQA", "TAT-DQA") for sp in ("dev", "test", "train")],
}


def hf_download(mode: str = "quick", log=print) -> pathlib.Path:
    """Fetch the dataset via `hf download` (falls back to the huggingface_hub
    snapshot API if the `hf` CLI is unavailable). Returns the local dir."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {list(MODES)}")
    include = MODES[mode]["include"]
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    hf = shutil.which("hf")
    if hf:
        cmd = [hf, "download", REPO_ID, "--repo-type", "dataset",
               "--revision", REVISION, "--local-dir", str(DATASET_DIR)]
        for pat in include or []:
            cmd += ["--include", pat]
        log(f"$ {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    else:
        log("hf CLI not found — using huggingface_hub.snapshot_download (same HTTP download, not git)")
        from huggingface_hub import snapshot_download
        snapshot_download(
            REPO_ID, repo_type="dataset", revision=REVISION,
            local_dir=str(DATASET_DIR), allow_patterns=include,
        )
    return DATASET_DIR


def split_dir(subset: str, split: str) -> pathlib.Path:
    return DATASET_DIR / "data" / subset / split


def main() -> None:
    ap = argparse.ArgumentParser(description="Download T2-RAGBench (fixed revision) via hf download")
    ap.add_argument("mode", nargs="?", default="quick", choices=list(MODES))
    ap.add_argument("--dir", help="download target (default $DATASET_DIR or ~/t2-ragbench)")
    args = ap.parse_args()
    if args.dir:
        global DATASET_DIR
        DATASET_DIR = pathlib.Path(args.dir).expanduser()
    print(f"Downloading {REPO_ID}@{REVISION[:12]} [{MODES[args.mode]['desc']}] -> {DATASET_DIR}")
    hf_download(args.mode)
    for subset, split in SPLITS[args.mode]:
        d = split_dir(subset, split)
        n = len(list((d / "pdf").glob("*.pdf"))) if (d / "pdf").is_dir() else 0
        print(f"  {subset}/{split}: {n} PDFs, metadata.jsonl={'yes' if (d/'metadata.jsonl').is_file() else 'no'}")


if __name__ == "__main__":
    main()
