<!--
SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Third-Party Data Attribution

The Deploy UI and notebook can download and ingest a public benchmark dataset,
**T²-RAGBench**, as demo/benchmark data. That dataset and its source corpora are
third-party works with their own licenses and attribution requirements. This file
records them.

> The NeMo Retriever code in this repository does **not** include or redistribute
> this data. It is downloaded at runtime, on the user's request, from Hugging Face
> at a pinned revision (see [`download_dataset.py`](./download_dataset.py)).

---

## T²-RAGBench (the dataset we download)

| | |
|---|---|
| Name | **T²-RAGBench** — Text-and-Table Benchmark for Evaluating RAG |
| Hugging Face | [`G4KMU/t2-ragbench`](https://huggingface.co/datasets/G4KMU/t2-ragbench) |
| Pinned revision | `adf7fe1541ac37351ce1142544d8e3b43010ed92` |
| License | **CC BY 4.0** (Creative Commons Attribution 4.0 International) |
| Paper | Strich et al., *T²-RAGBench*, [arXiv:2506.12071](https://arxiv.org/abs/2506.12071) |
| Size / scope | ~2.98 GB, **7,353 PDF documents**, 23,088 context-independent QA pairs |
| What we use | The original **PDF files** (`data/<subset>/<split>/pdf/*.pdf`) and the per-split **`metadata.jsonl`** (fields: `question`, `program_answer`, `original_answer`, `file_name`, …). |

**Attribution (required by CC BY 4.0):**

> T²-RAGBench (G4KMU/t2-ragbench), Strich et al., 2025, University of Hamburg —
> licensed under CC BY 4.0. https://huggingface.co/datasets/G4KMU/t2-ragbench

CC BY 4.0 permits commercial use, sharing, and adaptation provided you give
appropriate credit, link the license, and indicate any changes.

---

## Source datasets (bundled into T²-RAGBench)

T²-RAGBench is assembled from three public financial-document QA datasets. PDFs
for FinQA and ConvFinQA derive from FinTabNet; TAT-DQA ships its own PDFs.

| Subset | Source | Upstream license | Docs | QA pairs |
|--------|--------|------------------|------|----------|
| **FinQA** | [FinQA](https://github.com/czyssrs/FinQA) (SEC filings via FinTabNet) | MIT | 2,789 | 8,281 |
| **ConvFinQA** | [ConvFinQA](https://github.com/czyssrs/ConvFinQA) (FinTabNet) | MIT | 1,806 | 3,458 |
| **TAT-DQA** | [TAT-DQA](https://github.com/NExTplusplus/TAT-DQA) (SEC filings) | MIT | 2,723 | 11,349 |

> Exact upstream license terms are governed by each source project; verify against
> the linked repositories and the T²-RAGBench dataset card before redistribution.
> The aggregate T²-RAGBench release is licensed **CC BY 4.0**.

---

## Was the data filtered or processed?

Yes. Relevant to commercial/benchmark use:

- **Reformulated questions.** T²-RAGBench rewrote the original questions with
  **Llama-3.3-70B** to make them *context-independent* (answerable without seeing
  the surrounding passage). Answers are the verified originals.
- **Curated & deduplicated** across the three source datasets into a single
  benchmark of 23,088 QA pairs over 7,353 documents.
- **What this launchable does with it:** downloads a **pinned revision** (no
  silent drift), and ingests the **PDF files only** through the NeMo Retriever
  pipeline (extract → embed → LanceDB). We read `metadata.jsonl` to associate each
  PDF with its `file_name`, `question`, and answer, and to surface a sample
  question in the playground. We do **not** modify or redistribute the source
  files.
- **Quick Demo** ingests only the **FinQA `dev`** split (~72 MB). **Full
  Benchmark** ingests the entire corpus (2.98 GB, 7,353 PDFs) on demand.

---

## Sources

- T²-RAGBench dataset: <https://huggingface.co/datasets/G4KMU/t2-ragbench>
- T²-RAGBench paper: <https://arxiv.org/abs/2506.12071>
- FinQA: <https://github.com/czyssrs/FinQA>
- ConvFinQA: <https://github.com/czyssrs/ConvFinQA>
- TAT-DQA: <https://github.com/NExTplusplus/TAT-DQA>
