# aidp-rag

NVIDIA / AIDP retrieval half of the **AIDP × tensormesh — CacheBlend on Brev** demo.
**Owner:** NVIDIA (Yiwen) · part of [Tensormesh-Collaboration](https://github.com/Tensormesh-Collaboration)

**NeMo Retriever RAG** — ingestion + retrieval (NIM microservices; NGC-gated):
- Extraction (PDF/table/chart → chunks) · Embedding · Reranking
- vector DB for semantic search

## Where it fits
**This retriever** → retrieved context → `cacheblend-inf-stack` engine (generation + CacheBlend) → dual-chat UI.
Retrieved context must be **segmented + pre-tokenized** so CacheBlend can reuse non-prefix KV.

📋 Project hub: Partners / NVIDIA → *AIDP × tensormesh — CacheBlend on Brev*.
<img width="1711" height="1180" alt="image" src="https://github.com/user-attachments/assets/619b7faa-2f6e-43f1-b8d0-3fcd1f0411a1" />
