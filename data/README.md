# Data Directory

This directory contains datasets and scripts for data preparation.

## Structure

```
data/
├── hotpotqa/          # HotpotQA dataset (downloaded)
├── indexes/           # FAISS and BM25 indexes
└── scripts/           # Data preparation scripts
    ├── download_hotpotqa.py
    └── build_corpus.py
```

## Usage

### Download HotpotQA

```bash
python data/scripts/download_hotpotqa.py --output data/hotpotqa
```

### Build Indexes

```bash
python data/scripts/build_corpus.py --bm25 --dense --output data/indexes
```

## Dataset Format

HotpotQA samples are stored as JSONL with the following structure:

```json
{
  "id": "5a8b57f25542995d1e6f1371",
  "question": "What is the capital of France?",
  "answer": "Paris",
  "context": [
    ["Paris", ["Paris is the capital...", "..."]]
  ],
  "supporting_facts": [
    ["Paris", 0]
  ]
}
```

## Chunking Strategy

Documents are split into chunks of 256-512 tokens with 32-64 token overlap for retrieval.
