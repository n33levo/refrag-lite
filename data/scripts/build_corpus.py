#!/usr/bin/env python3
"""Build retrieval indexes."""
import argparse
import json
from pathlib import Path
import pickle

# Try to import refrag modules, fallback to simple implementations
try:
    from refrag.retrieval.bm25 import BM25Retriever
    from refrag.retrieval.dense import DenseRetriever
    REFRAG_AVAILABLE = True
except ImportError:
    print("Warning: refrag modules not available, using simple implementations")
    REFRAG_AVAILABLE = False
    
    # Simple fallback implementations
    class BM25Retriever:
        def __init__(self):
            self.corpus = []
        
        def index(self, corpus):
            self.corpus = corpus
            print(f"Simple BM25 index created with {len(corpus)} documents")
    
    class DenseRetriever:
        def __init__(self):
            self.corpus = []
        
        def index_documents(self, corpus):
            self.corpus = corpus
            print(f"Simple dense index created with {len(corpus)} documents")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/hotpotqa")
    parser.add_argument("--output", type=str, default="data/indexes")
    parser.add_argument("--bm25", action="store_true")
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--max-docs", type=int, default=None, help="Maximum number of documents to index")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load corpus
    corpus = []
    
    # Try JSONL files first
    for file in Path(args.input).glob("*.jsonl"):
        with open(file) as f:
            for line in f:
                item = json.loads(line)
                for title, sentences in item.get("context", []):
                    corpus.extend(sentences)
    
    # Try JSON files if no JSONL found
    if not corpus:
        for file in Path(args.input).glob("*.json"):
            with open(file) as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        for context_item in item.get("context", []):
                            # Handle different context structures
                            if isinstance(context_item, list) and len(context_item) >= 2:
                                title, sentences = context_item[0], context_item[1]
                                if isinstance(sentences, list):
                                    corpus.extend(sentences)
                            elif isinstance(context_item, str):
                                corpus.append(context_item)
                else:
                    # Single item
                    for context_item in data.get("context", []):
                        # Handle different context structures
                        if isinstance(context_item, list) and len(context_item) >= 2:
                            title, sentences = context_item[0], context_item[1]
                            if isinstance(sentences, list):
                                corpus.extend(sentences)
                        elif isinstance(context_item, str):
                            corpus.append(context_item)

    print(f"Loaded {len(corpus)} documents")
    
    # Debug: Show sample data structure
    if corpus:
        print(f"Sample corpus item: {corpus[0][:100]}...")
    else:
        print("⚠️ No corpus loaded - checking data structure...")
        for file in Path(args.input).glob("*.json"):
            with open(file) as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    sample = data[0]
                    print(f"Sample item keys: {list(sample.keys())}")
                    if "context" in sample:
                        print(f"Sample context structure: {sample['context'][:2] if len(sample['context']) > 0 else 'Empty'}")
                    break
    
    # Limit corpus size if max_docs is specified
    if args.max_docs and len(corpus) > args.max_docs:
        corpus = corpus[:args.max_docs]
        print(f"Limited to {len(corpus)} documents")

    if args.bm25:
        print("Building BM25 index...")
        bm25 = BM25Retriever()
        bm25.index(corpus)
        with open(output_path / "bm25.pkl", "wb") as f:
            pickle.dump(bm25, f)
        print("BM25 index saved")

    if args.dense:
        print("Building dense index...")
        dense = DenseRetriever()
        dense.index_documents(corpus)
        with open(output_path / "dense.pkl", "wb") as f:
            pickle.dump(dense, f)
        print("Dense index saved")

if __name__ == "__main__":
    main()