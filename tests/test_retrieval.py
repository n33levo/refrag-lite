"""Test retrieval."""
import pytest
from refrag.retrieval.bm25 import BM25Retriever

def test_bm25():
    retriever = BM25Retriever()
    corpus = ["Paris is the capital", "London is a city"]
    retriever.index(corpus)
    results = retriever.retrieve("capital", top_k=1)
    assert len(results) > 0