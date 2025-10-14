"""Real retrieval implementation with BM25 and dense retrieval."""
import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import pickle
import json
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi
import re


class RealBM25Retriever:
    """Real BM25 retrieval implementation."""
    
    def __init__(self, index_path: str = None):
        """Initialize BM25 retriever.
        
        Args:
            index_path: Path to saved BM25 index
        """
        self.index_path = index_path
        self.bm25 = None
        self.corpus = []
        self.corpus_ids = []
        
        if index_path and Path(index_path).exists():
            self.load_index(index_path)
    
    def build_index(self, corpus: List[Dict[str, Any]], save_path: str = None):
        """Build BM25 index from corpus.
        
        Args:
            corpus: List of documents with 'text' and 'id' fields
            save_path: Path to save the index
        """
        self.corpus = []
        self.corpus_ids = []
        
        # Extract text and IDs
        for doc in corpus:
            text = doc.get('text', '')
            if text.strip():
                self.corpus.append(text)
                self.corpus_ids.append(doc.get('id', len(self.corpus_ids)))
        
        # Tokenize corpus
        tokenized_corpus = [self._tokenize(text) for text in self.corpus]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # Save index
        if save_path:
            self.save_index(save_path)
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float, int]]:
        """Search for relevant documents.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of (text, score, doc_id) tuples
        """
        if self.bm25 is None:
            raise ValueError("BM25 index not built. Call build_index() first.")
        
        # Tokenize query
        tokenized_query = self._tokenize(query)
        
        # Get scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k results
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only return documents with positive scores
                results.append((
                    self.corpus[idx],
                    scores[idx],
                    self.corpus_ids[idx]
                ))
        
        return results
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25."""
        # Simple tokenization - can be improved
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return text.split()
    
    def save_index(self, save_path: str):
        """Save BM25 index to disk."""
        index_data = {
            'bm25': self.bm25,
            'corpus': self.corpus,
            'corpus_ids': self.corpus_ids
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(index_data, f)
    
    def load_index(self, load_path: str):
        """Load BM25 index from disk."""
        with open(load_path, 'rb') as f:
            index_data = pickle.load(f)
        
        self.bm25 = index_data['bm25']
        self.corpus = index_data['corpus']
        self.corpus_ids = index_data['corpus_ids']


class RealDenseRetriever:
    """Real dense retrieval implementation with FAISS."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 index_path: str = None, device: str = "auto"):
        """Initialize dense retriever.
        
        Args:
            model_name: Sentence transformer model name
            index_path: Path to saved FAISS index
            device: Device to run on
        """
        self.model_name = model_name
        self.index_path = index_path
        self.device = device if device != "auto" else self._get_device()
        
        # Load sentence transformer
        self.encoder = SentenceTransformer(model_name, device=self.device)
        
        # FAISS index
        self.index = None
        self.corpus = []
        self.corpus_ids = []
        
        if index_path and Path(index_path).exists():
            self.load_index(index_path)
    
    def _get_device(self) -> str:
        """Get the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def build_index(self, corpus: List[Dict[str, Any]], save_path: str = None):
        """Build FAISS index from corpus.
        
        Args:
            corpus: List of documents with 'text' and 'id' fields
            save_path: Path to save the index
        """
        self.corpus = []
        self.corpus_ids = []
        
        # Extract text and IDs
        texts = []
        for doc in corpus:
            text = doc.get('text', '')
            if text.strip():
                texts.append(text)
                self.corpus.append(text)
                self.corpus_ids.append(doc.get('id', len(self.corpus_ids)))
        
        # Encode texts
        print(f"Encoding {len(texts)} documents...")
        embeddings = self.encoder.encode(texts, show_progress_bar=True, batch_size=32)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings.astype('float32'))
        
        print(f"Built FAISS index with {self.index.ntotal} documents")
        
        # Save index
        if save_path:
            self.save_index(save_path)
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float, int]]:
        """Search for relevant documents.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of (text, score, doc_id) tuples
        """
        if self.index is None:
            raise ValueError("FAISS index not built. Call build_index() first.")
        
        # Encode query
        query_embedding = self.encoder.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.corpus):  # Valid index
                results.append((
                    self.corpus[idx],
                    float(score),
                    self.corpus_ids[idx]
                ))
        
        return results
    
    def save_index(self, save_path: str):
        """Save FAISS index to disk."""
        index_data = {
            'index': self.index,
            'corpus': self.corpus,
            'corpus_ids': self.corpus_ids,
            'model_name': self.model_name
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(index_data, f)
    
    def load_index(self, load_path: str):
        """Load FAISS index from disk."""
        with open(load_path, 'rb') as f:
            index_data = pickle.load(f)
        
        self.index = index_data['index']
        self.corpus = index_data['corpus']
        self.corpus_ids = index_data['corpus_ids']


class RealHybridRetriever:
    """Real hybrid retrieval combining BM25 and dense retrieval."""
    
    def __init__(self, bm25_retriever: RealBM25Retriever, 
                 dense_retriever: RealDenseRetriever,
                 bm25_weight: float = 0.5, dense_weight: float = 0.5):
        """Initialize hybrid retriever.
        
        Args:
            bm25_retriever: BM25 retriever instance
            dense_retriever: Dense retriever instance
            bm25_weight: Weight for BM25 scores
            dense_weight: Weight for dense scores
        """
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float, int]]:
        """Search using hybrid approach.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of (text, score, doc_id) tuples
        """
        # Get results from both retrievers
        bm25_results = self.bm25_retriever.search(query, top_k * 2)
        dense_results = self.dense_retriever.search(query, top_k * 2)
        
        # Create score dictionaries
        bm25_scores = {doc_id: score for _, score, doc_id in bm25_results}
        dense_scores = {doc_id: score for _, score, doc_id in dense_results}
        
        # Get all unique document IDs
        all_doc_ids = set(bm25_scores.keys()) | set(dense_scores.keys())
        
        # Combine scores
        combined_scores = {}
        for doc_id in all_doc_ids:
            bm25_score = bm25_scores.get(doc_id, 0.0)
            dense_score = dense_scores.get(doc_id, 0.0)
            
            # Normalize scores (simple min-max normalization)
            combined_score = (self.bm25_weight * bm25_score + 
                            self.dense_weight * dense_score)
            combined_scores[doc_id] = combined_score
        
        # Get top-k results
        sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Create final results
        results = []
        for doc_id, score in sorted_docs:
            # Find the document text
            text = None
            for _, _, d_id in bm25_results + dense_results:
                if d_id == doc_id:
                    text = next((t for t, _, d_id in bm25_results + dense_results if d_id == doc_id), None)
                    break
            
            if text:
                results.append((text, score, doc_id))
        
        return results


def build_corpus_from_hotpotqa(data_path: str, max_docs: int = None) -> List[Dict[str, Any]]:
    """Build corpus from HotpotQA dataset.
    
    Args:
        data_path: Path to HotpotQA data directory
        max_docs: Maximum number of documents to process
        
    Returns:
        List of documents for indexing
    """
    from refrag.data.hotpotqa import HotpotQADataset
    
    # Load dataset
    dataset = HotpotQADataset(data_path, split="train", max_samples=max_docs)
    
    # Extract documents
    corpus = []
    for i in range(len(dataset)):
        sample = dataset[i]
        
        # Extract context chunks
        from refrag.data.hotpotqa import HotpotQADataset
        chunks = HotpotQADataset.get_context_chunks(sample)
        
        # Add each chunk as a separate document
        for j, chunk in enumerate(chunks):
            corpus.append({
                'text': chunk,
                'id': f"{sample['id']}_{j}",
                'question_id': sample['id'],
                'chunk_index': j,
                'question': sample['question'],
                'answer': sample['answer']
            })
    
    return corpus


def create_retrievers(data_path: str, index_path: str = "data/indexes") -> Tuple[RealBM25Retriever, RealDenseRetriever]:
    """Create and build retrieval indexes.
    
    Args:
        data_path: Path to HotpotQA data
        index_path: Path to save indexes
        
    Returns:
        Tuple of (BM25 retriever, Dense retriever)
    """
    # Create index directory
    Path(index_path).mkdir(parents=True, exist_ok=True)
    
    # Build corpus
    print("Building corpus from HotpotQA...")
    corpus = build_corpus_from_hotpotqa(data_path, max_docs=1000)  # Limit for demo
    print(f"Built corpus with {len(corpus)} documents")
    
    # Create retrievers
    bm25_retriever = RealBM25Retriever()
    dense_retriever = RealDenseRetriever()
    
    # Build indexes
    print("Building BM25 index...")
    bm25_retriever.build_index(corpus, f"{index_path}/bm25_index.pkl")
    
    print("Building dense index...")
    dense_retriever.build_index(corpus, f"{index_path}/dense_index.pkl")
    
    return bm25_retriever, dense_retriever
