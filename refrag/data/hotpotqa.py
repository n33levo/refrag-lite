"""HotpotQA dataset loading and processing."""
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


class HotpotQADataset(Dataset):
    """HotpotQA dataset for training and evaluation."""
    
    def __init__(self, data_path: str, split: str = "train", max_samples: int = None):
        """Initialize HotpotQA dataset.
        
        Args:
            data_path: Path to HotpotQA data directory
            split: Dataset split ("train", "dev", "test")
            max_samples: Maximum number of samples to load (None for all)
        """
        self.data_path = Path(data_path)
        self.split = split
        self.max_samples = max_samples
        
        # Load data
        self.data = self._load_data()
        
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load HotpotQA data from JSON files."""
        # Try multiple possible file names
        possible_files = []
        
        if self.split == "train":
            possible_files = [
                "hotpot_train_v1.1.json",
                "train.json",
                "hotpot_train_v1.json"
            ]
        elif self.split == "dev":
            possible_files = [
                "hotpot_dev_distractor_v1.json",
                "validation.json",
                "dev.json"
            ]
        else:
            possible_files = [
                "hotpot_test_v1.json",
                "test.json"
            ]
        
        file_path = None
        for filename in possible_files:
            candidate_path = self.data_path / filename
            if candidate_path.exists():
                file_path = candidate_path
                break
                
        if not file_path:
            raise FileNotFoundError(f"HotpotQA {self.split} file not found. Tried: {possible_files}")
            
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        if self.max_samples:
            data = data[:self.max_samples]
            
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample from the dataset."""
        sample = self.data[idx]
        
        return {
            "question": sample["question"],
            "answer": sample["answer"],
            "context": sample.get("context", []),
            "supporting_facts": sample.get("supporting_facts", []),
            "level": sample.get("level", "medium"),
            "type": sample.get("type", "comparison"),
            "id": sample.get("_id", str(idx))
        }
    
    @staticmethod
    def get_context_chunks(sample: Dict[str, Any], chunk_size: int = 256) -> List[str]:
        """Extract context chunks from supporting facts and context."""
        chunks = []
        
        # Get supporting facts and context
        supporting_facts = sample.get("supporting_facts", {})
        context = sample.get("context", {})
        
        # Handle HotpotQA format: {"title": [...], "sentences": [[...], [...]]}
        if isinstance(context, dict) and "title" in context and "sentences" in context:
            titles = context["title"]
            sentences_list = context["sentences"]
            
            # Create context mapping: title -> sentences
            context_map = {title: sentences for title, sentences in zip(titles, sentences_list)}
            
            # Extract chunks from supporting facts
            if isinstance(supporting_facts, dict) and "title" in supporting_facts and "sent_id" in supporting_facts:
                fact_titles = supporting_facts["title"]
                fact_sent_ids = supporting_facts["sent_id"]
                
                for title, sent_idx in zip(fact_titles, fact_sent_ids):
                    if title in context_map:
                        sentences = context_map[title]
                        if sent_idx < len(sentences):
                            chunk = sentences[sent_idx]
                            if len(chunk) > chunk_size:
                                # Split long chunks
                                words = chunk.split()
                                for i in range(0, len(words), chunk_size // 4):
                                    chunk_text = " ".join(words[i:i + chunk_size // 4])
                                    if chunk_text.strip():
                                        chunks.append(chunk_text)
                            else:
                                chunks.append(chunk)
            
            # Add some additional context chunks if we have few supporting facts
            if len(chunks) < 3:
                for title, sentences in context_map.items():
                    for sent in sentences[:2]:  # Take first 2 sentences from each context
                        if sent not in chunks and len(sent) > 50:
                            chunks.append(sent)
                            if len(chunks) >= 5:  # Limit to 5 chunks
                                break
                    if len(chunks) >= 5:
                        break
        
        # Fallback: handle old format
        elif isinstance(context, list):
            context_map = {item[0]: item[1] for item in context}
            
            # Extract chunks from supporting facts
            for fact in supporting_facts:
                title, sent_idx = fact
                if title in context_map:
                    sentences = context_map[title]
                    if sent_idx < len(sentences):
                        chunk = sentences[sent_idx]
                        if len(chunk) > chunk_size:
                            # Split long chunks
                            words = chunk.split()
                            for i in range(0, len(words), chunk_size // 4):
                                chunk_text = " ".join(words[i:i + chunk_size // 4])
                                if chunk_text.strip():
                                    chunks.append(chunk_text)
                        else:
                            chunks.append(chunk)
            
            # Add some additional context chunks if we have few supporting facts
            if len(chunks) < 3:
                for title, sentences in context_map.items():
                    for sent in sentences[:2]:  # Take first 2 sentences from each context
                        if sent not in chunks and len(sent) > 50:
                            chunks.append(sent)
                            if len(chunks) >= 5:  # Limit to 5 chunks
                                break
                    if len(chunks) >= 5:
                        break
        
        return chunks[:5]  # Return max 5 chunks


def create_dataloader(data_path: str, split: str = "train", batch_size: int = 4, 
                     max_samples: int = None, num_workers: int = 2) -> DataLoader:
    """Create a DataLoader for HotpotQA dataset.
    
    Args:
        data_path: Path to HotpotQA data directory
        split: Dataset split ("train", "dev", "test")
        batch_size: Batch size for DataLoader
        max_samples: Maximum number of samples to load
        num_workers: Number of worker processes
        
    Returns:
        DataLoader instance
    """
    dataset = HotpotQADataset(data_path, split, max_samples)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=collate_fn
    )


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for batching samples."""
    return {
        "questions": [sample["question"] for sample in batch],
        "answers": [sample["answer"] for sample in batch],
        "contexts": [sample["context"] for sample in batch],
        "supporting_facts": [sample["supporting_facts"] for sample in batch],
        "levels": [sample["level"] for sample in batch],
        "types": [sample["type"] for sample in batch],
        "ids": [sample["id"] for sample in batch]
    }


def load_hotpotqa_samples(data_path: str, split: str = "dev", num_samples: int = 100) -> List[Dict[str, Any]]:
    """Load a subset of HotpotQA samples for evaluation.
    
    Args:
        data_path: Path to HotpotQA data directory
        split: Dataset split ("dev" or "test")
        num_samples: Number of samples to load
        
    Returns:
        List of sample dictionaries
    """
    dataset = HotpotQADataset(data_path, split, num_samples)
    return [dataset[i] for i in range(len(dataset))]
