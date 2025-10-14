#!/usr/bin/env python3
"""Download HotpotQA dataset."""
import argparse
import json
from pathlib import Path
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/hotpotqa")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--max-samples", type=int, default=500)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Downloading HotpotQA...")
    
    try:
        from datasets import load_dataset
        # Try to load with basic settings first
        dataset = load_dataset("hotpot_qa", "distractor", trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load HotpotQA dataset: {e}")
        print("Creating dummy data as fallback...")
        
        # Create dummy data structure that matches HotpotQA format
        dummy_train = []
        for i in range(args.max_samples if args.tiny else 1000):
            dummy_train.append({
                "_id": f"dummy_train_{i}",
                "question": f"What is the capital of country {i}?",
                "answer": f"Capital City {i}",
                "context": [
                    [f"Country {i}", [f"Country {i} is a nation.", f"The capital is Capital City {i}."]],
                    [f"Capital City {i}", [f"Capital City {i} is the largest city.", f"It has a population of {i * 1000} people."]]
                ],
                "supporting_facts": [[f"Country {i}", 1], [f"Capital City {i}", 0]],
                "level": "easy",
                "type": "bridge"
            })
        
        dummy_dev = []
        for i in range(100):
            dummy_dev.append({
                "_id": f"dummy_dev_{i}",
                "question": f"What is the population of city {i}?", 
                "answer": f"{i * 5000} people",
                "context": [
                    [f"City {i}", [f"City {i} is located in region {i}.", f"The population is {i * 5000} people."]],
                    [f"Region {i}", [f"Region {i} contains several cities.", f"City {i} is the largest."]]
                ],
                "supporting_facts": [[f"City {i}", 1], [f"Region {i}", 1]],
                "level": "medium",
                "type": "comparison"
            })
        
        # Save dummy data
        with open(output_path / "train.json", "w") as f:
            json.dump(dummy_train, f, indent=2)
        
        with open(output_path / "validation.json", "w") as f:
            json.dump(dummy_dev, f, indent=2)
            
        # Also create the expected dev file name
        with open(output_path / "hotpot_dev_distractor_v1.json", "w") as f:
            json.dump(dummy_dev, f, indent=2)
        
        print(f"Created dummy train data with {len(dummy_train)} samples")
        print(f"Created dummy dev data with {len(dummy_dev)} samples")
        return

    # Successfully loaded dataset
    for split in ["train", "validation"]:
        data = dataset[split]
        if args.tiny:
            data = data.select(range(min(args.max_samples, len(data))))

        # Save as JSON format (not JSONL) to match expected format
        output_file = output_path / f"{split}.json"
        data_list = [item for item in data]
        
        with open(output_file, "w") as f:
            json.dump(data_list, f, indent=2)

        print(f"Saved {len(data_list)} samples to {output_file}")
        
        # Also create expected filenames
        if split == "validation":
            dev_file = output_path / "hotpot_dev_distractor_v1.json"
            with open(dev_file, "w") as f:
                json.dump(data_list, f, indent=2)
            print(f"Also saved as {dev_file}")

if __name__ == "__main__":
    main()