#!/bin/bash
set -e

echo "Setting up real data and indexes for refrag-lite..."

# Download HotpotQA dataset
echo "Step 1: Downloading HotpotQA dataset..."
python data/scripts/download_hotpotqa.py --output data/hotpotqa --tiny --max-samples 1000

# Build retrieval indexes
echo "Step 2: Building retrieval indexes..."
python data/scripts/build_corpus.py --input data/hotpotqa --output data/indexes --bm25 --dense --max-docs 1000

echo "Real data setup complete!"
echo "Dataset: data/hotpotqa/"
echo "Indexes: data/indexes/"
