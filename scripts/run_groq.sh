#!/bin/bash
set -e

echo "Running refrag-lite with Groq API..."

# Check if GROQ_API_KEY is set
if [ -z "$GROQ_API_KEY" ]; then
    echo "ERROR: GROQ_API_KEY environment variable not set"
    echo "Please set it with: export GROQ_API_KEY='your_api_key_here'"
    echo "Get your API key from: https://console.groq.com/"
    exit 1
fi

echo "Using Groq API for LLM inference..."

# Stage 1: Download data and build indexes
echo "Stage 1: Setting up data and indexes..."
if [ ! -d "data/hotpotqa" ]; then
    echo "Downloading HotpotQA dataset..."
    python data/scripts/download_hotpotqa.py --output data/hotpotqa --tiny --max-samples 1000
fi

if [ ! -d "data/indexes" ]; then
    echo "Building retrieval indexes..."
    python data/scripts/build_corpus.py --input data/hotpotqa --output data/indexes --bm25 --dense --max-docs 1000
fi

# Stage 2: Pretrain reconstruction (local, no LLM needed)
echo "Stage 2: Pretraining reconstruction..."
if [ ! -f "checkpoints/groq/pretrain_recon.pt" ]; then
    python -m refrag.train.pretrain_recon --config configs/groq.yaml
else
    echo "Reconstruction pretraining already complete, skipping..."
fi

# Stage 3: Pretrain continual pretraining (local, no LLM needed)
echo "Stage 3: Pretraining continual pretraining..."
if [ ! -f "checkpoints/groq/pretrain_cpt.pt" ]; then
    python -m refrag.train.pretrain_cpt --config configs/groq.yaml
else
    echo "Continual pretraining already complete, skipping..."
fi

# Stage 4: Supervised fine-tuning (local, no LLM needed)
echo "Stage 4: Supervised fine-tuning..."
if [ ! -f "checkpoints/groq/sft_qa.pt" ]; then
    python -m refrag.train.sft_qa --config configs/groq.yaml
else
    echo "SFT already complete, skipping..."
fi

# Stage 5: Train RL policy (uses Groq API)
echo "Stage 5: Training RL policy with Groq API..."
if [ ! -f "checkpoints/groq/rl_policy_linucb.pt" ]; then
    python -m refrag.rl.train_policy --config configs/groq.yaml --algo linucb
else
    echo "RL policy training already complete, skipping..."
fi

# Stage 6: Run baselines (uses Groq API)
echo "Stage 6: Running baselines with Groq API..."
bash scripts/run_baselines.sh

# Stage 7: Evaluation (uses Groq API)
echo "Stage 7: Running evaluation with Groq API..."
if [ ! -f "outputs/groq/qa_results.json" ]; then
    python -m refrag.eval.qa_eval --config configs/groq.yaml
else
    echo "QA evaluation already complete, skipping..."
fi

if [ ! -f "outputs/groq/speed_results.json" ]; then
    python -m refrag.eval.speed_eval --config configs/groq.yaml
else
    echo "Speed evaluation already complete, skipping..."
fi

# Stage 8: Generate report
echo "Stage 8: Generating report..."
python -m refrag.eval.report --config configs/groq.yaml

echo "Groq pipeline complete!"
echo "Results saved to outputs/groq/"
echo "Checkpoints saved to checkpoints/groq/"
