#!/bin/bash
set -e

echo "Running refrag-lite full pipeline..."

# Build indexes if not already done
if [ ! -d "data/indexes" ]; then
    echo "Building indexes..."
    python3 data/scripts/build_corpus.py --input data/hotpotqa --output data/indexes --bm25 --dense
fi

# Pretrain reconstruction
echo "Stage 1: Pretraining reconstruction..."
python -m refrag.train.pretrain_recon --config configs/default.yaml

# Pretrain continual pretraining
echo "Stage 2: Pretraining continual pretraining..."
python -m refrag.train.pretrain_cpt --config configs/default.yaml

# SFT training
echo "Stage 3: Supervised fine-tuning..."
python -m refrag.train.sft_qa --config configs/default.yaml

# Train RL policy
echo "Stage 4: Training RL policy..."
python -m refrag.rl.train_policy --config configs/rl_bandit.yaml --algo linucb

# Run evaluation
echo "Stage 5: Running evaluation..."
bash scripts/run_baselines.sh
python -m refrag.eval.qa_eval --config configs/eval.yaml
python -m refrag.eval.speed_eval --config configs/eval.yaml
python -m refrag.eval.report --config configs/eval.yaml

echo "refrag-lite pipeline complete!"
