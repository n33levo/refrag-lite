#!/bin/bash
set -e

echo "Running tiny pipeline with checkpoint resuming..."

# Build indexes if not already done
if [ ! -d "data/indexes" ]; then
    echo "Building indexes..."
    python3 data/scripts/build_corpus.py --input data/hotpotqa --output data/indexes --bm25 --dense
else
    echo "Indexes already exist, skipping..."
fi

# Stage 1: Pretrain reconstruction
if [ ! -f "checkpoints/tiny/pretrain_recon.pt" ]; then
    echo "Stage 1: Pretraining reconstruction..."
    python -m refrag.train.pretrain_recon --config configs/tiny.yaml
else
    echo "Stage 1: Already complete, skipping..."
fi

# Stage 2: Pretrain continual pretraining
if [ ! -f "checkpoints/tiny/pretrain_cpt.pt" ]; then
    echo "Stage 2: Pretraining continual pretraining..."
    python -m refrag.train.pretrain_cpt --config configs/tiny.yaml
else
    echo "Stage 2: Already complete, skipping..."
fi

# Stage 3: SFT training
if [ ! -f "checkpoints/tiny/sft_qa.pt" ]; then
    echo "Stage 3: Supervised fine-tuning..."
    python -m refrag.train.sft_qa --config configs/tiny.yaml
else
    echo "Stage 3: Already complete, skipping..."
fi

# Stage 4: Train RL policy
if [ ! -f "checkpoints/tiny/rl_policy_linucb.pt" ]; then
    echo "Stage 4: Training RL policy..."
    python -m refrag.rl.train_policy --config configs/tiny.yaml --algo linucb
else
    echo "Stage 4: Already complete, skipping..."
fi

# Stage 5: Run evaluation
echo "Stage 5: Running evaluation..."
bash scripts/run_baselines.sh

if [ ! -f "outputs/tiny/qa_results.json" ]; then
    echo "Running QA evaluation..."
    python -m refrag.eval.qa_eval --config configs/tiny.yaml
else
    echo "QA evaluation already complete, skipping..."
fi

if [ ! -f "outputs/tiny/speed_results.json" ]; then
    echo "Running speed evaluation..."
    python -m refrag.eval.speed_eval --config configs/tiny.yaml
else
    echo "Speed evaluation already complete, skipping..."
fi

if [ ! -f "outputs/tiny/evaluation_report.md" ]; then
    echo "Generating report..."
    python -m refrag.eval.report --config configs/tiny.yaml
else
    echo "Report already generated, skipping..."
fi

echo "Tiny pipeline complete!"