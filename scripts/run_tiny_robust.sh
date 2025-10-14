#!/bin/bash
# Robust version with error handling and resume capability

set -e

echo "Running tiny pipeline with robust error handling..."

# Function to run a stage with error handling
run_stage() {
    local stage_name="$1"
    local checkpoint_file="$2"
    local command="$3"
    
    if [ -f "$checkpoint_file" ]; then
        echo "$stage_name: Already complete, skipping..."
        return 0
    fi
    
    echo "$stage_name: Starting..."
    if $command; then
        echo "$stage_name: Completed successfully!"
        return 0
    else
        echo "ERROR: $stage_name failed!"
        echo "Checkpoint file: $checkpoint_file"
        echo "You can resume by running this script again, or manually run:"
        echo "  $command"
        exit 1
    fi
}

# Build indexes if not already done
if [ ! -d "data/indexes" ]; then
    echo "Building indexes..."
    python3 data/scripts/build_corpus.py --input data/hotpotqa --output data/indexes --bm25 --dense
else
    echo "Indexes already exist, skipping..."
fi

# Stage 1: Pretrain reconstruction
run_stage "Stage 1: Pretraining reconstruction" \
    "checkpoints/tiny/pretrain_recon.pt" \
    "python -m refrag.train.pretrain_recon --config configs/tiny.yaml"

# Stage 2: Pretrain continual pretraining
run_stage "Stage 2: Pretraining continual pretraining" \
    "checkpoints/tiny/pretrain_cpt.pt" \
    "python -m refrag.train.pretrain_cpt --config configs/tiny.yaml"

# Stage 3: SFT training
run_stage "Stage 3: Supervised fine-tuning" \
    "checkpoints/tiny/sft_qa.pt" \
    "python -m refrag.train.sft_qa --config configs/tiny.yaml"

# Stage 4: Train RL policy
run_stage "Stage 4: Training RL policy" \
    "checkpoints/tiny/rl_policy_linucb.pt" \
    "python -m refrag.rl.train_policy --config configs/tiny.yaml --algo linucb"

# Stage 5: Run evaluation
echo "Stage 5: Running evaluation..."

# Baselines (always run)
echo "Running baselines..."
bash scripts/run_baselines.sh

# QA evaluation
if [ ! -f "outputs/tiny/qa_results.json" ]; then
    echo "Running QA evaluation..."
    python -m refrag.eval.qa_eval --config configs/tiny.yaml
else
    echo "QA evaluation already complete, skipping..."
fi

# Speed evaluation
if [ ! -f "outputs/tiny/speed_results.json" ]; then
    echo "Running speed evaluation..."
    python -m refrag.eval.speed_eval --config configs/tiny.yaml
else
    echo "Speed evaluation already complete, skipping..."
fi

# Report generation
if [ ! -f "outputs/tiny/evaluation_report.md" ]; then
    echo "Generating report..."
    python -m refrag.eval.report --config configs/tiny.yaml
else
    echo "Report already generated, skipping..."
fi

echo "Tiny pipeline complete!"
echo ""
echo "Checkpoints saved:"
ls -la checkpoints/tiny/
echo ""
echo "Outputs generated:"
ls -la outputs/tiny/
