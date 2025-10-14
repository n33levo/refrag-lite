.PHONY: setup data data-tiny tiny train rl eval test clean docker-build docker-run

# Python interpreter
PYTHON := python3

# Setup environment
setup:
	@echo "Setting up refrag-lite environment..."
	$(PYTHON) -m venv venv || true
	@echo "Activate with: source venv/bin/activate"
	@echo "Then run: pip install -r requirements.txt"

# Download and index data
data:
	@echo "Downloading HotpotQA dataset..."
	$(PYTHON) data/scripts/download_hotpotqa.py --output data/hotpotqa
	@echo "Building indexes..."
	$(PYTHON) data/scripts/build_corpus.py --bm25 --dense --output data/indexes

# Download tiny dataset for quick testing
data-tiny:
	@echo "Downloading tiny HotpotQA subset..."
	$(PYTHON) data/scripts/download_hotpotqa.py --output data/hotpotqa --tiny --max-samples 500

# Run tiny end-to-end pipeline
tiny:
	@echo "Running tiny pipeline..."
	bash scripts/run_all_tiny.sh

# Run tiny pipeline with robust error handling
tiny-robust:
	@echo "Running tiny pipeline with robust error handling..."
	bash scripts/run_tiny_robust.sh

# Run on Lambda Labs GPU Cloud
lambda-labs:
	@echo "Setting up for Lambda Labs deployment..."
	bash scripts/setup_lambda_labs.sh

# Deploy to Lambda Labs
deploy-lambda:
	@echo "Deploying to Lambda Labs GPU Cloud..."
	bash scripts/deploy_lambda_labs.sh

# Run with Groq API (no local GPU needed)
groq:
	@echo "Running with Groq API..."
	bash scripts/run_groq.sh

# Run full training pipeline
train:
	@echo "Running pretraining..."
	$(PYTHON) -m refrag.train.pretrain_recon --config configs/default.yaml
	$(PYTHON) -m refrag.train.pretrain_cpt --config configs/default.yaml
	@echo "Running SFT..."
	$(PYTHON) -m refrag.train.sft_qa --config configs/default.yaml

# Train RL policy
rl:
	@echo "Training RL policy..."
	$(PYTHON) -m refrag.rl.train_policy --config configs/rl_bandit.yaml

# Run evaluation
eval:
	@echo "Running baselines..."
	bash scripts/run_baselines.sh
	@echo "Running evaluation..."
	$(PYTHON) -m refrag.eval.qa_eval --config configs/eval.yaml
	$(PYTHON) -m refrag.eval.speed_eval --config configs/eval.yaml
	$(PYTHON) -m refrag.eval.report --config configs/eval.yaml

# Run tests
test:
	@echo "Running tests..."
	pytest tests/ -v --cov=refrag --cov-report=html

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -rf outputs/ checkpoints/ logs/ reports/ mlruns/
	rm -rf **/__pycache__ **/*.pyc **/*.pyo **/.pytest_cache
	rm -rf .coverage htmlcov/
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

# Docker commands
docker-build:
	@echo "Building Docker image..."
	docker build -t refrag-lite .

docker-run:
	@echo "Running Docker container..."
	docker run -it --gpus all -v $(PWD)/data:/app/data -v $(PWD)/outputs:/app/outputs refrag-lite

# Help
help:
	@echo "refrag-lite Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  setup       - Create virtual environment"
	@echo "  data        - Download and index full dataset"
	@echo "  data-tiny   - Download tiny dataset for testing"
	@echo "  tiny        - Run tiny end-to-end pipeline"
	@echo "  train       - Run full training pipeline"
	@echo "  rl          - Train RL policy"
	@echo "  eval        - Run evaluation and generate report"
	@echo "  test        - Run unit tests"
	@echo "  clean       - Clean generated files"
	@echo "  docker-build- Build Docker image"
	@echo "  docker-run  - Run Docker container"
	@echo "  help        - Show this help message"
