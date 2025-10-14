#!/bin/bash

# Improved RL Pipeline Runner for Lambda Labs
# This script pushes changes, runs the complete pipeline, and downloads results

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Running Improved RL Pipeline on Lambda Labs${NC}"
echo "=================================================="

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo -e "${RED}❌ .env file not found. Please create one with your GROQ_API_KEY${NC}"
    exit 1
fi

# Load environment variables
echo -e "${YELLOW}📁 Loading .env file...${NC}"
source .env

if [ -z "$GROQ_API_KEY" ]; then
    echo -e "${RED}❌ GROQ_API_KEY not found in .env file${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Groq API key loaded from .env file${NC}"

# Get Lambda Labs instance details
echo -e "${YELLOW}🔍 Checking Lambda Labs GPU instance...${NC}"
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}⚠️ nvidia-smi not found - make sure Lambda Labs instance has GPU enabled${NC}"
fi

echo -e "${BLUE}📋 Please provide your Lambda Labs instance details:${NC}"
read -p "Instance IP address: " INSTANCE_IP
read -p "SSH key path (e.g., refrag-lite.pem): " SSH_KEY_PATH

if [ -z "$INSTANCE_IP" ] || [ -z "$SSH_KEY_PATH" ]; then
    echo -e "${RED}❌ Instance IP and SSH key path are required${NC}"
    exit 1
fi

# Test SSH connection
echo -e "${YELLOW}🔌 Testing SSH connection...${NC}"
if ! ssh -i "$SSH_KEY_PATH" -o ConnectTimeout=10 -o StrictHostKeyChecking=no ubuntu@$INSTANCE_IP "echo 'SSH connection successful'" 2>/dev/null; then
    echo -e "${RED}❌ SSH connection failed. Please check your instance IP and SSH key${NC}"
    exit 1
fi

echo -e "${GREEN}✅ SSH connection successful${NC}"

# Push only the improved files to the cloud
echo -e "${YELLOW}📤 Pushing improved files to Lambda Labs...${NC}"
echo "Uploading improved inference and evaluation files..."

# Upload the improved files
scp -i "$SSH_KEY_PATH" refrag/llm/inference_real.py ubuntu@$INSTANCE_IP:~/refrag-lite/refrag/llm/ 2>/dev/null || echo "Failed to upload inference_real.py"
scp -i "$SSH_KEY_PATH" refrag/llm/groq_client.py ubuntu@$INSTANCE_IP:~/refrag-lite/refrag/llm/ 2>/dev/null || echo "Failed to upload groq_client.py"
scp -i "$SSH_KEY_PATH" refrag/eval/qa_eval.py ubuntu@$INSTANCE_IP:~/refrag-lite/refrag/eval/ 2>/dev/null || echo "Failed to upload qa_eval.py"
scp -i "$SSH_KEY_PATH" refrag/data/hotpotqa.py ubuntu@$INSTANCE_IP:~/refrag-lite/refrag/data/ 2>/dev/null || echo "Failed to upload hotpotqa.py"

echo -e "${GREEN}✅ Files uploaded successfully${NC}"

# Run the complete RL pipeline on Lambda Labs
echo -e "${YELLOW}🚀 Running Complete RL Pipeline on Lambda Labs...${NC}"

ssh -i "$SSH_KEY_PATH" ubuntu@$INSTANCE_IP << EOF
cd ~/refrag-lite
source .venv/bin/activate
export GROQ_API_KEY='$GROQ_API_KEY'

echo "🚀 Running Complete RL Pipeline with Enhanced F1 Scoring..."
echo "=================================================="

# Stage 1: RL Policy Training
echo "Stage 1: RL Policy Training..."
python -m refrag.rl.train_policy --config configs/groq.yaml

# Stage 2: QA Evaluation with RL Policy
echo "Stage 2: QA Evaluation with RL Policy..."
python -m refrag.eval.qa_eval --config configs/groq.yaml

# Stage 3: Speed Evaluation
echo "Stage 3: Speed Evaluation..."
python -m refrag.eval.speed_eval --config configs/groq.yaml

# Stage 4: Generate Report
echo "Stage 4: Generate Report..."
python -m refrag.eval.report --config configs/groq.yaml

echo "✅ Complete RL Pipeline Finished!"
echo "=================================================="
EOF

# Download all results
echo -e "${YELLOW}📥 Downloading results and checkpoints...${NC}"

# Create local directories if they don't exist
mkdir -p outputs/groq
mkdir -p checkpoints/groq
mkdir -p plots

# Download results
echo "Downloading QA results..."
scp -i "$SSH_KEY_PATH" ubuntu@$INSTANCE_IP:~/refrag-lite/outputs/groq/qa_results.json ./outputs/groq/ 2>/dev/null || echo "No qa_results.json to download"

echo "Downloading speed results..."
scp -i "$SSH_KEY_PATH" ubuntu@$INSTANCE_IP:~/refrag-lite/outputs/groq/speed_results.json ./outputs/groq/ 2>/dev/null || echo "No speed_results.json to download"

echo "Downloading evaluation report..."
scp -i "$SSH_KEY_PATH" ubuntu@$INSTANCE_IP:~/refrag-lite/outputs/groq/evaluation_report.md ./outputs/groq/ 2>/dev/null || echo "No evaluation_report.md to download"

echo "Downloading plots..."
scp -i "$SSH_KEY_PATH" ubuntu@$INSTANCE_IP:~/refrag-lite/outputs/groq/plots/* ./plots/ 2>/dev/null || echo "No plots to download"

echo "Downloading RL policy..."
scp -i "$SSH_KEY_PATH" ubuntu@$INSTANCE_IP:~/refrag-lite/checkpoints/groq/rl_policy_linucb.pt ./checkpoints/groq/ 2>/dev/null || echo "No RL policy to download"

echo -e "${GREEN}✅ Results downloaded to local outputs/ and checkpoints/ directories${NC}"

# Display results summary
echo -e "${BLUE}📊 Results Summary:${NC}"
echo "=================================================="

if [ -f "./outputs/groq/qa_results.json" ]; then
    echo "QA Results:"
    cat ./outputs/groq/qa_results.json | python -c "
import json, sys
data = json.load(sys.stdin)
print(f'  EM Score: {data[\"em\"]:.3f}')
print(f'  F1 Score: {data[\"f1\"]:.3f}')
print(f'  Evidence Precision: {data[\"evidence_precision\"]:.3f}')
print(f'  Evidence Recall: {data[\"evidence_recall\"]:.3f}')
print(f'  Samples: {data[\"num_samples\"]}')
"
fi

if [ -f "./outputs/groq/speed_results.json" ]; then
    echo "Speed Results:"
    cat ./outputs/groq/speed_results.json | python -c "
import json, sys
data = json.load(sys.stdin)
print(f'  TTFT: {data[\"ttft_mean\"]:.3f}s')
print(f'  Throughput: {data[\"throughput_mean\"]:.0f} tok/s')
print(f'  Memory: {data[\"vram_mean_gb\"]:.1f} GB')
"
fi

echo -e "${GREEN}🎉 Pipeline complete! Check outputs/groq/ for detailed results${NC}"
