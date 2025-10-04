#!/bin/bash

# FB60K+NYT10 Reproduction Script for ULTRA
# This script reproduces FB60K+NYT10 results using ULTRA framework

echo "=========================================="
echo "FB60K+NYT10 Reproduction with ULTRA"
echo "=========================================="

# Configuration
DATA_PATH="${1:-$HOME/Documents/RAG-EE/dataset/FB60K+NYT10}"
CHECKPOINT="${2:-ckpts/ultra_4g.pth}"
GPUS="${3:-null}"  # Default to CPU for Mac compatibility
BATCH_SIZE="${4:-8}"

# Expand tilde if present
DATA_PATH="${DATA_PATH/#\~/$HOME}"

# Convert checkpoint to absolute path
CHECKPOINT="$(pwd)/$CHECKPOINT"

echo "Configuration:"
echo "  Data Path: $DATA_PATH"
echo "  Checkpoint: $CHECKPOINT"
echo "  GPUs: $GPUS"
echo "  Batch Size: $BATCH_SIZE"
echo ""

# Check if running on Mac and warn about CPU usage
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Note: Running on macOS - using CPU for compatibility with ULTRA's rspmm kernel"
    echo "MPS (Apple Silicon GPU) is not supported by ULTRA's custom operations"
    echo ""
fi

# Check if data path exists
if [ ! -d "$DATA_PATH" ]; then
    echo "Error: Data path $DATA_PATH does not exist"
    echo "Please provide the correct path to FB60K+NYT10 dataset"
    echo "Usage: $0 [data_path] [checkpoint] [gpus] [batch_size]"
    exit 1
fi

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint $CHECKPOINT does not exist"
    echo "Available checkpoints:"
    ls -la ckpts/*.pth 2>/dev/null || echo "  No checkpoints found in ckpts/"
    exit 1
fi

echo "Starting ULTRA evaluation on FB60K+NYT10..."
echo "This will reproduce results in the same format as RAG-EE run_kgc.py"
echo ""

# Run the evaluation
python evaluate_fb60k_nyt10.py \
    --data_path "$DATA_PATH" \
    --checkpoint "$CHECKPOINT" \
    --gpus "$GPUS" \
    --batch_size "$BATCH_SIZE"

echo ""
echo "=========================================="
echo "Reproduction completed!"
echo ""
echo "To compare with RAG-EE results, run:"
echo "cd ../RAG-EE && python run_kgc.py --dataset FB60K+NYT10"
echo ""
echo "Key differences:"
echo "- ULTRA: Neural message passing on KG structure"
echo "- RAG-EE: Retrieval + text generation"
echo "- Both: Predict missing entities in knowledge graph triplets"
echo "=========================================="