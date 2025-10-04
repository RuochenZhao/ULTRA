# Reproducing FB60K+NYT10 Results with ULTRA

This guide explains how to reproduce FB60K+NYT10 knowledge graph completion results using the ULTRA framework, similar to the results obtained with RAG-EE.

## Overview

- **RAG-EE**: Uses retrieval-augmented generation for knowledge graph completion
- **ULTRA**: Uses neural message passing on knowledge graph structure with pre-trained foundation models
- **Goal**: Compare the two approaches on the same FB60K+NYT10 dataset

## Prerequisites

1. **ULTRA Environment**: Follow the installation instructions in the main README.md
2. **FB60K+NYT10 Dataset**: Ensure you have the RAG-EE dataset directory with the following structure:
   ```
   RAG-EE/dataset/FB60K+NYT10/
   ├── kg/
   │   ├── train.txt
   │   ├── valid.txt
   │   ├── test.txt
   │   └── vocab/
   ├── entity2label.txt
   ├── relations.txt
   └── relation_freq.json
   ```

## Files Added for FB60K+NYT10 Support

1. **`ultra/fb60k_nyt10_dataset.py`**: Custom dataset class for FB60K+NYT10
2. **`config/transductive/fb60k_nyt10_inference.yaml`**: Configuration file for experiments
3. **`run_fb60k_nyt10.py`**: Main script to run ULTRA inference
4. **`evaluate_fb60k_nyt10.py`**: Evaluation script that matches RAG-EE output format

## Quick Start

### Method 1: Using the Evaluation Script (Recommended)

```bash
cd ULTRA
python evaluate_fb60k_nyt10.py \
    --data_path ~/Documents/RAG-EE/dataset/FB60K+NYT10 \
    --checkpoint ckpts/ultra_4g.pth \
    --gpus "null" \
    --batch_size 8
```

**Note for Mac users**: The script automatically uses CPU for compatibility with ULTRA's custom operations. MPS (Apple Silicon GPU) is not supported by ULTRA's rspmm kernel.

### Method 2: Using the Configuration File

```bash
cd ULTRA
python script/run.py \
    -c config/transductive/fb60k_nyt10_inference.yaml \
    --dataset FB60KNYT10 \
    --epochs 0 \
    --bpe null \
    --gpus "[0]" \
    --ckpt $(pwd)/ckpts/ultra_4g.pth
```

### Method 3: Using the Standalone Script

```bash
cd ULTRA
python run_fb60k_nyt10.py \
    --data_path ~/Documents/RAG-EE/dataset/FB60K+NYT10 \
    --checkpoint ckpts/ultra_4g.pth \
    --gpus "[0]" \
    --batch_size 8
```

## Expected Output Format

The evaluation will produce output in the same format as RAG-EE's `run_kgc.py`:

```
---------TOTAL-------------
Final hits@1 out of XXXX samples: X.XXXX
Final hits@3 out of XXXX samples: X.XXXX
Final hits@5 out of XXXX samples: X.XXXX
Final hits@10 out of XXXX samples: X.XXXX
Final MRR: X.XXXX

---------EVALUATING WITH FREQUENCY-------------
[Frequency-based evaluation if relation_freq.json is available]

---------HEAD HITS-------------
[Head prediction results]

---------TAIL HITS-------------
[Tail prediction results]
```

## Checkpoint Options

Choose the appropriate pre-trained checkpoint:

- **`ultra_3g.pth`**: Trained on FB15k237, WN18RR, CoDExMedium (800K steps)
- **`ultra_4g.pth`**: Trained on FB15k237, WN18RR, CoDExMedium, NELL995 (400K steps) - **Recommended**
- **`ultra_50g.pth`**: Trained on 50 graphs (best for larger graphs)

## Key Differences: ULTRA vs RAG-EE

| Aspect | ULTRA | RAG-EE |
|--------|-------|--------|
| **Approach** | Neural message passing on KG structure | Retrieval + text generation |
| **Knowledge Source** | Pre-trained on multiple KGs | External text retrieval |
| **Entity Handling** | Relative representations, no entity embeddings | Can handle new entities via text |
| **Computational Cost** | Lower inference cost | Higher due to retrieval |
| **Strengths** | Structural reasoning, transferability | External knowledge, new entities |

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure ULTRA is in your Python path
   ```bash
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   ```

2. **CUDA Memory Error**: Reduce batch size
   ```bash
   python evaluate_fb60k_nyt10.py --batch_size 4 [other args]
   ```

3. **Dataset Path Error**: Ensure the data_path points to the FB60K+NYT10 directory
   ```bash
   ls ~/Documents/RAG-EE/dataset/FB60K+NYT10/kg/
   # Should show: train.txt valid.txt test.txt vocab/
   ```

### GPU Configuration

- **Single GPU**: `--gpus "[0]"`
- **Multiple GPUs**: `--gpus "[0,1]"`
- **CPU only**: `--gpus "null"`

## Understanding the Results

### Metrics Explanation

- **Hits@K**: Percentage of correct entities in top-K predictions
- **MRR**: Mean Reciprocal Rank (higher is better)
- **MR**: Mean Rank (lower is better)

### Comparison Guidelines

When comparing ULTRA vs RAG-EE results:

1. **Direct Metrics**: Compare Hits@1, Hits@3, Hits@10, and MRR
2. **Approach Differences**: Consider that ULTRA uses structural reasoning while RAG-EE uses textual knowledge
3. **Complementary Strengths**: ULTRA may excel on structural patterns, RAG-EE on entities with rich textual descriptions

## Advanced Usage

### Custom Evaluation

To modify the evaluation for specific analysis:

```python
# Example: Focus on specific relation types
from ultra.fb60k_nyt10_dataset import FB60KNYT10
dataset = FB60KNYT10(root="kg-datasets/", data_path="path/to/FB60K+NYT10")
# Access vocabularies and filter specific relations
```

### Batch Processing

For processing multiple checkpoints:

```bash
for ckpt in ultra_3g.pth ultra_4g.pth ultra_50g.pth; do
    echo "Evaluating $ckpt"
    python evaluate_fb60k_nyt10.py \
        --checkpoint ckpts/$ckpt \
        --data_path ~/Documents/RAG-EE/dataset/FB60K+NYT10
done
```

## Citation

If you use this reproduction setup, please cite both:

```bibtex
@inproceedings{galkin2023ultra,
    title={Towards Foundation Models for Knowledge Graph Reasoning},
    author={Mikhail Galkin and Xinyu Yuan and Hesham Mostafa and Jian Tang and Zhaocheng Zhu},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024}
}
```

And the original RAG-EE work for the FB60K+NYT10 dataset.