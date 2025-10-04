#!/usr/bin/env python3
"""
Evaluation script to reproduce FB60K+NYT10 results in the exact same format as RAG-EE.
This script processes ULTRA results and formats them exactly like run_kgc.py output.
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from collections import defaultdict

# Add ULTRA to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ultra import util
from ultra.models import Ultra
from script.run import test


def evaluate_ultra_on_fb60k_nyt10(data_path, checkpoint_path, gpus='[0]', batch_size=8):
    """
    Run ULTRA evaluation on FB60K+NYT10 and return results in RAG-EE format
    """
    
    # Convert string representation of list to actual list
    gpus = eval(gpus) if gpus != 'null' else None
    
    # Create config dictionary
    config = {
        'output_dir': '~/git/ULTRA/output',
        'dataset': {
            'class': 'FB60KNYT10',
            'root': '~/git/ULTRA/kg-datasets/',
            'data_path': data_path
        },
        'model': {
            'class': 'Ultra',
            'relation_model': {
                'class': 'RelNBFNet',
                'input_dim': 64,
                'hidden_dims': [64, 64, 64, 64, 64, 64],
                'message_func': 'distmult',
                'aggregate_func': 'sum',
                'short_cut': True,
                'layer_norm': True
            },
            'entity_model': {
                'class': 'EntityNBFNet',
                'input_dim': 64,
                'hidden_dims': [64, 64, 64, 64, 64, 64],
                'message_func': 'distmult',
                'aggregate_func': 'sum',
                'short_cut': True,
                'layer_norm': True
            }
        },
        'task': {
            'name': 'TransductiveInference',
            'num_negative': 256,
            'strict_negative': True,
            'adversarial_temperature': 1,
            'metric': ['mr', 'mrr', 'hits@1', 'hits@3', 'hits@5', 'hits@10']
        },
        'train': {
            'gpus': None,  # Force CPU usage for Mac compatibility
            'batch_size': batch_size,
            'num_epoch': 0,  # Zero-shot inference
            'log_interval': 100,
            'batch_per_epoch': None
        },
        'checkpoint': os.path.abspath(checkpoint_path)  # Use absolute path
    }
    
    # Convert to EasyDict
    import easydict
    cfg = easydict.EasyDict(config)
    
    # Set up working directory
    working_dir = util.create_working_directory(cfg)
    
    # Set random seed
    torch.manual_seed(1024 + util.get_rank())
    
    # Get logger
    logger = util.get_root_logger()
    if util.get_rank() == 0:
        logger.warning("Random seed: %d" % 1024)
        logger.warning("Running ULTRA evaluation on FB60K+NYT10")
    
    # Build dataset
    dataset = util.build_dataset(cfg)
    device = util.get_device(cfg)
    
    # Load data
    train_data, valid_data, test_data = dataset[0], dataset[1], dataset[2]
    train_data = train_data.to(device)
    valid_data = valid_data.to(device)
    test_data = test_data.to(device)
    
    # Build model
    model = Ultra(
        rel_model_cfg=cfg.model.relation_model,
        entity_model_cfg=cfg.model.entity_model,
    )
    
    # Load checkpoint
    if cfg.checkpoint and os.path.exists(cfg.checkpoint):
        if util.get_rank() == 0:
            logger.warning(f"Loading checkpoint from {cfg.checkpoint}")
        state = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
    else:
        if util.get_rank() == 0:
            logger.warning("No checkpoint found, using random initialization")
    
    model = model.to(device)
    
    # For transductive setting, use the whole graph for filtered ranking
    from torch_geometric.data import Data
    filtered_data = Data(
        edge_index=dataset._data.target_edge_index, 
        edge_type=dataset._data.target_edge_type, 
        num_nodes=dataset[0].num_nodes
    )
    filtered_data = filtered_data.to(device)
    
    # Run evaluation on test set
    test_metrics = test(cfg, model, test_data, filtered_data=filtered_data, 
                       device=device, logger=logger, return_metrics=True)
    
    return test_metrics, dataset


def print_rag_ee_format_results(metrics, dataset, data_path):
    """
    Print results in the exact same format as RAG-EE run_kgc.py
    """
    
    # Calculate total samples (this would be the number of test triplets * 2 for head/tail prediction)
    num_test_triplets = dataset[2].target_edge_index.shape[1]
    total_samples = num_test_triplets * 2  # Both head and tail prediction
    
    print('---------TOTAL-------------')
    if 'hits@1' in metrics:
        print('Final hits@1 out of {} samples: {}'.format(total_samples, metrics['hits@1']))
    if 'hits@3' in metrics:
        print('Final hits@3 out of {} samples: {}'.format(total_samples, metrics['hits@3']))
    if 'hits@5' in metrics:
        print('Final hits@5 out of {} samples: {}'.format(total_samples, metrics['hits@5']))
    if 'hits@10' in metrics:
        print('Final hits@10 out of {} samples: {}'.format(total_samples, metrics['hits@10']))
    if 'mrr' in metrics:
        print('Final MRR: {}'.format(metrics['mrr']))
    
    # Load relation frequency data if available
    relation_freq_path = os.path.join(data_path, 'relation_freq.json')
    if os.path.exists(relation_freq_path):
        with open(relation_freq_path) as json_file:
            relation_freq = json.load(json_file)
        
        print('---------EVALUATING WITH FREQUENCY-------------')
        print('Note: Frequency-based evaluation requires individual triplet results')
        print('ULTRA provides aggregate metrics, so frequency breakdown is not available')
    else:
        print('---------EVALUATING WITH FREQUENCY-------------')
        print('Relation frequency file not found, skipping frequency-based evaluation')
    
    # Note about head/tail breakdown
    print('---------HEAD HITS-------------')
    print('Note: ULTRA provides aggregate head+tail metrics')
    print('Individual head/tail breakdown requires modification of ULTRA evaluation')
    
    print('---------TAIL HITS-------------')
    print('Note: ULTRA provides aggregate head+tail metrics')
    print('Individual head/tail breakdown requires modification of ULTRA evaluation')
    
    print('ULTRA evaluation completed!')
    print("Note: ULTRA uses neural message passing while RAG-EE uses retrieval+generation")


def main():
    parser = argparse.ArgumentParser(description='Evaluate ULTRA on FB60K+NYT10 in RAG-EE format')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to FB60K+NYT10 dataset directory')
    parser.add_argument('--checkpoint', type=str, default='ckpts/ultra_4g.pth',
                        help='Path to ULTRA checkpoint')
    parser.add_argument('--gpus', type=str, default='[0]',
                        help='GPU devices to use, e.g., "[0]" or "[0,1]"')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for inference')
    
    args = parser.parse_args()
    
    print("Starting ULTRA evaluation on FB60K+NYT10...")
    print(f"Data path: {args.data_path}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"GPUs: {args.gpus}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 50)
    
    # Run ULTRA evaluation
    try:
        metrics, dataset = evaluate_ultra_on_fb60k_nyt10(
            args.data_path, 
            args.checkpoint, 
            args.gpus, 
            args.batch_size
        )
        
        # Print results in RAG-EE format
        print_rag_ee_format_results(metrics, dataset, args.data_path)
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())