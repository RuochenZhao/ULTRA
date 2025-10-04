#!/usr/bin/env python3
"""
Script to run ULTRA inference on FB60K+NYT10 dataset.
This reproduces the knowledge graph completion results similar to RAG-EE.
"""

import os
import sys
import argparse
import torch

# Add ULTRA to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ultra import util
from ultra.models import Ultra
from script.run import test


def main():
    parser = argparse.ArgumentParser(description='Run ULTRA inference on FB60K+NYT10')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to FB60K+NYT10 dataset directory')
    parser.add_argument('--checkpoint', type=str, default='ckpts/ultra_4g.pth',
                        help='Path to ULTRA checkpoint')
    parser.add_argument('--gpus', type=str, default='[0]',
                        help='GPU devices to use, e.g., "[0]" or "[0,1]"')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for inference')
    parser.add_argument('--output_dir', type=str, default='~/git/ULTRA/output',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Convert string representation of list to actual list
    gpus = eval(args.gpus) if args.gpus != 'null' else None
    
    # Create config dictionary
    config = {
        'output_dir': args.output_dir,
        'dataset': {
            'class': 'FB60KNYT10',
            'root': '~/git/ULTRA/kg-datasets/',
            'data_path': args.data_path
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
            'metric': ['mr', 'mrr', 'hits@1', 'hits@3', 'hits@10']
        },
        'train': {
            'gpus': gpus,
            'batch_size': args.batch_size,
            'num_epoch': 0,  # Zero-shot inference
            'log_interval': 100,
            'batch_per_epoch': None
        },
        'checkpoint': args.checkpoint
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
        logger.warning("Config: %s" % cfg)
    
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
    
    # Run evaluation
    if util.get_rank() == 0:
        logger.warning("=" * 30)
        logger.warning("Evaluate on validation set")
    valid_metrics = test(cfg, model, valid_data, filtered_data=filtered_data, 
                        device=device, logger=logger, return_metrics=True)
    
    if util.get_rank() == 0:
        logger.warning("=" * 30)
        logger.warning("Evaluate on test set")
    test_metrics = test(cfg, model, test_data, filtered_data=filtered_data, 
                       device=device, logger=logger, return_metrics=True)
    
    # Print results in RAG-EE format
    if util.get_rank() == 0:
        logger.warning("=" * 50)
        logger.warning("FINAL RESULTS (RAG-EE format)")
        logger.warning("=" * 50)
        
        if test_metrics:
            logger.warning("TEST SET RESULTS:")
            if 'hits@1' in test_metrics:
                logger.warning(f"Final hits@1: {test_metrics['hits@1']:.4f}")
            if 'hits@3' in test_metrics:
                logger.warning(f"Final hits@3: {test_metrics['hits@3']:.4f}")
            if 'hits@10' in test_metrics:
                logger.warning(f"Final hits@10: {test_metrics['hits@10']:.4f}")
            if 'mrr' in test_metrics:
                logger.warning(f"Final MRR: {test_metrics['mrr']:.4f}")
            if 'mr' in test_metrics:
                logger.warning(f"Final MR: {test_metrics['mr']:.4f}")
        
        logger.warning("=" * 50)
        logger.warning("Comparison with RAG-EE:")
        logger.warning("RAG-EE uses retrieval-augmented generation for KG completion")
        logger.warning("ULTRA uses neural message passing on the KG structure")
        logger.warning("Both approaches aim to predict missing entities in triplets")
        logger.warning("=" * 50)


if __name__ == "__main__":
    main()