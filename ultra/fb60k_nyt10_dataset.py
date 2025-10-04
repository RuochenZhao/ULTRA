import os
import torch
from torch_geometric.data import Data, InMemoryDataset, download_url
from ultra.tasks import build_relation_graph


class FB60KNYT10(InMemoryDataset):
    """
    FB60K+NYT10 dataset for ULTRA framework.
    This dataset is used in RAG-EE for knowledge graph completion.
    """
    
    name = "FB60K+NYT10"
    delimiter = "\t"
    
    def __init__(self, root, data_path=None, transform=None, pre_transform=build_relation_graph, **kwargs):
        """
        Args:
            root: Root directory for storing processed data
            data_path: Path to the FB60K+NYT10 dataset (should contain kg/ folder)
        """
        self.data_path = data_path
        if data_path is None:
            raise ValueError("data_path must be provided pointing to FB60K+NYT10 dataset directory")
        
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return ["train.txt", "valid.txt", "test.txt"]
    
    @property
    def processed_file_names(self):
        return "data.pt"
    
    @property
    def raw_dir(self):
        return os.path.join(self.root, "FB60K+NYT10", "raw")
    
    @property
    def processed_dir(self):
        return os.path.join(self.root, "FB60K+NYT10", "processed")
    
    @property
    def num_relations(self):
        return int(self.data.edge_type.max()) + 1

    def download(self):
        """Copy files from the provided data_path to raw_dir"""
        import shutil
        os.makedirs(self.raw_dir, exist_ok=True)
        
        # Copy the kg files
        kg_path = os.path.join(self.data_path, "kg")
        for filename in self.raw_file_names:
            src = os.path.join(kg_path, filename)
            dst = os.path.join(self.raw_dir, filename)
            if os.path.exists(src):
                shutil.copy2(src, dst)
            else:
                raise FileNotFoundError(f"Required file {src} not found")
    
    def load_file(self, triplet_file, inv_entity_vocab={}, inv_rel_vocab={}):
        """Load triplets from file and build vocabularies"""
        triplets = []
        entity_cnt, rel_cnt = len(inv_entity_vocab), len(inv_rel_vocab)

        with open(triplet_file, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split(self.delimiter)
                if len(parts) != 3:
                    continue
                    
                h_token, r_token, t_token = parts
                
                # Build entity vocabulary
                if h_token not in inv_entity_vocab:
                    inv_entity_vocab[h_token] = entity_cnt
                    entity_cnt += 1
                if t_token not in inv_entity_vocab:
                    inv_entity_vocab[t_token] = entity_cnt
                    entity_cnt += 1
                
                # Build relation vocabulary
                if r_token not in inv_rel_vocab:
                    inv_rel_vocab[r_token] = rel_cnt
                    rel_cnt += 1
                
                h = inv_entity_vocab[h_token]
                r = inv_rel_vocab[r_token]
                t = inv_entity_vocab[t_token]
                
                triplets.append((h, t, r))

        return {
            "triplets": triplets,
            "num_node": len(inv_entity_vocab),
            "num_relation": rel_cnt,
            "inv_entity_vocab": inv_entity_vocab,
            "inv_rel_vocab": inv_rel_vocab
        }
    
    def process(self):
        """Process the raw files and create PyG Data objects"""
        train_files = [os.path.join(self.raw_dir, f) for f in self.raw_file_names]

        # Load all files and build unified vocabularies
        train_results = self.load_file(train_files[0], inv_entity_vocab={}, inv_rel_vocab={})
        valid_results = self.load_file(train_files[1], 
                        train_results["inv_entity_vocab"], train_results["inv_rel_vocab"])
        test_results = self.load_file(train_files[2],
                        train_results["inv_entity_vocab"], train_results["inv_rel_vocab"])
        
        # Use the final vocabularies (may include new entities/relations from valid/test)
        num_node = test_results["num_node"] 
        num_relations = test_results["num_relation"]

        train_triplets = train_results["triplets"]
        valid_triplets = valid_results["triplets"]
        test_triplets = test_results["triplets"]

        # Convert to tensors
        train_target_edges = torch.tensor([[t[0], t[1]] for t in train_triplets], dtype=torch.long).t()
        train_target_etypes = torch.tensor([t[2] for t in train_triplets])

        valid_edges = torch.tensor([[t[0], t[1]] for t in valid_triplets], dtype=torch.long).t()
        valid_etypes = torch.tensor([t[2] for t in valid_triplets])

        test_edges = torch.tensor([[t[0], t[1]] for t in test_triplets], dtype=torch.long).t()
        test_etypes = torch.tensor([t[2] for t in test_triplets])

        # Create bidirectional edges for the knowledge graph (add inverse relations)
        train_edges = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        train_etypes = torch.cat([train_target_etypes, train_target_etypes + num_relations])

        # Create Data objects for train, valid, test
        train_data = Data(
            edge_index=train_edges, 
            edge_type=train_etypes, 
            num_nodes=num_node,
            target_edge_index=train_target_edges, 
            target_edge_type=train_target_etypes, 
            num_relations=num_relations * 2
        )
        
        valid_data = Data(
            edge_index=train_edges, 
            edge_type=train_etypes, 
            num_nodes=num_node,
            target_edge_index=valid_edges, 
            target_edge_type=valid_etypes, 
            num_relations=num_relations * 2
        )
        
        test_data = Data(
            edge_index=train_edges, 
            edge_type=train_etypes, 
            num_nodes=num_node,
            target_edge_index=test_edges, 
            target_edge_type=test_etypes, 
            num_relations=num_relations * 2
        )

        # Apply pre_transform (build relation graphs)
        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        # Save processed data
        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])
        
        # Save vocabularies for later use
        vocab_path = os.path.join(self.processed_dir, "vocabularies.pt")
        torch.save({
            'entity_vocab': test_results["inv_entity_vocab"],
            'relation_vocab': test_results["inv_rel_vocab"]
        }, vocab_path)

    def __repr__(self):
        return f"{self.__class__.__name__}()"


def FB60KNYT10_dataset(root, data_path):
    """Factory function to create FB60K+NYT10 dataset"""
    return FB60KNYT10(root=root, data_path=data_path)