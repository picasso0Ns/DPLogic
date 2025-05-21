import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random

class KnowledgeGraphDataset:
    """
    Dataset class for knowledge graphs.
    """
    def __init__(self, dataset_name):
        """
        Initialize the dataset.
        
        Args:
            dataset_name (str): Name of the dataset
        """
        self.dataset_name = dataset_name
        self.data_path = os.path.join('.', dataset_name)
        
        # Load entity and relation dictionaries
        self.entity2id, self.id2entity = self._load_entities()
        self.relation2id, self.id2relation = self._load_relations()
        
        # Load triples
        self.train_triples = self._load_triples('train.txt')
        self.test_triples = self._load_triples('test.txt')
        
        # Load all facts for rule mining
        self.all_facts = self._load_facts()
        
        # Set properties
        self.num_entities = len(self.entity2id)
        self.num_relations = len(self.relation2id)
        
        print(f"Loaded {dataset_name} dataset with {self.num_entities} entities, {self.num_relations} relations")
        print(f"Train: {len(self.train_triples)}, Test: {len(self.test_triples)}")
    
    def _load_entities(self):
        """
        Load entities from the dataset.
        
        Returns:
            entity2id, id2entity dictionaries
        """
        entity2id = {}
        id2entity = {}
        
        # Check if entities file exists, if not infer from triples
        entities_file = os.path.join(self.data_path, 'entities.txt')
        if os.path.exists(entities_file):
            with open(entities_file, 'r') as f:
                for i, line in enumerate(f):
                    entity = line.strip()
                    entity2id[entity] = i
                    id2entity[i] = entity
        else:
            # Collect entities from train, test, and facts files
            entities = set()
            for file_name in ['train.txt', 'test.txt', 'facts.txt']:
                file_path = os.path.join(self.data_path, file_name)
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        for line in f:
                            if line.strip():
                                h, _, t = line.strip().split('\t')
                                entities.add(h)
                                entities.add(t)
            
            # Assign IDs
            for i, entity in enumerate(sorted(entities)):
                entity2id[entity] = i
                id2entity[i] = entity
        
        return entity2id, id2entity
    
    def _load_relations(self):
        """
        Load relations from the dataset.
        
        Returns:
            relation2id, id2relation dictionaries
        """
        relation2id = {}
        id2relation = {}
        
        # Read relations file
        with open(os.path.join(self.data_path, 'relations.txt'), 'r') as f:
            for i, line in enumerate(f):
                relation = line.strip()
                relation2id[relation] = i
                id2relation[i] = relation
        
        return relation2id, id2relation
    
    def _load_triples(self, file_name):
        """
        Load triples from a file.
        
        Args:
            file_name (str): Name of the file to load
            
        Returns:
            List of triples as numpy arrays
        """
        triples = []
        with open(os.path.join(self.data_path, file_name), 'r') as f:
            for line in f:
                if line.strip():
                    h, r, t = line.strip().split('\t')
                    # Convert to IDs
                    h_id = int(h) if h.isdigit() else self.entity2id.get(h, 0)
                    r_id = self.relation2id.get(r, 0)
                    t_id = int(t) if t.isdigit() else self.entity2id.get(t, 0)
                    
                    triples.append(np.array([h_id, r_id, t_id]))
        
        return np.array(triples)
    
    def _load_facts(self):
        """
        Load all facts (triples) for rule mining.
        
        Returns:
            List of all facts (triples)
        """
        facts = []
        facts_file = os.path.join(self.data_path, 'facts.txt')
        
        if os.path.exists(facts_file):
            with open(facts_file, 'r') as f:
                for line in f:
                    if line.strip():
                        h, r, t = line.strip().split('\t')
                        # Convert to IDs
                        h_id = int(h) if h.isdigit() else self.entity2id.get(h, 0)
                        r_id = self.relation2id.get(r, 0)
                        t_id = int(t) if t.isdigit() else self.entity2id.get(t, 0)
                        
                        facts.append(np.array([h_id, r_id, t_id]))
            
            return np.array(facts)
        else:
            # If no facts.txt, combine train and test
            return np.concatenate([self.train_triples, self.test_triples], axis=0)
    
    def get_train_dataloader(self, batch_size, neg_ratio=1):
        """
        Create a DataLoader for training data.
        
        Args:
            batch_size (int): Batch size
            neg_ratio (int): Negative sampling ratio
            
        Returns:
            DataLoader for training
        """
        return DataLoader(
            KGTrainDataset(self.train_triples, self.num_entities, neg_ratio),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=KGTrainDataset.collate_fn
        )
    
    def get_test_dataloader(self, batch_size=100):
        """
        Create a DataLoader for test data.
        
        Args:
            batch_size (int): Batch size
            
        Returns:
            DataLoader for testing
        """
        return DataLoader(
            KGTestDataset(self.test_triples, self.num_entities),
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )


class KGTrainDataset(Dataset):
    """
    Dataset for training knowledge graph embeddings with negative sampling.
    """
    def __init__(self, triples, num_entities, neg_ratio=1):
        """
        Initialize the dataset.
        
        Args:
            triples: Array of triples (h, r, t)
            num_entities: Number of entities in the KG
            neg_ratio: Negative sampling ratio
        """
        self.triples = triples
        self.num_entities = num_entities
        self.neg_ratio = neg_ratio
    
    def __len__(self):
        return len(self.triples)
    
    def __getitem__(self, idx):
        pos_triple = self.triples[idx]
        h, r, t = pos_triple
        
        # Generate negative samples
        neg_triples = []
        for _ in range(self.neg_ratio):
            if random.random() < 0.5:  # Corrupt head
                h_neg = random.randint(0, self.num_entities - 1)
                while h_neg == h:  # Avoid false negatives
                    h_neg = random.randint(0, self.num_entities - 1)
                neg_triples.append([h_neg, r, t])
            else:  # Corrupt tail
                t_neg = random.randint(0, self.num_entities - 1)
                while t_neg == t:  # Avoid false negatives
                    t_neg = random.randint(0, self.num_entities - 1)
                neg_triples.append([h, r, t_neg])
        
        return torch.LongTensor(pos_triple), torch.LongTensor(neg_triples)
    
    @staticmethod
    def collate_fn(batch):
        pos_triples = torch.stack([item[0] for item in batch])
        neg_triples = torch.cat([item[1] for item in batch])
        return pos_triples, neg_triples


class KGTestDataset(Dataset):
    """
    Dataset for testing knowledge graph embeddings with rank computation.
    """
    def __init__(self, triples, num_entities):
        """
        Initialize the dataset.
        
        Args:
            triples: Array of triples (h, r, t)
            num_entities: Number of entities in the KG
        """
        self.triples = triples
        self.num_entities = num_entities
    
    def __len__(self):
        return len(self.triples)
    
    def __getitem__(self, idx):
        h, r, t = self.triples[idx]
        return torch.LongTensor([h, r, t])
