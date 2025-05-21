import numpy as np
from collections import defaultdict
import torch
import itertools
import networkx as nx
from tqdm import tqdm
import time
import random

class PredicateLogic:
    """
    Predicate Logic Representation component for constructing MLN structure.
    Uses GNN-inspired approach for representing node dependencies.
    """
    def __init__(self, dataset, rule_threshold=0.5, max_rule_length=2, max_rules=100, max_time=300, 
                 subgraph_size_ratio=0.2, max_nodes_per_relation=500):
        """
        Initialize the predicate logic component.
        
        Args:
            dataset: KnowledgeGraphDataset object
            rule_threshold: Confidence threshold for rule selection
            max_rule_length: Maximum length of rules to mine
            max_rules: Maximum number of rules to extract
            max_time: Maximum time (in seconds) for rule extraction
            subgraph_size_ratio: Maximum ratio of total triples to include in subgraph
            max_nodes_per_relation: Maximum number of nodes per relation to include
        """
        self.dataset = dataset
        self.rule_threshold = rule_threshold
        self.max_rule_length = max_rule_length
        self.max_rules = max_rules
        self.max_time = max_time
        self.subgraph_size_ratio = subgraph_size_ratio
        self.max_nodes_per_relation = max_nodes_per_relation
        
        # Store the MLN structure
        self.MLN_nodes = set()  # Set of (entity1, relation, entity2) tuples
        self.MLN_edges = set()  # Set of ((e1,r1,e2), (e3,r3,e4)) edges
        
        # Store rules and their confidences
        self.rules = []  # List of (head_relation, (body_relation1, body_relation2, ...)) tuples
        self.rule_weights = {}  # Dictionary mapping rules to weights
        
        # Store observed and unobserved facts
        self.observed_facts = set()
        self.unobserved_facts = set()
        
        # Calculate max subgraph size
        self.max_subgraph_size = int(len(dataset.train_triples) * subgraph_size_ratio)
        
        # Extract rules and build MLN structure
        self._extract_rules()
        self._build_mln_structure_efficient()
    
    def _extract_rules(self):
        """
        Extract Horn rules from the knowledge graph.
        Uses a simplified approach based on co-occurrence patterns.
        """
        print("Extracting Horn rules...")
        
        # Create mappings for fact retrieval
        relation_to_entity_pairs = defaultdict(set)
        entity_pair_to_relations = defaultdict(set)
        
        # Fill mappings from all facts - this can be time-consuming for large KGs
        # so we add a progress bar
        print("Building relation mappings...")
        for h, r, t in tqdm(self.dataset.all_facts, desc="Loading facts"):
            relation_to_entity_pairs[r].add((h, t))
            entity_pair_to_relations[(h, t)].add(r)
        
        # Add training facts to observed facts
        for h, r, t in self.dataset.train_triples:
            self.observed_facts.add((h, r, t))
        
        # Calculate relation statistics
        relation_stats = {r: len(pairs) for r, pairs in relation_to_entity_pairs.items()}
        
        # For efficiency, focus on the most frequent relations first
        relation_items = sorted(relation_stats.items(), key=lambda x: x[1], reverse=True)
        top_relations = [r for r, _ in relation_items[:min(20, len(relation_items))]]
        
        # Set a time limit
        start_time = time.time()
        
        # Try all possible rule combinations up to max_rule_length
        print(f"Mining rules with confidence threshold {self.rule_threshold}...")
        
        # Process only rules involving the most frequent relations
        total_pairs = len(top_relations) * len(top_relations)
        with tqdm(total=total_pairs, desc="Checking relation pairs") as pbar:
            for head_rel in top_relations:
                # Check time limit
                if time.time() - start_time > self.max_time:
                    print(f"Time limit of {self.max_time}s reached. Stopping rule extraction.")
                    break
                
                # Check rule limit
                if len(self.rules) >= self.max_rules:
                    print(f"Maximum number of rules ({self.max_rules}) reached.")
                    break
                
                # For rules with a single body relation
                for body_rel in top_relations:
                    pbar.update(1)
                    
                    if head_rel == body_rel:
                        continue
                    
                    # Calculate support and confidence
                    head_pairs = relation_to_entity_pairs[head_rel]
                    body_pairs = relation_to_entity_pairs[body_rel]
                    
                    if not body_pairs:
                        continue
                    
                    # Calculate overlap of entity pairs
                    overlap = head_pairs.intersection(body_pairs)
                    
                    # Calculate rule confidence
                    confidence = len(overlap) / len(body_pairs) if body_pairs else 0
                    
                    if confidence >= self.rule_threshold and len(overlap) > 0:
                        # Convert list to tuple for hashing
                        rule = (head_rel, (body_rel,))  # Use a tuple instead of a list
                        self.rules.append(rule)
                        self.rule_weights[rule] = confidence
        
        # For rules with two body relations, we'll limit to a small number
        # This is much more time-consuming
        if self.max_rule_length >= 2 and len(self.rules) < self.max_rules:
            print("Checking for 2-hop rules (this may take a while)...")
            # Sample a smaller set of relations to reduce computation
            sample_size = min(5, len(top_relations))
            sampled_relations = top_relations[:sample_size]
            
            total_triples = len(sampled_relations) * len(sampled_relations) * len(sampled_relations)
            with tqdm(total=total_triples, desc="Checking relation triples") as pbar:
                for head_rel in sampled_relations:
                    # Check time limit
                    if time.time() - start_time > self.max_time:
                        print(f"Time limit of {self.max_time}s reached. Stopping rule extraction.")
                        break
                    
                    # Check rule limit
                    if len(self.rules) >= self.max_rules:
                        print(f"Maximum number of rules ({self.max_rules}) reached.")
                        break
                    
                    for body_rel1 in sampled_relations:
                        for body_rel2 in sampled_relations:
                            pbar.update(1)
                            
                            if head_rel == body_rel1 or head_rel == body_rel2 or body_rel1 == body_rel2:
                                continue
                            
                            # For efficiency, estimate confidence with sampling
                            max_samples = 100  # Limit samples
                            
                            # Find entity pairs connected by body_rel1
                            rel1_pairs = list(relation_to_entity_pairs[body_rel1])
                            if not rel1_pairs:
                                continue
                            
                            # Sample pairs
                            if len(rel1_pairs) > max_samples:
                                rel1_pairs = random.sample(rel1_pairs, max_samples)
                            
                            # Count paths
                            path_pairs = set()
                            
                            for h1, t1 in rel1_pairs:
                                # Find pairs where t1 is connected via body_rel2
                                for h2, t2 in relation_to_entity_pairs[body_rel2]:
                                    if h2 == t1:
                                        path_pairs.add((h1, t2))
                            
                            # Calculate overlap with head relation
                            if path_pairs:
                                head_pairs = relation_to_entity_pairs[head_rel]
                                overlap = path_pairs.intersection(head_pairs)
                                
                                # Calculate rule confidence
                                confidence = len(overlap) / len(path_pairs)
                                
                                if confidence >= self.rule_threshold and len(overlap) > 1:
                                    rule = (head_rel, (body_rel1, body_rel2))
                                    self.rules.append(rule)
                                    self.rule_weights[rule] = confidence
        
        elapsed = time.time() - start_time
        print(f"Extracted {len(self.rules)} rules with confidence >= {self.rule_threshold} in {elapsed:.2f} seconds")
    
    def _build_mln_structure_efficient(self):
        """
        Build an efficient MLN structure inspired by GNN approaches.
        Limits subgraph size and uses a simplified dependency model.
        """
        print("Building efficient MLN structure...")
        
        # If no rules were extracted, use a fallback approach
        if not self.rules:
            print("No rules were extracted. Using fallback approach.")
            self._build_fallback_mln()
            return
        
        # Create a graph for efficient neighborhood construction
        print("Creating knowledge graph...")
        kg_graph = nx.DiGraph()
        
        # Add edges for all facts from training set
        for h, r, t in tqdm(self.dataset.train_triples, desc="Building graph"):
            kg_graph.add_edge(h, t, relation=r)
            self.observed_facts.add((h, r, t))
        
        # Add test facts to unobserved facts
        for h, r, t in self.dataset.test_triples:
            self.unobserved_facts.add((h, r, t))
        
        # Create a mapping from relation to entity pairs
        relation_to_pairs = defaultdict(list)
        for h, r, t in self.observed_facts:
            relation_to_pairs[r].append((h, t))
        
        # Process each rule and collect relevant nodes
        print("Collecting relevant nodes for rules...")
        for head_rel, body_rels in tqdm(self.rules, desc="Processing rules"):
            all_rels = [head_rel] + list(body_rels)
            
            for rel in all_rels:
                # Get entity pairs for this relation, limited by max_nodes_per_relation
                pairs = relation_to_pairs.get(rel, [])
                if len(pairs) > self.max_nodes_per_relation:
                    pairs = random.sample(pairs, self.max_nodes_per_relation)
                
                # Add these direct relation nodes to MLN
                for h, t in pairs:
                    self.MLN_nodes.add((h, rel, t))
                
                # Find 1-hop neighbors efficiently (more like a GNN approach)
                for h, t in pairs:
                    if h in kg_graph and t in kg_graph:
                        # Get direct neighbors (1-hop only)
                        h_neighbors = list(kg_graph.neighbors(h))
                        if len(h_neighbors) > 5:  # Limit neighbors
                            h_neighbors = h_neighbors[:5]
                        
                        t_neighbors = list(kg_graph.neighbors(t))
                        if len(t_neighbors) > 5:
                            t_neighbors = t_neighbors[:5]
                        
                        # Create edges for these neighbors
                        for e1 in [h] + h_neighbors[:1]:  # Include h and at most 1 neighbor
                            for e2 in [t] + t_neighbors[:1]:  # Include t and at most 1 neighbor
                                for body_rel in body_rels:
                                    # Add nodes for relation between these entities
                                    self.MLN_nodes.add((e1, body_rel, e2))
                                    # Add edges between related nodes
                                    self.MLN_edges.add(((h, rel, t), (e1, body_rel, e2)))
        
        # Check if the MLN is too large and reduce if necessary
        if len(self.MLN_nodes) > self.max_subgraph_size:
            print(f"MLN too large ({len(self.MLN_nodes)} nodes). Reducing to {self.max_subgraph_size} nodes...")
            sampled_nodes = random.sample(list(self.MLN_nodes), self.max_subgraph_size)
            self.MLN_nodes = set(sampled_nodes)
            
            # Rebuild edges
            self.MLN_edges = set()
            for node1 in self.MLN_nodes:
                h1, r1, t1 = node1
                # Connect nodes that share entities or are related by rules
                for node2 in self.MLN_nodes:
                    if node1 == node2:
                        continue
                    
                    h2, r2, t2 = node2
                    # Connect if they share entities
                    if h1 == h2 or t1 == t2:
                        self.MLN_edges.add((node1, node2))
                    
                    # Connect if they are part of the same rule
                    for head_rel, body_rels in self.rules:
                        if (r1 == head_rel and r2 in body_rels) or (r2 == head_rel and r1 in body_rels):
                            self.MLN_edges.add((node1, node2))
        
        # Update unobserved facts
        for node in self.MLN_nodes:
            if node not in self.observed_facts:
                self.unobserved_facts.add(node)
        
        print(f"Built efficient MLN with {len(self.MLN_nodes)} nodes and {len(self.MLN_edges)} edges")
        print(f"Observed facts: {len(self.observed_facts)}, Unobserved facts: {len(self.unobserved_facts)}")
    
    def _build_fallback_mln(self):
        """
        Build a simple MLN structure as a fallback.
        """
        self.MLN_nodes.clear()
        self.MLN_edges.clear()
        self.observed_facts.clear()
        self.unobserved_facts.clear()
        
        # Add all training triples as observed facts
        for h, r, t in self.dataset.train_triples:
            self.MLN_nodes.add((h, r, t))
            self.observed_facts.add((h, r, t))
        
        # Add a subset of test triples as unobserved facts
        max_test = min(len(self.dataset.test_triples), int(0.2 * len(self.dataset.train_triples)))
        if len(self.dataset.test_triples) > max_test:
            test_sample = random.sample(list(self.dataset.test_triples), max_test)
        else:
            test_sample = self.dataset.test_triples
            
        for h, r, t in test_sample:
            self.MLN_nodes.add((h, r, t))
            self.unobserved_facts.add((h, r, t))
        
        # Create a simplified edge structure based on entity sharing
        entity_to_nodes = defaultdict(list)
        for node in self.MLN_nodes:
            h, r, t = node
            entity_to_nodes[h].append(node)
            entity_to_nodes[t].append(node)
        
        # Connect nodes that share an entity (limit connections per entity)
        for entity, nodes in entity_to_nodes.items():
            # Limit number of nodes per entity to reduce edges
            if len(nodes) > 10:
                nodes = random.sample(nodes, 10)
            
            for i, node1 in enumerate(nodes):
                for node2 in nodes[i+1:]:
                    self.MLN_edges.add((node1, node2))
                    self.MLN_edges.add((node2, node1))
        
        print(f"Built fallback MLN with {len(self.MLN_nodes)} nodes and {len(self.MLN_edges)} edges")
    
    def get_markov_blanket(self, node):
        """
        Get the Markov blanket of a node in the MLN.
        In GNN-style, this represents the 1-hop neighborhood.
        
        Args:
            node: Tuple (entity1, relation, entity2)
            
        Returns:
            Set of nodes in the Markov blanket
        """
        markov_blanket = set()
        
        # Find directly connected nodes (1-hop neighborhood)
        for node1, node2 in self.MLN_edges:
            if node == node1:
                markov_blanket.add(node2)
            elif node == node2:
                markov_blanket.add(node1)
        
        # If blanket is too large, sample to limit size
        max_blanket_size = 10
        if len(markov_blanket) > max_blanket_size:
            markov_blanket = set(random.sample(list(markov_blanket), max_blanket_size))
        
        # If blanket is empty, add some related nodes based on entity sharing
        if not markov_blanket:
            h, r, t = node
            
            # Find nodes that share entities
            for other_node in self.MLN_nodes:
                if other_node == node:
                    continue
                    
                h2, r2, t2 = other_node
                if h == h2 or t == t2:
                    markov_blanket.add(other_node)
                    if len(markov_blanket) >= max_blanket_size:
                        break
        
        return markov_blanket
    
    def get_rule_groundings(self, rule):
        """
        Get a sample of groundings for a rule.
        
        Args:
            rule: Tuple (head_relation, (body_relation1, body_relation2, ...))
            
        Returns:
            List of groundings, each a tuple of nodes
        """
        head_rel, body_rels = rule
        groundings = []
        
        # Limit number of groundings for efficiency
        max_groundings = 100
        
        # Find facts with the head relation
        head_facts = [(h, t) for h, r, t in self.MLN_nodes if r == head_rel]
        
        # Sample head facts if there are too many
        if len(head_facts) > max_groundings:
            head_facts = random.sample(head_facts, max_groundings)
        
        for h, t in head_facts:
            valid_grounding = True
            body_facts = []
            
            for body_rel in body_rels:
                if (h, body_rel, t) in self.MLN_nodes:
                    body_facts.append((h, body_rel, t))
                else:
                    valid_grounding = False
                    break
            
            if valid_grounding:
                grounding = [(h, head_rel, t)] + body_facts
                groundings.append(grounding)
        
        return groundings
    
    def get_rules(self):
        """
        Get all extracted rules.
        
        Returns:
            List of rules and their weights
        """
        return self.rules, self.rule_weights
    
    def get_mln_structure(self):
        """
        Get the MLN structure.
        
        Returns:
            MLN nodes and edges
        """
        return self.MLN_nodes, self.MLN_edges
    
    def get_observed_unobserved(self):
        """
        Get observed and unobserved facts.
        
        Returns:
            Observed and unobserved facts
        """
        return self.observed_facts, self.unobserved_facts