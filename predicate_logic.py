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
    def __init__(self, dataset, rule_threshold=0.2, max_rule_length=3, max_rules=100, max_time=300, 
                 subgraph_size_ratio=0.1, max_nodes_per_relation=1000):
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
        针对大型数据集的优化版本
        """
        print("Extracting Horn rules...")
        
        # Create mappings for fact retrieval
        relation_to_entity_pairs = defaultdict(set)
        entity_pair_to_relations = defaultdict(set)
        
        # Fill mappings from all facts
        print("Building relation mappings...")
        for h, r, t in tqdm(self.dataset.all_facts, desc="Loading facts"):
            relation_to_entity_pairs[r].add((h, t))
            entity_pair_to_relations[(h, t)].add(r)
        
        # Add training facts to observed facts
        for h, r, t in self.dataset.train_triples:
            self.observed_facts.add((h, r, t))
        
        # Calculate relation statistics
        relation_stats = {r: len(pairs) for r, pairs in relation_to_entity_pairs.items()}
        print(f"Total relations in dataset: {len(relation_stats)}")
        
        # 根据数据集大小动态调整参数
        total_relations = len(relation_stats)
        total_facts = len(self.dataset.all_facts)
        
        if total_facts > 100000:  # 大型数据集 (如FB15k237, WN18RR)
            max_relations_1hop = min(100, total_relations)  # 增加到100个关系
            max_relations_2hop = min(30, total_relations)   # 增加到30个关系
            max_time_limit = 1800  # 30分钟
            max_samples_per_rule = 1000  # 增加采样数
            print(f"大型数据集模式: 关系数={total_relations}, 事实数={total_facts}")
        elif total_facts > 10000:  # 中型数据集
            max_relations_1hop = min(50, total_relations)
            max_relations_2hop = min(15, total_relations)
            max_time_limit = 900  # 15分钟
            max_samples_per_rule = 500
            print(f"中型数据集模式: 关系数={total_relations}, 事实数={total_facts}")
        else:  # 小型数据集 (如family, kinship)
            max_relations_1hop = total_relations  # 全部关系
            max_relations_2hop = total_relations
            max_time_limit = 300  # 5分钟
            max_samples_per_rule = 100
            print(f"小型数据集模式: 关系数={total_relations}, 事实数={total_facts}")
        
        # For efficiency, focus on the most frequent relations first
        relation_items = sorted(relation_stats.items(), key=lambda x: x[1], reverse=True)
        top_relations_1hop = [r for r, _ in relation_items[:max_relations_1hop]]
        top_relations_2hop = [r for r, _ in relation_items[:max_relations_2hop]]
        
        print(f"使用 {len(top_relations_1hop)} 个关系进行1跳规则挖掘")
        print(f"使用 {len(top_relations_2hop)} 个关系进行2跳规则挖掘")
        
        # Set a time limit
        start_time = time.time()
        
        # Try all possible rule combinations up to max_rule_length
        print(f"Mining rules with confidence threshold {self.rule_threshold}...")
        
        # 1跳规则: A -> B
        total_pairs = len(top_relations_1hop) * len(top_relations_1hop)
        rules_found = 0
        
        with tqdm(total=total_pairs, desc="检查1跳规则") as pbar:
            for head_rel in top_relations_1hop:
                # Check time limit
                if time.time() - start_time > max_time_limit:
                    print(f"时间限制 {max_time_limit}s 到达，停止规则提取")
                    break
                
                # Check rule limit
                if len(self.rules) >= self.max_rules:
                    print(f"达到最大规则数量 ({self.max_rules})")
                    break
                
                # For rules with a single body relation
                for body_rel in top_relations_1hop:
                    pbar.update(1)
                    
                    if head_rel == body_rel:
                        continue
                    
                    # Calculate support and confidence
                    head_pairs = relation_to_entity_pairs[head_rel]
                    body_pairs = relation_to_entity_pairs[body_rel]
                    
                    if not body_pairs or len(body_pairs) < 5:  # 至少需要5个支持
                        continue
                    
                    # Calculate overlap of entity pairs
                    overlap = head_pairs.intersection(body_pairs)
                    
                    # Calculate rule confidence
                    confidence = len(overlap) / len(body_pairs) if body_pairs else 0
                    support = len(overlap)
                    
                    # 添加支持度要求，避免低频规则
                    min_support = max(3, int(len(body_pairs) * 0.1))  # 至少3个或10%支持
                    
                    if confidence >= self.rule_threshold and support >= min_support:
                        rule = (head_rel, (body_rel,))
                        self.rules.append(rule)
                        self.rule_weights[rule] = confidence
                        rules_found += 1
                        
                        if rules_found % 10 == 0:
                            print(f"已找到 {rules_found} 条1跳规则")
        
        print(f"找到 {rules_found} 条1跳规则")
        
        # 2跳规则: A,B -> C (如果还有时间和空间)
        if self.max_rule_length >= 2 and len(self.rules) < self.max_rules and time.time() - start_time < max_time_limit:
            print("开始2跳规则挖掘...")
            
            # 更智能的2跳规则挖掘
            rules_2hop_found = 0
            total_combinations = len(top_relations_2hop) ** 3
            processed = 0
            
            with tqdm(total=min(total_combinations, 10000), desc="检查2跳规则") as pbar:
                for head_rel in top_relations_2hop:
                    if time.time() - start_time > max_time_limit:
                        break
                    
                    if len(self.rules) >= self.max_rules:
                        break
                    
                    head_pairs = relation_to_entity_pairs[head_rel]
                    if len(head_pairs) < 10:  # 头关系需要足够的支持
                        continue
                    
                    for body_rel1 in top_relations_2hop:
                        if processed >= 10000:  # 限制处理数量
                            break
                        
                        for body_rel2 in top_relations_2hop:
                            processed += 1
                            pbar.update(1)
                            
                            if head_rel == body_rel1 or head_rel == body_rel2 or body_rel1 == body_rel2:
                                continue
                            
                            # 高效的路径查找
                            rel1_pairs = relation_to_entity_pairs[body_rel1]
                            rel2_pairs = relation_to_entity_pairs[body_rel2]
                            
                            if not rel1_pairs or not rel2_pairs:
                                continue
                            
                            # 构建中间实体映射
                            rel1_by_head = defaultdict(set)  # head -> set of tails
                            for h, t in rel1_pairs:
                                rel1_by_head[h].add(t)
                            
                            rel2_by_head = defaultdict(set)  # head -> set of tails
                            for h, t in rel2_pairs:
                                rel2_by_head[h].add(t)
                            
                            # 找到路径: (h, body_rel1, intermediate) AND (intermediate, body_rel2, t)
                            path_pairs = set()
                            path_count = 0
                            max_paths_to_check = max_samples_per_rule
                            
                            for h in rel1_by_head:
                                if path_count >= max_paths_to_check:
                                    break
                                
                                for intermediate in rel1_by_head[h]:
                                    if intermediate in rel2_by_head:
                                        for t in rel2_by_head[intermediate]:
                                            path_pairs.add((h, t))
                                            path_count += 1
                                            if path_count >= max_paths_to_check:
                                                break
                                    if path_count >= max_paths_to_check:
                                        break
                            
                            # 计算与头关系的重叠
                            if len(path_pairs) >= 5:  # 至少5个路径
                                overlap = head_pairs.intersection(path_pairs)
                                confidence = len(overlap) / len(path_pairs)
                                support = len(overlap)
                                
                                min_support_2hop = max(2, int(len(path_pairs) * 0.05))
                                
                                if confidence >= self.rule_threshold and support >= min_support_2hop:
                                    rule = (head_rel, (body_rel1, body_rel2))
                                    self.rules.append(rule)
                                    self.rule_weights[rule] = confidence
                                    rules_2hop_found += 1
                                    
                                    if rules_2hop_found % 5 == 0:
                                        print(f"已找到 {rules_2hop_found} 条2跳规则")
                                    
                                    if len(self.rules) >= self.max_rules:
                                        break
                        
                        if len(self.rules) >= self.max_rules or processed >= 10000:
                            break
                    
                    if len(self.rules) >= self.max_rules or processed >= 10000:
                        break
            
            print(f"找到 {rules_2hop_found} 条2跳规则")
        
        elapsed = time.time() - start_time
        print(f"总共提取了 {len(self.rules)} 条规则，置信度 >= {self.rule_threshold}，耗时 {elapsed:.2f} 秒")
        
        # 打印一些统计信息
        if self.rules:
            confidences = list(self.rule_weights.values())
            print(f"规则置信度统计: 最小={min(confidences):.3f}, 最大={max(confidences):.3f}, 平均={np.mean(confidences):.3f}")
            
            # 按置信度排序并展示前10条规则
            sorted_rules = sorted(self.rules, key=lambda r: self.rule_weights[r], reverse=True)
            print("\n置信度最高的前10条规则:")
            for i, rule in enumerate(sorted_rules[:10]):
                head_rel, body_rels = rule
                confidence = self.rule_weights[rule]
                print(f"{i+1:2d}. {head_rel} <- {body_rels}, 置信度: {confidence:.3f}")
        else:
            print("警告: 没有提取到任何规则！请考虑降低置信度阈值。")
    
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
            # print(f"MLN too large ({len(self.MLN_nodes)} nodes). Reducing to {self.max_subgraph_size} nodes...")
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
