import torch
import torch.nn as nn
import torch.nn.functional as F

class RelationRuleEmbedding(nn.Module):
    """
    Relation-Rule Joint Embedding component.
    
    Implements:
    1. Construction of logical operator (Neural network for conjunction)
    2. Representation of rule weights through relation-specific embeddings
    """
    def __init__(self, transH, dim=100, hidden_dim=200):
        """
        Initialize the relation-rule joint embedding component.
        
        Args:
            transH: TransH model with entity and relation embeddings
            dim: Dimension of entity and relation embeddings
            hidden_dim: Hidden dimension for the logical operator network
        """
        super(RelationRuleEmbedding, self).__init__()
        
        self.transH = transH
        self.dim = dim
        self.hidden_dim = hidden_dim
        
        # Create separate networks for different number of input relations
        # Network for 1 relation in body
        self.single_layer1 = nn.Linear(2 * dim, hidden_dim)
        self.single_layer2 = nn.Linear(hidden_dim, 2 * dim)
        
        # Network for 2 relations in body
        self.double_layer1 = nn.Linear(4 * dim, hidden_dim)
        self.double_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.double_layer3 = nn.Linear(hidden_dim, 2 * dim)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.single_layer1.weight)
        nn.init.xavier_uniform_(self.single_layer2.weight)
        nn.init.xavier_uniform_(self.double_layer1.weight)
        nn.init.xavier_uniform_(self.double_layer2.weight)
        nn.init.xavier_uniform_(self.double_layer3.weight)
    
    def logical_operator(self, norm_vecs, trans_vecs):
        """
        Implement the logical AND operator for relation vectors.
        
        Args:
            norm_vecs: List of normal vectors for relations
            trans_vecs: List of translation vectors for relations
            
        Returns:
            Combined normal and translation vectors
        """
        device = self.single_layer1.weight.device
        
        # Check if inputs are not empty
        if not norm_vecs or not trans_vecs:
            # Return default vectors if inputs are empty
            n_combined = torch.zeros(self.dim, device=device)
            t_combined = torch.zeros(self.dim, device=device)
            return F.normalize(n_combined, p=2, dim=0), t_combined
        
        # Choose appropriate network based on number of relations
        if len(norm_vecs) == 1:
            # Single relation case - simply return the original vectors
            return norm_vecs[0].squeeze(0), trans_vecs[0].squeeze(0)
            
        elif len(norm_vecs) == 2:
            # Two relations case - use the double network
            # Concatenate all vectors
            n1, n2 = norm_vecs[0].squeeze(0), norm_vecs[1].squeeze(0)
            t1, t2 = trans_vecs[0].squeeze(0), trans_vecs[1].squeeze(0)
            
            x = torch.cat([n1, t1, n2, t2])
            
            # Forward pass through MLP
            x = F.relu(self.double_layer1(x))
            x = F.relu(self.double_layer2(x))
            output = self.double_layer3(x)
            
            # Split output into normal and translation vectors
            n_raw, t_raw = torch.split(output, self.dim)
            
            # Normalize normal vector to unit length
            n_combined = F.normalize(n_raw, p=2, dim=0)
            
            # For translation vector, we don't normalize but scale to similar magnitude
            t_norm = torch.norm(t_raw) + 1e-10
            t_combined = t_raw / t_norm
            
            return n_combined, t_combined
        else:
            # More than 2 relations - process pairs iteratively
            # Start with the first pair
            n1, n2 = norm_vecs[0].squeeze(0), norm_vecs[1].squeeze(0)
            t1, t2 = trans_vecs[0].squeeze(0), trans_vecs[1].squeeze(0)
            
            x = torch.cat([n1, t1, n2, t2])
            
            # Forward pass through MLP for first pair
            x = F.relu(self.double_layer1(x))
            x = F.relu(self.double_layer2(x))
            output = self.double_layer3(x)
            
            # Split output into normal and translation vectors
            n_combined, t_combined = torch.split(output, self.dim)
            n_combined = F.normalize(n_combined, p=2, dim=0)
            t_combined = t_combined / (torch.norm(t_combined) + 1e-10)
            
            # Process remaining relations
            for i in range(2, len(norm_vecs)):
                ni = norm_vecs[i].squeeze(0)
                ti = trans_vecs[i].squeeze(0)
                
                x = torch.cat([n_combined, t_combined, ni, ti])
                
                # Forward pass through MLP
                x = F.relu(self.double_layer1(x))
                x = F.relu(self.double_layer2(x))
                output = self.double_layer3(x)
                
                # Split and normalize
                n_combined, t_combined = torch.split(output, self.dim)
                n_combined = F.normalize(n_combined, p=2, dim=0)
                t_combined = t_combined / (torch.norm(t_combined) + 1e-10)
            
            return n_combined, t_combined
    
    def compute_rule_weight(self, rule, entity_pairs):
        """
        Compute the weight of a rule based on projection patterns.
        
        Args:
            rule: Tuple (head_relation, (body_relation1, body_relation2, ...))
            entity_pairs: List of (head, tail) entity pairs for grounding
            
        Returns:
            Computed rule weight
        """
        head_rel, body_rels = rule
        
        # Get embeddings for all relations
        _, relation_norm_vecs, relation_trans_vecs = self.transH.get_embeddings()
        
        # Ensure relation indices are valid
        max_rel_idx = len(relation_norm_vecs) - 1
        head_rel = min(head_rel, max_rel_idx)
        
        # Extract relation embeddings
        head_norm = relation_norm_vecs[head_rel].unsqueeze(0)
        head_trans = relation_trans_vecs[head_rel].unsqueeze(0)
        
        # Body relations are now in a tuple - convert to list of valid indices
        valid_body_rels = [min(rel, max_rel_idx) for rel in body_rels]
        
        body_norms = [relation_norm_vecs[rel].unsqueeze(0) for rel in valid_body_rels]
        body_trans = [relation_trans_vecs[rel].unsqueeze(0) for rel in valid_body_rels]
        
        try:
            # Apply logical operator to get combined body embeddings
            body_combined_norm, body_combined_trans = self.logical_operator(body_norms, body_trans)
            
            # If no entity pairs for grounding, return default weight
            if not entity_pairs:
                return torch.tensor(-1.0, device=head_norm.device)
            
            # Compute scores for all entity pairs
            total_score = torch.tensor(0.0, device=head_norm.device)
            entity_embeds, _, _ = self.transH.get_embeddings()
            max_entity_idx = len(entity_embeds) - 1
            
            valid_pairs = 0
            for h, t in entity_pairs:
                try:
                    # Ensure entity indices are valid
                    h = min(h, max_entity_idx)
                    t = min(t, max_entity_idx)
                    
                    # Get entity embeddings
                    h_embed = entity_embeds[h].unsqueeze(0)
                    t_embed = entity_embeds[t].unsqueeze(0)
                    
                    # Project entities onto head relation hyperplane
                    h_proj_head = self.transH.project_entity(h_embed, head_norm)
                    t_proj_head = self.transH.project_entity(t_embed, head_norm)
                    
                    # Project entities onto body relation hyperplane
                    h_proj_body = self.transH.project_entity(
                        h_embed, 
                        body_combined_norm.unsqueeze(0) if body_combined_norm.dim() == 1 else body_combined_norm
                    )
                    t_proj_body = self.transH.project_entity(
                        t_embed, 
                        body_combined_norm.unsqueeze(0) if body_combined_norm.dim() == 1 else body_combined_norm
                    )
                    
                    # Calculate projection patterns
                    p_head = h_proj_head + head_trans - t_proj_head
                    
                    body_trans_for_proj = (
                        body_combined_trans.unsqueeze(0) 
                        if body_combined_trans.dim() == 1 
                        else body_combined_trans
                    )
                    
                    p_body = h_proj_body + body_trans_for_proj - t_proj_body
                    
                    # Calculate squared Euclidean distance between patterns
                    distance = torch.norm(p_head - p_body, p=2) ** 2
                    
                    # Negative distance as score (smaller distance = higher confidence)
                    total_score -= distance
                    valid_pairs += 1
                except (IndexError, RuntimeError) as e:
                    # Skip problematic entity pairs
                    continue
            
            # Average score over all valid entity pairs
            if valid_pairs > 0:
                weight = total_score / valid_pairs
            else:
                weight = torch.tensor(-1.0, device=head_norm.device)
            
            return weight
        except Exception as e:
            print(f"Error computing weight for rule {rule}: {e}")
            device = next(self.parameters()).device
            return torch.tensor(-1.0, device=device)
    
    def normalize_weights(self, weights):
        """
        Normalize rule weights using softmax.
        
        Args:
            weights: Dictionary mapping rules to weights
            
        Returns:
            Normalized weights
        """
        if not weights:
            return {}
        
        # Convert weights to tensor
        try:
            # Filter out any None or NaN values
            valid_weights = {k: v for k, v in weights.items() 
                            if v is not None and not torch.isnan(v).any()}
            
            if not valid_weights:
                return weights
                
            weight_values = torch.stack(list(valid_weights.values()))
            
            # Apply softmax normalization
            normalized_weights = F.softmax(weight_values, dim=0)
            
            # Create new dictionary with normalized weights
            normalized_dict = {}
            for i, rule in enumerate(valid_weights.keys()):
                normalized_dict[rule] = normalized_weights[i]
            
            # Add back any rules that were filtered out with a small default weight
            device = next(iter(normalized_dict.values())).device
            default_weight = torch.tensor(0.01, device=device)
            
            for rule in weights:
                if rule not in normalized_dict:
                    normalized_dict[rule] = default_weight
            
            return normalized_dict
        except (RuntimeError, ValueError) as e:
            # Fallback if weights cannot be stacked
            print(f"Warning: Could not normalize weights ({e}), returning original values")
            return weights
    
    def forward(self, rules, groundings):
        """
        Forward pass to compute rule weights.
        
        Args:
            rules: List of rules (head_relation, (body_relations))
            groundings: Dictionary mapping rules to entity pairs
            
        Returns:
            Dictionary of computed weights for each rule
        """
        if not rules:
            return {}
        
        weights = {}
        
        for rule in rules:
            try:
                # Extract entity pairs from groundings
                rule_groundings = groundings.get(rule, [])
                entity_pairs = []
                
                # Process groundings to extract entity pairs
                for grounding in rule_groundings:
                    if grounding:
                        head_fact = grounding[0]  # First fact is head fact
                        if head_fact:
                            h, _, t = head_fact
                            entity_pairs.append((h, t))
                
                # Compute weight for this rule
                weight = self.compute_rule_weight(rule, entity_pairs)
                weights[rule] = weight
            except Exception as e:
                # Skip problematic rules
                print(f"Error processing rule {rule}: {e}")
                # Use default weight
                weights[rule] = torch.tensor(-1.0, device=next(self.parameters()).device)
        
        # Normalize weights
        normalized_weights = self.normalize_weights(weights)
        
        return normalized_weights