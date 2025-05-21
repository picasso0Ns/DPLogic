import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SimpleRuleEnhancedTransH(nn.Module):
    """
    基于简单规则增强的TransH模型。
    直接整合规则知识，避免复杂的MLN结构。
    """
    def __init__(self, transH, dataset, rule_weight=0.5):
        super(SimpleRuleEnhancedTransH, self).__init__()
        self.transH = transH
        self.dataset = dataset
        self.rule_weight = rule_weight
        
        # 抽取简单规则
        self.rules = self._extract_simple_rules()
        print(f"抽取了 {len(self.rules)} 条简单规则")
    
    def _extract_simple_rules(self):
        """
        抽取简单的关系规则：如果关系r1经常与r2共现，则r1(h,t) => r2(h,t)
        只提取高置信度的规则
        """
        rules = []
        # 关系到实体对的映射
        rel_to_pairs = {}
        
        # 统计关系到实体对的映射
        for h, r, t in self.dataset.train_triples:
            if r not in rel_to_pairs:
                rel_to_pairs[r] = set()
            rel_to_pairs[r].add((h, t))
        
        # 检查关系之间的蕴含
        relations = list(rel_to_pairs.keys())
        for r1 in relations:
            pairs1 = rel_to_pairs[r1]
            if len(pairs1) < 10:  # 忽略罕见关系
                continue
                
            for r2 in relations:
                if r1 == r2:
                    continue
                
                pairs2 = rel_to_pairs.get(r2, set())
                if len(pairs2) < 10:
                    continue
                
                # 计算r1 => r2的支持度
                overlap = pairs1.intersection(pairs2)
                conf = len(overlap) / len(pairs1) if pairs1 else 0
                
                # 只保留高置信度规则
                if conf > 0.7 and len(overlap) > 5:
                    rules.append((r1, r2, conf))
        
        # 按置信度排序并返回前20个规则
        rules.sort(key=lambda x: x[2], reverse=True)
        return rules[:20]
    
    def forward(self, pos_triples, neg_triples):
        """
        带规则增强的TransH损失
        """
        # 处理批次大小不匹配
        pos_batch_size = pos_triples.size(0)
        neg_batch_size = neg_triples.size(0)
        neg_ratio = neg_batch_size // pos_batch_size
        expanded_pos_triples = pos_triples.repeat_interleave(neg_ratio, dim=0)
        
        # 基本TransH损失
        basic_loss = self.transH(expanded_pos_triples, neg_triples)
        
        # 规则增强损失
        rule_loss = torch.tensor(0.0, device=pos_triples.device)
        
        if self.rules:
            for i, (h, r, t) in enumerate(pos_triples):
                for r1, r2, conf in self.rules:
                    if r == r1:
                        # 如果当前三元组的关系是规则的前提
                        # 鼓励结论关系也成立
                        rule_score = self.transH.get_score(
                            torch.tensor([h], device=pos_triples.device),
                            torch.tensor([r2], device=pos_triples.device),
                            torch.tensor([t], device=pos_triples.device)
                        )
                        # 使用规则的置信度作为权重
                        rule_loss -= conf * rule_score.mean()
        
        # 总损失
        total_loss = basic_loss + self.rule_weight * rule_loss
        
        return total_loss
    
    def get_score(self, head, relation, tail):
        """
        计算三元组(h, r, t)的分数，包括规则增强。
        
        参数:
            head: 头实体索引张量
            relation: 关系索引张量
            tail: 尾实体索引张量
            
        返回:
            带规则增强的分数张量
        """
        # 基本TransH分数
        basic_scores = self.transH.get_score(head, relation, tail)
        
        # 检查是否为单个三元组评分
        if head.size(0) == 1 and tail.size(0) == 1:
            return basic_scores
        
        # 规则增强分数
        rule_scores = torch.zeros_like(basic_scores)
        
        # 应用规则 - 仅在评估模式下应用
        if not self.training and self.rules:
            h = head[0].item()  # 假设在评估时所有头实体相同
            r = relation[0].item()  # 假设在评估时所有关系相同
            
            for r1, r2, conf in self.rules:
                if r == r1:
                    # 如果查询关系是规则前提，检查结论关系
                    for i, t in enumerate(tail):
                        # 计算r2(h,t)的得分
                        rule_score = self.transH.get_score(
                            torch.tensor([h], device=head.device),
                            torch.tensor([r2], device=relation.device),
                            torch.tensor([t.item()], device=tail.device)
                        )
                        rule_scores[i] += conf * rule_score.item()
        
        # 返回组合分数
        return basic_scores + self.rule_weight * rule_scores
    
    def evaluate(self, dataset):
        """
        在测试数据上评估模型。
        
        参数:
            dataset: KnowledgeGraphDataset
            
        返回:
            MRR, Hits@1, Hits@3, Hits@10
        """
        print("评估模型...")
        
        hits1 = 0
        hits3 = 0
        hits10 = 0
        mrr = 0
        count = 0
        
        # 获取测试数据
        test_triples = dataset.test_triples
        
        # 使用批处理进行评估
        batch_size = 100
        num_batches = (len(test_triples) + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc="评估"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(test_triples))
            batch_triples = test_triples[start_idx:end_idx]
            
            for h, r, t in batch_triples:
                # 跳过无效的实体或关系ID
                if h > self.max_entity_id or t > self.max_entity_id or r > self.max_relation_id:
                    continue
                
                # 限制候选实体数量以提高效率
                max_candidates = 500
                if dataset.num_entities > max_candidates:
                    # 使用一些实际实体 + 正确实体
                    sampled_entities = np.random.choice(
                        dataset.num_entities, 
                        size=max_candidates-1, 
                        replace=False
                    ).tolist()
                    if t not in sampled_entities:
                        sampled_entities.append(t)
                    candidates = sampled_entities
                else:
                    candidates = list(range(dataset.num_entities))
                
                # 为所有候选实体计算分数
                h_tensor = torch.tensor([h] * len(candidates), device=self.device)
                r_tensor = torch.tensor([r] * len(candidates), device=self.device)
                t_tensor = torch.tensor(candidates, device=self.device)
                
                try:
                    # 获取TransH分数
                    combined_tensor = torch.cat([
                    h_tensor.view(-1, 1),  
                    r_tensor.view(-1, 1),  
                    t_tensor.view(-1, 1)   
                    ], dim=1)  

                    # 调用 scoreOp 方法
                    scores = self.transH.scoreOp(combined_tensor)
                    
                    # 应用规则增强
                    rule_scores = self._apply_rule_enhancement(h, r, candidates)
                    
                    # 组合分数
                    final_scores = scores + self.beta * rule_scores
                    
                    # 排序分数
                    _, sorted_indices = torch.sort(final_scores, descending=True)
                    
                    # 找到正确实体的排名
                    correct_idx = candidates.index(t)
                    rank = (sorted_indices == correct_idx).nonzero().item() + 1
                    
                    # 更新指标
                    if rank <= 10:
                        hits10 += 1
                        if rank <= 3:
                            hits3 += 1
                            if rank == 1:
                                hits1 += 1
                    
                    mrr += 1.0 / rank
                    count += 1
                    
                except Exception as e:
                    print(f"评估错误: {e}")
                    continue
        
        # 计算最终指标
        if count == 0:
            return 0, 0, 0, 0
            
        hits1 = hits1 / count
        hits3 = hits3 / count
        hits10 = hits10 / count
        mrr = mrr / count
        
        return mrr, hits1, hits3, hits10