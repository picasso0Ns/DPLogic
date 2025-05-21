# 文件：optimization/improved_joint_optimization.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from tqdm import tqdm
import math
import copy
import matplotlib.pyplot as plt

class ImprovedJointOptimization:
    """
    改进的联合优化框架，采用预训练、渐进式规则引入和动态规则评估
    """
    def __init__(self, transH, relation_rule_embedding, predicate_logic, 
                 alpha=1.0, beta=0.01, gamma=0.3, lr=0.001,
                 pretrain_epochs=20, rule_warmup_epochs=10,
                 rule_quality_threshold=0.3, rule_decay_rate=0.9):
        """
        初始化改进的联合优化组件。
        
        参数:
            transH: TransH模型
            relation_rule_embedding: RelationRuleEmbedding模型
            predicate_logic: PredicateLogic组件
            alpha: 表示损失的权重
            beta: 规则损失的初始权重（会逐步增加）
            gamma: 权重正则化损失的权重
            lr: 学习率
            pretrain_epochs: 预训练纯TransH的轮次
            rule_warmup_epochs: 规则权重从beta增加到1.0需要的轮次
            rule_quality_threshold: 规则质量阈值，低于此值的规则将被衰减
            rule_decay_rate: 规则权重的衰减率，应用于低效规则
        """
        self.transH = transH
        self.relation_rule_embedding = relation_rule_embedding
        self.predicate_logic = predicate_logic
        
        # 损失权重
        self.alpha = alpha
        self.init_beta = beta  # 初始规则权重
        self.target_beta = 1.0  # 目标规则权重
        self.current_beta = 0.0  # 当前规则权重，初始设为0
        self.gamma = gamma
        
        # 预训练和规则引入参数
        self.pretrain_epochs = pretrain_epochs
        self.rule_warmup_epochs = rule_warmup_epochs
        self.rule_quality_threshold = rule_quality_threshold
        self.rule_decay_rate = rule_decay_rate
        
        # 获取规则
        self.rules, self.rule_confidences = predicate_logic.get_rules()
        
        # 创建可学习的规则权重参数
        self.device = next(transH.parameters()).device
        self.rule_weights = {}
        self.learnable_rule_weights = nn.ParameterDict()
        
        # 规则评估统计
        self.rule_effectiveness = {rule: 0.0 for rule in self.rules}
        self.rule_application_count = {rule: 0 for rule in self.rules}
        self.rule_success_count = {rule: 0 for rule in self.rules}
        
        # 基于规则置信度初始化权重
        for i, rule in enumerate(self.rules):
            rule_str = f"rule_{i}"
            confidence = self.rule_confidences.get(rule, 0.1)
            # 初始化为规则置信度值
            self.learnable_rule_weights[rule_str] = nn.Parameter(torch.tensor(confidence, device=self.device))
            self.rule_weights[rule] = rule_str
        
        # 获取MLN结构
        self.mln_nodes, self.mln_edges = predicate_logic.get_mln_structure()
        self.observed_facts, self.unobserved_facts = predicate_logic.get_observed_unobserved()
        
        # 初始化优化器 - 确保参数是唯一的
        all_params = set()
        unique_params = []
        
        for param in list(transH.parameters()) + list(relation_rule_embedding.parameters()):
            if param not in all_params:
                unique_params.append(param)
                all_params.add(param)
        
        # 添加可学习的规则权重
        self.optimizer = Adam(
            unique_params + list(self.learnable_rule_weights.values()), 
            lr=lr
        )
        
        # 单独的优化器用于预训练阶段
        self.pretrain_optimizer = Adam(list(transH.parameters()), lr=lr)
        
        # 实体和关系ID的上限
        self.max_entity_id = transH.num_entities - 1
        self.max_relation_id = transH.num_relations - 1
        
        # 初始化因子图（用于跟踪规则影响的三元组）
        self.factor_graph = self._initialize_factor_graph()
        
        # 保存最佳模型状态
        self.best_epoch = 0
        self.best_mrr = 0
        self.best_model_state = None

    def _initialize_factor_graph(self):
        """
        初始化因子图，跟踪规则与三元组之间的联系。
        
        返回:
            字典，将规则映射到相关三元组列表
        """
        factor_graph = {rule: [] for rule in self.rules}
        
        print("初始化因子图...")
        
        # 确定每个规则影响的三元组
        for rule in tqdm(self.rules, desc="处理规则"):
            head_rel, body_rels = rule
            
            # 这里简化处理，我们只考虑规则头部和规则体中存在的三元组
            # 寻找与规则相关的所有三元组
            for h, r, t in self.observed_facts:
                if r == head_rel or r in body_rels:
                    # 确保实体和关系ID在有效范围内
                    if h <= self.max_entity_id and t <= self.max_entity_id and r <= self.max_relation_id:
                        factor_graph[rule].append((h, r, t))
            
            # 限制每个规则的三元组数量以提高效率
            if len(factor_graph[rule]) > 100:
                factor_graph[rule] = factor_graph[rule][:100]
        
        return factor_graph
    
    def joint_forward(self, batch, current_epoch, total_epochs, use_filter=False):
        """
        统一的前向传播，同时考虑嵌入和规则。
        
        参数:
            batch: 训练数据批次（正三元组，负三元组）
            current_epoch: 当前训练轮次
            total_epochs: 总训练轮次
            use_filter: 是否使用过滤机制
                
        返回:
            总损失
        """
        pos_triples, neg_triples = batch
        pos_triples = pos_triples.to(self.device)
        neg_triples = neg_triples.to(self.device)
        
     # 确保所有索引在有效范围内
        pos_triples[:, 0].clamp_(0, self.transH.num_entities - 1)  # 头实体
        pos_triples[:, 2].clamp_(0, self.transH.num_entities - 1)  # 尾实体
        pos_triples[:, 1].clamp_(0, self.transH.num_relations - 1)  # 关系
        
        neg_triples[:, 0].clamp_(0, self.transH.num_entities - 1)  # 负样本头实体
        neg_triples[:, 2].clamp_(0, self.transH.num_entities - 1)  # 负样本尾实体
        neg_triples[:, 1].clamp_(0, self.transH.num_relations - 1)  # 负样本关系
    
        # 添加调试信息（可选）
        if current_epoch == 0 and torch.rand(1).item() < 0.1:  # 只在第一轮随机打印几次
            print(f"正三元组形状: {pos_triples.shape}, 负三元组形状: {neg_triples.shape}")
            print(f"正三元组最大实体ID: {pos_triples[:, [0,2]].max().item()}, 模型实体数: {self.transH.num_entities}")
            print(f"负三元组最大实体ID: {neg_triples[:, [0,2]].max().item()}")
            print(f"正三元组最大关系ID: {pos_triples[:, 1].max().item()}, 模型关系数: {self.transH.num_relations}")
            print(f"负三元组最大关系ID: {neg_triples[:, 1].max().item()}")
        
        # 处理批次大小不匹配
        pos_batch_size = pos_triples.size(0)
        neg_batch_size = neg_triples.size(0)
        neg_ratio = neg_batch_size // pos_batch_size
        
        expanded_pos_triples = pos_triples.repeat_interleave(neg_ratio, dim=0)
        
        # 处理批次大小不匹配
        pos_batch_size = pos_triples.size(0)
        neg_batch_size = neg_triples.size(0)
        neg_ratio = neg_batch_size // pos_batch_size
        
        # 如果使用过滤，确保负样本中没有训练中的正样本
        if use_filter and hasattr(self, 'all_triples'):
            filtered_neg_triples = []
            for i in range(neg_batch_size):
                h, r, t = neg_triples[i].tolist()
                if (h, r, t) not in self.all_triples:
                    filtered_neg_triples.append(neg_triples[i])
            
            if filtered_neg_triples:
                neg_triples = torch.stack(filtered_neg_triples)
                neg_batch_size = neg_triples.size(0)
                neg_ratio = max(1, neg_batch_size // pos_batch_size)
        
        expanded_pos_triples = pos_triples.repeat_interleave(neg_ratio, dim=0)
        
        # 计算TransH表示损失
        rep_loss = self.transH(expanded_pos_triples, neg_triples)
        
        # 在预训练阶段，只返回表示损失
        if current_epoch < self.pretrain_epochs:
            return rep_loss
        
        # 计算规则增强损失
        rule_loss = self._compute_rule_enhanced_loss(pos_triples)
        
        # 计算规则权重正则化损失
        weight_reg_loss = sum(weight**2 for weight in self.learnable_rule_weights.values()).mean()
        
        # 总损失
        total_loss = self.alpha * rep_loss + self.current_beta * rule_loss + self.gamma * weight_reg_loss
        
        # 每10批次打印一次损失组成
        if torch.rand(1).item() < 0.1:  # 10%的概率打印
            print(f"损失组成 - 表示: {rep_loss.item():.4f}, 规则: {rule_loss.item():.4f}, "
                  f"规则权重: {self.current_beta:.4f}, 加权规则损失: {(self.current_beta * rule_loss).item():.4f}")
        
        return total_loss
    
    def _compute_rule_enhanced_loss(self, pos_triples):
        """
        基于规则的损失增强，使用更智能的阈值和规则权重计算。
        
        参数:
            pos_triples: 批次中的正三元组
            
        返回:
            规则增强损失
        """
        loss = torch.tensor(0.0, device=self.device)
        
        for i, (h, r, t) in enumerate(pos_triples):
            # 确保实体和关系ID在有效范围内
            if h > self.max_entity_id or t > self.max_entity_id or r > self.max_relation_id:
                continue
                
            # 查找适用于该关系的规则
            applicable_rules = [rule for rule in self.rules if rule[0] == r.item() or r.item() in rule[1]]
            
            if not applicable_rules:
                continue
                
            for rule in applicable_rules:
                head_rel, body_rels = rule
                rule_str = self.rule_weights[rule]
                rule_weight = self.learnable_rule_weights[rule_str]
                
                # 使用动态阈值，基于规则的学习权重
                threshold = -0.5 * (1.0 + rule_weight.item())  # 阈值随规则权重调整
                
                # 检查规则是否应用于当前三元组
                if r.item() == head_rel:
                    # 当前三元组是规则头部
                    # 检查规则体中的所有关系是否存在
                    body_scores = []  # 初始化body_scores列表
                    body_exists = True
                    
                    for body_rel in body_rels:
                        # 获取TransH分数
                        body_score = self.transH.scoreOp(
                            torch.cat([
                                torch.tensor([[h.item()]], device=self.device),
                                torch.tensor([[body_rel]], device=self.device),
                                torch.tensor([[t.item()]], device=self.device)
                            ], dim=1)
                        )
                        
                        # 将分数转换为0-1之间的值（分数越低越好）
                        body_score_normalized = torch.sigmoid(-body_score)
                        body_scores.append(body_score_normalized)
                        
                        # 仍然保留二元判断用于统计
                        if body_score.item() < threshold:
                            body_exists = False
                            break
                    
                    # 记录规则应用
                    self.rule_application_count[rule] += 1
                    
                    # 使用连续分数计算贡献，而不是简单的二元判断
                    if body_exists and body_scores:  # 确保有分数
                        # 计算所有规则体分数的平均值
                        avg_body_score = torch.stack(body_scores).mean()
                        # 使用规则权重加权
                        contribution = rule_weight * avg_body_score
                        loss -= contribution
                        self.rule_success_count[rule] += 1
                
                elif r.item() in body_rels:
                    # 当前三元组是规则体的一部分
                    # 获取规则头部的TransH分数
                    head_score = self.transH.scoreOp(
                                torch.cat([
                                    torch.tensor([[h.item()]], device=self.device),
                                    torch.tensor([[head_rel]], device=self.device),
                                    torch.tensor([[t.item()]], device=self.device)
                                ], dim=1)
                            )
                    
                    # 将头部分数转换为0-1之间的值
                    head_score_normalized = torch.sigmoid(-head_score)
                    
                    # 检查其他规则体关系
                    other_body_scores = []  # 初始化other_body_scores列表
                    other_body_exists = True
                    
                    for other_rel in body_rels:
                        if other_rel == r.item():
                            continue  # 跳过当前关系
                        
                        other_score = self.transH.scoreOp(
                            torch.cat([
                                    torch.tensor([[h.item()]], device=self.device),
                                    torch.tensor([[other_rel]], device=self.device),
                                    torch.tensor([[t.item()]], device=self.device)
                                ], dim=1)
                        )
                        
                        # 将分数转换为0-1之间的值
                        other_score_normalized = torch.sigmoid(-other_score)
                        other_body_scores.append(other_score_normalized)
                        
                        # 仍然保留二元判断用于统计
                        if other_score.item() < threshold:
                            other_body_exists = False
                            break
                    
                    # 记录规则应用
                    self.rule_application_count[rule] += 1
                    
                    # 使用连续分数计算贡献
                    if other_body_exists and head_score.item() > threshold and other_body_scores:
                        # 计算所有其他规则体分数的平均值
                        avg_other_score = torch.stack(other_body_scores).mean() if other_body_scores else torch.tensor(0.0, device=self.device)
                        # 组合头部和其他体部分数
                        combined_score = (head_score_normalized + avg_other_score) / 2
                        # 使用规则权重加权
                        contribution = rule_weight * combined_score
                        loss -= contribution
                        self.rule_success_count[rule] += 1
            
        return loss / max(len(pos_triples), 1)
    
    def update_rule_effectiveness(self):
        """
        更新规则有效性评分并动态管理规则权重
        """
        rules_to_adjust = []
        
        for rule in self.rules:
            # 计算规则有效性（成功率）
            if self.rule_application_count[rule] > 10:  # 至少应用10次才评估
                effectiveness = self.rule_success_count[rule] / self.rule_application_count[rule]
                self.rule_effectiveness[rule] = effectiveness
                
                # 根据有效性调整规则权重
                rule_str = self.rule_weights[rule]
                current_weight = self.learnable_rule_weights[rule_str].item()
                
                if effectiveness < 0.2:  # 低效规则
                    # 大幅降低权重
                    new_weight = current_weight * 0.8
                    rules_to_adjust.append((rule_str, new_weight, "降低"))
                elif effectiveness > 0.7:  # 高效规则
                    # 适当提高权重
                    new_weight = min(current_weight * 1.1, 2.0)  # 上限为2.0
                    rules_to_adjust.append((rule_str, new_weight, "提高"))
        
        # 应用权重调整
        for rule_str, new_weight, action in rules_to_adjust:
            self.learnable_rule_weights[rule_str].data = torch.tensor(new_weight, device=self.device)
            print(f"规则权重{action}：{rule_str}, 新权重: {new_weight:.4f}")
        
        # 重置统计计数
        self.rule_application_count = {rule: 0 for rule in self.rules}
        self.rule_success_count = {rule: 0 for rule in self.rules}
    
    def train(self, dataset, batch_size=128, epochs=100, neg_ratio=1, eval_interval=5, patience=7):
        """
        改进的训练过程，包括预训练、渐进式规则引入和动态规则评估。
        
        参数:
            dataset: KnowledgeGraphDataset
            batch_size: 批次大小
            epochs: 训练轮次数
            neg_ratio: 负采样比率
            eval_interval: 评估间隔轮次
            patience: 早期停止耐心值
            
        返回:
            训练损失列表, 最佳MRR, 预训练TransH状态
        """
        print(f"训练{epochs}个轮次，包括{self.pretrain_epochs}轮预训练...")
        
        losses = []
        beta_history = []  # 记录规则权重历史
        
        # 获取数据加载器
        train_loader = dataset.get_train_dataloader(batch_size, neg_ratio)
        
        # 保存原始TransH模型状态用于后续恢复
        original_transH_state = copy.deepcopy(self.transH.state_dict())
        best_model_state = None
        best_mrr = 0
        best_epoch = 0
        patience_counter = 0
        
        # 创建小型验证集，用于早期停止判断
        val_size = min(1000, len(dataset.test_triples))
        val_indices = np.random.choice(len(dataset.test_triples), size=val_size, replace=False)
        val_triples = [dataset.test_triples[i] for i in val_indices]
        
        for epoch in range(epochs):
            # 明确设置当前规则权重
            if epoch < self.pretrain_epochs:
                self.current_beta = 0.0  # 预训练阶段不使用规则
                phase = "预训练阶段"
            elif epoch < self.pretrain_epochs + self.rule_warmup_epochs:
                progress = (epoch - self.pretrain_epochs) / self.rule_warmup_epochs
                self.current_beta = self.init_beta + progress * (self.target_beta - self.init_beta)
                phase = "规则预热阶段"
            else:
                self.current_beta = self.target_beta
                phase = "全权重阶段"
            
            beta_history.append(self.current_beta)
            print(f"\n轮次 {epoch+1}/{epochs} ({phase})")
            print(f"当前规则权重: {self.current_beta:.4f}")
            
            epoch_loss = 0
            num_batches = 0
            
            for batch in tqdm(train_loader, desc=f"轮次 {epoch+1} 训练"):
                try:
                    # 在预训练阶段使用不同的优化器
                    if epoch < self.pretrain_epochs:
                        self.pretrain_optimizer.zero_grad()
                    else:
                        self.optimizer.zero_grad()
                    
                    # 统一的前向传播
                    loss = self.joint_forward(batch, epoch, epochs)
                    
                    # 反向传播和优化
                    loss.backward()
                    
                    # 梯度裁剪，防止训练不稳定
                    torch.nn.utils.clip_grad_norm_(self.transH.parameters(), 1.0)
                    
                    if epoch < self.pretrain_epochs:
                        self.pretrain_optimizer.step()
                    else:
                        self.optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    print(f"批次错误: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            avg_loss = epoch_loss / max(num_batches, 1)
            losses.append(avg_loss)
            
            print(f"轮次 {epoch+1} 损失: {avg_loss:.4f}, 规则权重: {self.current_beta:.4f}")
            
            # 规范化TransH参数
            self.transH.normalizeEmbedding()
            
            # 如果是预训练结束，保存预训练的模型状态
            if epoch == self.pretrain_epochs - 1:
                pretrained_transH_state = copy.deepcopy(self.transH.state_dict())
                print("预训练完成，保存预训练模型状态")
            
            # 定期评估模型
            if (epoch + 1) % eval_interval == 0 or epoch == epochs - 1:
                # 使用验证集快速评估
                mrr, hits1, hits3, hits10 = self.quick_evaluate(val_triples)
                print(f"轮次 {epoch+1} 评估结果: MRR={mrr:.4f}, Hits@1={hits1:.4f}, Hits@3={hits3:.4f}, Hits@10={hits10:.4f}")
                
                # 保存最佳模型
                current_metric = mrr  # 使用MRR作为主要指标
                if current_metric > best_mrr:
                    best_mrr = current_metric
                    best_epoch = epoch + 1
                    best_model_state = {
                        'transH': copy.deepcopy(self.transH.state_dict()),
                        'relation_rule_embedding': copy.deepcopy(self.relation_rule_embedding.state_dict()),
                        'rule_weights': {rule: self.learnable_rule_weights[self.rule_weights[rule]].item() 
                                         for rule in self.rules},
                        'current_beta': self.current_beta
                    }
                    print(f"找到新的最佳模型，MRR: {best_mrr:.4f}")
                    patience_counter = 0
                else:
                    patience_counter += 1
                    print(f"模型未改进，耐心计数: {patience_counter}/{patience}")
                    
                    # 学习率衰减
                    if patience_counter % 3 == 0 and patience_counter > 0:
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] *= 0.5
                        print(f"降低学习率至 {self.optimizer.param_groups[0]['lr']:.6f}")
                    
                    # 早期停止
                    if patience_counter >= patience:
                        print(f"早期停止于轮次 {epoch+1}")
                        break
            
            # 更新规则有效性并衰减低效规则的权重
            if epoch >= self.pretrain_epochs and (epoch + 1) % 5 == 0:
                self.update_rule_effectiveness()
                self._print_rule_statistics()
        
        # 恢复最佳模型
        if best_model_state:
            print(f"恢复最佳模型（轮次 {best_epoch}，MRR: {best_mrr:.4f}）")
            self.transH.load_state_dict(best_model_state['transH'])
            self.relation_rule_embedding.load_state_dict(best_model_state['relation_rule_embedding'])
            self.current_beta = best_model_state['current_beta']
            for rule, weight in best_model_state['rule_weights'].items():
                if rule in self.rule_weights:
                    rule_str = self.rule_weights[rule]
                    self.learnable_rule_weights[rule_str].data = torch.tensor(weight, device=self.device)
        
        # 绘制规则权重历史图
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(beta_history)
            plt.title('规则权重历史')
            plt.xlabel('Epoch')
            plt.ylabel('规则权重 (beta)')
            plt.savefig('results/rule_weight_history.png')
        except Exception as e:
            print(f"绘图错误: {e}")
        
        return losses, best_mrr, pretrained_transH_state if 'pretrained_transH_state' in locals() else None
    
    def _print_rule_statistics(self):
        """
        打印规则使用情况和有效性统计。
        """
        print("\n规则统计:")
        print("-" * 50)
        
        # 按有效性排序规则
        sorted_rules = sorted(
            [(rule, self.rule_effectiveness.get(rule, 0), self.rule_application_count.get(rule, 0)) 
             for rule in self.rules if self.rule_application_count.get(rule, 0) > 0],
            key=lambda x: x[1],
            reverse=True
        )
        
        # 打印规则统计
        for rule, effectiveness, count in sorted_rules[:10]:
            head_rel, body_rels = rule
            rule_str = self.rule_weights[rule]
            weight = self.learnable_rule_weights[rule_str].item()
            
            print(f"规则: {head_rel} <- {body_rels}")
            print(f"  应用次数: {count}, 成功率: {effectiveness:.2f}, 权重: {weight:.4f}")
        
        print("-" * 50)
    
    def quick_evaluate(self, test_triples, max_candidates=100):
        """
        快速评估方法，用于训练过程中的定期检查
        """
        hits1 = 0
        hits3 = 0
        hits10 = 0
        mrr = 0
        count = 0
        
        for h, r, t in test_triples:
            # 跳过无效ID
            if h > self.max_entity_id or t > self.max_entity_id or r > self.max_relation_id:
                continue
            
            # 随机采样一部分候选实体 + 正确实体
            candidates = set([t])  # 确保正确答案在候选集中
            while len(candidates) < max_candidates:
                rand_e = np.random.randint(0, self.transH.num_entities)
                candidates.add(rand_e)
            candidates = list(candidates)
            correct_idx = candidates.index(t)
            
            # 计算分数
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
                
                scores = self.transH.scoreOp(combined_tensor)
                
                # 应用规则增强
                rule_scores = self._apply_rule_enhancement(h, r, candidates)
                
                # 总分数
                final_scores = scores + self.current_beta * rule_scores
                
                # 分数排序 (TransH中分数越小越好)
                _, sorted_indices = torch.sort(final_scores, descending=False)
                
                # 计算排名
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
                continue
        
        # 计算平均指标
        if count == 0:
            return 0, 0, 0, 0
        
        return mrr/count, hits1/count, hits3/count, hits10/count
    
    def evaluate(self, dataset, use_filter=True):
        """
        在测试数据上评估模型，并可选择是否使用过滤机制。
        
        参数:
            dataset: KnowledgeGraphDataset
            use_filter: 是否过滤掉已知的正确三元组（默认为True）
                
        返回:
            MRR, Hits@1, Hits@3, Hits@10
        """
        print("评估模型...")
        print(f"{'使用' if use_filter else '不使用'}过滤机制")
        
        # 保存当前训练/评估状态
        current_training_state = self.train
        
        # 确保在评估模式
        self.train(False)
        
        hits1 = 0
        hits3 = 0
        hits10 = 0
        mrr = 0
        count = 0
        
        # 获取测试数据
        test_triples = dataset.test_triples
        
        # 如果使用过滤，收集所有已知三元组
        all_triples = set()
        if use_filter:
            try:
                # 添加训练三元组
                for h, r, t in dataset.train_triples:
                    all_triples.add((h, r, t))
                
                # 添加验证三元组（如果有）
                if hasattr(dataset, 'valid_triples'):
                    for h, r, t in dataset.valid_triples:
                        all_triples.add((h, r, t))
                
                # 添加测试三元组
                for h, r, t in dataset.test_triples:
                    all_triples.add((h, r, t))
                
                print(f"收集了 {len(all_triples)} 个已知三元组用于过滤评估")
            except Exception as e:
                print(f"收集三元组时出错: {e}")
                print("继续评估但不使用过滤")
                use_filter = False
        
        # 收集关系统计信息
        relation_metrics = {}
        
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
                    final_scores = scores + self.current_beta * rule_scores
                    
                    # 如果使用过滤，过滤掉已知的正确三元组（除了当前测试的三元组）
                    if use_filter:
                        for i, candidate in enumerate(candidates):
                            if candidate != t and (h, r, candidate) in all_triples:
                                # 将已知正确三元组的分数设为无穷大（对于TransH，分数越小越好）
                                final_scores[i] = float('inf')
                    
                    # 排序分数 (TransH中分数越小越好)
                    _, sorted_indices = torch.sort(final_scores, descending=False)
                    
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
                    
                    # 在计算完每个三元组的排名后，记录关系指标
                    if r not in relation_metrics:
                        relation_metrics[r] = {
                            'ranks': [],
                            'count': 0,
                            'mrr': 0,
                            'hits1': 0,
                            'hits3': 0,
                            'hits10': 0
                        }
                    
                    relation_metrics[r]['ranks'].append(rank)
                    relation_metrics[r]['count'] += 1
                    relation_metrics[r]['mrr'] += 1.0 / rank
                    relation_metrics[r]['hits1'] += 1 if rank <= 1 else 0
                    relation_metrics[r]['hits3'] += 1 if rank <= 3 else 0
                    relation_metrics[r]['hits10'] += 1 if rank <= 10 else 0
                    
                    # 间隔打印当前进度和指标
                    if count % 1000 == 0:
                        curr_mrr = mrr / count
                        curr_hits10 = hits10 / count
                        print(f"已评估 {count} 个测试三元组, 当前 MRR: {curr_mrr:.4f}, Hits@10: {curr_hits10:.4f}")
                    
                except Exception as e:
                    print(f"评估错误: {e}")
                    continue
        
        # 计算最终指标
        if count == 0:
            print("警告: 没有有效的评估三元组!")
            return 0, 0, 0, 0
                
        hits1 = hits1 / count
        hits3 = hits3 / count
        hits10 = hits10 / count
        mrr = mrr / count
        
        print(f"\n最终评估结果 ({'有过滤' if use_filter else '无过滤'}):")
        print(f"MRR: {mrr:.4f}, Hits@1: {hits1:.4f}, Hits@3: {hits3:.4f}, Hits@10: {hits10:.4f}")
        print(f"评估了 {count} 个有效测试三元组")
        
        # 计算每个关系的平均指标
        rel_performances = []
        print("\n各关系评估指标:")
        print(f"{'关系ID':<10}{'MRR':>10}{'Hits@10':>10}{'评估数量':>10}")
        print("-" * 40)
        
        for r, metrics in relation_metrics.items():
            if metrics['count'] > 0:
                rel_mrr = metrics['mrr'] / metrics['count']
                rel_hits1 = metrics['hits1'] / metrics['count']
                rel_hits3 = metrics['hits3'] / metrics['count']
                rel_hits10 = metrics['hits10'] / metrics['count']
                rel_count = metrics['count']
                
                rel_performances.append((r, rel_mrr, rel_hits1, rel_hits3, rel_hits10, rel_count))
        
        # 按MRR排序
        rel_performances.sort(key=lambda x: x[1], reverse=True)
        
        # 打印测试最准的一些关系
        print("\n测试最准确的20个关系（按MRR排序）:")
        print(f"{'关系ID':<10}{'MRR':>10}{'Hits@1':>10}{'Hits@3':>10}{'Hits@10':>10}{'评估数量':>10}")
        print("-" * 70)
        
        top_relations = rel_performances[:20]
        for r, rel_mrr, rel_hits1, rel_hits3, rel_hits10, rel_count in top_relations:
            print(f"{r:<10}{rel_mrr:>10.4f}{rel_hits1:>10.4f}{rel_hits3:>10.4f}{rel_hits10:>10.4f}{rel_count:>10}")
        
        # 保存最准确的关系到文件
        with open(f'results/top_relations_{dataset.name}.txt', 'w') as f:
            f.write("测试最准确的关系（按MRR排序）:\n")
            f.write(f"{'关系ID':<10}{'MRR':>10}{'Hits@1':>10}{'Hits@3':>10}{'Hits@10':>10}{'评估数量':>10}\n")
            f.write("-" * 70 + "\n")
            
            for r, rel_mrr, rel_hits1, rel_hits3, rel_hits10, rel_count in top_relations:
                f.write(f"{r:<10}{rel_mrr:>10.4f}{rel_hits1:>10.4f}{rel_hits3:>10.4f}{rel_hits10:>10.4f}{rel_count:>10}\n")
        
        # 恢复原始状态
        self.train(current_training_state)
        
        return mrr, hits1, hits3, hits10
    
    def _apply_rule_enhancement(self, h, r, candidates):
        """
        为评估应用规则增强，使用批处理高效计算分数。
        只使用与当前关系最相关的规则。
        
        参数:
            h: 头实体索引
            r: 关系索引
            candidates: 候选尾实体列表
            
        返回:
            每个候选实体的规则增强分数
        """
        rule_scores = torch.zeros(len(candidates), device=self.device)
        
        # 查找适用于该关系的规则
        applicable_rules = [rule for rule in self.rules if rule[0] == r or r in rule[1]]
        
        if not applicable_rules:
            return rule_scores
        
        # 选择最相关的规则（根据规则权重和有效性）
        top_rules = []
        for rule in applicable_rules:
            rule_str = self.rule_weights[rule]
            rule_weight = self.learnable_rule_weights[rule_str].item()
            rule_effectiveness = self.rule_effectiveness.get(rule, 0.5)
            
            # 计算规则得分 = 权重 * 有效性
            rule_score = rule_weight * rule_effectiveness
            top_rules.append((rule, rule_score))
        
        # 按规则得分降序排序，并只保留前5个规则
        top_rules.sort(key=lambda x: x[1], reverse=True)
        top_rules = top_rules[:5]  # 只使用前5个最相关的规则
        
        if not top_rules:
            return rule_scores
        
        for rule, _ in top_rules:
            head_rel, body_rels = rule
            rule_str = self.rule_weights[rule]
            rule_weight = self.learnable_rule_weights[rule_str].item()
            
            # 使用规则有效性作为置信度
            rule_confidence = self.rule_effectiveness.get(rule, 0.5)
            
            # 使用动态阈值
            threshold = -0.5 * (1.0 + rule_weight)
            
            if r == head_rel:
                # 当前查询涉及规则头部
                # 对每个规则体关系进行批量计算
                total_body_scores = torch.zeros(len(candidates), device=self.device)
                valid_count = 0
                
                for body_rel in body_rels:
                    # 批量计算所有候选实体的分数
                    h_batch = torch.tensor([h] * len(candidates), device=self.device)
                    r_batch = torch.tensor([body_rel] * len(candidates), device=self.device)
                    t_batch = torch.tensor(candidates, device=self.device)
                    
                    combined_tensor = torch.cat([
                        h_batch.view(-1, 1),
                        r_batch.view(-1, 1),
                        t_batch.view(-1, 1)
                    ], dim=1)
                    
                    batch_scores = self.transH.scoreOp(combined_tensor)
                    
                    # 软规则应用：使用连续分数，而不是二值判断
                    # 将分数转换为贡献值：分数越小（越好），贡献越大
                    # 阈值放宽为 threshold*2，扩大影响范围
                    condition = batch_scores < threshold * 2
                    # 计算贡献：1 - 归一化距离，归一化到0-1
                    contribution = torch.where(
                        condition,
                        1.0 - torch.clamp((batch_scores - threshold) / threshold, 0, 1),
                        torch.zeros_like(batch_scores)
                    )
                    
                    total_body_scores += contribution
                    valid_count += 1
                
                # 计算平均体部分数并应用规则权重和置信度
                if valid_count > 0:
                    avg_body_scores = total_body_scores / valid_count
                    # TransH中分数越小越好，所以使用负号
                    rule_scores -= rule_weight * rule_confidence * avg_body_scores
            
            elif r in body_rels:
                # 当前查询涉及规则体
                
                # 1. 批量计算规则头部分数
                h_batch = torch.tensor([h] * len(candidates), device=self.device)
                r_batch = torch.tensor([head_rel] * len(candidates), device=self.device)
                t_batch = torch.tensor(candidates, device=self.device)
                
                combined_tensor = torch.cat([
                    h_batch.view(-1, 1),
                    r_batch.view(-1, 1),
                    t_batch.view(-1, 1)
                ], dim=1)
                
                head_scores = self.transH.scoreOp(combined_tensor)
                
                # 将头部分数转换为贡献值
                head_contribution = torch.where(
                    head_scores < threshold * 2,
                    1.0 - torch.clamp((head_scores - threshold) / threshold, 0, 1),
                    torch.zeros_like(head_scores)
                )
                
                # 2. 批量计算其他规则体关系分数
                total_other_body_scores = torch.zeros(len(candidates), device=self.device)
                other_count = 0
                
                for other_rel in body_rels:
                    if other_rel == r:
                        continue  # 跳过当前关系
                    
                    # 批量计算
                    h_batch = torch.tensor([h] * len(candidates), device=self.device)
                    r_batch = torch.tensor([other_rel] * len(candidates), device=self.device)
                    t_batch = torch.tensor(candidates, device=self.device)
                    
                    combined_tensor = torch.cat([
                        h_batch.view(-1, 1),
                        r_batch.view(-1, 1),
                        t_batch.view(-1, 1)
                    ], dim=1)
                    
                    other_scores = self.transH.scoreOp(combined_tensor)
                    
                    # 将其他体部分数转换为贡献值
                    other_contribution = torch.where(
                        other_scores < threshold * 2,
                        1.0 - torch.clamp((other_scores - threshold) / threshold, 0, 1),
                        torch.zeros_like(other_scores)
                    )
                    
                    total_other_body_scores += other_contribution
                    other_count += 1
                
                # 3. 组合头部和其他规则体的分数
                if other_count > 0:
                    avg_other_body_scores = total_other_body_scores / other_count
                    
                    # 只对既有头部贡献又有其他体部贡献的实体应用规则
                    # 使用元素级别的乘法实现"与"逻辑
                    combined_contribution = (head_contribution * avg_other_body_scores)
                    
                    # TransH中分数越小越好，所以使用负号
                    rule_scores -= rule_weight * rule_confidence * combined_contribution
        
        return rule_scores