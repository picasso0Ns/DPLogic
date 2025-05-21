import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import copy

from models.transH import TransH
from models.relation_rule_embedding import RelationRuleEmbedding
from models.predicate_logic import PredicateLogic
from models.SimpleRuleEnhancedTransH import SimpleRuleEnhancedTransH
# from optimization.joint_optimization import JointOptimization
# # 导入新的改进联合优化类
from optimization.joint_optimization import ImprovedJointOptimization
from data_utils.dataset import KnowledgeGraphDataset
from utils import create_dirs, save_model, save_results, set_seed

# 修改后的评估函数 - 只打印每5000个样本的性能
def evaluate_transH(model, dataset, device, test_triples=None, batch_size=128, sample_size=None):
    model.eval()
    
    # 获取测试三元组
    if test_triples is None:
        try:
            test_loader = dataset.get_test_dataloader(batch_size=1)
            test_triples = []
            for batch in test_loader:
                if isinstance(batch, tuple) and len(batch) >= 1:
                    test_triples.append(batch[0])
                else:
                    test_triples.append(batch)
            test_triples = torch.cat(test_triples, dim=0).to(device)
        except (AttributeError, TypeError):
            try:
                test_triples = dataset.test_triples
                if not isinstance(test_triples, torch.Tensor):
                    test_triples = torch.tensor(test_triples, dtype=torch.long)
                test_triples = test_triples.to(device)
            except (AttributeError, TypeError):
                print("错误: 无法访问测试三元组")
                return None
    
    # 如果指定了样本大小，随机采样测试三元组
    if sample_size is not None and sample_size < len(test_triples):
        indices = torch.randperm(len(test_triples))[:sample_size]
        test_triples = test_triples[indices]
    
    # 获取所有已知三元组，用于过滤设置
    all_triples = set()
    try:
        # 添加训练三元组
        for h, r, t in dataset.train_triples:
            all_triples.add((h, r, t))
        
        # 添加测试三元组
        for h, r, t in dataset.test_triples:
            all_triples.add((h, r, t))
    except:
        pass
    
    # 初始化指标
    ranks = []
    filtered_ranks = []
    
    # 调试统计信息
    score_stats = {
        'true_scores': [],
        'head_mean_scores': [],
        'tail_mean_scores': [],
        'head_filtered_count': [],
        'tail_filtered_count': []
    }
    
    with torch.no_grad():
        for i, triple in enumerate(test_triples):            
            h, r, t = triple
            
            # 确保索引在有效范围内
            h_idx = h.item() if isinstance(h, torch.Tensor) else h
            r_idx = r.item() if isinstance(r, torch.Tensor) else r
            t_idx = t.item() if isinstance(t, torch.Tensor) else t
            
            h_idx = min(max(0, h_idx), dataset.num_entities - 1)
            r_idx = min(max(0, r_idx), dataset.num_relations - 1)
            t_idx = min(max(0, t_idx), dataset.num_entities - 1)
            
            # 转换为张量
            h = torch.tensor(h_idx, device=device)
            r = torch.tensor(r_idx, device=device)
            t = torch.tensor(t_idx, device=device)
            
            # 构建真实三元组
            true_triple = torch.tensor([[h_idx, r_idx, t_idx]], device=device)
            true_score = model.scoreOp(true_triple)
            score_stats['true_scores'].append(true_score.item())
            
            # --- 替换头实体 ---
            head_scores = []
            
            # 分批处理所有实体，避免内存问题
            for start_idx in range(0, dataset.num_entities, batch_size):
                end_idx = min(start_idx + batch_size, dataset.num_entities)
                curr_batch_size = end_idx - start_idx
                
                # 创建替换头实体的批量三元组
                head_batch = torch.zeros((curr_batch_size, 3), dtype=torch.long, device=device)
                head_batch[:, 0] = torch.arange(start_idx, end_idx, device=device)
                head_batch[:, 1] = r
                head_batch[:, 2] = t
                
                # 计算分数
                batch_scores = model.scoreOp(head_batch)
                head_scores.extend(batch_scores.tolist())
            
            head_scores = np.array(head_scores)
            score_stats['head_mean_scores'].append(np.mean(head_scores))
            
            # 计算头部排名 (raw) - TransH中分数是距离，较小的值更好
            head_rank_raw = np.sum(head_scores < true_score.item()) + 1
            
            # 计算头部排名 (filtered) - 过滤掉已知正确三元组
            head_scores_filt = head_scores.copy()
            filter_count = 0
            if all_triples:
                for e in range(dataset.num_entities):
                    if e != h_idx and (e, r_idx, t_idx) in all_triples:
                        head_scores_filt[e] = float('inf')  # 将已知正确三元组设为无穷大
                        filter_count += 1
            
            score_stats['head_filtered_count'].append(filter_count)
            head_rank_filt = np.sum(head_scores_filt < true_score.item()) + 1
            
            # --- 替换尾实体 ---
            tail_scores = []
            
            # 分批处理所有实体
            for start_idx in range(0, dataset.num_entities, batch_size):
                end_idx = min(start_idx + batch_size, dataset.num_entities)
                curr_batch_size = end_idx - start_idx
                
                # 创建替换尾实体的批量三元组
                tail_batch = torch.zeros((curr_batch_size, 3), dtype=torch.long, device=device)
                tail_batch[:, 0] = h
                tail_batch[:, 1] = r
                tail_batch[:, 2] = torch.arange(start_idx, end_idx, device=device)
                
                # 计算分数
                batch_scores = model.scoreOp(tail_batch)
                tail_scores.extend(batch_scores.tolist())
            
            tail_scores = np.array(tail_scores)
            score_stats['tail_mean_scores'].append(np.mean(tail_scores))
            
            # 计算尾部排名 (raw)
            tail_rank_raw = np.sum(tail_scores < true_score.item()) + 1
            
            # 计算尾部排名 (filtered)
            tail_scores_filt = tail_scores.copy()
            filter_count = 0
            if all_triples:
                for e in range(dataset.num_entities):
                    if e != t_idx and (h_idx, r_idx, e) in all_triples:
                        tail_scores_filt[e] = float('inf')
                        filter_count += 1
            
            score_stats['tail_filtered_count'].append(filter_count)
            tail_rank_filt = np.sum(tail_scores_filt < true_score.item()) + 1
            
            # 将排名添加到列表中
            ranks.extend([head_rank_raw, tail_rank_raw])
            filtered_ranks.extend([head_rank_filt, tail_rank_filt])
            
            # 只在每5000个三元组处打印进度
            if i % 5000 == 0 and i > 0:
                mean_rank = np.mean(filtered_ranks)
                mean_mrr = np.mean(1.0/np.array(filtered_ranks))
                print(f"已评估 {i}/{len(test_triples)} 个测试三元组, Mean Rank: {mean_rank:.2f}, MRR: {mean_mrr:.4f}")
    
    # 计算指标
    if not filtered_ranks:
        print("警告: 评估过程中没有计算出有效的排名。")
        return None
    
    # 使用过滤后的排名计算指标
    ranks_np = np.array(filtered_ranks)
    mrr = np.mean(1.0 / ranks_np)
    hits1 = np.mean(ranks_np <= 1)
    hits3 = np.mean(ranks_np <= 3)
    hits10 = np.mean(ranks_np <= 10)
    
    # 打印结果
    print(f"\n最终结果:")
    print(f"MRR: {mrr:.4f}, Hits@1: {hits1:.4f}, Hits@3: {hits3:.4f}, Hits@10: {hits10:.4f}")
    print(f"Mean Rank: {np.mean(filtered_ranks):.2f}")
    
    model.train()  # 将模型设回训练模式
    
    return {
        'MRR': mrr,
        'Hits@1': hits1,
        'Hits@3': hits3,
        'Hits@10': hits10,
        'Mean Rank': np.mean(filtered_ranks),
        'Raw Mean Rank': np.mean(ranks)
    }

def parse_args():
    parser = argparse.ArgumentParser(description='Train and evaluate the model')
    
    # Dataset and model parameters
    parser.add_argument('--dataset', type=str, default='family', 
                        choices=['family', 'kinship', 'UMLS', 'FB15k237', 'WN18rr'],
                        help='Dataset name')
    parser.add_argument('--dim', type=int, default=200, 
                        help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=300, 
                        help='Hidden dimension for logical operator')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10, 
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01, 
                        help='Learning rate')
    parser.add_argument('--margin', type=float, default=1.0, 
                        help='Margin for TransH loss')
    parser.add_argument('--neg_ratio', type=int, default=10, 
                        help='Negative sampling ratio')
    
    # Rule mining parameters
    parser.add_argument('--rule_threshold', type=float, default=0.2, 
                        help='Confidence threshold for rule selection')
    parser.add_argument('--max_rule_length', type=int, default=3, 
                        help='Maximum length of rules to mine')
    
    # Loss weights
    parser.add_argument('--alpha', type=float, default=1.0, 
                        help='Weight for representation loss')
    parser.add_argument('--beta', type=float, default=0.5, 
                        help='Weight for distribution loss')
    parser.add_argument('--gamma', type=float, default=0.3, 
                        help='Weight for weight loss')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=5, 
                        help='Random seed')
    parser.add_argument('--gpu', type=int, default=0, 
                        help='GPU ID, -1 for CPU')
    parser.add_argument('--save_model', action='store_true', 
                        help='Save model after training')
    
    # TransH specific parameters
    parser.add_argument('--L', type=int, default=2, choices=[1, 2],
                        help='L norm for TransH distance')
    parser.add_argument('--C', type=float, default=1.0,
                        help='Parameter C for TransH regularization')
    parser.add_argument('--eps', type=float, default=0.01,
                        help='Epsilon for orthogonality constraint in TransH')
    
    # Model type parameter
    parser.add_argument('--model_type', type=str, default='joint', 
                    choices=['transH', 'joint', 'simple'],
                    help='Model type: TransH, Joint Optimization, or Simple Rule Enhanced')
    parser.add_argument('--rule_weight', type=float, default=0.5, 
                    help='Weight for rule loss in SimpleRuleEnhancedTransH')
    
    # Early stopping parameters
    parser.add_argument('--patience', type=int, default=5,
                    help='Patience for early stopping')
    parser.add_argument('--eval_interval', type=int, default=10,
                    help='Epochs between evaluations')
    
    # Evaluation parameters
    parser.add_argument('--eval_sample_size', type=int, default=None,
                    help='Number of test samples to use for intermediate evaluations')
    
    # 改进的联合优化参数
    parser.add_argument('--improved_joint', action='store_true',
                    help='使用改进的联合优化')
    parser.add_argument('--pretrain_epochs', type=int, default=20, 
                    help='预训练轮次数')
    parser.add_argument('--rule_warmup_epochs', type=int, default=10, 
                    help='规则权重预热轮次数')
    parser.add_argument('--initial_rule_weight', type=float, default=0.01, 
                    help='初始规则权重')
    parser.add_argument('--target_rule_weight', type=float, default=1.0, 
                    help='目标规则权重')
    parser.add_argument('--rule_quality_threshold', type=float, default=0.3, 
                    help='规则质量阈值')
    parser.add_argument('--early_stopping_patience', type=int, default=7, 
                    help='早期停止耐心值')
    
    return parser.parse_args()

def train_transH_with_early_stopping(model, dataset, optimizer, args, device):
    """带早停机制的TransH训练函数"""
    train_loader = dataset.get_train_dataloader(args.batch_size, args.neg_ratio)
    
    losses = []
    best_mrr = 0
    patience_counter = 0
    best_model_state = None
    best_metrics = None  # 添加变量存储最佳性能指标
    
    for epoch in range(args.epochs):
        epoch_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            pos_triples, neg_triples = batch
            
            # 检查无效索引并修复
            pos_triples = pos_triples.to(device)
            neg_triples = neg_triples.to(device)
            
            # 将索引限制在有效范围内
            pos_triples[:, 0].clamp_(0, dataset.num_entities - 1)
            pos_triples[:, 2].clamp_(0, dataset.num_entities - 1)
            pos_triples[:, 1].clamp_(0, dataset.num_relations - 1)
            
            neg_triples[:, 0].clamp_(0, dataset.num_entities - 1)
            neg_triples[:, 2].clamp_(0, dataset.num_entities - 1)
            neg_triples[:, 1].clamp_(0, dataset.num_relations - 1)
            
            # 处理批量大小不匹配
            pos_batch_size = pos_triples.size(0)
            neg_batch_size = neg_triples.size(0)
            
            # 当每个正样本有多个负样本时，需要扩展正样本
            if neg_batch_size > pos_batch_size:
                neg_ratio = neg_batch_size // pos_batch_size
                # 重复每个正三元组 neg_ratio 次
                expanded_pos_triples = pos_triples.repeat_interleave(neg_ratio, dim=0)
                
                optimizer.zero_grad()
                loss = model(expanded_pos_triples, neg_triples)
            else:
                # 正负样本 1:1 关系的情况
                optimizer.zero_grad()
                loss = model(pos_triples, neg_triples)
            
            # 反向传播与优化
            loss.backward()
            
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            # 归一化TransH参数
            model.normalizeEmbedding()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / max(num_batches, 1)
        losses.append(avg_loss)
        
        # 打印当前epoch的损失
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
        
        # 阶段性评估
        if (epoch + 1) % args.eval_interval == 0 or (epoch + 1) == args.epochs:
            print(f"在Epoch {epoch+1}进行评估...")
            
            # 使用较小的样本集进行快速评估
            metrics = evaluate_transH(
                model, 
                dataset, 
                device, 
                sample_size=args.eval_sample_size
            )
            
            if metrics and metrics['MRR'] > best_mrr:
                best_mrr = metrics['MRR']
                best_metrics = metrics  # 保存所有最佳指标
                patience_counter = 0
                # 保存最佳模型状态
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                print(f"找到新的最佳MRR: {best_mrr:.4f}")
            else:
                patience_counter += 1
                print(f"没有改进。耐心值: {patience_counter}/{args.patience}")
                
                if patience_counter >= args.patience:
                    print(f"早停于Epoch {epoch+1}")
                    # 恢复最佳模型
                    if best_model_state:
                        model.load_state_dict(best_model_state)
                    break
    
    # 如果有最佳模型状态，恢复到最佳模型
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # 打印训练过程中获得的最佳性能
    if best_metrics:
        print("\n===== 训练期间最佳性能 =====")
        print(f"最佳 MRR: {best_metrics['MRR']:.4f}")
        print(f"最佳 Hits@1: {best_metrics['Hits@1']:.4f}")
        print(f"最佳 Hits@3: {best_metrics['Hits@3']:.4f}")
        print(f"最佳 Hits@10: {best_metrics['Hits@10']:.4f}")
        print(f"最佳 Mean Rank: {best_metrics['Mean Rank']:.2f}")
    
    return losses, model, best_metrics

def train_and_evaluate(args):
    # 创建目录
    create_dirs()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据集
    print(f"加载 {args.dataset} 数据集...")
    dataset = KnowledgeGraphDataset(args.dataset)
    
    # 检查数据集中的最大ID
    train_triples = torch.tensor(dataset.train_triples)
    max_entity_id = max(train_triples[:, 0].max().item(), train_triples[:, 2].max().item())
    max_relation_id = train_triples[:, 1].max().item()
    
    # 确保实体和关系数量足够
    if max_entity_id >= dataset.num_entities:
        dataset.num_entities = max_entity_id + 1
        
    if max_relation_id >= dataset.num_relations:
        dataset.num_relations = max_relation_id + 1
    
    # 初始化 TransH 模型
    transH = TransH(
        entityNum=dataset.num_entities,
        relationNum=dataset.num_relations,
        embeddingDim=args.dim,
        margin=args.margin,
        L=args.L,
        C=args.C,
        eps=args.eps
    ).to(device)
    
    # 根据模型类型选择模型
    if args.model_type == 'transH':
        # 只使用 TransH 模型
        model = transH
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        # 使用早停机制训练模型
        losses, model = train_transH_with_early_stopping(
            model=model,
            dataset=dataset,
            optimizer=optimizer,
            args=args,
            device=device
        )
        
    elif args.model_type == 'simple':
        # 使用简单规则增强的 TransH 模型
        model = SimpleRuleEnhancedTransH(
            transH=transH,
            dataset=dataset,
            rule_weight=args.rule_weight
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        # 使用早停机制训练模型
        losses, model = train_transH_with_early_stopping(
            model=model,
            dataset=dataset,
            optimizer=optimizer,
            args=args,
            device=device
        )
        
    else:  # 'joint'
        # 使用联合优化
        predicate_logic = PredicateLogic(
            dataset=dataset,
            rule_threshold=args.rule_threshold,
            max_rule_length=args.max_rule_length
        )
        
        relation_rule_embedding = RelationRuleEmbedding(
            transH=transH,
            dim=args.dim,
            hidden_dim=args.hidden_dim
        ).to(device)
        
        # 根据参数选择使用改进的联合优化或原始联合优化
        if args.improved_joint:
            print("使用改进的联合优化...")
            # 使用改进的联合优化器
            joint_optimizer = ImprovedJointOptimization(
                transH=transH,
                relation_rule_embedding=relation_rule_embedding,
                predicate_logic=predicate_logic,
                alpha=args.alpha,
                beta=args.initial_rule_weight,
                gamma=args.gamma,
                lr=args.lr,
                pretrain_epochs=args.pretrain_epochs,
                rule_warmup_epochs=args.rule_warmup_epochs,
                rule_quality_threshold=args.rule_quality_threshold,
                rule_decay_rate=0.9
            )
            
            # 使用改进的联合优化进行训练
            try:
                losses, best_mrr, pretrained_transH = joint_optimizer.train(
                    dataset=dataset,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    neg_ratio=args.neg_ratio,
                    eval_interval=args.eval_interval,
                    patience=args.early_stopping_patience
                )
                
                # 比较预训练TransH和联合优化效果
                if pretrained_transH:
                    print("\n===== 比较预训练TransH和联合优化效果 =====")
                    # 临时保存当前模型状态
                    current_state = copy.deepcopy(transH.state_dict())
                    
                    # 恢复预训练状态进行评估
                    transH.load_state_dict(pretrained_transH)
                    pretrain_metrics = evaluate_transH(transH, dataset, device)
                    
                    if pretrain_metrics:
                        print("预训练TransH效果:")
                        print(f"MRR: {pretrain_metrics['MRR']:.4f}")
                        print(f"Hits@1: {pretrain_metrics['Hits@1']:.4f}")
                        print(f"Hits@3: {pretrain_metrics['Hits@3']:.4f}")
                        print(f"Hits@10: {pretrain_metrics['Hits@10']:.4f}")
                    
                    # 恢复联合优化后的状态
                    transH.load_state_dict(current_state)
                    
                    print("\n联合优化效果:")
                    print(f"MRR: {best_mrr:.4f}")
                    
                    # 计算相对提升
                    if pretrain_metrics:
                        pretrain_mrr = pretrain_metrics['MRR']
                        improvement = ((best_mrr - pretrain_mrr) / pretrain_mrr) * 100 if pretrain_mrr > 0 else 0
                        print(f"相对提升: {improvement:.2f}%")
                
            except Exception as e:
                print(f"改进的联合优化训练失败，错误: {str(e)}")
                import traceback
                traceback.print_exc()
                return
        else:
            print("使用原始联合优化...")
            # 使用原始联合优化
            joint_optimizer = JointOptimization(
                transH=transH,
                relation_rule_embedding=relation_rule_embedding,
                predicate_logic=predicate_logic,
                alpha=args.alpha,
                beta=args.beta,
                gamma=args.gamma,
                lr=args.lr
            )
            
            # 使用联合优化进行训练
            try:
                losses = joint_optimizer.train(
                    dataset=dataset,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    neg_ratio=args.neg_ratio
                )
            except Exception as e:
                print(f"联合优化训练失败，错误: {str(e)}")
                import traceback
                traceback.print_exc()
                return
    
    # 绘制训练损失
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title(f'训练损失 - {args.dataset} - {args.model_type}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(f'results/{args.dataset}_{args.model_type}_loss.png')
    except Exception as e:
        print(f"无法保存损失图: {str(e)}")
    
    # 最终评估
    print("最终评估...")
    try:
        if args.model_type in ['transH', 'simple']:
            # 使用新的评估函数
            metrics = evaluate_transH(model, dataset, device)
            
            if metrics:
                print(f"在 {args.dataset} 上使用 {args.model_type} 的最终结果:")
                print(f"MRR: {metrics['MRR']:.4f}")
                print(f"Hits@1: {metrics['Hits@1']:.4f}")
                print(f"Hits@3: {metrics['Hits@3']:.4f}")
                print(f"Hits@10: {metrics['Hits@10']:.4f}")
                print(f"Mean Rank: {metrics['Mean Rank']:.2f}")
                
                # 保存结果
                results = {
                    'dataset': args.dataset,
                    'model': args.model_type,
                    'dim': args.dim,
                    'margin': args.margin,
                    'learning_rate': args.lr,
                    'L': args.L,
                    'C': args.C,
                    'metrics': metrics
                }
                save_results(results, f'results/{args.dataset}_{args.model_type}_results.json')
        else:
            # 联合优化的评估
            if args.improved_joint:
                # 使用改进的联合优化评估
                mrr, hits1, hits3, hits10 = joint_optimizer.evaluate(dataset)
            else:
                # 使用原始联合优化评估
                mrr, hits1, hits3, hits10 = joint_optimizer.evaluate(dataset)
            
            # 打印结果
            print(f"在 {args.dataset} 上使用 {args.model_type} 的结果:")
            print(f"MRR: {mrr:.4f}")
            print(f"Hits@1: {hits1:.4f}")
            print(f"Hits@3: {hits3:.4f}")
            print(f"Hits@10: {hits10:.4f}")
            
            # 保存结果
            results = {
                'dataset': args.dataset,
                'model': args.model_type + ('_improved' if args.improved_joint else ''),
                'dim': args.dim,
                'margin': args.margin,
                'learning_rate': args.lr,
                'metrics': {
                    'MRR': mrr,
                    'Hits@1': hits1,
                    'Hits@3': hits3,
                    'Hits@10': hits10
                }
            }
            save_results(results, f'results/{args.dataset}_{args.model_type}_results.json')
    except Exception as e:
        print(f"评估失败: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # 如果需要，保存模型
    if args.save_model:
        try:
            model_suffix = '_improved' if args.model_type == 'joint' and args.improved_joint else ''
            if args.model_type == 'transH':
                save_model(model, f'checkpoints/{args.dataset}_transH.pt')
            elif args.model_type == 'simple':
                save_model(model, f'checkpoints/{args.dataset}_simple_transH.pt')
            else:
                save_model(transH, f'checkpoints/{args.dataset}_transH{model_suffix}.pt')
                save_model(relation_rule_embedding, f'checkpoints/{args.dataset}_relation_rule_embedding{model_suffix}.pt')
        except Exception as e:
            print(f"保存模型失败: {str(e)}")

if __name__ == '__main__':
    args = parse_args()
    train_and_evaluate(args)