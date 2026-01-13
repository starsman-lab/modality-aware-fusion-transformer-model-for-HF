# preprocessing/graph_utils.py
import torch
import numpy as np
import joblib
from torch_geometric.data import HeteroData
from collections import defaultdict
from tqdm import tqdm
from utils import logger, clear_memory
from config import DEVICE, GNN_EPOCHS, GNN_LR

def build_heterogeneous_graph_generic(codes_list_series, node_type_name, edge_type_key_name, min_cooccurrence):
    """
    对应论文：构建共病图拓扑结构。
    将 ICD 编码或药物列表转化为异构图，保留共现次数超过阈值的边。
    """
    logger.info(f"Building heterogeneous graph for {node_type_name}...")
    codes_list = codes_list_series.apply(lambda x: x if isinstance(x, list) else ['UNKNOWN']).tolist()

    all_unique_codes = set()
    for codes in codes_list:
        all_unique_codes.update(c for c in codes if c != 'UNKNOWN')

    if not all_unique_codes:
        return HeteroData(), {}

    code_mapping = {code: i for i, code in enumerate(sorted(list(all_unique_codes)))}
    num_nodes = len(code_mapping)

    hetero_data = HeteroData()
    # 初始特征使用单位矩阵 (Identity Matrix)，对应论文中的初始拓扑编码
    hetero_data[node_type_name].x = torch.eye(num_nodes, dtype=torch.float32)

    # 计算共现矩阵
    cooccurrence_edges = defaultdict(int)
    for codes in codes_list:
        valid_codes = [c for c in codes if c in code_mapping]
        for i in range(len(valid_codes)):
            for j in range(i + 1, len(valid_codes)):
                u, v = code_mapping[valid_codes[i]], code_mapping[valid_codes[j]]
                pair = tuple(sorted((u, v)))
                cooccurrence_edges[pair] += 1

    edge_list_src, edge_list_dst, edge_weights = [], [], []
    for (u, v), count in cooccurrence_edges.items():
        if count >= min_cooccurrence:
            edge_list_src.extend([u, v])
            edge_list_dst.extend([v, u])
            edge_weights.extend([float(count), float(count)])

    edge_type = (node_type_name, f"{edge_type_key_name}_{node_type_name}", node_type_name)
    if edge_list_src:
        hetero_data[edge_type].edge_index = torch.tensor([edge_list_src, edge_list_dst], dtype=torch.long)
        hetero_data[edge_type].edge_weight = torch.tensor(edge_weights, dtype=torch.float32)
        logger.info(f"Graph built: {num_nodes} nodes, {len(edge_list_src)//2} edges.")
    
    return hetero_data, code_mapping

def train_heterognn_generic(model, hetero_data, node_type_name, edge_type_key, num_epochs=GNN_EPOCHS, lr=GNN_LR):
    """
    对应论文：自监督链路预测训练。
    通过学习哪些诊断代码常在一起出现，来获得节点（ICD）的生理表示。
    """
    edge_type = (node_type_name, f"{edge_type_key}_{node_type_name}", node_type_name)
    
    if node_type_name not in hetero_data.node_types or hetero_data[edge_type].num_edges == 0:
        logger.warning(f"No edges for {node_type_name}, skipping training.")
        return model.state_dict()

    model = model.to(DEVICE)
    hetero_data = hetero_data.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = torch.nn.BCEWithLogitsLoss()

    pos_edge_index = hetero_data[edge_type].edge_index
    num_nodes = hetero_data[node_type_name].num_nodes

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # 负采样：随机抽取不存在的边作为负样本
        neg_edge_index = torch.randint(0, num_nodes, (2, pos_edge_index.size(1)), device=DEVICE)

        # Forward
        node_embs_dict = model(hetero_data.x_dict, hetero_data.edge_index_dict)
        embs = node_embs_dict[node_type_name]

        # 计算正样本得分和负样本得分 (内积)
        pos_scores = (embs[pos_edge_index[0]] * embs[pos_edge_index[1]]).sum(dim=-1)
        neg_scores = (embs[neg_edge_index[0]] * embs[neg_edge_index[1]]).sum(dim=-1)

        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])

        loss = criterion(scores, labels)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"GNN {node_type_name} Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    return model.state_dict()

def generate_patient_level_gnn_embeddings_generic(gnn_model, hetero_data, node_mapping, patient_codes_series, 
                                                 node_type_name, out_channels, fallback_embeddings=None):
    """
    对应论文：患者级特征聚合。
    将图学习到的节点 Embedding 聚合为患者层面的特征向量，并处理回退逻辑。
    """
    if gnn_model is None or not node_mapping:
        logger.warning(f"GNN missing for {node_type_name}, using fallback.")
        return fallback_embeddings if fallback_embeddings is not None else np.zeros((len(patient_codes_series), out_channels))

    gnn_model.eval()
    with torch.no_grad():
        node_embs_dict = gnn_model(hetero_data.x_dict.to(DEVICE), hetero_data.edge_index_dict.to(DEVICE))
    
    target_embs = node_embs_dict[node_type_name].cpu().numpy()
    patient_codes = patient_codes_series.apply(lambda x: x if isinstance(x, list) else ['UNKNOWN']).tolist()
    
    final_embs = np.zeros((len(patient_codes), out_channels), dtype=np.float32)
    for i, codes in enumerate(patient_codes):
        valid_idx = [node_mapping[c] for c in codes if c in node_mapping]
        if valid_idx:
            final_embs[i, :] = np.mean(target_embs[valid_idx], axis=0)
        elif fallback_embeddings is not None:
            final_embs[i, :] = fallback_embeddings[i]
            
    return final_embs