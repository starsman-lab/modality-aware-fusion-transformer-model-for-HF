import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, HeteroConv
from utils import logger

class HeteroGNN(torch.nn.Module):
    """
    MFT-HF 核心组件：异构图神经网络编码器。
    
    设计要点：
    1. 自适应线性投影：处理不同规模节点集的初始特征（如 Identity Matrix）。
    2. 异构卷积封装：支持未来扩展多类型节点（如诊断、药物、实验室检查）。
    3. 多头注意力 (GAT)：捕捉共病节点间非均匀的重要性权重。
    4. 残差连接：防止深层图网络中的过度平滑（Over-smoothing）问题。
    """
    def __init__(self, hidden_channels, out_channels, hetero_data,
                 node_type_name='icd',  
                 edge_type_key_name='co_occurs_with'):
        super().__init__()
        
        self.node_type = node_type_name
        # 构造边类型元组，例如: ('icd', 'co_occurs_with_icd', 'icd')
        self.edge_type = (node_type_name, f"{edge_type_key_name}_{node_type_name}", node_type_name)

        logger.info(f"Initializing HeteroGNN for {node_type_name}...")

        # 1. 动态确定输入维度
        # 在医学图谱中，输入维度通常等于节点总数（Identity特征），因此需要线性映射
        try:
            actual_input_dim = hetero_data[node_type_name].x.size(1)
        except Exception:
            logger.error(f"Could not determine input_dim for {node_type_name}. Check hetero_data.")
            actual_input_dim = hidden_channels

        # 线性投射层：将稀疏/高维特征映射到统一的隐空间
        self.lin_dict = nn.ModuleDict({
            node_type_name: nn.Linear(actual_input_dim, hidden_channels)
        })

        # 2. 第一层卷积：多头图注意力网络 (GAT)
        # 使用 2 个 Attention Head，concat=False 取平均以保持维度一致性
        self.conv1 = HeteroConv({
            self.edge_type: GATConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                heads=2, 
                dropout=0.3, 
                concat=False 
            )
        }, aggr='mean')

        # 3. 第二层卷积：输出层
        self.conv2 = HeteroConv({
            self.edge_type: GATConv(
                in_channels=hidden_channels,
                out_channels=out_channels, 
                heads=1, 
                dropout=0.3, 
                concat=False 
            )
        }, aggr='mean')

        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x_dict, edge_index_dict, edge_weight_dict=None):
        """
        Args:
            x_dict: 节点特征字典 {'node_type': Tensor}
            edge_index_dict: 边索引字典
            edge_weight_dict: 边权重字典（包含共现频率信息）
        """
        if self.node_type not in x_dict:
            return {}

        # 第一步：线性投影与初步转换
        x_processed_dict = {}
        h = x_dict[self.node_type]
        h = self.lin_dict[self.node_type](h)
        h = self.dropout(self.act(h))
        x_processed_dict[self.node_type] = h

        # 第二步：Conv1 + 残差连接
        out_conv1_dict = self.conv1(x_processed_dict, edge_index_dict, edge_attr_dict=edge_weight_dict)
        
        h_conv1 = out_conv1_dict[self.node_type]
        # 残差连接：将 Conv1 的输出与投影后的输入相加
        if h_conv1.shape == h.shape:
            h = self.dropout(self.act(h_conv1 + h))
        else:
            h = self.dropout(self.act(h_conv1))
        
        # 准备 Conv2 的输入字典
        x_inter_dict = {self.node_type: h}

        # 第三步：Conv2 (输出层)
        out_conv2_dict = self.conv2(x_inter_dict, edge_index_dict, edge_attr_dict=edge_weight_dict)
        

        return out_conv2_dict
