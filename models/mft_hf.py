import torch
import torch.nn as nn
import torch.nn.functional as F

"""
MFT-HF: Modality-Aware Fusion Transformer for Heart Failure Prediction
Paper: Modality-Aware Deep Learning Model for Risk Prediction of Stage A Heart Failure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MFTHF(nn.Module):
    def __init__(self, 
                 struct_dim, 
                 icd_embed_dim, 
                 drug_embed_dim, 
                 hidden_dim=192,       # 对齐 config 中的 FUSION_HIDDEN_DIM
                 num_heads=8, 
                 num_classes=1, 
                 dropout_rate=0.2,
                 use_interaction_path=True,   # 消融开关 A
                 use_independence_path=True): # 消融开关 B
        super(MFTHF, self).__init__()
        
        self.use_interaction = use_interaction_path
        self.use_independence = use_independence_path
        self.hidden_dim = hidden_dim

        # 1. Modality-Specific Encoders (对齐论文流程图：Linear -> LayerNorm -> GELU)
        self.struct_enc = nn.Sequential(
            nn.Linear(struct_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        self.icd_enc = nn.Sequential(
            nn.Linear(icd_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

        self.drug_enc = nn.Sequential(
            nn.Linear(drug_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

        # 2. Path A: Cross-Modal Interaction (Equation 1)
        if self.use_interaction:
            self.multihead_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim, 
                num_heads=num_heads,
                dropout=dropout_rate,
                batch_first=True
            )
            
            self.transformer_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, 
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout_rate,
                activation='gelu',
                batch_first=True
            )
            
            # Attention Pooling for Global Context Vector (Equation 2)
            self.attn_pooling = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Softmax(dim=1) 
            )

        # 3. Path B: Modality-Specific Independence (Fig 6A 的基础)
        if self.use_independence:
            # 学习到的 ws, wc, wd
            self.modality_weights = nn.Parameter(torch.ones(3)) 
        
        # 4. Final Classification Head
        # 动态计算输入维度：Interaction 提供 1*d, Independence 提供 3*d
        fusion_dim = 0
        if self.use_interaction: fusion_dim += hidden_dim
        if self.use_independence: fusion_dim += (hidden_dim * 3)
        
        # 如果两条路径都关了（Standard MLP 情况），则直接处理拼接的特征
        if fusion_dim == 0: fusion_dim = hidden_dim * 3

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, num_classes)
        )

    def forward(self, x_s, x_c, x_d, return_attn=False):
        """
        Args:
            return_attn: 如果为 True，则额外返回注意力权重
        """
        # Step 1: Modality-Specific Encoding
        h_s = self.struct_enc(x_s) 
        h_c = self.icd_enc(x_c)    
        h_d = self.drug_enc(x_d)    
        
        fusion_outputs = []

        # --- Path A: Cross-Modal Interaction ---
        attn_weights = None
        if self.use_interaction:
            H = torch.stack([h_s, h_c, h_d], dim=1) # [Batch, 3, d]
            
            # Equation (1): Multi-head Self-Attention

            attn_out, attn_weights = self.multihead_attn(H, H, H)
            H_prime = self.transformer_layer(H + attn_out) 
            
            # Equation (2): Attention Pooling
            alpha = self.attn_pooling(H_prime) # [Batch, 3, 1]
            c_inter = torch.sum(H_prime * alpha, dim=1) # [Batch, d]
            fusion_outputs.append(c_inter)

        # --- Path B: Modality-Specific Independence ---
        if self.use_independence:
            w = F.softmax(self.modality_weights, dim=0) # ws, wc, wd
            h_weighted = torch.cat([
                w[0] * h_s, 
                w[1] * h_c, 
                w[2] * h_d
            ], dim=1) # [Batch, 3*d]
            fusion_outputs.append(h_weighted)

        # --- Final Fusion ---
        if not fusion_outputs: # 特殊情况处理
            f_final = torch.cat([h_s, h_c, h_d], dim=1)
        else:
            f_final = torch.cat(fusion_outputs, dim=1)
        
        logits = self.classifier(f_final)
        
        if return_attn:
            return logits, attn_weights
        return logits

class StandardMLP(nn.Module):
    """Baseline Model: Standard Multi-Layer Perceptron"""
    def __init__(self, input_dim, hidden_dims=[256, 128], output_dim=1, dropout_rate=0.3):
        super().__init__()
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)