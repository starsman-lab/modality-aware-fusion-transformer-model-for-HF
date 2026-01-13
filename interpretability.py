import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 导入配置和模型
import config
from utils import logger, load_artifacts
from models.mft_hf import MFTHF
from preprocessing.feature_engineering import process_features
from preprocessing.data_loader import get_train_val_datasets

class MFT_HF_Interpreter:
    def __init__(self, model_name="mft_hf_best"):
        self.model_name = model_name
        self.output_dir = config.FIGURES_DIR / "interpretability"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载数据以获取特征维度
        df_raw = get_train_val_datasets()
        _, self.dims = process_features(df_raw, 'icd_codes', 'drugs', train_mode=False)
        
        # 加载模型
        self.model = MFTHF(
            self.dims['struct'], self.dims['icd'], self.dims['drug'], 
            config.MODEL_PARAMS['hidden_dim']
        ).to(config.DEVICE)
        self.model, _ = load_artifacts(self.model, model_name)
        self.model.eval()

    # =========================================================================
    # 1. 可视化全局模态重要性 (对应论文 Fig 6A)
    # =========================================================================
    def plot_global_modality_importance(self):
        """提取 Path B 中学习到的 ws, wc, wd 权重"""
        logger.info("Extracting global modality weights...")
        with torch.no_grad():
            # 从模型参数中获取原始权重并进行 Softmax 归一化
            raw_weights = self.model.modality_weights
            weights = torch.softmax(raw_weights, dim=0).cpu().numpy()
        
        modality_names = ['Structured', 'ICD (GNN)', 'Drug (Multi-hot)']
        
        plt.figure(figsize=(8, 6))
        sns.barplot(x=modality_names, y=weights, palette="viridis")
        plt.ylabel('Normalized Weight (Importance)')
        plt.title('Global Learned Modality Importance (Fig 6A)')
        plt.ylim(0, 0.5)
        
        # 在柱状图上方标注具体数值
        for i, v in enumerate(weights):
            plt.text(i, v + 0.01, f"{v:.3f}", ha='center', fontweight='bold')
            
        plt.savefig(self.output_dir / "fig6a_global_importance.png", dpi=300)
        logger.info("Saved Fig 6A.")

    # =========================================================================
    # 2. 跨模态注意力热图 (对应论文 Fig 6B)
    # =========================================================================
    def plot_attention_heatmap(self, patient_index=0):
        """
        可视化 Transformer 层的注意力矩阵。
        展示模型在决策时，不同模态之间是如何相互参照的。
        """
        logger.info(f"Generating Attention Heatmap for Patient {patient_index}...")
        
        # 准备一个病人的数据
        df_raw = get_train_val_datasets().iloc[[patient_index]]
        feat_df, _ = process_features(df_raw, 'icd_codes', 'drugs', train_mode=False)
        
        ns, icd, dr = self.dims['struct'], self.dims['icd'], self.dims['drug']
        xs = torch.tensor(feat_df.iloc[:, :ns].values, dtype=torch.float32).to(config.DEVICE)
        xc = torch.tensor(feat_df.iloc[:, ns:ns+icd].values, dtype=torch.float32).to(config.DEVICE)
        xd = torch.tensor(feat_df.iloc[:, ns+icd:].values, dtype=torch.float32).to(config.DEVICE)

        with torch.no_grad():
            # 编码
            h_s = self.model.ln(self.model.struct_enc(xs))
            h_c = self.model.ln(self.model.icd_enc(xc))
            h_d = self.model.ln(self.model.drug_enc(xd))
            H = torch.stack([h_s, h_c, h_d], dim=1) # [1, 3, d]
            
            # 提取 MultiheadAttention 的权重
            # 注意：需确保模型中的 multihead_attn.forward 返回了 attn_output_weights
            _, attn_weights = self.model.multihead_attn(H, H, H)
            attn_matrix = attn_weights[0].cpu().numpy() # [3, 3]

        labels = ['Struct', 'ICD', 'Drug']
        plt.figure(figsize=(7, 6))
        sns.heatmap(attn_matrix, annot=True, cmap="YlGnBu", xticklabels=labels, yticklabels=labels)
        plt.title('Cross-Modal Attention Heatmap (Fig 6B)')
        plt.xlabel('Key Modalities (Attended To)')
        plt.ylabel('Query Modalities (Attending From)')
        
        plt.savefig(self.output_dir / "fig6b_attention_heatmap.png", dpi=300)
        logger.info("Saved Fig 6B.")

    # =========================================================================
    # 3. 单个病例决策模式 (对应论文 Fig 6C-E)
    # =========================================================================
    def analyze_patient_case(self, patient_df, case_name="Case_Study"):
        """
        分析特定病例的模态权重分配。
        展示模型在面对特殊临床场景（如脓毒症或多病共存）时的动态调整。
        """
        logger.info(f"Analyzing {case_name}...")
        # 提取逻辑与全局相似，但关注局部 Attention Pooling 的输出
        # ... (此处实现提取特定样本在 Path A 中 alpha 权重的逻辑)
        pass

if __name__ == "__main__":
    interpreter = MFT_HF_Interpreter()
    
    # 1. 生成全局模态权重图
    interpreter.plot_global_modality_importance()
    
    # 2. 生成注意力交互热图
    interpreter.plot_attention_heatmap(patient_index=10) # 随机取一个病人演示
    
    logger.info("Interpretability analysis complete. Check the 'figures/interpretability' folder.")