import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, 
    average_precision_score, calibration_curve
)
from pathlib import Path

# 导入项目模块
from config import (
    DEVICE, MODEL_PARAMS, FIGURES_DIR, 
    LABEL_COLUMN, SAVED_MODELS_DIR
)
from utils import logger, calculate_metrics, load_artifacts
from preprocessing.data_loader import get_train_val_datasets, get_external_test_dataset
from preprocessing.feature_engineering import process_features
from models.mft_hf import MFTHF

class Evaluator:
    def __init__(self, model_name="mft_hf_best"):
        self.model_name = model_name
        self.figures_path = FIGURES_DIR
        self.figures_path.mkdir(exist_ok=True)

    def _get_predictions(self, model, df, dims):
        """执行推理并获取预测概率"""
        model.eval()
        ns, icd, dr = dims['struct'], dims['icd'], dims['drug']
        
        with torch.no_grad():
            x_s = torch.tensor(df.iloc[:, :ns].values, dtype=torch.float32).to(DEVICE)
            x_c = torch.tensor(df.iloc[:, ns:ns+icd].values, dtype=torch.float32).to(DEVICE)
            x_d = torch.tensor(df.iloc[:, ns+icd:].values, dtype=torch.float32).to(DEVICE)
            
            logits = model(x_s, x_c, x_d)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
        return probs

    def plot_discrimination_curves(self, y_true, y_prob, dataset_name="MIMIC-IV"):
        """绘制 ROC 和 PR 曲线 (对应论文 Fig 3A, 3B)"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax1.set_title(f'ROC Curve ({dataset_name})')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.legend(loc="lower right")

        # PR Curve
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        ax2.plot(recall, precision, color='blue', lw=2, label=f'AUPRC = {pr_auc:.3f}')
        ax2.set_title(f'Precision-Recall Curve ({dataset_name})')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.legend(loc="lower left")

        plt.tight_layout()
        plt.savefig(self.figures_path / f"{dataset_name}_discrimination.png", dpi=300)
        logger.info(f"Saved discrimination curves for {dataset_name}")

    def plot_calibration(self, y_true, y_prob, dataset_name="MIMIC-IV"):
        """绘制校准曲线 (对应论文 Fig 3C)"""
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        
        plt.figure(figsize=(6, 6))
        plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='MFT-HF')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
        
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title(f'Calibration Curve ({dataset_name})')
        plt.legend()
        plt.savefig(self.figures_path / f"{dataset_name}_calibration.png", dpi=300)

    def plot_dca(self, y_true, y_prob, dataset_name="MIMIC-IV"):
        """
        绘制决策曲线分析 (DCA) - 临床效用的核心展示 (对应论文 Fig 3D)
        """
        thresholds = np.linspace(0.01, 0.99, 100)
        net_benefit_model = []
        net_benefit_all = []

        for thresh in thresholds:
            # 计算模型净收益
            y_pred = (y_prob >= thresh).astype(int)
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            n = len(y_true)
            
            nb = (tp / n) - (fp / n) * (thresh / (1 - thresh))
            net_benefit_model.append(nb)
            
            # 计算 "全部干预" 策略的净收益
            tp_all = np.sum(y_true == 1)
            fp_all = np.sum(y_true == 0)
            nb_all = (tp_all / n) - (fp_all / n) * (thresh / (1 - thresh))
            net_benefit_all.append(nb_all)

        plt.figure(figsize=(8, 6))
        plt.plot(thresholds, net_benefit_model, color='red', label='MFT-HF')
        plt.plot(thresholds, net_benefit_all, color='gray', linestyle='--', label='Treat All')
        plt.axhline(y=0, color='black', label='Treat None')
        
        plt.ylim(-0.05, max(net_benefit_model) * 1.2)
        plt.xlabel('Threshold Probability')
        plt.ylabel('Net Benefit')
        plt.title(f'Decision Curve Analysis ({dataset_name})')
        plt.legend()
        plt.savefig(self.figures_path / f"{dataset_name}_dca.png", dpi=300)

    def run_full_evaluation(self, dataset_type="internal"):
        """一键运行所有评估"""
        logger.info(f"Running full evaluation for {dataset_type} dataset...")
        
        # 1. 加载数据
        if dataset_type == "internal":
            df_raw = get_train_val_datasets()
            name = "MIMIC-IV"
        else:
            df_raw = get_external_test_dataset()
            name = "eICU"

        # 2. 特征工程 (train_mode=False，确保使用训练集的参数)
        features_df, dims = process_features(df_raw, 'icd_codes', 'drugs', train_mode=False)
        y_true = features_df[LABEL_COLUMN].values

        # 3. 加载模型
        model = MFTHF(dims['struct'], dims['icd'], dims['drug'], MODEL_PARAMS['hidden_dim']).to(DEVICE)
        model, metadata = load_artifacts(model, self.model_name)

        # 4. 获取预测结果
        y_prob = self._get_predictions(model, features_df, dims)

        # 5. 绘图
        self.plot_discrimination_curves(y_true, y_prob, name)
        self.plot_calibration(y_true, y_prob, name)
        self.plot_dca(y_true, y_prob, name)
        
        # 6. 打印核心指标
        metrics = calculate_metrics(y_true, y_prob)
        print(f"\n--- {name} Final Results ---")
        for k, v in metrics.items():
            print(f"{k.upper()}: {v:.4f}")

if __name__ == "__main__":
    evaluator = Evaluator()
    # 评估内部验证集 (Figure 3)
    evaluator.run_full_evaluation(dataset_type="internal")
    # 评估外部验证集 (Figure 4)
    evaluator.run_full_evaluation(dataset_type="external")