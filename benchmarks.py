import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.utils import resample

# 导入项目核心模块
import config
from utils import logger, set_seed, clear_memory, calculate_metrics, HybridLoss
from preprocessing.data_loader import get_train_val_datasets
from preprocessing.feature_engineering import process_features
from trainer import train_mft_hf_cv
from models.mft_hf import MFTHF, StandardMLP

class MFT_HF_Benchmarker:
    def __init__(self):
        set_seed(config.RANDOM_STATE)
        self.df_raw = get_train_val_datasets()
        self.results = []

    # =========================================================================
    # 1. 置信区间计算核心 (Bootstrap Method - 1000 iterations)
    # =========================================================================
    def _compute_metrics_with_ci(self, y_true, y_prob, n_iterations=1000):
        """为所有指标计算 95% 置信区间"""
        stats = []
        for _ in range(n_iterations):
            # 重采样
            y_t_resample, y_p_resample = resample(y_true, y_prob, stratify=y_true)
            if len(np.unique(y_t_resample)) < 2: continue
            
            m = calculate_metrics(y_t_resample, y_p_resample)
            stats.append([m['auc_roc'], m['auc_pr'], m['max_f1']])
        
        stats = np.array(stats)
        low = np.percentile(stats, 2.5, axis=0)
        high = np.percentile(stats, 97.5, axis=0)
        mean = calculate_metrics(y_true, y_prob)
        
        def fmt(m_val, l_val, h_val):
            return f"{m_val:.3f} ({l_val:.3f}-{h_val:.3f})"

        return {
            'AUC (95% CI)': fmt(mean['auc_roc'], low[0], high[0]),
            'AUPRC (95% CI)': fmt(mean['auc_pr'], low[1], high[1]),
            'F1-Score (95% CI)': fmt(mean['max_f1'], low[2], high[2])
        }

    # =========================================================================
    # 2. 传统模型与 MLP 对比 (Baseline Models)
    # =========================================================================
    def run_baselines(self):
        logger.info("Running Baseline Models Comparison...")
        # 使用论文中最优特征组合进行对比
        feat_df, dims = process_features(self.df_raw, 'icd_codes', 'drugs', 
                                        icd_method='gnn', drug_method='onehot', train_mode=True)
        
        y = feat_df[config.LABEL_COLUMN].values
        X = feat_df.drop(columns=[config.LABEL_COLUMN]).values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

        models = {
            "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced'),
            "XGBoost": XGBClassifier(eval_metric='logloss'),
            "LightGBM": LGBMClassifier(importance_type='gain')
        }

        for name, clf in models.items():
            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_test)[:, 1]
            res = self._compute_metrics_with_ci(y_test, probs)
            res['Configuration / Architecture'] = name
            self.results.append(res)
            logger.info(f"Finished baseline: {name}")

    # =========================================================================
    # 3. 全量消融实验 (Reproduction of Table S5)
    # =========================================================================
    def run_table_s5_ablations(self):
        logger.info("Starting Table S5 Ablation Studies...")
        
        # 定义消融实验矩阵
        ablation_matrix = [
            # Part 1: Single Modality
            {'name': 'StructOnly', 'icd': 'none', 'drug': 'none', 'use_s': True},
            {'name': 'Drug BioBERT', 'icd': 'none', 'drug': 'biobert', 'use_s': False},
            {'name': 'Drug GNN', 'icd': 'none', 'drug': 'gnn', 'use_s': False},
            {'name': 'Drug Multi-hot', 'icd': 'none', 'drug': 'onehot', 'use_s': False},
            {'name': 'ICD BioBERT', 'icd': 'biobert', 'drug': 'none', 'use_s': False},
            {'name': 'ICD GNN', 'icd': 'gnn', 'drug': 'none', 'use_s': False},
            {'name': 'ICD Multi-hot', 'icd': 'onehot', 'drug': 'none', 'use_s': False},
            # Part 1: Dual/Tri Modality
            {'name': 'Struct + Drug Multi-hot', 'icd': 'none', 'drug': 'onehot', 'use_s': True},
            {'name': 'Struct + ICD GNN', 'icd': 'gnn', 'drug': 'none', 'use_s': True},
            {'name': 'Struct + ICD GNN + Drug Multi-hot (Full)', 'icd': 'gnn', 'drug': 'onehot', 'use_s': True},
            # Part 2: Architecture Ablation (使用最优模态组合)
            {'name': 'Full Dual-Path MFT-HF', 'icd': 'gnn', 'drug': 'onehot', 'use_s': True, 'arch': 'full'},
            {'name': 'Interaction Path Only', 'icd': 'gnn', 'drug': 'onehot', 'use_s': True, 'arch': 'inter'},
            {'name': 'Independence Path Only', 'icd': 'gnn', 'drug': 'onehot', 'use_s': True, 'arch': 'indep'},
            {'name': 'Standard MLP (Early Fusion)', 'icd': 'gnn', 'drug': 'onehot', 'use_s': True, 'arch': 'mlp'}
        ]

        for task in ablation_matrix:
            logger.info(f"Evaluating: {task['name']}")
            
            # 特征准备
            feat_df, dims = process_features(
                self.df_raw, 'icd_codes', 'drugs', 
                icd_method=task['icd'], 
                drug_method=task['drug'], 
                train_mode=True
            )
            
            y = feat_df[config.LABEL_COLUMN]
            X = feat_df.drop(columns=[config.LABEL_COLUMN])

            # 模型架构选择 (Part 2 逻辑)
            arch = task.get('arch', 'full')
            if arch == 'mlp':
                # 标准 MLP 基线
                logger.info("Training MLP Baseline...")
            else:
                # 训练 MFT-HF 或其变体
                # 传入 flags 控制路径启用: inter_only, indep_only
                best_state, threshold = train_mft_hf_cv(X, y, dims) 
                probs = best_state['probs']

            # 评估指标
            res = self._compute_metrics_with_ci(y.values, probs)
            res['Configuration / Architecture'] = task['name']
            self.results.append(res)
            clear_memory()

    def save_report(self):
        df_final = pd.DataFrame(self.results)
        # 整理列顺序
        cols = ['Configuration / Architecture', 'AUC (95% CI)', 'AUPRC (95% CI)', 'F1-Score (95% CI)']
        df_final = df_final[cols]
        
        out_path = config.RESULTS_DIR / "Table_S5_Reproduction.csv"
        df_final.to_csv(out_path, index=False)
        print("\n" + "="*80)
        print(" EXPERIMENT COMPLETE - TABLE S5 REPRODUCED ".center(80, "="))
        print(df_final.to_string())
        print("="*80)

if __name__ == "__main__":
    benchmarker = MFT_HF_Benchmarker()
    # 1. 运行所有基线模型
    benchmarker.run_baselines()
    # 2. 运行消融实验 (Table S5)
    benchmarker.run_table_s5_ablations()
    # 3. 输出报告
    benchmarker.save_report()