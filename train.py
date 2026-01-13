"""
MFT-HF: train.py
Main entry point for training the final Stage A Heart Failure prediction model.
"""

import torch
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# 导入项目核心模块
import config
from utils import logger, set_seed, clear_memory, save_artifacts
from preprocessing.data_loader import get_train_val_datasets, get_external_test_dataset
from preprocessing.feature_engineering import process_features
from trainer import train_mft_hf_cv, calibrate_model
from models.mft_hf import MFTHF

def run_training_pipeline():
    # 1. 环境初始化
    set_seed(config.RANDOM_STATE)
    logger.info("==========================================")
    logger.info("   MFT-HF MODEL TRAINING COMMENCED       ")
    logger.info("==========================================")

    # 2. 数据加载 (内部开发集: MIMIC-IV)
    df_raw = get_train_val_datasets()
    if df_raw is None:
        logger.error("Training data not found. Please run main_extraction.py first.")
        return

    # 3. 特征工程 (训练模式)
    # 逻辑：训练GNN -> 提取BioBERT -> 标准化数值特征
    logger.info("Step 1: Feature Engineering (Training Mode)...")
    train_features_df, dims = process_features(
        df_raw, 
        icd_col=config.TEXT_FEATURE_ICD, 
        drug_col=config.TEXT_FEATURE_DRUG,
        icd_method='gnn',    # 论文核心：GNN处理共病
        drug_method='onehot', # 论文核心：Multi-hot处理药物
        train_mode=True      # 保存Scaler和PCA分量
    )

    y = train_features_df[config.LABEL_COLUMN]
    X = train_features_df.drop(columns=[config.LABEL_COLUMN])

    # 4. K-Fold 交叉验证训练
    logger.info(f"Step 2: Starting {config.CV_FOLDS}-Fold Cross-Validation...")
    best_state_dict, optimal_threshold = train_mft_hf_cv(X, y, dims)

    # 5. 模型实例化与保存
    final_model = MFTHF(
        struct_dim=dims['struct'],
        icd_dim=dims['icd'],
        drug_dim=dims['drug'],
        hidden_dim=config.MODEL_PARAMS['hidden_dim']
    ).to(config.DEVICE)
    
    final_model.load_state_dict(best_state_dict)

    # 6. 概率校准 (Calibration)
    logger.info("Step 3: Performing Probability Calibration...")
    calibrator = calibrate_model(final_model, X, y, dims)

    # 7. 保存最终成果 (Artifacts)
    metrics_placeholder = {"status": "trained_successfully"} # 实际可传入评估指标
    save_artifacts(
        model=final_model,
        model_name="mft_hf_best",
        metrics=metrics_placeholder,
        config_dict={
            "dims": dims,
            "threshold": float(optimal_threshold),
            "params": config.MODEL_PARAMS
        }
    )
    
    # 保存校准器
    joblib.dump(calibrator, config.SAVED_MODELS_DIR / "mft_hf_best" / "calibrator.joblib")
    
    logger.info("==========================================")
    logger.info("✅ Training Pipeline Complete!")
    logger.info(f"Model saved to: {config.SAVED_MODELS_DIR}/mft_hf_best")
    logger.info("Next step: Run 'python evaluate.py' for external validation.")
    logger.info("==========================================")

if __name__ == "__main__":
    run_training_pipeline()