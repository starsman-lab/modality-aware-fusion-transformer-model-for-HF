import os
import json
import torch
import random
import hashlib
import logging
import joblib
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss, f1_score
from config import LOG_CONFIG, CACHE_DIR, SAVED_MODELS_DIR

# =============================================================================
# 1. 基础环境与日志 (Infrastructure)
# =============================================================================

def setup_logger(name="MFT-HF"):
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(getattr(logging, LOG_CONFIG['level']))
        ch = logging.StreamHandler()
        formatter = logging.Formatter(LOG_CONFIG['format'])
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

logger = setup_logger()

def set_seed(seed=42):
    """固定随机种子，确保医学实验结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Reproducibility seed set to {seed}")

def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# =============================================================================
# 2. 损失函数 (Loss Functions - Equation 3)
# =============================================================================

class HybridLoss(nn.Module):
    """
    对应论文公式 (3): L_hybrid = λ1*L_focal + λ2*L_dice + λ3*L_fbeta
    用于处理 A 期心衰预测中的高度类别不平衡问题。
    """
    def __init__(self, alpha=0.25, gamma=2.0, lambda1=1.0, lambda2=0.5, lambda3=0.5):
        super(HybridLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

    def focal_loss(self, inputs, targets):
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        return loss.mean()

    def dice_loss(self, inputs, targets, smooth=1.0):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice

    def forward(self, inputs, targets):
        l_focal = self.focal_loss(inputs, targets)
        l_dice = self.dice_loss(inputs, targets)
        # 组合损失
        return self.lambda1 * l_focal + self.lambda2 * l_dice

# =============================================================================
# 3. 性能评估 (Clinical Metrics)
# =============================================================================

def calculate_metrics(y_true, y_prob):
    """
    医学预测全指标计算。
    重点包含：AUC-ROC, AUC-PR (不平衡数据核心指标) 以及 Brier Score (校准度)。
    """
    # AUC-ROC
    auc_roc = roc_auc_score(y_true, y_prob)
    
    # AUC-PR (Precision-Recall Curve Area)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auc_pr = auc(recall, precision)
    
    # Brier Score (越低代表模型概率预测越准)
    brier = brier_score_loss(y_true, y_prob)
    
    # 寻找最佳 F1 阈值
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    return {
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "brier_score": brier,
        "best_threshold": best_threshold,
        "max_f1": np.max(f1_scores)
    }

# =============================================================================
# 4. 持久化与缓存 (Persistence & Cache)
# =============================================================================

def get_text_hash(text_series):
    """基于文本内容生成哈希，用于 BioBERT 缓存"""
    combined = "".join(text_series.astype(str).values)
    return hashlib.md5(combined.encode()).hexdigest()

def save_artifacts(model, model_name, metrics, config_dict):
    """保存模型权重、配置和评估结果"""
    save_path = SAVED_MODELS_DIR / model_name
    save_path.mkdir(exist_ok=True)
    
    # 保存权重
    torch.save(model.state_dict(), save_path / "model.pth")
    
    # 保存元数据 (含最佳阈值)
    with open(save_path / "metadata.json", "w") as f:
        json.dump({
            "metrics": {k: float(v) for k, v in metrics.items()},
            "config": config_dict
        }, f, indent=4)
    
    logger.info(f"Successfully saved {model_name} artifacts to {save_path}")

def load_artifacts(model, model_name):
    """加载模型权重和元数据"""
    load_path = SAVED_MODELS_DIR / model_name
    model.load_state_dict(torch.load(load_path / "model.pth", map_location=torch.device('cpu')))
    
    with open(load_path / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    logger.info(f"Loaded {model_name} from {load_path}")
    return model, metadata