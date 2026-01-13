import torch
import pandas as pd
import numpy as np
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.calibration import IsotonicRegression, CalibratedClassifierCV
from tqdm import tqdm

from config import (
    DEVICE, RANDOM_STATE, CV_FOLDS, MODEL_PARAMS, 
    MLP_CONFIG, SAVED_MODELS_DIR, LABEL_COLUMN
)
from utils import logger, clear_memory, calculate_metrics
from models.mft_hf import MFTHF

# =============================================================================
# 1. 损失函数 (Advanced Hybrid Loss - Equation 3)
# =============================================================================

class ImprovedHybridLoss(nn.Module):
    """
    对应论文公式 (3): 结合 Focal, Dice 和 F-beta 的复合损失函数。
    核心逻辑：动态权重分配，在训练初期关注整体分布，后期关注难分类样本。
    """
    def __init__(self, alpha=0.25, gamma=2.0, dice_weight_range=(0.2, 0.4)):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_start, self.dice_end = dice_weight_range
        self.eps = 1e-8

    def forward(self, inputs, targets, epoch=0, total_epochs=50):
        # 1. 动态权重计算 (随着 epoch 增加，增加 Dice Loss 的权重以优化 F1)
        progress = epoch / total_epochs
        w_dice = self.dice_start + (self.dice_end - self.dice_start) * progress
        w_focal = 1.0 - w_dice

        # 2. Focal Loss: 处理难分类的 Stage A 患者
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = (alpha_t * (1 - p_t).pow(self.gamma) * ce_loss).mean()

        # 3. Dice Loss: 优化预测标签的重叠度 (对极少数正类敏感)
        intersection = (p * targets).sum()
        dice_loss = 1 - (2 * intersection + self.eps) / (p.sum() + targets.sum() + self.eps)

        return w_focal * focal_loss + w_dice * dice_loss

# =============================================================================
# 2. 训练辅助类 (Training Utilities)
# =============================================================================

class EarlyStopping:
    """早停机制：防止模型在训练集上过拟合，保留验证集指标最好的权重。"""
    def __init__(self, patience=10, mode='max'):
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_score = -np.inf if mode == 'max' else np.inf
        self.early_stop = False
        self.best_state = None

    def __call__(self, score, model):
        improved = (score > self.best_score) if self.mode == 'max' else (score < self.best_score)
        if improved:
            self.best_score = score
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

# =============================================================================
# 3. MFT-HF 专属训练引擎 (MFT-HF Training Engine)
# =============================================================================

def train_mft_hf_cv(X_df, y_series, dims):
    """
    针对 MFT-HF 模型的 K-Fold 交叉验证训练逻辑。
    包含：三模态数据切分、OOF 阈值寻优、模型训练。
    """
    logger.info(f"Starting {CV_FOLDS}-Fold CV for MFT-HF...")
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    oof_probs = np.zeros(len(X_df))
    best_overall_state = None
    max_val_auc = 0

    for fold, (t_idx, v_idx) in enumerate(kf.split(X_df, y_series)):
        logger.info(f"--- Fold {fold+1} ---")
        
        # 数据切分 (三模态拼接数据)
        X_train, X_val = X_df.iloc[t_idx], X_df.iloc[v_idx]
        y_train, y_val = y_series.iloc[t_idx], y_series.iloc[v_idx]

        # 准备 DataLoader
        def get_loader(features, labels, batch_size):
            # 将拼接的长向量切回三模态输入
            ns, icd, dr = dims['struct'], dims['icd'], dims['drug']
            ds = TensorDataset(
                torch.tensor(features.iloc[:, :ns].values, dtype=torch.float32),
                torch.tensor(features.iloc[:, ns:ns+icd].values, dtype=torch.float32),
                torch.tensor(features.iloc[:, ns+icd:].values, dtype=torch.float32),
                torch.tensor(labels.values, dtype=torch.float32).unsqueeze(1)
            )
            return DataLoader(ds, batch_size=batch_size, shuffle=True)

        train_loader = get_loader(X_train, y_train, MODEL_PARAMS['batch_size'])
        
        # 初始化模型与优化器
        model = MFTHF(dims['struct'], dims['icd'], dims['drug'], MODEL_PARAMS['hidden_dim']).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=MODEL_PARAMS['lr'], weight_decay=MODEL_PARAMS['weight_decay'])
        scheduler = CosineAnnealingLR(optimizer, T_max=MODEL_PARAMS['epochs'])
        criterion = ImprovedHybridLoss()
        stopper = EarlyStopping(patience=10)

        # 训练循环
        for epoch in range(MODEL_PARAMS['epochs']):
            model.train()
            for xs, xc, xd, y in train_loader:
                xs, xc, xd, y = xs.to(DEVICE), xc.to(DEVICE), xd.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(xs, xc, xd)
                loss = criterion(outputs, y, epoch, MODEL_PARAMS['epochs'])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            # 验证
            model.eval()
            with torch.no_grad():
                ns, icd, dr = dims['struct'], dims['icd'], dims['drug']
                v_xs = torch.tensor(X_val.iloc[:, :ns].values, dtype=torch.float32).to(DEVICE)
                v_xc = torch.tensor(X_val.iloc[:, ns:ns+icd].values, dtype=torch.float32).to(DEVICE)
                v_xd = torch.tensor(X_val.iloc[:, ns+icd:].values, dtype=torch.float32).to(DEVICE)
                v_out = model(v_xs, v_xc, v_xd)
                v_prob = torch.sigmoid(v_out).cpu().numpy().flatten()
            
            val_metrics = calculate_metrics(y_val.values, v_prob)
            if stopper(val_metrics['auc_pr'], model): break
            scheduler.step()

        # 记录 OOF 预测用于最终阈值寻优
        oof_probs[v_idx] = v_prob
        if val_metrics['auc_roc'] > max_val_auc:
            max_val_auc = val_metrics['auc_roc']
            best_overall_state = stopper.best_state

    # 使用所有 OOF 预测寻找全局最佳阈值 
    final_metrics = calculate_metrics(y_series.values, oof_probs)
    logger.info(f"CV Finished. Best OOF AUC-PR: {final_metrics['auc_pr']:.4f}, Best Threshold: {final_metrics['best_threshold']:.4f}")
    
    return best_overall_state, final_metrics['best_threshold']

# =============================================================================
# 4. 概率校准 (Probability Calibration)
# =============================================================================

def calibrate_model(model, X_val_df, y_val_series, dims):
    """
    使用 Isotonic Regression 对模型输出的 logits 进行校准。
    确保模型输出的概率与真实患病率一致（对应论文中 Figure 3C 的 Calibration Curve）。
    """
    logger.info("Training Isotonic Calibrator...")
    model.eval()
    ns, icd, dr = dims['struct'], dims['icd'], dims['drug']
    
    with torch.no_grad():
        x_s = torch.tensor(X_val_df.iloc[:, :ns].values, dtype=torch.float32).to(DEVICE)
        x_c = torch.tensor(X_val_df.iloc[:, ns:ns+icd].values, dtype=torch.float32).to(DEVICE)
        x_d = torch.tensor(X_val_df.iloc[:, ns+icd:].values, dtype=torch.float32).to(DEVICE)
        logits = model(x_s, x_c, x_d).cpu().numpy().flatten()
    
    calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    calibrator.fit(logits, y_val_series.values)
    return calibrator