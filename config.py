import torch
from pathlib import Path

# =============================================================================
# 1. 路径配置 (Paths Configuration)
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent

# 数据与结果目录
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
CACHE_DIR = BASE_DIR / "cache"
SAVED_MODELS_DIR = BASE_DIR / "saved_models"
ICD9_GEM_PATH = DATA_DIR / "2017_I9gem.txt"  # 官方 GEM 映射文件路径

# 确保目录存在
for p in [DATA_DIR, RESULTS_DIR, FIGURES_DIR, CACHE_DIR, SAVED_MODELS_DIR]:
    p.mkdir(exist_ok=True)

# 数据文件路径
TRAIN_DATA_PATH = DATA_DIR / "merged_df.csv"
EXTERNAL_VAL_DATA_PATH = DATA_DIR / "eicu_validation_data.csv"

# 预训练模型路径
BIOBERT_MODEL_PATH = Path("/root/autodl-tmp/models--dmis-lab--biobert-v1.1/snapshots") 

# =============================================================================
# 2. 临床特征定义 (Clinical Feature Definitions)
# =============================================================================
LABEL_COLUMN = "heart_failure"

# 结构化数值特征
NUMERIC_FEATURES_BASE = [
    'age',          # 患者年龄
    'lab_50912',    # 肌酐 (Creatinine) - 评估肾功能，心衰常伴随心肾综合征
    'lab_51301',    # 白细胞计数 (WBC) - 评估全身炎症反应或感染
    'lab_50931',    # 血糖 (Glucose) - 糖尿病是心衰进展的核心代谢风险因素
    'lab_51006',    # 尿素氮 (BUN) - 评估容量状态及肾灌注
    'lab_50971',    # 钾 (Potassium) - 电解质，直接影响心肌电生理稳定性
    'lab_50960',    # 镁 (Magnesium) - 电解质，低镁与心律失常风险相关
    'lab_50893',    # 总钙 (Calcium, Total) - 参与心肌收缩偶联
    'lab_51237',    # 国际标准化比值 (INR) - 评估凝血功能及肝脏淤血情况
    'lab_51265',    # 血小板计数 (Platelet Count) - 基础血液学指标
    'lab_50983',    # 钠 (Sodium) - 评估水钠潴留及容量负荷
    'lab_50902'     # 氯 (Chloride) - 评估酸碱平衡及利尿剂反应
]

# 特征映射表 (用于 SHAP 或可视化绘图时的标签显示)
FEATURE_LABEL_MAPPING = {
    'age': 'Age',
    'lab_50912': 'Creatinine (Kidney)',
    'lab_51301': 'White Blood Cells (Inflammation)',
    'lab_50931': 'Glucose (Metabolic)',
    'lab_51006': 'Urea Nitrogen (BUN)',
    'lab_50971': 'Potassium (Electrolyte)',
    'lab_50960': 'Magnesium (Electrolyte)',
    'lab_50893': 'Calcium (Electrolyte)',
    'lab_51237': 'INR (Coagulation)',
    'lab_51265': 'Platelet Count',
    'lab_50983': 'Sodium (Volume)',
    'lab_50902': 'Chloride (Acid-Base)'
}

# 结构化类别特征
CATEGORICAL_FEATURES_BASE = [
    'gender', 'atelectasis', 'cardiomegaly', 'consolidation', 'edema',
    'enlarged_cardiomediastinum', 'fracture', 'lung_lesion', 'lung_opacity',
    'no_finding', 'pleural_effusion', 'pleural_other', 'pneumonia',
    'pneumothorax', 'support_devices'
]

# 文本/序列特征
TEXT_FEATURE_ICD = 'icd_codes'
TEXT_FEATURE_DRUG = 'drugs'

# =============================================================================
# 3. 模态特定参数 (Modality Specific Params)
# =============================================================================

# --- ICD GNN 参数 ---
GNN_ICD_PARAMS = {
    'out_channels': 64,
    'hidden_channels': 128,
    'epochs': 50,
    'lr': 0.005,
    'min_cooccurrence': 5  # 构边的最小共现次数
}

# --- Drug GNN 参数 ---
GNN_DRUG_PARAMS = {
    'out_channels': 64,
    'hidden_channels': 128,
    'epochs': 50,
    'lr': 0.005,
    'min_cooccurrence': 5
}

# --- BioBERT 提取参数 ---
BIOBERT_PARAMS = {
    'batch_size': 64,
    'max_len': 128,
    'pca_components': 128  # 当回退到BioBERT时降维到的维度
}

# =============================================================================
# 4. 主模型 (MFT-HF) 超参数
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_STATE = 42

MODEL_PARAMS = {
    'hidden_dim': 192,      # 论文中的 d
    'num_heads': 8,         # Transformer 的 head 数
    'dropout': 0.2,
    'epochs': 50,
    'lr': 0.0005,
    'weight_decay': 1e-5,
    'batch_size': 256,
    'grad_clip': 1.0        # 梯度裁剪
}

# =============================================================================
# 5. 基线模型与消融实验 (Baselines & Ablation)
# =============================================================================
CV_FOLDS = 5
RUN_CONTROL = {
    'train_fusion': True,
    'train_baselines': True,
    'run_ablation': True,
    'run_shap': True,
    'external_val': True
}

# 基线 MLP 架构
MLP_CONFIG = {
    'hidden_dims': [256, 128],
    'lr': 0.001,
    'epochs': 30,
    'batch_size': 256
}

# =============================================================================
# 6. 日志配置 (Logging)
# =============================================================================
LOG_CONFIG = {
    'level': "INFO",
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
}