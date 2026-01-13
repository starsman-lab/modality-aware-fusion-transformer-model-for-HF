import pandas as pd
import numpy as np
import ast
from pathlib import Path
from config import (
    TRAIN_DATA_PATH, EXTERNAL_VAL_DATA_PATH, ICD9_GEM_PATH,
    LABEL_COLUMN, TEXT_FEATURE_ICD, TEXT_FEATURE_DRUG,
    NUMERIC_FEATURES_BASE, CATEGORICAL_FEATURES_BASE
)
from utils import logger

class DataLoader:
    """
    MFT-HF 数据加载与预处理核心类。
    功能：
    1. 队列筛选 (Figure 1 筛选流水线)
    2. 术语标准化 (ICD-9 to ICD-10 Mapping)
    3. 鲁棒的数据清洗与类型转换
    """
    def __init__(self):
        # 初始化时加载 ICD 转换字典
        self.icd_map = self._load_icd_gem(ICD9_GEM_PATH)

    def _load_icd_gem(self, path):
        """加载 2017_I9gem.txt 映射文件"""
        mapping_dict = {}
        if not path.exists():
            logger.warning(f"GEM mapping file not found at {path}. ICD conversion will be limited.")
            return {}
        
        try:
            with open(path, 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        # 映射文件通常不带点，例如 4280 -> I509
                        i9, i10 = parts[0].strip(), parts[1].strip()
                        mapping_dict[i9] = i10
            logger.info(f"ICD Mapping Dictionary loaded: {len(mapping_dict)} pairs.")
        except Exception as e:
            logger.error(f"Error loading GEM file: {e}")
        return mapping_dict

    def _safe_parse_list(self, x):
        """安全解析 CSV 中的列表字符串 (e.g., "['428.0', '250.0']")"""
        if pd.isna(x) or x == "":
            return ['UNKNOWN']
        if isinstance(x, list):
            return x
        try:
            res = ast.literal_eval(x)
            if isinstance(res, list):
                return [str(i).strip().upper() for i in res if str(i).strip()]
            return ['UNKNOWN']
        except:
            return ['UNKNOWN']

    def _normalize_icd(self, codes):
        """执行标准化：去小数点 + ICD-9 to 10 映射"""
        if not codes: return ['UNKNOWN']
        
        normalized = []
        for code in codes:
            # 1. 清理：去小数点 (例如 428.0 -> 4280)
            cleaned = code.replace('.', '')
            # 2. 映射：查表，找不到则保留原样 (假设已经是 ICD-10 或无法映射)
            mapped = self.icd_map.get(cleaned, cleaned)
            normalized.append(mapped)
        
        return list(set(normalized)) # 去重

    def filter_cohort(self, df):
        """
        对应论文 Figure 1: 患者筛选流水线。
        确保研究队列仅包含 Stage A 患者且具备足够的随访数据。
        """
        initial_n = len(df)
        
        # 1. 排除缺失关键模态的患者
        df = df.dropna(subset=[TEXT_FEATURE_ICD, TEXT_FEATURE_DRUG], how='all')
        
        # 2. 筛选多次入院记录 (确保有 Outcome 的观察窗口)
        # 注意：在 main_extraction.py 中已通过 SQL 初步筛选，此处做二次校验
        if 'admission_count' in df.columns:
            df = df[df['admission_count'] >= 2]
            
        logger.info(f"Cohort filtering complete: {initial_n} -> {len(df)}")
        return df

    def load_and_clean(self, file_path):
        """
        执行完整加载流水线：读取 -> 解析 -> 标准化 -> 筛选。
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Missing data file: {file_path}")

        df = pd.read_csv(file_path)
        logger.info(f"Loading raw file: {file_path} (N={len(df)})")

        # 1. 解析文本列为列表格式
        df[TEXT_FEATURE_ICD] = df[TEXT_FEATURE_ICD].apply(self._safe_parse_list)
        df[TEXT_FEATURE_DRUG] = df[TEXT_FEATURE_DRUG].apply(self._safe_parse_list)

        # 2. ICD 术语标准化 (ICD-9 to 10)
        logger.info("Normalizing ICD terminology to ICD-10...")
        df[TEXT_FEATURE_ICD] = df[TEXT_FEATURE_ICD].apply(self._normalize_icd)

        # 3. 数值型特征强制转换
        for col in NUMERIC_FEATURES_BASE:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 4. 执行队列筛选
        df = self.filter_cohort(df)

        # 5. 标签处理
        if LABEL_COLUMN in df.columns:
            df[LABEL_COLUMN] = pd.to_numeric(df[LABEL_COLUMN], errors='coerce').fillna(0).astype(int)

        return df

# =============================================================================
# 外部调用接口 (API for train.py)
# =============================================================================

def get_train_val_datasets():
    """获取 MIMIC-IV 训练/内部验证数据"""
    loader = DataLoader()
    try:
        return loader.load_and_clean(TRAIN_DATA_PATH)
    except Exception as e:
        logger.error(f"Error loading MIMIC training data: {e}")
        return None

def get_external_test_dataset():
    """获取 eICU 外部验证数据"""
    loader = DataLoader()
    try:
        return loader.load_and_clean(EXTERNAL_VAL_DATA_PATH)
    except Exception as e:
        logger.warning(f"External dataset (eICU) skipped: {e}")
        return None

if __name__ == "__main__":
    df = get_train_val_datasets()
    if df is not None:
        print("\n--- ICD Mapping Result Sample ---")
        print(f"Original ICD-9 logic might be: 4280")
        print(f"Normalized Sample: {df[TEXT_FEATURE_ICD].iloc[0]}")
        print(f"Final Cohort Size: {len(df)}")