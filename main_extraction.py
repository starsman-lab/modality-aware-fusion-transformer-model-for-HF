"""
MFT-HF: main_extraction.py
Description: 
    This script automates the cohort selection process for Stage A heart failure 
    from MIMIC-IV and eICU databases. It follows the exact inclusion/exclusion 
    criteria described in the research paper.
"""

import os
import json
import pandas as pd
from pathlib import Path
from config import DATA_DIR, LOG_CONFIG
from utils import logger, setup_logger
from extraction.mimic_extractor import MIMICExtractor
from extraction.eicu_extractor import EICUExtractor

# =============================================================================
# 1. 数据库安全连接设置 (Security & Credentials)
# =============================================================================
def get_db_config(db_name):
    """
    用户应通过环境变量设置：export DB_PASSWORD='your_pass'
    """
    password = os.getenv('DB_PASSWORD', 'default_password') 
    
    configs = {
        'mimic': {
            'host': '127.0.0.1',
            'user': 'postgres',
            'password': password,
            'dbname': 'mimiciv',
            'port': 5432
        },
        'eicu': {
            'host': '127.0.0.1',
            'user': 'postgres',
            'password': password,
            'dbname': 'eicu',
            'port': 5432
        }
    }
    return configs.get(db_name)

# =============================================================================
# 2. 提取流水线 (Extraction Pipeline)
# =============================================================================

def run_extraction_for_database(db_type="mimic"):
    """
    通用的数据库提取流水线。
    db_type: "mimic" 或 "eicu"
    """
    logger.info(f"🚀 Starting Extraction Pipeline for {db_type.upper()}...")
    
    # 动态初始化对应的提取器
    config = get_db_config(db_type)
    if db_type == "mimic":
        extractor = MIMICExtractor(config)
    else:
        extractor = EICUExtractor(config)
    
    try:
        # Step 1: 队列筛选 (Figure 1: Patient Selection Flowchart)
        # 内部逻辑包含：高危筛选 -> 排除首诊异常 -> 筛选多次入院
        cohort_pids = extractor.get_stage_a_cohort()
        
        # Step 2: 结局指标提取 (Extract Outcome Labels)
        df_outcomes = extractor.extract_outcomes(cohort_pids)
        
        # Step 3: 标准化输出
        # 将 ID 统一重命名为 patient_id，以便下游通用处理
        id_col = 'subject_id' if db_type == "mimic" else 'uniquepid'
        df_outcomes.rename(columns={id_col: 'patient_id'}, inplace=True)
        
        # Step 4: 保存结果
        save_name = f"{db_type}_stage_a_cohort.csv"
        save_path = DATA_DIR / save_name
        df_outcomes.to_csv(save_path, index=False)
        
        # 生成筛选摘要报告
        logger.info(f"✅ {db_type.upper()} Pipeline Finished.")
        logger.info(f"   - Final Cohort Size: {len(df_outcomes)}")
        logger.info(f"   - Outcome (HF) Rate: {df_outcomes['heart_failure'].mean():.2%}")
        logger.info(f"   - Data saved to: {save_path}")
        
        return df_outcomes

    except Exception as e:
        logger.error(f"❌ Error during {db_type} extraction: {str(e)}")
        return None
    finally:
        extractor.close()

# =============================================================================
# 3. 主程序入口 (Main Entry)
# =============================================================================

def main():
    logger.info("==========================================")
    logger.info("   MFT-HF COHORT EXTRACTION COMMENCED    ")
    logger.info("==========================================")
    
    # 确保保存目录存在
    DATA_DIR.mkdir(exist_ok=True)
    
    # 1. 提取训练集/内部验证集 (MIMIC-IV)
    mimic_df = run_extraction_for_database("mimic")
    
    # 2. 提取外部验证集 (eICU)
    eicu_df = run_extraction_for_database("eicu")
    
    if mimic_df is not None and eicu_df is not None:
        logger.info("🎉 All extractions completed successfully.")
        logger.info("Next steps: Run 'python preprocessing/feature_engineering.py' to generate embeddings.")
    else:
        logger.warning("Extraction completed with errors. Please check the logs.")

if __name__ == "__main__":
    main()