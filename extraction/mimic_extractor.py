import psycopg2
import pandas as pd
from utils import logger
from config import DEVICE

class MIMICExtractor:
    """
    MIMIC-IV 阶段 A 心衰队列提取器。
    实现逻辑：Figure 1 筛选流程 (A-B-C-D)。
    """
    
    # 临床定义：高危因素 (Inclusion) 与 既往心脏病排除 (Exclusion)
    # 适配 ICD-9 和 ICD-10
    CRITERIA = {
        'high_risk': {
            'hypertension': "((icd_version = 9 AND icd_code BETWEEN '401' AND '405') OR (icd_version = 10 AND icd_code LIKE 'I1[0-5]%%'))",
            'diabetes': "((icd_version = 9 AND icd_code LIKE '250%%') OR (icd_version = 10 AND icd_code BETWEEN 'E10' AND 'E14'))",
            'cad_ihd': "((icd_version = 9 AND icd_code BETWEEN '410' AND '414') OR (icd_version = 10 AND icd_code BETWEEN 'I20' AND 'I25'))",
            'obesity': "((icd_version = 9 AND icd_code LIKE '278.0%%') OR (icd_version = 10 AND icd_code LIKE 'E66%%'))",
            'metabolic': "((icd_version = 9 AND icd_code = '277.7') OR (icd_version = 10 AND icd_code = 'E88.81'))"
        },
        'cardiac_exclusion': {
            'hf': "((icd_version = 9 AND icd_code LIKE '428%%') OR (icd_version = 10 AND icd_code LIKE 'I50%%'))",
            'mi': "((icd_version = 9 AND icd_code LIKE '410%%') OR (icd_version = 10 AND icd_code LIKE 'I21%%'))",
            'arrhythmia': "((icd_version = 9 AND icd_code LIKE '427%%') OR (icd_version = 10 AND icd_code LIKE 'I4[7-9]%%'))"
        }
    }

    def __init__(self, db_config):
        self.conn = psycopg2.connect(**db_config)
        self.cursor = self.conn.cursor()
        logger.info("MIMICExtractor: Connected to database.")

    def get_stage_a_cohort(self):
        """
        核心队列筛选逻辑：合并步骤 A, B, C, D 为一个高效的 SQL 视图。
        """
        logger.info("Executing optimized cohort selection query...")

        # 构建 SQL 条件字符串
        high_risk_sql = " OR ".join(self.CRITERIA['high_risk'].values())
        cardiac_excl_sql = " OR ".join(self.CRITERIA['cardiac_exclusion'].values())

        query = f"""
        WITH patient_admissions AS (
            -- 步骤 A & B: 获取所有入院记录并标记首诊，同时计算总入院次数
            SELECT 
                subject_id, hadm_id, admittime,
                RANK() OVER (PARTITION BY subject_id ORDER BY admittime, hadm_id) as admit_rank,
                COUNT(hadm_id) OVER (PARTITION BY subject_id) as total_admissions
            FROM mimiciv_hosp.admissions
        ),
        first_admission_diagnoses AS (
            -- 提取首诊的诊断信息
            SELECT 
                pa.subject_id, pa.hadm_id, pa.total_admissions,
                MAX(CASE WHEN {high_risk_sql} THEN 1 ELSE 0 END) as is_high_risk,
                MAX(CASE WHEN {cardiac_excl_sql} THEN 1 ELSE 0 END) as has_preexisting_cardiac
            FROM patient_admissions pa
            JOIN mimiciv_hosp.diagnoses_icd dx ON pa.hadm_id = dx.hadm_id
            WHERE pa.admit_rank = 1
            GROUP BY pa.subject_id, pa.hadm_id, pa.total_admissions
        )
        -- 最终筛选：首诊有高危因素 AND 首诊无心脏病 AND 入院次数 >= 2
        SELECT subject_id 
        FROM first_admission_diagnoses
        WHERE is_high_risk = 1 
          AND has_preexisting_cardiac = 0 
          AND total_admissions >= 2;
        """
        
        self.cursor.execute(query)
        rows = self.cursor.fetchall()
        cohort_pids = [row[0] for row in rows]
        logger.info(f"Cohort selection finished. Stage A N = {len(cohort_pids)}")
        return cohort_pids

    def extract_outcomes(self, cohort_pids):
        """
        结局查询：在随后的入院中寻找心衰信号。
        """
        if not cohort_pids: return pd.DataFrame()

        # 同样使用窗口函数处理，避免 Python 端的复杂循环
        pids_tuple = tuple(cohort_pids)
        hf_sql = self.CRITERIA['cardiac_exclusion']['hf']

        query = f"""
        WITH subsequent_admissions AS (
            SELECT 
                subject_id, hadm_id,
                RANK() OVER (PARTITION BY subject_id ORDER BY admittime, hadm_id) as admit_rank
            FROM mimiciv_hosp.admissions
            WHERE subject_id IN %s
        )
        SELECT 
            subject_id,
            MAX(CASE WHEN {hf_sql} THEN 1 ELSE 0 END) as heart_failure
        FROM subsequent_admissions sa
        JOIN mimiciv_hosp.diagnoses_icd dx ON sa.hadm_id = dx.hadm_id
        WHERE sa.admit_rank > 1 -- 只看后续入院
        GROUP BY subject_id;
        """
        
        self.cursor.execute(query, (pids_tuple,))
        results = self.cursor.fetchall()
        return pd.DataFrame(results, columns=['patient_id', 'heart_failure'])

    def close(self):
        self.cursor.close()
        self.conn.close()