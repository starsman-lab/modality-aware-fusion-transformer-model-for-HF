import psycopg2
import pandas as pd
from utils import logger

class EICUExtractor:
    """
    eICU-CRD 阶段 A 心衰队列提取器（外部验证集）。
    实现逻辑：Figure 1 筛选流程，与 MIMIC 提取逻辑严格对齐。
    """
    
    # 临床定义：主要基于 ICD-9 (eICU 核心编码系统)
    CRITERIA = {
        'high_risk': {
            'hypertension': "individual_code ~ '^40[1-5]'",
            'diabetes': "individual_code LIKE '250%%'",
            'cad_ihd': "individual_code ~ '^41[0-4]'",
            'obesity': "individual_code LIKE '278.0%%'",
            'metabolic': "individual_code = '277.7'"
        },
        'cardiac_exclusion': {
            'hf': "individual_code LIKE '428%%'",
            'mi': "individual_code LIKE '410%%'",
            'arrhythmia': "individual_code LIKE '427%%'"
        }
    }

    def __init__(self, db_config):
        self.conn = psycopg2.connect(**db_config)
        self.cursor = self.conn.cursor()
        logger.info("EICUExtractor: Connected to eICU database.")

    def get_stage_a_cohort(self):
        """
        核心队列筛选：利用窗口函数识别首次入院，并执行筛选。
        """
        logger.info("Executing optimized cohort selection for eICU...")

        high_risk_sql = " OR ".join(self.CRITERIA['high_risk'].values())
        cardiac_excl_sql = " OR ".join(self.CRITERIA['cardiac_exclusion'].values())

        # eICU 特有逻辑：处理可能存在的逗号分隔 ICD 码并进行解析
        query = f"""
        WITH patient_stays AS (
            -- 步骤 A & B: 识别多次入院并标记首诊顺序
            SELECT 
                uniquepid, patientunitstayid, hospitaladmitoffset,
                RANK() OVER (PARTITION BY uniquepid ORDER BY hospitaladmitoffset ASC) as admit_rank,
                COUNT(patientunitstayid) OVER (PARTITION BY uniquepid) as total_stays
            FROM eicuii.patient
        ),
        diagnoses_unfolded AS (
            -- 处理 eICU 的字符串格式 ICD 码
            SELECT 
                ps.uniquepid, ps.patientunitstayid, ps.admit_rank, ps.total_stays,
                UNNEST(STRING_TO_ARRAY(REPLACE(dx.icd9code, ' ', ''), ',')) AS individual_code
            FROM patient_stays ps
            JOIN eicuii.diagnosis dx ON ps.patientunitstayid = dx.patientunitstayid
        ),
        first_admission_status AS (
            -- 步骤 C & D: 评估首诊状态
            SELECT 
                uniquepid,
                MAX(CASE WHEN {high_risk_sql} THEN 1 ELSE 0 END) as is_high_risk,
                MAX(CASE WHEN {cardiac_excl_sql} THEN 1 ELSE 0 END) as has_preexisting_cardiac,
                MAX(total_stays) as total_stays
            FROM diagnoses_unfolded
            WHERE admit_rank = 1
            GROUP BY uniquepid
        )
        -- 最终筛选
        SELECT uniquepid 
        FROM first_admission_status
        WHERE is_high_risk = 1 
          AND has_preexisting_cardiac = 0 
          AND total_stays >= 2;
        """
        
        self.cursor.execute(query)
        rows = self.cursor.fetchall()
        cohort_pids = [row[0] for row in rows]
        logger.info(f"eICU Stage A Cohort Filtered: {len(cohort_pids)} patients.")
        return cohort_pids

    def extract_outcomes(self, cohort_pids):
        """
        结局查询：在首次入院之后的记录中提取心衰（Incident HF）标签。
        """
        if not cohort_pids: return pd.DataFrame()

        pids_tuple = tuple(cohort_pids)
        hf_sql = self.CRITERIA['cardiac_exclusion']['hf']

        query = f"""
        WITH subsequent_stays AS (
            SELECT 
                ps.uniquepid, dx.icd9code,
                RANK() OVER (PARTITION BY ps.uniquepid ORDER BY ps.hospitaladmitoffset ASC) as admit_rank
            FROM eicuii.patient ps
            JOIN eicuii.diagnosis dx ON ps.patientunitstayid = dx.patientunitstayid
            WHERE ps.uniquepid IN %s
        ),
        unfolded_subsequent AS (
            SELECT uniquepid, UNNEST(STRING_TO_ARRAY(REPLACE(icd9code, ' ', ''), ',')) AS individual_code
            FROM subsequent_stays
            WHERE admit_rank > 1
        )
        SELECT 
            uniquepid,
            MAX(CASE WHEN {hf_sql} THEN 1 ELSE 0 END) as heart_failure
        FROM unfolded_subsequent
        GROUP BY uniquepid;
        """
        
        self.cursor.execute(query, (pids_tuple,))
        results = self.cursor.fetchall()
        # 统一输出列名，确保与 MIMIC 提取器返回的 DataFrame 格式一致
        df = pd.DataFrame(results, columns=['patient_id', 'heart_failure'])
        
        # 补充那些在后续入院中完全没有诊断记录的患者（设为 0）
        all_pids_df = pd.DataFrame(cohort_pids, columns=['patient_id'])
        final_df = all_pids_df.merge(df, on='patient_id', how='left').fillna(0)
        final_df['heart_failure'] = final_df['heart_failure'].astype(int)
        
        return final_df

    def close(self):
        self.cursor.close()
        self.conn.close()