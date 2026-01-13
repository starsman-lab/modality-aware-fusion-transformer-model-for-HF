# preprocessing/medical_mapping.py
import ast
from utils import logger
from config import ICD9_GEM_PATH

class ICDConverter:
    """
    ICD 编码转换器：负责将 ICD-9 映射为 ICD-10。
    医学背景：使用 GEMs (General Equivalence Mappings) 确保跨数据库的语义一致性。
    """
    def __init__(self, mapping_path=ICD9_GEM_PATH):
        self.mapping = self._load_gem_mapping(mapping_path)

    def _load_gem_mapping(self, path):
        """加载 2017_I9gem.txt 映射字典"""
        mapping_dict = {}
        if not path.exists():
            logger.warning(f"GEM mapping file not found at {path}. Skip conversion.")
            return {}
        
        try:
            with open(path, 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        # 映射文件通常不带点，我们需要统一格式
                        i9, i10 = parts[0].strip(), parts[1].strip()
                        mapping_dict[i9] = i10
            logger.info(f"Loaded {len(mapping_dict)} ICD mapping pairs.")
        except Exception as e:
            logger.error(f"Error loading GEM mapping: {e}")
        return mapping_dict

    def convert(self, codes):
        """
        转换输入的代码列表。
        输入可以是 ['428.0', '250.0'] 或字符串格式。
        """
        if not codes or not isinstance(codes, list):
            return ['UNKNOWN']

        converted = []
        for code in codes:
            # 临床代码清理：转大写、去空格、去小数点
            cleaned = str(code).strip().upper().replace('.', '')
            # 查找映射，找不到则保留原样（可能是已经是ICD-10或无法映射的代码）
            target_code = self.mapping.get(cleaned, cleaned)
            converted.append(target_code)
        
        return list(set(converted)) # 去重