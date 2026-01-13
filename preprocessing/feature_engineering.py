import pandas as pd
import numpy as np
import torch
import joblib
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoTokenizer, AutoModel

from config import (
    DEVICE, BIOBERT_MODEL_PATH, BIOBERT_BATCH_SIZE, BIOBERT_MAX_LEN, SAVED_MODELS_DIR, CACHE_DIR,
    GNN_OUT_CHANNELS, GNN_HIDDEN_CHANNELS, GNN_EPOCHS, GNN_LR, GNN_MIN_COOCCURRENCE,
    PCA_DRUG_COMPONENTS, NUMERIC_FEATURES_BASE, CATEGORICAL_FEATURES_BASE
)
from utils import logger, clear_memory, get_text_hash
from models.gnn_encoder import HeteroGNN
from preprocessing.graph_utils import (
    build_heterogeneous_graph_generic, 
    train_heterognn_generic, 
    generate_patient_level_gnn_embeddings_generic
)

# --- 1. BioBERT 语义嵌入 (带缓存逻辑) ---
def get_biobert_embeddings(text_lists_series, feature_name="biobert_embed"):
    """
    使用 BioBERT 提取文本特征。
    利用哈希缓存避免重复计算耗时的 BioBERT 嵌入。
    """
    text_hash = get_text_hash(text_lists_series)
    cache_path = CACHE_DIR / f"{feature_name}_{text_hash}.npy"

    if cache_path.exists():
        logger.info(f"Loading cached {feature_name} embeddings.")
        return np.load(cache_path)

    logger.info(f"Generating BioBERT embeddings for {len(text_lists_series)} samples...")
    text_lists = text_lists_series.apply(lambda x: " ".join(x) if isinstance(x, list) else "").tolist()

    tokenizer = AutoTokenizer.from_pretrained(BIOBERT_MODEL_PATH)
    model = AutoModel.from_pretrained(BIOBERT_MODEL_PATH).to(DEVICE)
    model.eval()

    all_embeddings = []
    for i in tqdm(range(0, len(text_lists), BIOBERT_BATCH_SIZE), desc="BioBERT Infer"):
        batch_texts = text_lists[i : i + BIOBERT_BATCH_SIZE]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=BIOBERT_MAX_LEN, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 使用 [CLS] 向量作为句表示
        batch_embs = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(batch_embs)
        clear_memory()

    final_embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    np.save(cache_path, final_embeddings)
    return final_embeddings

# --- 2. 文本特征的简单编码 (One-hot/Count) ---
def get_simple_text_embeddings(text_lists_series, prefix, max_features=100, train_mode=True):
    """
    处理高频药物或诊断。
    """
    processed_texts = text_lists_series.apply(lambda lst: " ".join(str(i) for i in lst) if isinstance(lst, list) else "")
    vectorizer_path = SAVED_MODELS_DIR / f"{prefix}vectorizer.pkl"

    if train_mode:
        vectorizer = CountVectorizer(binary=True, max_features=max_features, min_df=5)
        embeddings_sparse = vectorizer.fit_transform(processed_texts)
        joblib.dump(vectorizer, vectorizer_path)
    else:
        vectorizer = joblib.load(vectorizer_path)
        embeddings_sparse = vectorizer.transform(processed_texts)

    return pd.DataFrame(embeddings_sparse.toarray(), index=text_lists_series.index, 
                        columns=[f"{prefix}_{i}" for i in range(embeddings_sparse.shape[1])])

# --- 3. process_features ---
def process_features(df, icd_col, drug_col, icd_method='gnn', drug_method='biobert', train_mode=True):
    """
    对应论文：多模态特征工程。
    整合结构化数据、ICD 图嵌入和药物语义。
    """
    X = df.copy()
    logger.info(f"Running Feature Engineering (Train={train_mode})...")

    # A. 结构化特征处理 (数值 + 类别)
    X_numeric = X[NUMERIC_FEATURES_BASE].apply(pd.to_numeric, errors='coerce')
    
    # 状态持久化：确保测试集使用训练集的 Imputer 和 Scaler
    imputer_path = SAVED_MODELS_DIR / "numeric_imputer.pkl"
    scaler_path = SAVED_MODELS_DIR / "scaler_numeric.pkl"

    if train_mode:
        imputer = KNNImputer(n_neighbors=5).fit(X_numeric)
        scaler = StandardScaler().fit(imputer.transform(X_numeric))
        joblib.dump(imputer, imputer_path)
        joblib.dump(scaler, scaler_path)
    else:
        imputer = joblib.load(imputer_path)
        scaler = joblib.load(scaler_path)

    X_struct_final = pd.DataFrame(scaler.transform(imputer.transform(X_numeric)), 
                                  columns=NUMERIC_FEATURES_BASE, index=X.index)

    # B. ICD 编码处理 (GNN vs BioBERT)
    icd_embed_df = pd.DataFrame(index=X.index)
    if icd_method == 'gnn':
        graph_path = SAVED_MODELS_DIR / "icd_graph.pkl"
        model_path = SAVED_MODELS_DIR / "icd_gnn_weights.pth"
        
        if train_mode:
            # 构建图并训练自监督 GNN
            graph, node_map = build_heterogeneous_graph_generic(X[icd_col], 'icd', 'co_occurs', GNN_MIN_COOCCURRENCE)
            joblib.dump((graph, node_map), graph_path)
            
            gnn_model = HeteroGNN(GNN_HIDDEN_CHANNELS, GNN_OUT_CHANNELS, graph, 'icd', 'co_occurs')
            state_dict = train_heterognn_generic(gnn_model, graph, 'icd', 'co_occurs')
            torch.save(state_dict, model_path)
        else:
            graph, node_map = joblib.load(graph_path)
            gnn_model = HeteroGNN(GNN_HIDDEN_CHANNELS, GNN_OUT_CHANNELS, graph, 'icd', 'co_occurs')
            gnn_model.load_state_dict(torch.load(model_path, map_location=DEVICE))

        # 回退逻辑：如果患者没有任何 ICD 落在图中，使用 BioBERT + PCA 补全
        icd_fallback = get_biobert_embeddings(X[icd_col], "icd_fallback")
        pca_path = SAVED_MODELS_DIR / "icd_pca_fallback.pkl"
        
        if train_mode:
            pca = PCA(n_components=GNN_OUT_CHANNELS).fit(icd_fallback)
            joblib.dump(pca, pca_path)
        else:
            pca = joblib.load(pca_path)
        
        icd_values = generate_patient_level_gnn_embeddings_generic(
            gnn_model, graph, node_map, X[icd_col], 'icd', GNN_OUT_CHANNELS, pca.transform(icd_fallback)
        )
        icd_embed_df = pd.DataFrame(icd_values, index=X.index, columns=[f"icd_gnn_{i}" for i in range(GNN_OUT_CHANNELS)])

    # C. 药物编码处理 (BioBERT + PCA)
    drug_embed_df = pd.DataFrame(index=X.index)
    if drug_method == 'biobert':
        drug_raw = get_biobert_embeddings(X[drug_col], "drug_biobert")
        drug_pca_path = SAVED_MODELS_DIR / "drug_pca.pkl"
        
        if train_mode:
            pca_drug = PCA(n_components=PCA_DRUG_COMPONENTS).fit(drug_raw)
            joblib.dump(pca_drug, drug_pca_path)
        else:
            pca_drug = joblib.load(drug_pca_path)
            
        drug_values = pca_drug.transform(drug_raw)
        drug_embed_df = pd.DataFrame(drug_values, index=X.index, columns=[f"drug_pca_{i}" for i in range(PCA_DRUG_COMPONENTS)])

    # D. 特征合并
    final_df = pd.concat([X_struct_final, icd_embed_df, drug_embed_df], axis=1).astype(np.float32)
    
    # 记录各模态维度，供模型初始化使用
    dims = {
        "struct": X_struct_final.shape[1],
        "icd": icd_embed_df.shape[1],
        "drug": drug_embed_df.shape[1]
    }
    
    return final_df, dims