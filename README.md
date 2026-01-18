# MFT-HF: Modality-Aware Fusion Transformer for Stage A Heart Failure Prediction

[![Status](https://img.shields.io/badge/Status-Manuscript--Submitted-blue.svg)](https://www.thelancet.com/journals/eclinm/home)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Official implementation of the paper: **"Modality-Aware Deep Learning Model for Risk Prediction of Stage A Heart Failure: A Retrospective Cohort Study"**.

> [!IMPORTANT]
> This manuscript has been **submitted** to *eClinicalMedicine* for consideration.

---

## 🌟 Research Highlights
Stage A Heart Failure (Stage A HF) represents a critical yet overlooked window for preventive intervention. **MFT-HF** addresses the challenges of subtle risk signals and heterogeneous EHR data through several innovations:

*   **Comorbidity Topology Modeling**: Uses self-supervised **Graph Neural Networks (GNNs)** to explicitly encode the relational structure of ICD diagnostic codes, capturing patterns like the "Cardiovascular-Kidney-Metabolic (CKM)" syndrome.
*   **Dual-Path Fusion Architecture**: A novel **Modality-Aware Transformer** that captures complex cross-modal interactions (Interaction Path) while preserving independent predictive signals from specific clinical domains (Independence Path).
*   **Algorithmic Fairness**: Demonstrated robust performance across biological sex and age groups, specifically improving risk stratification for historically underdiagnosed female patients.
*   **Clinical Interpretability**: Multi-level attention mechanisms align model logic with clinical reasoning, identifying high-impact variables like loop diuretics, beta-blockers, and renal function.

---

<!-- ## 🏗️ Architecture
![Model Architecture](figures/architecture.png)  
*Figure 2: Overview of the MFT-HF framework, featuring modality-specific encoding, cross-modal interaction, and independent signal preservation.*

--- -->

## 📊 Experimental Results

We evaluated the **MFT-HF** framework against five state-of-the-art baseline models. Statistical stability was assessed via 1,000 non-parametric bootstrap resampling iterations to generate 95% Confidence Intervals (CI).

---

### 1. Internal Validation (MIMIC-IV Test Set, $n=19,701$)
| Model | AUC-ROC (95% CI) | AUC-PR (95% CI) | F1-Score (95% CI) | Precision (95% CI) | Sensitivity (95% CI) | Specificity (95% CI) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.730 (0.710-0.750) | 0.390 (0.355-0.432) | 0.416 (0.391-0.445) | 0.302 (0.275-0.343) | 0.675 (0.584-0.730) | 0.687 (0.646-0.778) |
| MLP | 0.696 (0.675-0.720) | 0.359 (0.322-0.399) | 0.384 (0.353-0.418) | 0.300 (0.235-0.377) | 0.570 (0.417-0.758) | 0.716 (0.534-0.851) |
| LightGBM | 0.722 (0.701-0.744) | 0.372 (0.335-0.409) | 0.405 (0.378-0.433) | 0.290 (0.254-0.321) | 0.679 (0.622-0.793) | 0.666 (0.533-0.713) |
| XGBoost | 0.719 (0.697-0.740) | 0.372 (0.338-0.408) | 0.395 (0.368-0.423) | 0.278 (0.243-0.321) | 0.689 (0.565-0.798) | 0.640 (0.535-0.759) |
| Random Forest | 0.703 (0.681-0.725) | 0.356 (0.322-0.395) | 0.391 (0.362-0.421) | 0.292 (0.247-0.340) | 0.605 (0.503-0.736) | 0.700 (0.566-0.786) |
| **MFT-HF (Ours)** | **0.752 (0.730-0.773)** | **0.433 (0.396-0.472)** | **0.428 (0.393-0.466)** | **0.321 (0.267-0.393)** | **0.660 (0.535-0.785)** | **0.714 (0.583-0.831)** |

### 2. External Validation (eICU-CRD Database, $n=2,949$)
| Model | AUC-ROC (95% CI) | AUC-PR (95% CI) | F1-Score (95% CI) | Precision (95% CI) | Sensitivity (95% CI) | Specificity (95% CI) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.698 (0.676-0.721) | 0.330 (0.300-0.368) | 0.387 (0.363-0.410) | 0.268 (0.246-0.293) | 0.695 (0.596-0.739) | 0.620 (0.590-0.714) |
| MLP | 0.666 (0.643-0.689) | 0.306 (0.276-0.343) | 0.358 (0.335-0.385) | 0.257 (0.221-0.322) | 0.627 (0.426-0.769) | 0.624 (0.485-0.816) |
| LightGBM | 0.679 (0.655-0.701) | 0.304 (0.274-0.341) | 0.369 (0.344-0.396) | 0.264 (0.233-0.305) | 0.622 (0.510-0.720) | 0.650 (0.554-0.745) |
| XGBoost | 0.624 (0.600-0.650) | 0.278 (0.244-0.311) | 0.333 (0.309-0.357) | 0.245 (0.211-0.294) | 0.535 (0.375-0.684) | 0.665 (0.502-0.815) |
| Random Forest | 0.629 (0.606-0.652) | 0.259 (0.230-0.291) | 0.334 (0.310-0.357) | 0.226 (0.203-0.281) | 0.663 (0.393-0.788) | 0.541 (0.418-0.798) |
| **MFT-HF (Ours)** | **0.704 (0.682-0.726)** | **0.371 (0.333-0.410)** | **0.390 (0.363-0.417)** | **0.280 (0.247-0.322)** | **0.652 (0.552-0.747)** | **0.662 (0.572-0.754)** |

> [!TIP]
> **AUPRC (AUC-PR)** is the primary metric for this task due to high class imbalance (prevalence ≈ 16.6%). MFT-HF achieved an 11.0% relative improvement in AUPRC over Logistic Regression.

---

## 💻 Environment Setup
This code is optimized for **Ubuntu 22.04** with the following specifications:
*   **Python**: 3.12
*   **PyTorch**: 2.5.1
*   **CUDA**: 12.4

### Installation
```bash
# Clone the repository
git clone https://github.com/starsman-lab/modality-aware-fusion-transformer-model-for-HF.git
cd MFT-HF

# Install PyTorch 2.5.1 for CUDA 12.4
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install PyTorch Geometric and its dependencies
pip install torch_geometric
pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.5.1+cu124.html

# Install remaining requirements
pip install -r requirements.txt
```

---
## 📂 Project Structure
```text
MFT-HF/
├── 📁 data/               # Data mapping files
├── 📁 extraction/         # SQL scripts for MIMIC-IV & eICU-CRD cohort extraction
├── 📁 models/             # Core architecture (GNN Encoder, Transformer, Dual-path fusion)
├── 📁 preprocessing/      # Feature engineering, KNN imputation, and ICD mapping
├── 📄 benchmarks.py       # Comprehensive baseline comparisons (LR, XGBoost, etc.)
├── 📄 config.py           # Global hyperparameters and environment settings
├── 📄 evaluate.py         # Performance evaluation and visualization script
├── 📄 interpretability.py # Multi-level attention analysis
├── 📄 main_extraction.py  # Main entry point for database cohort extraction
├── 📄 requirements.txt    # Project dependencies and environment requirements
├── 📄 train.py            # Main entry point for training and calibration
├── 📄 trainer.py          # Encapsulated training logic, loops, and validation
└── 📄 utils.py            # Helper functions for logging, metrics, and data loading
```

---

## 📚 Selected References

This research is grounded in the latest clinical guidelines and state-of-the-art deep learning methodologies:

*   **Clinical Guidelines for Stage A HF**: Targeted prevention strategies based on the 2022 AHA/ACC/HFSA Guideline for the Management of Heart Failure. *Circulation*. 2022;145(18). [DOI: 10.1161/CIR.0000000000001063]
*   **CKM Syndrome**: Aligning with the Cardiovascular-Kidney-Metabolic (CKM) Health Presidential Advisory from the American Heart Association. *Circulation*. 2023;148(20). [DOI: 10.1161/CIR.0000000000001184]
*   **Gender Equity in Cardiology**: Addressing sex-specific risk factors identified by the Lancet Commission on women’s cardiovascular disease. *The Lancet*. 2021;397(10292). [DOI: 10.1016/S0140-6736(21)00684-X]
*   **Relational Learning in EHR**: Inspired by multi-layer representation learning for medical concepts (Gram/Med2Vec) and Graph Neural Networks for comorbidity modeling. *ACM SIGKDD*. 2016. [DOI: 10.1145/2939672.2939823]

---

## 📄 Citation
If you find this work useful for your research, please cite:
```bibtex
@article{wang2026modality,
  title={Modality-Aware Deep Learning Model for Risk Prediction of Stage A Heart Failure: A Retrospective Cohort Study},
  author={Wang, Zixing and Wang, Yang and Sun, Zhaohong and Gao, Xiaoyuan and Yang, Zhan and Zhang, Xue and Yuan, Jing and Zhao, Wei},
  journal={Manuscript submitted for publication (eClinicalMedicine)},
  year={2026}
}
```

## ⚖️ Disclaimer
* This code is for **academic research only**. 
* The risk scores generated by this model should not be used for direct clinical diagnosis or medical decision-making. The authors are not responsible for any clinical consequences resulting from the use of this software.





