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

### 1. Internal Validation (MIMIC-IV Test Set, $n=19,701$)
| Model | AUC-ROC (95% CI) | AUC-PR (95% CI) | F1-Score (95% CI) | Precision (95% CI) | Recall (95% CI) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.730 (0.709-0.752) | 0.390 (0.351-0.431) | 0.367 (0.351-0.383) | 0.312 (0.296-0.328) | 0.445 (0.429-0.461) |
| LightGBM | 0.722 (0.701-0.744) | 0.372 (0.334-0.411) | 0.402 (0.385-0.418) | 0.365 (0.349-0.381) | 0.448 (0.432-0.464) |
| XGBoost | 0.719 (0.699-0.739) | 0.372 (0.338-0.412) | 0.405 (0.389-0.421) | 0.368 (0.352-0.384) | 0.450 (0.434-0.466) |
| MLP | 0.696 (0.673-0.719) | 0.359 (0.322-0.402) | 0.386 (0.370-0.402) | 0.335 (0.319-0.351) | 0.452 (0.436-0.468) |
| Random Forest | 0.703 (0.682-0.726) | 0.356 (0.322-0.394) | 0.398 (0.382-0.414) | 0.359 (0.343-0.375) | 0.446 (0.430-0.462) |
| **MFT-HF (Ours)** | **0.752 (0.732-0.772)** | **0.433 (0.396-0.471)** | **0.440 (0.424-0.456)** | **0.386 (0.370-0.402)** | **0.508 (0.492-0.524)** |

### 2. External Validation (eICU-CRD Database, $n=2,949$)
| Model | AUC-ROC (95% CI) | AUC-PR (95% CI) | F1-Score (95% CI) | Sensitivity (95% CI) | Specificity (95% CI) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.698 (0.675-0.720) | 0.332 (0.299-0.367) | 0.295 (0.255-0.335) | 0.542 (0.478-0.606) | 0.648 (0.629-0.667) |
| MLP | 0.666 (0.643-0.689) | 0.308 (0.274-0.340) | 0.278 (0.238-0.318) | 0.513 (0.449-0.577) | 0.663 (0.644-0.682) |
| LightGBM | 0.680 (0.658-0.701) | 0.307 (0.275-0.339) | 0.275 (0.235-0.315) | 0.510 (0.445-0.575) | 0.660 (0.640-0.680) |
| XGBoost | 0.624 (0.597-0.648) | 0.280 (0.250-0.314) | 0.249 (0.209-0.289) | 0.433 (0.369-0.497) | 0.759 (0.741-0.777) |
| Random Forest | 0.629 (0.605-0.654) | 0.259 (0.232-0.289) | 0.267 (0.227-0.307) | 0.471 (0.407-0.535) | 0.742 (0.724-0.760) |
| **MFT-HF (Ours)** | **0.704 (0.682-0.726)** | **0.373 (0.337-0.412)** | **0.329 (0.289-0.369)** | **0.583 (0.519-0.647)** | **0.694 (0.675-0.713)** |

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
git clone https://github.com/starsman-lab/MFT-HF.git
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
├── 📁 extraction/         # SQL scripts for MIMIC-IV & eICU-CRD cohort extraction
├── 📁 models/             # Core architecture (GNN Encoder, Transformer, Dual-path fusion)
├── 📁 preprocessing/      # Feature engineering, KNN imputation, and ICD mapping
├── 📁 results/            # Performance logs and ablation study outputs
├── 📁 figures/            # Reproducible ROC, PR, DCA, and Attention Heatmaps
├── 📄 config.py           # Global hyperparameters and environment paths
├── 📄 train.py            # Main training and probability calibration pipeline
├── 📄 evaluate.py         # Performance evaluation and visualization script
├── 📄 benchmarks.py       # Comprehensive baseline comparisons (LR, XGBoost, etc.)
└── 📄 interpretability.py # Multi-level attention analysis (Figure 6)
```

---

## 📚 Selected References
1. **Clinical Guidelines**: Targeted Stage A HF prevention based on **ACC/AHA/HFSA** guidelines [3, 6].
2. **CKM Syndrome**: Aligning with the **AHA Presidential Advisory** on Cardiovascular-Kidney-Metabolic health [33].
3. **Gender Equity**: Addressing sex-specific risk factors as identified by the **Lancet Commission** [32].
4. **Relational Learning**: Utilizing GNNs for medical knowledge prior encoding [21, 27].

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


