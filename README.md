# Predictive Modeling of Enzymes from Protein Data using Machine Learning

## 📌 Project Overview  
Enzymes are essential proteins that catalyze biochemical reactions and play a vital role in plant growth, metabolism, and stress adaptation. Identifying enzymes in crops like **rice (Oryza sativa)** is crucial for **agricultural biotechnology and functional genomics**.  

This project develops a **machine learning pipeline** to classify enzymes vs. non-enzymes in the **rice genome dataset released by the Government of India**. Using **UniProt protein sequences** and **biochemical/sequence-derived features**, models were trained to predict enzyme functions with high accuracy.  

🔑 **Key Highlights:**  
- Dataset: **62,730 protein sequences** curated from India Govt. rice genome + UniProt mapping  
- Features: **28 extracted features** including molecular weight, isoelectric point, GRAVY, instability index, amino acid composition, sequence length  
- Models: Random Forest, XGBoost, LightGBM, and an **Ensemble Voting Classifier**  
- Metrics: **F1 Score = 0.768, ROC AUC = 0.94**   (Voting Classifier results on test set)
- Tools: **Python, Biopython, Scikit-learn, Imbalanced-learn**  

---

## ⚙️ Methodology  

### 1. Data Acquisition  
- Rice genome dataset obtained from the **Government of India** public release  
- Protein identifiers mapped to **UniProt** for retrieving full protein sequences  
- Proteins annotated as **enzymes** or **non-enzymes** using Enzyme Commission (EC) numbers  

### 2. Feature Extraction  
Extracted 28 biochemical and sequence-derived features using **Python & Biopython**:  
- Molecular weight  
- Isoelectric point (pI)  
- GRAVY (hydrophobicity score)  
- Instability index  
- Amino acid composition (20 amino acids)  
- Sequence length  

### 3. Preprocessing  
- Removed duplicates & incomplete entries  
- Standardized features with **StandardScaler**  
- Addressed class imbalance using **SMOTE (Synthetic Minority Over-sampling Technique)**  

### 4. Model Training  
Implemented and optimized multiple models:  
- Random Forest  
- XGBoost  
- LightGBM  
- **Ensemble Voting Classifier (soft voting)**  

Optimization:  
- Hyperparameter tuning with **RandomizedSearchCV**  
- 3-fold cross-validation for generalization  

### 5. Evaluation Metrics  
- Accuracy  
- Precision  
- Recall  
- **F1 Score (primary metric due to imbalance)**  
- ROC AUC  

---

## 📊 Results  

The **Voting Classifier** achieved the best performance:  

| Model              | F1 Score | Precision | Recall | Accuracy | ROC AUC |  
|--------------------|----------|-----------|--------|----------|---------|  
| Voting Classifier  | 0.767    | 0.746     | 0.789  | 0.881    | 0.936   |  
| XGBoost            | 0.765    | 0.750     | 0.780  | 0.890    | 0.934   |  
| LightGBM           | 0.752    | 0.718     | 0.789  | 0.881    | 0.931   |  
| Random Forest      | 0.745    | 0.712     | 0.780  | 0.877    | 0.928   |  

✅ Ensemble learning outperformed individual models.  
✅ ROC AUC > 0.92 across all models indicates strong discriminative ability.  

---

## 🛠️ Tech Stack  

- **Programming**: Python (3.8+)  
- **Libraries**:  
  - Data Handling: Pandas, NumPy  
  - ML Models: Scikit-learn, XGBoost, LightGBM  
  - Bioinformatics: Biopython  
  - Imbalanced Data: imbalanced-learn (SMOTE)  
  - Visualization: Matplotlib, Seaborn  

---

