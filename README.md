# Predicting Drug Response

This repository contains the code for the machine-learning models described in the paper:  
**Predicting Drug Response with Multi-Task Gradient-Boosted Trees in Epilepsy** by Hellmig et al.

## Overview

The repository includes implementations of both baseline single-task models and a novel multi-task gradient-boosted trees model:

### Baseline Single-Task Models
- **Gradient-Boosted Trees Classifier**: `gradient-boosting.py`
- **Random Forest Classifier**: `random-forest.py`
- **Support Vector Machine (SVM)**: `svm.py`
- **Logistic Regression Model**: `logistic-regression.py`

### Multi-Task Model
- **Multi-Task Gradient-Boosted Trees Model**: `multi-task-gbt.py`

Each file contains the implementation and relevant comments for usage. Details of the models are also provided below.

---

## Model Details and Usage

### Logistic Regression
- **Input**: 
  - `data` (learning data)
  - `outcomes` (outcome vector)
  - `loss function`: `l1`, `l2`, or `elasticnet`
- **Output**: 
  - Mean Area under the Precision Recall Curve (AUC-PRC)
  - Standard Deviation (SD) of AUC-PRC

---

### SVM
- **Input**:
  - `data` (learning data)
  - `outcomes` (outcome vector)
  - `kernel`: `poly`, `rbf`, or `linear`
  - `degree`: (e.g., 2, 3, ...) *(ignored if kernel is not `poly`)*
- **Output**: 
  - Mean Area under the Precision Recall Curve (AUC-PRC)
  - Standard Deviation (SD) of AUC-PRC

---

### Random Forest
- **Input**:
  - `data` (learning data)
  - `outcomes` (outcome vector)
- **Output**: 
  - Mean Area under the Precision Recall Curve (AUC-PRC)
  - Standard Deviation (SD) of AUC-PRC

---

### Gradient-Boosted Trees
- **Input**:
  - `data` (learning data)
  - `outcomes` (outcome vector)
  - List of categorical columns in the dataset *(used for feature importance)* 
- **Output**: 
  - Mean Area under the Precision Recall Curve (AUC-PRC)
  - Standard Deviation (SD) of AUC-PRC
  - 10 most important features for prediction and their directions of influence

---

### Multi-Task Gradient-Boosted Trees
- **Input**:
  - `data` (learning data)
  - `outcomes` (outcome vector)
  - **Note**: Filepaths and task-identification columns need to be specified in the code.
- **Output**:
  - Mean Area under the Precision Recall Curve (AUC-PRC)
  - Standard Deviation (SD) of AUC-PRC
  - Global feature set
  - Task-specific feature sets

---

## Additional Notes
- For more details about the models, their implementation, and their evaluation, refer to the [paper](#) and the comments within the code files.
- The performance metrics reported are based on stratified, nested 5x5-fold cross-validation.

---

## Citation
If you use this code, please cite:  
*Hellmig et al., Predicting Drug Response with Multi-Task Gradient-Boosted Trees in Epilepsy.*
