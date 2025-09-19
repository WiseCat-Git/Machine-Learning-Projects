# Machine-Learning-Projects

# Machine Learning Projects - Master's in Applied AI

This repository contains comprehensive machine learning projects developed during my **Master's in Applied Artificial Intelligence** at IEP + Summa University. These projects demonstrate progressive mastery of ML concepts from supervised learning fundamentals to advanced semi-supervised techniques and model interpretability.

## Course Structure & Learning Progression

### Unit 1: Supervised Learning Foundations
**Files:** `solucion_caso_practico_iep_iaa_ml_u1.py`, `unidad_1_c2_machine_learning_caso_practico.py`

**Core Competencies:**
- Classification algorithm implementation and comparison
- Model evaluation methodologies and metrics
- Industry case study application (energy fraud detection)
- Feature engineering and preprocessing pipelines

### Unit 2: Unsupervised Learning & Clustering
**File:** `solucion_iep_ml_u2.py`

**Advanced Concepts:**
- Manual K-Means implementation from scratch
- Comparative analysis: manual vs. scikit-learn implementations
- Clustering optimization and evaluation techniques
- Data normalization impact analysis

### Unit 3: Advanced ML Techniques
**File:** `solucion_enunciados_i&ii_iep_iaa_ml_u3.py`

**Cutting-Edge Applications:**
- Semi-supervised learning with limited labeled data
- Model interpretability using LIME and SHAP
- Advanced evaluation techniques for real-world datasets
- Hyperparameter optimization with GridSearchCV

## Project Portfolio Overview

### 1. Multi-Algorithm Classification Study
**File:** `solucion_caso_practico_iep_iaa_ml_u1.py`

**Objective:** Comprehensive comparison of classification algorithms across multiple datasets

| Algorithm | Implementation Details | Key Metrics |
|-----------|----------------------|-------------|
| **Logistic Regression** | Linear classifier with regularization | Precision, Recall, F1-Score |
| **k-Nearest Neighbors** | Distance-based classification | Cross-validation accuracy |
| **Support Vector Machine** | Linear and RBF kernels | ROC-AUC, confusion matrices |
| **Decision Tree** | CART algorithm implementation | Feature importance analysis |
| **Random Forest** | Ensemble method with bagging | Out-of-bag error estimation |
| **Gradient Boosting** | Sequential ensemble learning | Learning curve analysis |

**Technical Achievements:**
- Implemented stratified cross-validation for robust evaluation
- Applied comprehensive metric analysis (accuracy, precision, recall, F1-score)
- Conducted dataset-specific algorithm performance analysis
- Demonstrated understanding of algorithm selection based on data characteristics

### 2. Energy Fraud Detection System
**File:** `unidad_1_c2_machine_learning_caso_practico.py`

**Industry Application:** Real-world fraud detection in energy sector

**Dataset Engineering:**
- Customer profiling (residential, commercial, industrial)
- Consumption pattern analysis and anomaly scoring
- Billing discrepancy detection
- Geographic and demographic feature integration

**Technical Implementation:**
- Feature engineering with categorical encoding and standardization
- Random Forest classifier with hyperparameter tuning
- Feature importance analysis for business insights
- Comprehensive evaluation with confusion matrices

**Business Impact Analysis:**
- Identified key fraud indicators (meter type, billing discrepancies, consumption patterns)
- Developed actionable recommendations for fraud prevention
- Created risk scoring methodology for targeted inspections

### 3. Advanced Clustering Implementation
**File:** `solucion_iep_ml_u2.py`

**Research Focus:** Manual implementation vs. production libraries

**Technical Components:**
- **Manual K-Means Algorithm:**
  - Centroid initialization strategies
  - Distance calculation and cluster assignment
  - Convergence criteria implementation
  - Iterative optimization process

- **Comparative Analysis:**
  - Performance benchmarking: manual vs. scikit-learn
  - Scalability assessment
  - Accuracy validation

- **Optimization Techniques:**
  - Elbow method for optimal k selection
  - Silhouette analysis for cluster quality
  - Data normalization impact studies

### 4. Semi-Supervised Learning & Interpretability
**File:** `solucion_enunciados_i&ii_iep_iaa_ml_u3.py`

**Research Challenge:** Learning with limited labeled data (6 labeled samples from 1000 total)

#### Case Study 1: Semi-Supervised Classification
**Methodologies Implemented:**
- **Label Propagation:** Graph-based semi-supervised learning
- **Self-Training Classifier:** Iterative pseudo-labeling approach
- **Hybrid Approaches:** Combined propagation and self-training

**Technical Results:**
- Label Propagation: 60% accuracy with 92.3% recall
- Self-Training SVM: 53% accuracy with improved precision
- Hyperparameter optimization across multiple kernels

#### Case Study 2: Model Interpretability
**Boston Housing Regression:**
- Random Forest implementation with feature importance analysis
- LIME (Local Interpretable Model-agnostic Explanations) integration
- SHAP (SHapley Additive exPlanations) value computation
- Global vs. local interpretability comparison

**Titanic Survival Classification:**
- Logistic Regression vs. Decision Tree comparison
- Advanced preprocessing pipeline design
- LIME explanations for individual predictions
- Business-interpretable feature analysis

## Technical Skills Demonstrated

### Machine Learning Algorithms
- **Supervised Learning:** Classification and regression across 7+ algorithms
- **Unsupervised Learning:** Clustering with manual and library implementations
- **Semi-Supervised Learning:** Advanced techniques for limited labeled data
- **Ensemble Methods:** Random Forest, Gradient Boosting with optimization

### Model Evaluation & Validation
- **Cross-Validation:** Stratified k-fold for robust assessment
- **Metrics:** Comprehensive evaluation (accuracy, precision, recall, F1, ROC-AUC)
- **Statistical Analysis:** Silhouette scores, within-cluster sum of squares
- **Hyperparameter Tuning:** GridSearchCV with multiple parameter spaces

### Data Processing & Engineering
- **Feature Engineering:** Categorical encoding, normalization, scaling
- **Data Quality:** Missing value imputation, outlier detection
- **Pipeline Design:** End-to-end preprocessing workflows
- **Validation:** Train-test splits with stratification

### Model Interpretability & Explainability
- **LIME:** Local explanations for individual predictions
- **SHAP:** Global feature importance and interaction analysis
- **Feature Importance:** Tree-based and permutation importance
- **Visualization:** Comprehensive plots for model understanding

## Advanced Technical Implementations

### Manual Algorithm Development
```python
# Example: Manual K-Means Implementation
def kmeans_manual(data, k, max_iters=100, tol=1e-4):
    centroids = initialize_centroids(data, k)
    for _ in range(max_iters):
        clusters = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, clusters, k)
        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        centroids = new_centroids
    return clusters, centroids
```

### Semi-Supervised Learning Pipeline
```python
# Label Propagation with Self-Training enhancement
label_prop_model = LabelPropagation(max_iter=2000)
label_prop_model.fit(X_train, y_train_unlabeled)
y_train_propagated = label_prop_model.transduction_

self_training_model = SelfTrainingClassifier(best_svm, max_iter=50)
self_training_model.fit(X_train, y_train_propagated)
```

## Real-World Application Impact

### Industry Case Studies
- **Energy Sector:** Fraud detection system with actionable business insights
- **Healthcare Data:** Semi-supervised learning for limited annotation scenarios
- **Financial Analysis:** Interpretable models for regulatory compliance

### Business Value Creation
- **Risk Assessment:** Automated scoring systems for fraud detection
- **Resource Optimization:** Targeted inspection strategies based on ML predictions
- **Decision Support:** Interpretable AI for stakeholder confidence

## Technology Stack

### Core Libraries
- **scikit-learn:** Production ML algorithms and evaluation
- **pandas/numpy:** Data manipulation and numerical computing
- **matplotlib/seaborn:** Statistical visualization and analysis

### Specialized Tools
- **LIME:** Local interpretability for black-box models
- **SHAP:** Advanced explainability with Shapley values
- **GridSearchCV:** Automated hyperparameter optimization

### Development Environment
- **Google Colab:** Cloud-based Jupyter environment
- **Drive Integration:** Seamless data pipeline management

## Academic Excellence & Research Rigor

**Institution:** IEP + Summa University  
**Program:** Master's in Applied Artificial Intelligence  
**Academic Year:** 2024-2025  
**Specialization:** Machine Learning Theory and Applications

### Learning Outcomes Achieved
1. **Algorithm Mastery:** Deep understanding of 7+ ML algorithms with manual implementations
2. **Evaluation Expertise:** Comprehensive model assessment using industry-standard metrics
3. **Real-World Application:** Successfully applied ML to industry case studies
4. **Research Skills:** Advanced techniques in semi-supervised learning and interpretability
5. **Technical Communication:** Clear documentation and results interpretation

## Connection to Professional AI/ML Development

These projects directly support advanced coursework and professional development in:
- **Deep Learning:** Foundation for neural network implementations
- **MLOps:** Model evaluation and validation pipelines
- **Production AI:** Interpretability requirements for enterprise deployment
- **Research & Development:** Semi-supervised learning for data-scarce scenarios

## Future Applications

The methodologies and implementations developed here serve as building blocks for:
- **Computer Vision:** Transfer learning with limited labeled data
- **Natural Language Processing:** Semi-supervised text classification
- **Time Series Forecasting:** Advanced ensemble methods
- **Recommender Systems:** Interpretable AI for user experience

---

*This repository demonstrates the progression from foundational ML concepts to advanced research-level implementations, bridging academic rigor with practical industry applications in artificial intelligence.*
