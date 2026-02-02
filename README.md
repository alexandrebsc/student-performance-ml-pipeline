# Student Performance Prediction – Applied Machine Learning Project

This repository contains an end-to-end **applied machine learning project** focused on predicting student academic performance using historical educational and behavioral data.

The project is structured as a **reproducible ML pipeline**, covering data ingestion, preprocessing, model selection via grid search, evaluation against strong baselines, and inference — with an optional **Streamlit application** for interactive predictions.

> ⚠️ This project is intended for educational, analytical, and portfolio purposes.

## 📌 Project Overview

Educational institutions often need early, data-driven signals to better understand student development and intervene proactively.

This project predicts a student’s **INDE (Educational Development Index)** using a limited set of indicators that are available *before the end of the academic year*.
The focus is not on deterministic outcomes, but on **estimating the current trajectory** of a student.

Key objectives:

* Build a clean and modular ML pipeline
* Use realistic feature availability constraints
* Compare model performance against naïve baselines
* Provide interpretable, actionable predictions
* Maintain reproducibility through deterministic splits and seeds


## 🎯 Practical Motivation & Real-World Context

This project was developed with the goal of **supporting student development** by predicting, *before the end of the academic year*, the **INDE (Educational Development Index)** of a student.

The prediction is performed using only a **subset of the indicators** traditionally used to compute the INDE, combined with basic administrative information. Specifically, the model uses:

- Student age  
- Current academic phase  
- Whether the student is a new entrant  
- **IAN** – Level Adequacy Indicator  
- **IEG** – Engagement Indicator  
- **IAA** – Self-Assessment Indicator  

### Minimal Data Collection Strategy

From a practical standpoint, the model was designed with **real-world applicability** in mind.

Among all required inputs:
- Most information is already available in existing administrative systems
- The **IEG** indicator is already collected as part of the official INDE calculation
- The **IAA** is the only additional data required, obtained through a short self-assessment questionnaire (five questions)

This makes the model feasible to apply without introducing significant operational overhead.


### 📌 Practical Applications

The model can be made available to both **educators** and **students**:

- **For teachers:**  
  - Enables efficient analysis of multiple students  
  - Helps identify students who may need additional support  
  - Serves as a tool to validate pedagogical intuition with data  

- **For students:**  
  - Provides visibility into their academic progress during the school year  
  - Encourages reflection on engagement and self-assessment behaviors  


### ⚠️ Responsible Use & Ethical Considerations

Predicting an educational performance index can be **highly sensitive**, as it may impact student motivation.

For this reason, the model was designed to estimate **what the INDE would be if no action is taken**, representing the student’s current trajectory — not a fixed outcome.

The intention is **not to limit expectations**, but to support improvement.  
Students can explore hypothetical scenarios, such as increasing engagement (IEG), to understand how behavioral changes may positively influence outcomes.

By doing so, the model reinforces the importance of engagement and self-assessment as indirect drivers of overall educational development, serving as a **motivational and reflective tool rather than a deterministic judgment**.


## 📊 Dataset

The dataset is stored in the `data/` directory.

* **File:** `pede_passos.csv`
* **Structure:** Multi-year student academic and behavioral indicators
* **Target:** `INDE` (Educational Development Index)

### Data Processing Highlights

The raw dataset is:

* Cleaned for corrupted and invalid rows
* Normalized across multiple academic years (2020–2022)
* Unpivoted into a **student-year format**
* Filtered to exclude undergraduate-level phases

The final modeling dataset contains **only primary and middle-school phases**, ensuring conceptual consistency.


## 🧠 Modeling Approach

* **Problem Type:** Regression
* **Model:** Random Forest Regressor
* **Selection Strategy:** Manual grid search using a validation set
* **Evaluation:** Train / Validation / Test split
* **Baselines:** Mean predictor evaluated on all splits

### Why Random Forest?

* Handles non-linear relationships well
* Robust to feature scaling imperfections
* Requires minimal assumptions about data distributions
* Performs reliably on moderate-sized tabular datasets


## 📊 Exploratory Data Analysis (EDA)

An initial exploratory analysis was conducted to assess data quality, structure, and distribution before model training.

### Dataset Overview
- **Number of records:** 2,247
- **Number of features:** 13
- **Target variable:** `INDE` (Student Performance Index)

**Feature types:**
- Numerical (float/int): Academic indicators, age, year, phase
- Categorical: Student identifier
- Boolean: Student admission status

### Data Quality Checks
- **Missing values:** None detected across all columns
- **Duplicate records:** No duplicates found

This confirms the dataset is **clean and well-structured**, requiring no imputation or deduplication steps.

### Statistical Summary
Key observations:
- Most academic indicators are bounded between **0 and 10**
- The target variable `INDE` has:
  - Mean ≈ **7.06**
  - Standard deviation ≈ **1.20**
- Age distribution is centered around **12–13 years**

These characteristics suggest a **moderate-variance regression problem**, where small absolute errors are meaningful.

### Distribution Analysis
A Shapiro–Wilk test was applied to assess normality:

- **W statistic:** 0.9594  
- **p-value:** < 0.001

This indicates that the target distribution **deviates from normality**, supporting the choice of a **tree-based model** rather than linear regression.


### Artifacts

Generated artifacts include:
- Feature distribution plots
- Correlation heatmaps

These can be found in the `figs/` directory.


## 🏗️ Project Structure

```
│   .gitignore
│   LICENSE
│   README.md
│   requirements.txt
│   
├─── data/                  # Dataset
├─── figs/                  # Generated plots
├─── models/                # Trained model artifacts
│
└───src                     # Source code
    │   run_app.py          # Streamlit app
    │   run_eda.py          # Exploratory data analysis
    │   run_train.py        # Model training
    │   __init__.py
    │
    └───utils               # ETL and constants
        │   constants.py
        │   evaluator.py
        │   pede_passos_loader.py
        │   pede_passos_pipeline.py
        │   pede_passos_preprocessor.py
        │   random_forest_regressor_model.py
        │   __init__.py
        └───
```


## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
pip install -r requirements.txt
```


## ▶️ Running the Project

### Run Exploratory Data Analysis
```bash
python src/run_eda.py
```

### Train the Model
```bash
python src/run_train.py
```

### Launch Streamlit App
```bash
streamlit run src/run_app.py
```


## 🧪 Results & Model Performance

The model is trained using a **train / validation / test split**, with all evaluations compared against a **mean baseline** trained on the training target distribution.

**Best hyperparameters:**
- Number of trees: 25
- Max features: `sqrt`
- Max depth: 12

### 🔹 Model Performance

| Split      | MAE    | MSE     | R²     |
| ---------- | ------ | ------- | ------ |
| Train      | 0.0262 | 0.00115 | 0.9214 |
| Validation | 0.0414 | 0.00291 | 0.7938 |
| Test       | 0.0438 | 0.00303 | 0.7558 |

### 🔹 Mean Baseline Performance

| Split      | MAE    | MSE     | R²      |
| ---------- | ------ | ------- | ------- |
| Train      | 0.0953 | 0.01460 | 0.0000  |
| Validation | 0.0941 | 0.01410 | -0.0001 |
| Test       | 0.0910 | 0.01254 | -0.0095 |

### Interpretation

* The model explains **~76% of the variance on unseen test data**
* Errors are small relative to the INDE scale (0–10)
* Performance is stable across validation and test sets
* Baseline comparison confirms the model captures **meaningful signal**

The train–test gap indicates **limited overfitting**, which is expected and acceptable given dataset size and feature constraints.


## ⚠️ Limitations

- Although a **grid search strategy** is implemented to explore multiple hyperparameter configurations, the search space is intentionally limited and does not guarantee a globally optimal model.
- The dataset size is relatively small, which constrains model generalization to unseen populations.
- A **data augmentation strategy** was tested to increase training data volume; however, it did not lead to measurable performance improvements and was therefore not used in the final training pipeline.
- No cross-validation was applied; model evaluation relies on a single train/validation/test split.
- The project does not include production-level concerns such as monitoring, automated retraining, or model drift detection.




## 🔮 Possible Improvements

- Expand hyperparameter search using randomized search or Bayesian optimization.
- Apply k-fold cross-validation for more robust evaluation.
- Perform feature importance analysis and feature selection.
- Experiment with alternative models (e.g., Gradient Boosting, XGBoost).
- Add experiment tracking and model versioning.
- Integrate the training pipeline with a data ingestion workflow.
- Deploy the model using a lightweight API or cloud-based service.



## 📄 License

This project is licensed under the MIT License.


## 👤 Author

Developed by **Alexandre** as part of a personal exploration into applied machine learning, data analysis, and predictive modeling.
