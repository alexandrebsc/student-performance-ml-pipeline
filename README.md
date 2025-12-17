# Student Performance Prediction – Applied Machine Learning Project

This repository contains an end-to-end **machine learning project** focused on predicting student performance indicators based on academic and behavioral features.

The project demonstrates a complete workflow including **data ingestion, exploratory analysis, feature preparation, model training, evaluation, and inference**, along with a simple **Streamlit application** for interactive predictions.

> ⚠️ This project is intended for educational and portfolio purposes.

## 📌 Project Overview

Educational institutions often need data-driven insights to better understand factors influencing student performance.  
This project explores how historical student data can be used to **predict performance indicators** using supervised machine learning techniques.

The main goals are:
- Apply structured data preparation (ETL)
- Train and evaluate regression models
- Provide reproducible and interpretable results
- Offer a simple interface for model inference

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

The dataset used in this project is provided in the `data/` directory.

- **File:** `pede_passos.csv`
- **Description:** Structured dataset containing academic and behavioral attributes related to student performance
- **Target:** Student performance index (INDE)

Basic preprocessing steps include:
- Handling missing values
- Feature selection
- Data normalization (where applicable)


## 🧠 Model & Approach

- **Model:** Random Forest Regressor
- **Problem Type:** Regression
- **Training Strategy:** Train/test split
- **Evaluation Metrics:**
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - R² Score

The Random Forest model was chosen due to its robustness to non-linear relationships and minimal feature scaling requirements.


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
├── data/                 # Dataset
├── models/               # Trained model artifacts
├── scr/                  # Source code
│   ├── run_eda.py        # Exploratory data analysis
│   ├── run_train.py     # Model training
│   ├── run_predict.py   # Batch prediction
│   ├── run_app.py       # Streamlit app
│   └── utils/            # ETL and constants
├── figs/                 # Generated plots
├── requirements.txt
├── README.md
└── LICENSE
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
python scr/run_eda.py
```

### Train the Model
```bash
python scr/run_train.py
```

### Run Predictions
```bash
python scr/run_predict.py
```

### Launch Streamlit App
```bash
streamlit run scr/run_app.py
```


## 🧪 Results & Model Performance

A **Random Forest Regressor** was trained using a grid search strategy to select optimal hyperparameters.

**Best hyperparameters:**
- Number of trees: 100
- Max features: `sqrt`
- Max depth: 12

### Performance Metrics

| Dataset | MAE | MSE | R² | Max Absolute Error |
-|
| Train | 0.028 | 0.0013 | 0.911 | 0.129 |
| Test | 0.044 | 0.0031 | 0.751 | 0.220 |
| Full Comparison | 0.032 | 0.0018 | 0.876 | 0.220 |

### Baseline Comparison

As a baseline, a naïve model that always predicts the **mean target value** would yield an R² score close to **0.0**.

Compared to this baseline:
- The trained model explains approximately **75% of the variance** on unseen data
- MAE remains low relative to the target scale (0–10)

### Interpretation

These results indicate that the model:
- Captures meaningful relationships between academic indicators and student performance
- Generalizes reasonably well given the dataset size
- Shows limited overfitting, as reflected by the train–test performance gap

The observed maximum error suggests that while predictions are generally accurate, **individual outliers remain**, which is expected in educational performance data.



## ⚠️ Limitations

- Although a **grid search strategy** is implemented to explore multiple hyperparameter configurations, the search space is intentionally limited and does not guarantee a globally optimal model.
- The dataset size is relatively small, which constrains model generalization to unseen populations.
- A **data augmentation strategy** was tested to increase training data volume; however, it did not lead to measurable performance improvements and was therefore not used in the final training pipeline.
- No cross-validation was applied; model evaluation relies on a single train/test split.
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
