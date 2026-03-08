## Predicting-Churn-for-bank-Customers
## Project Overview
This project focuses on **predicting customer churn in a bank** using Machine Learning techniques. Customer churn refers to customers who exit or close their bank accounts. The goal is to build a robust end-to-end ML pipeline that preprocesses data, trains models, evaluates performance, and makes reliable predictions on unseen test data.

## Dataset Description
The dataset contains customer-level information such as:
* Demographic details ( Geography, Gender)
* Financial attributes (Balance, Credit Score, Estimated Salary)
* Behavioral indicators ( Number of Products, Active Member)
* Target variable: **Exited** (1 = Customer churned, 0 = Retained)
  
## Code Sections:
### 1️.Library Imports
All required Python libraries are imported at the beginning, including:
* NumPy & Pandas for data handling
* Matplotlib for visualization
* Scikit-learn for preprocessing, modeling, and evaluation

This ensures a clean and organized environment for ML development.

### 2️.Data Loading & Initial Exploration
* Training and test datasets are loaded
* Shape, column names, and basic statistics are examined
* Missing values and data types are checked

 To understand data quality and structure before preprocessing.

### 3️.Feature Engineering
Key preprocessing steps applied on training data:
* Separation of numerical and categorical features
* Scaling numerical features using Min-Max normalization
* Encoding categorical variables (e.g., Geography, Gender)
* Storing `minVec` and `maxVec` for consistent test preprocessing

This step ensures the model receives clean and standardized input features.

### 4️.Model Training with Random Forest + GridSearchCV
* Features (`X_train`) and target (`y_train`) are separated
* Random Forest Classifier is selected due to its robustness
* Hyperparameters are optimized using **GridSearchCV**
* Evaluation metric used: **ROC-AUC**
* Best estimator is extracted and stored as `RF`

This approach improves generalization and avoids manual tuning.

### 5️.Custom Test Data Preprocessing Pipeline (DfPrepPipeline)
A reusable preprocessing function is implemented to ensure:
* Same scaling logic as training data
* Safe handling of missing categorical columns
* Encoding only when columns exist
* Alignment of test features with training columns

This avoids common issues like feature mismatch and KeyErrors.

### 6️.Test Data Preparation
* Test data is passed through `DfPrepPipeline`
* Infinite and missing values are handled
* Final shape of test data is verified

To ensure test data matches training feature space exactly.

### 7.Model Prediction on Test Data
* Class predictions using `RF.predict()`
* Probability predictions using `RF.predict_proba()`

These outputs are used for performance evaluation.

### 8️.Model Evaluation
The following metrics are computed:
* **Classification Report** (Precision, Recall, F1-score)
* **ROC-AUC Score** for overall model discrimination
  
These metrics provide a balanced view of model performance, especially for imbalanced data.

### 9️.ROC Curve Visualization
* False Positive Rate vs True Positive Rate plotted
* Model ROC curve compared against random classifier baseline

This visualization helps assess the model’s predictive power.

## Key Highlights
* End-to-end ML pipeline (Train → Test → Evaluate)
* Robust preprocessing with feature alignment
* Hyperparameter optimization using GridSearchCV
* Strong evaluation using ROC-AUC
* Interview-ready, production-style workflow
  
## Business Impact:
- Identifies high-risk churn customers
- Enables proactive retention strategies
- Helps reduce revenue loss
  
## Future Improvements
* Try advanced models (XGBoost, LightGBM)
* Handle class imbalance using SMOTE
* Deploy model using Flask or Streamlit
* Add SHAP-based feature importance for explainability
  
## 👩‍💻 Author
**Apurwa Khare**
MCA (AIML) | Artificial Intelligence & Machine Learning

This project demonstrates strong fundamentals in data preprocessing, machine learning, and model evaluation.
