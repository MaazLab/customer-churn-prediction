# customer-churn-prediction
This project predicts whether a customer is likely to churn (leave a service) based on their account activity and demographic features. It leverages machine learning techniques to help businesses identify at-risk customers and take proactive steps to retain them.

## 🔍 Overview

Customer churn is a major concern for subscription-based businesses. This project uses the **Telco Customer Churn** dataset to build and evaluate a classification model using **Random Forest**.

## 📊 Features Used

- Gender
- SeniorCitizen
- Partner
- Dependents
- Tenure
- PhoneService
- InternetService
- MonthlyCharges
- TotalCharges
- PaymentMethod
- Contract Type
- and more...

## 🧠 Model

- **Algorithm:** Random Forest Classifier
- **Preprocessing:**
  - Label Encoding for categorical variables
  - Standard Scaling for numerical features
- **Evaluation Metrics:**
  - Confusion Matrix
  - Precision, Recall, F1-Score

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Download the dataset
Get the **Telco Customer Churn** dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it inside the `data/` folder as `churn_data.csv`.

### 4. Run the model.
```bash
python churn_prediction.py
```
### 📚 Dependencies
- pandas
- scikit-learn


