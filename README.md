# Customer Churn Analysis

## 📌 Project Overview

This project focuses on analyzing customer churn and identifying customers who are likely to leave a service.

The project follows an end-to-end data analytics and machine learning workflow using **Excel, SQL, Python, and Power BI**.

## 🎯 Business Problem

Customer churn can negatively impact business revenue. The goal of this project is to analyze customer data, identify important factors related to churn, and build a machine learning model that can help businesses identify customers who may be at risk of leaving.

## 🛠️ Tools & Technologies

* **Excel** – Data cleaning and preprocessing
* **MySQL** – Data storage and SQL analysis
* **Python** – Data analysis and machine learning
* **Pandas & NumPy** – Data manipulation
* **Scikit-learn** – Machine learning
* **Power BI** – Interactive dashboard and visualization
* **Random Forest** – Churn prediction model

## 🔄 Project Workflow

```text
Raw Dataset
     ↓
Excel Data Cleaning
     ↓
MySQL Database
     ↓
SQL Analysis
     ↓
Python Data Analysis
     ↓
Machine Learning Model
     ↓
Power BI Dashboard
     ↓
Business Insights
```

## 📊 Dataset

The dataset contains customer information such as:

* Customer ID
* Credit Score
* Geography
* Gender
* Age
* Tenure
* Balance
* Number of Products
* Credit Card Status
* Active Member Status
* Estimated Salary
* Complaints
* Satisfaction Score
* Card Type
* Points Earned
* Churn Status

The target variable is **Exited**:

* `0` – Customer did not churn
* `1` – Customer churned

## 🧹 Data Cleaning

Data cleaning and preprocessing were performed using Excel and Python.

The process included:

* Handling missing values
* Checking duplicate records
* Checking data types
* Identifying inconsistencies
* Preparing the dataset for SQL analysis and machine learning

## 🗄️ SQL Analysis

MySQL was used to analyze customer behavior and identify patterns related to churn.

The analysis included:

* Customer segmentation
* Churn analysis by geography
* Churn analysis by age
* Churn analysis by gender
* Product and activity analysis
* Customer balance analysis
* Aggregated business metrics

## 🤖 Machine Learning

A **Random Forest Classifier** was used to predict customer churn.

A **Decision Tree Classifier** was also considered for comparison.

Because churn is an imbalanced classification problem, model evaluation was performed using:

* Accuracy
* Precision
* Recall
* F1-score

The model can help identify customers who may be at higher risk of churning.

## 📈 Power BI Dashboard

Power BI was used to create an interactive dashboard for monitoring customer churn.

The dashboard helps analyze:

* Total customers
* Churned customers
* Churn rate
* Customer demographics
* Geographic churn patterns
* Product usage
* Customer activity
* Other important churn-related factors

## 💡 Business Use Case

The analysis can help businesses identify customers who are more likely to churn.

Businesses can use these insights to:

* Identify high-risk customers
* Improve customer retention
* Design targeted offers
* Improve customer satisfaction
* Reduce potential revenue loss

## 📁 Project Structure

```text
customer-churn-analysis/
│
├── README.md
├── data/
├── excel/
├── sql/
├── python/
├── powerbi/
└── images/
```

## 👨‍💻 Author

**Sarvesh Dudhe**

Electronics Engineering | Data Analyst | Machine Learning

Skills: Python | SQL | Excel | Power BI | Machine Learning

