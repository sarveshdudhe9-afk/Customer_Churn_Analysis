import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
df = pd.read_csv("Customer-Churn-Records.csv")

print(df.head())

print("Dataset Shape:", df.shape)
print(df.columns.tolist())
print(df.isnull().sum())
print("Duplicate Rows:",df.duplicated().sum())

df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])
print(df.head)

print(df.columns.tolist())

print(df[["Geography","Gender","Card Type"]].head())

df = pd.get_dummies(
    df,
    columns = ["Geography", "Gender","Card Type"],
    drop_first = True
)
print(df.head())

df= df.astype(int)

X = df.drop("Exited",axis=1)
Y = df["Exited"]
print("Features:", X.shape)
print("Target:", Y.shape)

X_train, X_test,Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size = 0.20,

    random_state = 42,
    stratify = Y
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logistic_model = LogisticRegression(max_iter = 1000)
logistic_model.fit(X_train_scaled, Y_train)

Y_pred_logistic = logistic_model.predict(X_test_scaled)
logistic_accuracy = accuracy_score(Y_test, Y_pred_logistic)
print("Logistic Regression Accuracy:", round(logistic_accuracy * 100,2))

print(classification_report(Y_test, Y_pred_logistic))

cm_logistic = confusion_matrix(Y_test, Y_pred_logistic)
sns.heatmap(cm_logistic, annot= True, fmt = "d")
plt.title("Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

decision_tree = DecisionTreeClassifier(random_state = 42,
                                       max_depth= 5
)
decision_tree.fit(X_train, Y_train)
Y_pred_tree = decision_tree.predict(X_test)

tree_accuracy = accuracy_score(Y_test, Y_pred_tree)

print("Decision Tree Accuracy:",
      round(tree_accuracy * 100,2), "%")

print(classification_report(Y_test,Y_pred_tree))

random_forest = RandomForestClassifier(n_estimators = 200,
                                       random_state = 42,
                                       max_depth = 10)

random_forest.fit(X_train, Y_train)

Y_pred_rf = random_forest.predict(X_test)
rf_accuracy = accuracy_score(Y_test, Y_pred_rf)
print("Random Forest Accuracy:", round(rf_accuracy * 100, 2), "%")
print(classification_report(Y_test,Y_pred_rf))

cm_rf = confusion_matrix(Y_test,Y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt="d")
plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

logistic_accuracy = accuracy_score(Y_test, Y_pred_logistic)
tree_accuracy = accuracy_score(Y_test,Y_pred_tree)
rf_accuracy = accuracy_score(Y_test, Y_pred_rf)

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy": [
        logistic_accuracy,
        tree_accuracy,
        rf_accuracy
    ]
})
results["Accuracy"] = results["Accuracy"] * 100
print(results)

plt.figure(figsize =(8,5))
plt.bar(results["Model"],
        results["Accuracy"])
plt.title("Customer Churn Model Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy (%)")
plt.xticks(rotation = 15)
plt.show

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance":
    random_forest.feature_importances_
    })
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending = False

)
print(feature_importance)

plt.figure(figsize =(10, 6))
plt.barh(
    feature_importance["Feature"].head(10),
    feature_importance["Importance"].head(10)
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 Feature Affecting Customer Churn")

plt.gca().invert_yaxis()
plt.show()

sample_customer = X_test.iloc[[0]]
prediction = random_forest.predict(sample_customer)[0]
if prediction == 1:
    print("Prediction : Customer is likely to CHURN")
else:
    print("Prediction : Customer is likely to STAY")

probability = random_forest.predict_proba(sample_customer)[0][1]
print("Churn Probability : ", round(probability * 100,2), "%")

import joblib
joblib.dump(random_forest, "customer_churn_model.pkl")
joblib.dump(scaler,"customer_churn_scaler.pkl")

print("Model saved successfully.")

print("\n ==== CUSTOMER CHURN PROJECT SUMMARY ====")
print("Total Cutomers:", len(df))
print("Churned Customer:",Y.sum())
print("Retained Customers :",len(Y)- Y.sum())
print("Overall Churn Rate:",round(Y.mean()*100,2) ,"%")
print("Logistic Regression:", round(logistic_accuracy * 100, 2), "%")
print("Decision Tree:",round(tree_accuracy * 100, 2),"%")
print("Decision Tree:",round(tree_accuracy *100,2), "%")
print("Random Forest:", round(rf_accuracy * 100, 2),"%")