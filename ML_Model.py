import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/HP/3D Objects/Desktop/DA_using_Python/DA Projects/Project 2 (Railway AI ML DA)/Railway_Safety_Feature_Engineered.csv",low_memory=False)
print(df.columns.tolist())
print(df.columns)
print(df.isnull().sum().sort_values(ascending=False).head(10))
# Dependent and independent variable
features = [
    "Accident_Type",
    "Weather_Condition",
    "State_Name",
    "Equipment_Type",
    "Train_Speed",
    "Total_Casualties"]
target = "Risk_Category"
df_ml = df[features + [target]].dropna()

# Encoding
le_dict = {}
for col in df_ml.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df_ml[col] = le.fit_transform(df_ml[col])
    le_dict[col] = le

# Train Test Split
X = df_ml[features]
y = df_ml[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42)

# Training 3 different models and comparing and choosing the best one.
# Logistics Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# Random forest 
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# XGBoost
xgb = XGBClassifier(eval_metric="mlogloss", random_state=42)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)

# Evaluation of models through metrics measures
def evaluate_model(name, y_test, y_pred):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred, average="weighted"),
        "F1 Score": f1_score(y_test, y_pred, average="weighted")
        }

# Comparision of Models:
results = []
results.append(evaluate_model("Logistic Regression", y_test, lr_pred))
results.append(evaluate_model("Random Forest", y_test, rf_pred))
results.append(evaluate_model("XGBoost", y_test, xgb_pred))
results_df = pd.DataFrame(results)
print(results_df)

def plot_cm(y_test, y_pred, title):
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5,4))
    plot_cm(y_test, lr_pred, "Logistic Regression")
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    plot_cm(y_test, rf_pred, "Random Forest")
    plt.show()
    plot_cm(y_test, xgb_pred, "XGBoost")
    plt.show()
# Feature Importance
importances = rf.feature_importances_

feature_imp = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances}).sort_values(by="Importance", ascending=False)
print(feature_imp)

plt.figure(figsize=(8,5))
plt.bar(feature_imp["Feature"], feature_imp["Importance"])
plt.xticks(rotation=45)
plt.title("Feature Importance (Random Forest)")
plt.show()

from xgboost import plot_importance

plt.figure(figsize=(8,5))
plot_importance(xgb)
plt.title("XGBoost Feature Importance")
plt.show()