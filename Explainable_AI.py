import shap as sh
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
# Load dataset, low_memory=False avoids datatype warning issues in large datasets
df = pd.read_csv(
    "C:/Users/HP/3D Objects/Desktop/DA_using_Python/DA Projects/Project 2 (Railway AI ML DA)/Railway_Safety_Feature_Engineered.csv",
    low_memory=False)

# Print dataset structure and missing values for quick inspection
print(df.columns.tolist())
print(df.columns)
print(df.isnull().sum().sort_values(ascending=False).head(10))


# Selecting features and target variable
# These features are used to predict accident risk category
features = [
    "Accident_Type",
    "Weather_Condition",
    "State_Name",
    "Equipment_Type",
    "Train_Speed",
    "Total_Casualties"]

target = "Risk_Category"
df_ml = df[features + [target]].dropna()    # Remove rows where any selected column is missing

# Encoding categorical variables
le_dict = {}
for col in df_ml.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df_ml[col] = le.fit_transform(df_ml[col])
    le_dict[col] = le

# Train-Test Split
# X = input features
# y = target variable (Risk_Category)
X = df_ml[features]
y = df_ml[target]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# XGBoost Model Training
xgb = XGBClassifier(eval_metric="mlogloss", random_state=42)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)

# Explainable AI using SHAP (Explains how each feature contributes to a prediction)
# TreeExplainer works specifically with tree-based models like XGBoost
explainer = sh.TreeExplainer(xgb)

# Compute SHAP values for test dataset. Each value shows how much each feature pushes prediction higher or lower
shap_values = explainer.shap_values(X_test)


# Global Explanation (Summary Plot)
sh.summary_plot(shap_values, X_test)
sh.summary_plot(shap_values, X_test, plot_type="bar") # type: ignore