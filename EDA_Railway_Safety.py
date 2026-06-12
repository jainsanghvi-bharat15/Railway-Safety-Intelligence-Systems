import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Load Dataset
df_project = pd.read_csv("C:/Users/HP/3D Objects/Desktop/DA_using_Python/DA Projects/Project 2 (Railway AI ML DA)/Railway_Safety_Project_Dataset.csv")

# EDA 1: Dataset Overview
print(df_project.shape)
print(df_project.dtypes)
print(df_project.isnull().sum())

# EDA 2: Accident Trend Analysis    (Accidents by Year)
yearly_accidents = (df_project.groupby("Accident_Year_Clean").size().sort_index())
print("\nYearly Accidents:", yearly_accidents)

# EDA 3: Accident Type Analysis
accident_type = (df_project["Accident_Type"].value_counts())
print("\nAccident Types:", accident_type)

# EDA 4: Root Cause Analysis
primary_causes = (df_project["Primary_Accident_Cause"].value_counts().head(20))
print("\nPrimary Causes:", primary_causes)

# EDA 5: State Risk Analysis
state_accidents = (df_project.groupby("State_Name").size().sort_values(ascending=False))
print("\nState Accidents:", state_accidents)

# EDA 6: Fatality Analysis
df_project["Total_Casualties"] = (df_project["Total_Persons_Killed"] + df_project["Total_Persons_Injured"])
print("\nAverage Casualties by Accident Type:")
print(df_project[["Accident_Type", "Total_Casualties"]].groupby("Accident_Type").mean())

# EDA 7: Financial Impact
print("\nFinancial Impact:")
print(df_project["Total_Damage_Cost"].describe())

# EDA 8: Weather Impact
weather_accidents = (df_project["Weather_Condition"].value_counts())
print("\nWeather Accidents:")
print(weather_accidents)

# EDA 9: Speed Impact
print("\nAverage Speed by Accident Type:")
print(df_project.groupby("Accident_Type")["Train_Speed"].mean())

df_project["Severity_Score"] = (df_project["Total_Persons_Killed"] * 10 + df_project["Total_Persons_Injured"] * 5 + df_project["Total_Damage_Cost"] / 100000)
df_project["Risk_Category"] = pd.cut(   #pd.cut() function used to group continuous numerical data into discrete categories or "bins"
    df_project["Severity_Score"],
    bins=[-1,10,50,100,999999],
    labels=[
        "Low",
        "Medium",
        "High",
        "Critical"])

df_project.to_csv("C:/Users/HP/3D Objects/Desktop/DA_using_Python/DA Projects/Project 2 (Railway AI ML DA)/Railway_Safety_Feature_Engineered.csv", index=False)
print("Dataset Saved Successfully")