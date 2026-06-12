"""Exploratory Data Analysis
# Indian Railway Accident Dataset"""
import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned sheet
df = pd.read_excel("C:/Users/HP/3D Objects/Desktop/DA_using_Python/DA Projects/Project 2 (Railway AI ML DA)/Indian_Railway_Cleaned_Data.xlsx", sheet_name="Accidents")

# Dataset Overview
print("Shape:", df.shape)
print("\nColumns:\n",df.columns)
print("\nData Types:\n",df.dtypes)
print("\nMissing Values:\n",df.isnull().sum())

""" Yearly Accident Trend
    value_counts() counts frequency of values
    sort_index() arranges years chronologically"""

accidents_year = (df["Year"].value_counts().sort_index())
plt.figure(figsize=(12,6))
accidents_year.plot(kind="line",marker="o")
plt.title("Accidents Trend by Year")
plt.xlabel("Year")
plt.ylabel("Number of Accidents")
plt.grid(True)
plt.show()

""" Top States
    value_counts() gives category frequency"""
top_states = (df["State"].value_counts().head(10))
plt.figure(figsize=(10,6))
top_states.plot(kind="bar")
plt.title("Top 10 States by Accidents")
plt.xlabel("State")
plt.ylabel("Accident Count")
plt.show()

# Accident Type Distribution
accident_type = (df["Standard_Accident_Type"].value_counts())
plt.figure(figsize=(10,6))
accident_type.plot(kind="bar")
plt.title("Accident Type Distribution")
plt.ylabel("Count")
plt.show()

# Most Common Causes
top_causes = (df["Standard_Cause_Type"].value_counts().head(10))
plt.figure(figsize=(10,6))
top_causes.plot(kind="bar")
plt.title("Top Accident Causes")
plt.ylabel("Count")
plt.show()

# Monthly Accident Trend
if "Month" in df.columns:
    monthly = (df["Month"].dropna().value_counts().sort_index())
    if len(monthly) > 0:
        plt.figure(figsize=(10,6))
        monthly.plot(kind="bar")
        plt.title("Accidents by Month")
        plt.xlabel("Month")
        plt.ylabel("Count")
        plt.show()
    else:
        print("Month column contains no usable values")
else:
    print("Month column not found")

# Casualty Analysis
df["Total_Casualties"] = (df["Deaths"]+df["Injuries"])
print("\nCasualty Statistics")
print(df["Total_Casualties"].describe())

""" Top Deadliest Accidents
    nlargest() returns rows with largest values"""
deadliest = (df[["Accident_Name", "State", "Deaths"]].nlargest(10,"Deaths"))

print("\nTop Deadliest Accidents")
print(deadliest)

# Rescue Time Analysis
plt.figure(figsize=(10,6))
df["Rescue_Time_hrs"].hist(bins=15)
plt.title("Rescue Time Distribution")
plt.xlabel("Hours")
plt.show()
print("\nEDA Completed Successfully")