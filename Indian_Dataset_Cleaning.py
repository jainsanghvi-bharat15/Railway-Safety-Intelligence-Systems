# Railway Safety Supplementary Dataset Cleaning
""" Dataset 2:
    1. accidents_1902-2024
    2. funds_allocated
    3. rescue_time

    Purpose: Clean and prepare the Indian Railway datasets for:
                    EDA
                    SQL Analysis
                    Power BI Dashboard
                    Business Insights"""
import pandas as pd
import numpy as np
# Load Excel File
file_path = r"C:/Users/HP/3D Objects/Desktop/DA_using_Python/DA Projects/Project 2 (Railway AI ML DA)/Uncleaned Indian_Railways_Accidents_Dataset_1902_2024.xlsx"

# read_excel() loads a specific sheet from Excel into a DataFrame
accidents_df = pd.read_excel(file_path, sheet_name="accidents_1902-2024")
funds_df = pd.read_excel(file_path, sheet_name="funds_allocated")
rescue_df = pd.read_excel(file_path, sheet_name="rescue_time")

# BASIC DATASET EXPLORATION
print("\nACCIDENTS SHEET:\n", accidents_df.shape)
print("\nFUNDS SHEET:\n", funds_df.shape)
print("\nRESCUE SHEET:\n", rescue_df.shape)

# ACCIDENTS SHEET CLEANING
accidents_df.drop_duplicates(inplace=True)  # drop_duplicates() removes duplicate rows

"""Standardize column names
    str.strip() removes extra spaces
    str.replace() makes column names Python and SQL friendly"""

accidents_df.columns = (
    accidents_df.columns
    .str.strip()
    .str.replace(r"[^\w]", "_", regex=True)
    .str.replace(r"_+", "_", regex=True)
    .str.strip("_"))

""" Convert numerical columns to numeric datatype
    errors="coerce" converts invalid values to NaN"""
numeric_cols = [
    "Year",
    "Month",
    "Day",
    "Deaths",
    "Injuries",
    "Rescue_Time_hrs" ]

for col in numeric_cols:
    if col in accidents_df.columns:
        accidents_df[col] = pd.to_numeric(accidents_df[col], errors="coerce")

""" Fill missing numeric values using median
    median() is less affected by outliers""" 
num_cols = accidents_df.select_dtypes(include=["number"]).columns

for col in num_cols:
    median_value = accidents_df[col].median()
    if pd.notna(median_value):
        accidents_df[col] = accidents_df[col].fillna(median_value)

# Clean text columns
text_cols = accidents_df.select_dtypes(include=["object", "string"]).columns

for col in text_cols:
    accidents_df[col] = (
        accidents_df[col]
        .astype(str)
        .str.strip()
        .replace(["", "nan", "None"], np.nan))

    mode_value = (
        accidents_df[col].mode().iloc[0]
        if not accidents_df[col].mode().empty
        else "Unknown")
    
    accidents_df[col] = accidents_df[col].fillna(mode_value)

# Create a proper accident date
accidents_df["Accident_Date"] = pd.to_datetime(
    dict(
        year=accidents_df["Year"],
        month=accidents_df["Month"],
        day=accidents_df["Day"]
    ), errors="coerce")

""" Feature Engineering
    dt.day_name() extracts weekday name""" 

accidents_df["Weekday"] = (accidents_df["Accident_Date"].dt.day_name())

# dt.quarter extracts quarter
accidents_df["Quarter"] = (accidents_df["Accident_Date"].dt.quarter)

# Create casualty metric
accidents_df["Total_Casualties"] = (
    accidents_df["Deaths"]
    +
    accidents_df["Injuries"])

# FUNDS SHEET CLEANING
funds_df.drop_duplicates(inplace=True)  # Remove duplicate rows

# Standardize column names
funds_df.columns = (
    funds_df.columns
    .str.strip()
    .str.replace(r"[^\w]", "_", regex=True)
    .str.replace(r"_+", "_", regex=True)
    .str.strip("_"))

# Convert all columns to numeric
for col in funds_df.columns:
    funds_df[col] = pd.to_numeric(funds_df[col], errors="coerce")

# Fill missing values with median
for col in funds_df.columns:
    median_value = funds_df[col].median()
    if pd.notna(median_value):
        funds_df[col] = funds_df[col].fillna(median_value)

""" Create Business Metrics
    These metrics will be useful in Power BI"""
funds_df["Safety_Fund_Utilization_Percent"] = (
    funds_df["Railway_Safety_Fund_Actual_Expenditure_cr"]
    /
    funds_df["Railway_Safety_Fund_Total_Grant_cr"]) * 100

funds_df["DRF_Utilization_Percent"] = (
    funds_df["Depreciation_Reserve_Fund_Actual_Utilisation_cr"]
    /
    funds_df["Depreciation_Reserve_Fund_DRF"]) * 100

funds_df["Working_Expenditure_Utilization_Percent"] = (
    funds_df["Total_Working_Expenditure_Actual_expenditure_cr"]
    /
    funds_df["Total_Working_Expenditure_cr"]) * 100

funds_df["Ordinary_Expense_Utilization_Percent"] = (
    funds_df["Actual_ordinary_working_expense_cr"]
    /
    funds_df["Ordinary_working_expense_cr"]) * 100

# RESCUE TIME SHEET CLEANING
rescue_df.drop_duplicates(inplace=True) # Remove duplicates

# Standardize column names
rescue_df.columns = (
    rescue_df.columns
    .str.strip()
    .str.replace(r"[^\w]", "_", regex=True)
    .str.replace(r"_+", "_", regex=True)
    .str.strip("_"))

# Correct spelling mistake in column name
if "Loacation" in rescue_df.columns:
    rescue_df.rename(columns={"Loacation": "Location"}, inplace=True)

# Convert rescue time column to numeric
if "Average_Rescue_Time_hrs" in rescue_df.columns:
    rescue_df["Average_Rescue_Time_hrs"] = pd.to_numeric(
        rescue_df["Average_Rescue_Time_hrs"],
        errors="coerce")

# Fill missing numeric values
num_cols = rescue_df.select_dtypes(include=["number"]).columns

for col in num_cols:
    median_value = rescue_df[col].median()
    if pd.notna(median_value):
        rescue_df[col] = rescue_df[col].fillna(median_value)

# Fill missing text values using mode
text_cols = rescue_df.select_dtypes(include=["object", "string"]).columns

for col in text_cols:
    mode_value = (
        rescue_df[col].mode().iloc[0]
        if not rescue_df[col].mode().empty
        else "Unknown"
    )

    rescue_df[col] = rescue_df[col].fillna(mode_value)

# FINAL DATA QUALITY CHECK
print("ACCIDENTS DATASET")
print("Shape:", accidents_df.shape)
print("\nMissing Values: ",accidents_df.isnull().sum())

print("FUNDS DATASET")
print("Shape:", funds_df.shape)
print("\nMissing Values: ", funds_df.isnull().sum())


print("RESCUE DATASET")
print("Shape:", rescue_df.shape)
print("\nMissing Values: ", rescue_df.isnull().sum())

# SAVE ALL CLEANED DATASETS INTO A SINGLE EXCEL FILE
output_file = r"C:/Users/HP/3D Objects/Desktop/DA_using_Python/DA Projects/Project 2 (Railway AI ML DA)/Indian_Railway_Cleaned_Data.xlsx"

# ExcelWriter() creates a multi-sheet Excel workbook
with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    accidents_df.to_excel(writer, sheet_name="Accidents", index=False)
    funds_df.to_excel(writer, sheet_name="Funds_Allocated", index=False)
    rescue_df.to_excel(writer, sheet_name="Rescue_Time", index=False)

print("\n================================")
print("Cleaning Completed Successfully")
print("Output File Saved:")
print(output_file)
print("================================")