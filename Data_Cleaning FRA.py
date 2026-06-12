# FRA Rail Equipment Accident Incident Data Cleaning
import pandas as pd
import numpy as np

# Load Dataset
df = pd.read_excel(r"C:/Users/HP/3D Objects/Desktop/DA_using_Python/DA Projects/Project 2 (Railway AI ML DA)/Uncleaned Rail_Equipment_Accident_Incident_Data.xlsx",engine="openpyxl")

# 1. Basic Exploration
print("Original Shape:", df.shape)
print("Memory Usage (MB):",round(df.memory_usage(deep=True).sum() / 1024**2, 2))

# 2. Remove Duplicate Records
df.drop_duplicates(inplace=True)
print("After Duplicate Removal:", df.shape)

# 3. Standardize Column Names
df.columns = (df.columns.str.strip().str.replace(" ", "_", regex=False).str.replace("/", "_", regex=False).str.replace("-", "_", regex=False))

# 4. Remove Unnecessary Columns
columns_to_drop = [
    "PDF_Link",
    "Report_Key",
    "Incident_Key",
    "Grade_Crossing_ID",
    "Other_Railroad_Code",
    "Other_Railroad_Name",
    "Other_Accident_Number",
    "Other_Accident_Year",
    "Other_Accident_Month",
    "Maintenance_Railroad_Code",
    "Maintenance_Railroad_Name",
    "Maintenance_Accident_Number",
    "Maintenance_Accident_Year",
    "Maintenance_Accident_Month",
    "First_Car_Initials",
    "First_Car_Number",
    "Causing_Car_Initials",
    "Causing_Car_Number",
    "Train_Number",
    "Special_Study_1",
    "Special_Study_2",
    "Joint_CD",
    "Reporting_Parent_Railroad_Company_Code",
    "Other_Parent_Railroad_Company_Code",
    "Maintenance_Parent_Railroad_Company_Code",
    "Reporting_Parent_Railroad_Company_Name",
    "Other_Parent_Railroad_Company_Name",
    "Maintenance_Parent_Railroad_Company_Name",
    "Reporting_Railroad_Holding_Company",
    "Other_Railroad_Holding_Company",
    "Maintenance_Railroad_Holding_Company",
    "Reporting_Railroad_Company_Grouping",
    "Other_Railroad_Company_Grouping",
    "Maintenance_Railroad_Company_Grouping",
    "Reporting_Railroad_SMT_Grouping",
    "Other_Railroad_SMT_Grouping",
    "Maintenance_Railroad_SMT_Grouping",
    "Reporting_Railroad_Class",
    "Other_Railroad_Class",
    "Maintenance_Railroad_Class"]
df.drop(columns=[col for col in columns_to_drop if col in df.columns],inplace=True)
print("After Removing Unnecessary Columns:", df.shape)

# 5. Convert Date Columns
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"],errors="coerce")

# 6. Convert Important Numeric Columns
numeric_cols = [
    "Train_Speed",
    "Recorded_Estimated_Speed",
    "Maximum_Speed",
    "Gross_Tonnage",
    "Equipment_Damage_Cost",
    "Track_Damage_Cost",
    "Total_Damage_Cost",
    "Latitude",
    "Longitude",
    "Temperature",
    "Persons_Evacuated",
    "Positive_Alcohol_Tests",
    "Positive_Drug_Tests",
    "Passengers_Transported",
    "Total_Persons_Killed",
    "Total_Persons_Injured",
    "Railroad_Employees_Killed",
    "Railroad_Employees_Injured",
    "Passengers_Killed",
    "Passengers_Injured",
    "Others_Killed",
    "Others_Injured"]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col],errors="coerce")

# 7. Remove Columns With >80% Missing Data
missing_percent = df.isnull().mean() * 100
high_missing_cols = missing_percent[missing_percent > 80].index.tolist()
df.drop(columns=high_missing_cols,inplace=True)
print("High Missing Columns Removed:", len(high_missing_cols))
print("After Missing Column Removal:", df.shape)

# 8. Fill Missing Numeric Values
num_cols = df.select_dtypes(include=["number"]).columns
for col in num_cols:
    median_value = df[col].median()
    if pd.notna(median_value):
        df[col] = df[col].fillna(median_value)

# 9. Fill Missing Categorical Values
cat_cols = df.select_dtypes(include=["object","string"]).columns
for col in cat_cols:
    df[col] = (df[col].astype(str).str.strip().replace(["", "nan", "None"], np.nan))
    mode_value = (
        df[col].mode().iloc[0]
        if not df[col].mode().empty
        else "Unknown")
    df[col] = df[col].fillna(mode_value)

# 10. Fill Missing Dates
date_cols = df.select_dtypes(include=["datetime64[ns]"]).columns
for col in date_cols:
    median_date = df[col].median()
    if pd.notna(median_date):
        df[col] = df[col].fillna(median_date)

# 11. Remove Impossible Coordinates
if "Latitude" in df.columns:
    df.loc[
        (df["Latitude"] < -90) |
        (df["Latitude"] > 90),
        "Latitude"
    ] = np.nan

if "Longitude" in df.columns:
    df.loc[
        (df["Longitude"] < -180) |
        (df["Longitude"] > 180),
        "Longitude"
    ] = np.nan

# 12. Create Useful Features
if "Date" in df.columns:
    df["Accident_Year_Clean"] = (df["Date"].dt.year)
    df["Accident_Month_Clean"] = (df["Date"].dt.month)
    df["Accident_Weekday"] = (df["Date"].dt.day_name())

# 13. Memory Optimization
for col in df.select_dtypes(include=["int64"]).columns:
    df[col] = pd.to_numeric(
        df[col],
        downcast="integer")

for col in df.select_dtypes(include=["float64"]).columns:
    df[col] = pd.to_numeric(df[col], downcast="float")

# 14. Data Quality Report
quality_report = pd.DataFrame({
    "Column": df.columns,
    "Missing_Values":df.isnull().sum(),
    "Missing_Percent": round(df.isnull().mean() * 100, 2),
    "Data_Type": df.dtypes.astype(str)})

quality_report.sort_values(
    by="Missing_Percent",
    ascending=False,
    inplace=True)

# 15. Final Checks
print("\nFinal Shape:", df.shape)
print("\nTop 20 Missing Columns:\n")
print(quality_report.head(20))

print("\nFinal Memory Usage (MB):", round(df.memory_usage(deep=True).sum() / 1024**2, 2))

# 16. Save Outputs
print("\nCleaning Completed Successfully")
project_columns = [
    "Accident_Number",
    "Date",
    "Accident_Year",
    "Accident_Month",
    "Day",
    "Accident_Type",
    "Accident_Type_Code",
    "Primary_Accident_Cause",
    "Primary_Accident_Cause_Code",
    "Contributing_Accident_Cause",
    "Contributing_Accident_Cause_Code",
    "Accident_Cause",
    "Accident_Cause_Code",
    "Narrative",
    "State_Name",
    "County_Name",
    "Subdivision",
    "Division",
    "Station",
    "District",
    "Latitude",
    "Longitude",
    "Milepost",
    "Temperature",
    "Visibility",
    "Weather_Condition",
    "Track_Type",
    "Track_Class",
    "Track_Density",
    "Signalization",
    "Method_of_Operation",
    "Train_Speed",
    "Recorded_Estimated_Speed",
    "Maximum_Speed",
    "Train_Direction",
    "Equipment_Type",
    "Gross_Tonnage",
    "Positive_Alcohol_Tests",
    "Positive_Drug_Tests",
    "Hours_Engineers_On_Duty",
    "Minutes_Engineers_On_Duty",
    "Hours_Conductors_On_Duty",
    "Minutes_Conductors_On_Duty",
    "Railroad_Employees_Killed",
    "Railroad_Employees_Injured",
    "Passengers_Killed",
    "Passengers_Injured",
    "Others_Killed",
    "Others_Injured",
    "Total_Persons_Killed",
    "Total_Persons_Injured",
    "Persons_Evacuated",
    "Equipment_Damage_Cost",
    "Track_Damage_Cost",
    "Total_Damage_Cost",
    "Loaded_Freight_Cars",
    "Empty_Freight_Cars",
    "Derailed_Loaded_Freight_Cars",
    "Derailed_Empty_Freight_Cars",
    "Head_End_Locomotives",
    "Accident_Year_Clean",
    "Accident_Month_Clean",
    "Accident_Weekday"]

project_columns = [col for col in project_columns if col in df.columns]
df_project = df[project_columns]
print(df_project.shape)

df_project.to_excel("C:/Users/HP/3D Objects/Desktop/DA_using_Python/DA Projects/Project 2 (Railway AI ML DA)/Railway_Safety_Project_Dataset_Cleaned.xlsx",index=False)
print("\nProject Dataset Saved Successfully")