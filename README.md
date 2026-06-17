# Railway Safety Intelligence System
 
Railway accidents don't just happen -they follow patterns. This project started with that assumption and ended up uncovering a lot of them.
 
Built around two real-world datasets -FRA Railroad Accident & Incident data from the USA and Indian Railway accident records -this is an end-to-end Data Analytics and ML project aimed at making sense of where, why, and how railway accidents occur. The goal was never just to build models, but to surface insights that railway authorities and decision-makers could actually act on.
 
The stack includes Python, SQL, Power BI, Scikit-Learn, and SHAP Explainable AI. 
The analysis covers everything from raw data cleaning to geographic hotspot detection to interactive dashboards.
 
---
 
## What's Inside
 
- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- SQL Analytics
- Feature Engineering
- Machine Learning Models
- SHAP Explainable AI
- Geographic Hotspot Detection
- Power BI Dashboard
- Project Report
---
 
## What This Project Does
 
- Digs into 217,000+ FRA railway accident records to find patterns that aren't obvious at first glance
- Pulls in Indian Railway datasets covering accidents, rescue times, and safety fund allocation
- Cleans and prepares both datasets for consistent analysis
- Runs EDA and SQL-based investigations to explore accident trends and root causes
- Builds ML models to predict accident severity and flag high-risk scenarios
- Uses SHAP to explain what the models are actually picking up on -not just what they predict
- Maps accident hotspots geographically to identify the most dangerous corridors and zones
- Packages everything into Power BI dashboards for executive-level visibility
---
 
## Technologies Used
 
Python · Pandas · NumPy · SQL · Power BI · Scikit-Learn · SHAP · Excel
 
---
 
## Getting Started
 
### Prerequisites
 
Make sure you have the following installed before anything else:
 
- Python 3.10+
- Microsoft Power BI Desktop
- Microsoft Excel
- VS Code or Jupyter Notebook *(optional but recommended)*
---
 
### Installing Dependencies
 
```bash
pip install pandas numpy matplotlib seaborn scikit-learn shap openpyxl xgboost
```
 
---
 
### Project Structure
 
```
Railway-Safety-Intelligence-System
│
├── Data_Cleaning_FRA.py
├── Indian_Dataset_Cleaning.py
├── EDA_Railway_Safety.py
├── EDA_Indian_Accident.py
├── ML_Model.py
├── Explainable_AI.py
├── Dashboard_P2.pbix
├── Project Report.pdf
└── README.md
```
 
---
 
### Dataset Setup
 
Download the datasets from Kaggle and place them in the project directory. Update file paths in the scripts if your folder structure differs.
 
**FRA Railroad Accident & Incident Data (USA)**
https://www.kaggle.com/datasets/chrico03/railroad-accident-and-incident-data
 
**Indian Railways Accidents (1902–2024)**
https://www.kaggle.com/datasets/siddhanthkumardas/indian-railways-accidents-1902-2024
 
---
 
### Running the Project
 
Run the scripts in this order -each one builds on the previous:
 
```
1. Data_Cleaning_FRA.py
2. Indian_Dataset_Cleaning.py
3. EDA_Railway_Safety.py
4. EDA_Indian_Accident.py
5. ML_Model.py
6. Explainable_AI.py
```
 
---
 
### Power BI Dashboard
 
1. Open `Dashboard_P2.pbix` in Power BI Desktop
2. Update dataset source paths if prompted
3. Hit **Refresh**
4. Explore the dashboards
---
 
### What Gets Generated
 
Once everything runs, you'll have:
 
- Cleaned versions of both datasets
- Data quality reports
- EDA visualizations
- SQL analysis outputs
- Trained ML models and predictions
- SHAP explainability plots
- Geographic hotspot analysis
- The fully interactive Power BI dashboard
---
 
### A Few Things to Keep in Mind
 
- The FRA dataset is large (217,000+ records) -running it in a virtual environment with sufficient RAM is strongly recommended
- Double-check dataset file paths before kicking off the scripts
- The Power BI dashboard pulls from the cleaned datasets produced by the preprocessing scripts, so run those first
- A virtual environment keeps dependency conflicts out of the picture
---
 
## Author
 
**Bharat Jain Sanghvi**
Data Analytics · SQL · Python · Power BI · Machine Learning
