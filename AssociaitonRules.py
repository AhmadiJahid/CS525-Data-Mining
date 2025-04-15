import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

def load_data():
    patients_path = 'Data/patients.csv'
    admissions_path = 'Data/admissions.csv'
    diagnoses_path = 'Data/diagnoses_icd.csv'
    d_icd_diagnoses_path = 'Data/d_icd_diagnoses.csv'
    d_icd_procedures_path = 'Data/d_icd_procedures.csv'
    procedures_path = 'Data/procedures_icd.csv'


    patients = pd.read_csv(patients_path)
    admissions = pd.read_csv(admissions_path)
    diagnoses = pd.read_csv(diagnoses_path)
    d_icd_diagnoses = pd.read_csv(d_icd_diagnoses_path)
    d_icd_procedures = pd.read_csv(d_icd_procedures_path)
    procedures = pd.read_csv(procedures_path)


    print("Patints dataset shape: ", patients.shape)
    print("Admissions dataset shape: ", admissions.shape)
    print("Diagnoses dataset shape: ", diagnoses.shape)
    print("d_icd_diagnoses dataset shape: ", d_icd_diagnoses.shape)
    print("d_icd_procedures dataset shape: ", d_icd_procedures.shape)
    print("Procedures dataset shape: ", procedures.shape)

    print("Patients data preview: \n", patients.head())
    print("Admissions data preview: \n", admissions.head())
    print("Diagnoses data preview: \n", diagnoses.head())
    print("d_icd_diagnoses data preview: \n", d_icd_diagnoses.head())
    print("d_icd_procedures data preview: \n", d_icd_procedures.head())
    print("Procedures data preview: \n", procedures.head())

    #check for missing values
    print("Patients missing values: ", patients.isnull().sum())
    print("Admissions missing values: ", admissions.isnull().sum())
    print("Diagnoses missing values: ", diagnoses.isnull().sum())
    
    return patients, admissions, diagnoses, d_icd_diagnoses, d_icd_procedures, procedures

def preprocess_data(patients, admissions, diagnoses, d_icd_diagnoses, d_icd_procedures, procedures):
    def categorize_age(age):
        if age < 18:
            return 'Child'
        elif age < 30:
            return 'Young_Adult'
        elif age < 50:
            return 'Adult'
        elif age < 70:
            return 'Middle_Aged'
        else:
            return 'Elderly'
        
    print("Starting preprocessing...")
    print(f"Initial shapes - Patients: {patients.shape}, Admissions: {admissions.shape}, Diagnoses: {diagnoses.shape}")
    
    # 1. Handle missing values in patients
    # No action needed as only dod is missing which is expected

    #Handle missing values in admissions
    essential_columns = ['hadm_id', 'subject_id', 'admittime', 'dischtime', 'admission_type', "discharge_location", "race", "hospital_expire_flag"]
    admissions_subset = admissions[essential_columns].copy()

    admissions_subset["discharge_location"] = admissions_subset["discharge_location"].fillna("Unknown")

    print(f"Missing values in discharge_location after handling: {admissions_subset['discharge_location'].isna().sum()}")

    patients["age_category"] = patients["anchor_age"].apply(categorize_age)

    #Merge diagnoses with description
    diagnoses_with_desc= pd.merge(diagnoses, d_icd_diagnoses, how='left', left_on=["icd_code", "icd_version"], right_on=["icd_code", "icd_version"])
    missing_desc = diagnoses_with_desc[diagnoses_with_desc["long_title"].isnull()]

    if len(missing_desc) > 0:
        print(f"WARNING: {len(missing_desc)} diagnosis codes have no description in the dictionary")
        #fill missing descriptions with code itself 
        diagnoses_with_desc["long_title"] =  diagnoses_with_desc["long_title"].fillna("Unlabeled_" + diagnoses_with_desc["icd_code"].astype(str))
        
    #Merde patients with admissions (only essential columns)

    patient_admissions = pd.merge(admissions_subset, patients[["subject_id", "anchor_age", "gender", "age_category"]], how='left', on="subject_id")

    #get primary diagnosis

    primary_diagnosis = diagnoses_with_desc[diagnoses_with_desc["seq_num"] == 1].copy()

    #create base transaction dataset
    transactions_base = pd.merge(patient_admissions, primary_diagnosis[["subject_id", "hadm_id", "icd_code", "long_title"]], how='inner', on=["subject_id", "hadm_id"])

    transactions_base = transactions_base.rename(columns={"long_title": "primary_diagnosis", "icd_code": "primary_diagnosis_code"})

    #checking for missing values
    missing_values = transactions_base.isnull().sum()
    
    if missing_values.sum() > 0:
        print("WARNING: Missing values found in transactions_base")
        print(missing_values[missing_values > 0])
        
    print(f"Created base transaction dataset with {len(transactions_base)} rows and {transactions_base.shape[1]} columns")
    print(f"Columns in transactions_base: {transactions_base.columns.tolist()}")

    return transactions_base



patients, admissions, diagnoses, d_icd_diagnoses, d_icd_procedures, procedures = load_data()
transactions_base = preprocess_data(patients, admissions, diagnoses, d_icd_diagnoses, d_icd_procedures, procedures)