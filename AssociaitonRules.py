

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import os

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

def preprocess_data(patients, admissions, diagnoses, d_icd_diagnoses):
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

    return transactions_base, diagnoses_with_desc


def engineer_features(transactions_base, procedures, d_icd_procedures, diagnoses_with_desc):
    print("Starting feature engineering...")

    try:
        max_features_count = int(os.environ.get("MAX_FEATURES_COUNT", 1000))
        print(f"Max features count set to {max_features_count}")
    except (ValueError, TypeError):
        print("Invalid MAX_FEATURES_COUNT environment variable. Using default value of 1000.")
        max_features_count = 1000
    
    # 1. Merge procedures with descriptions
    procedures_with_desc = pd.merge(procedures, d_icd_procedures, how='left', left_on=["icd_code", "icd_version"], right_on=["icd_code", "icd_version"])
    missing_proc_desc = procedures_with_desc[procedures_with_desc["long_title"].isnull()]
    if len(missing_proc_desc) > 0:
        print(f"WARNING: {len(missing_proc_desc)} procedure codes have no description in the dictionary")
        procedures_with_desc["long_title"] = procedures_with_desc["long_title"].fillna("Unlabeled_" + procedures_with_desc["icd_code"].astype(str))

    #get comorbidities
    comorbidities = diagnoses_with_desc[diagnoses_with_desc["seq_num"] > 1].copy()

    print(f"Found {len(comorbidities)} comorbidities")

    #create procedure presence feature
    procedure_counts = procedures_with_desc["long_title"].value_counts()
    min_procedure_freq = 200

    common_procedures = procedure_counts[procedure_counts >= min_procedure_freq].index.tolist()
    print(f"Using {len(common_procedures)} common procedures for feature engineering out of {len(procedure_counts)} total procedures")

    procedures_filtered = procedures_with_desc[procedures_with_desc["long_title"].isin(common_procedures)]

    print(f"Filtered procedures dataset shape: {procedures_filtered.shape}")
    print(f"Filtered procedures dataset preview: \n{procedures_filtered.head()}")

    #checking if procedures are more than the frequency threshold
    if len(procedures_filtered) == 0:
        print("WARNING: No procedures match the frequency threshold. Reducing threshold.")
        min_procedure_freq = 10
        common_procedures = procedure_counts[procedure_counts >= min_procedure_freq].index.tolist()
        procedures_filtered = procedures_with_desc[procedures_with_desc['long_title'].isin(common_procedures)]
        print(f"Using {len(common_procedures)} procedures with reduced threshold")
    
    if len(procedures_filtered) > 0:
        procedures_pivot = pd.get_dummies(procedures_filtered[["hadm_id", "long_title"]], columns=["long_title"], prefix="Procedure", prefix_sep="_")
        procedures_by_admission= procedures_pivot.groupby("hadm_id").max()
        print(f"Created procedures with admissons with shape: {procedures_by_admission.shape}")
    else:
        print("WARNING: No procedures found after filtering. Skipping procedure feature engineering.")
        procedures_by_admission = pd.DataFrame(index = transactions_base["hadm_id"].unique())
    
    #create comorbidity presence feature
    comorbidity_counts = comorbidities["long_title"].value_counts()
    min_comorbidity_freq = 600
    common_comorbidities = comorbidity_counts[comorbidity_counts >= min_comorbidity_freq].index.tolist()
    print(f"Using {len(common_comorbidities)} common comorbidities for feature engineering out of {len(comorbidity_counts)} total comorbidities")
    comorbidities_filtered = comorbidities[comorbidities["long_title"].isin(common_comorbidities)]
    print(f"Filtered comorbidities dataset shape: {comorbidities_filtered.shape}")
    print(f"Filtered comorbidities dataset preview: \n{comorbidities_filtered.head()}")
    if len(comorbidities_filtered) == 0:
        print("WARNING: No comorbidities match the frequency threshold. Reducing threshold.")
        min_comorbidity_freq = 10
        common_comorbidities = comorbidity_counts[comorbidity_counts >= min_comorbidity_freq].index.tolist()
        comorbidities_filtered = comorbidities[comorbidities['long_title'].isin(common_comorbidities)]
        print(f"Using {len(common_comorbidities)} comorbidities with reduced threshold")
    if len(comorbidities_filtered) > 0:
        comorbidities_pivot = pd.get_dummies(comorbidities_filtered[["hadm_id", "long_title"]], columns=["long_title"], prefix="Comorbidity", prefix_sep="_")
        comorbidities_by_admission = comorbidities_pivot.groupby("hadm_id").max()
        print(f"Created comorbidities with admissons with shape: {comorbidities_by_admission.shape}")
    else:
        print("WARNING: No comorbidities found after filtering. Skipping comorbidity feature engineering.")
        comorbidities_by_admission = pd.DataFrame(index = transactions_base["hadm_id"].unique())
    
    #create demographic features
    demographic_cols = ["hadm_id", "anchor_age", "gender", "age_category", "admission_type", "discharge_location", "race", "hospital_expire_flag"]

    demographic_features = pd.get_dummies(transactions_base[demographic_cols], columns=["gender", "age_category", "admission_type", "discharge_location", "race"], prefix=["Gender", "Age", "AdmType", "Discharge", "Race"], prefix_sep="_")

    if "hospital_expire_flag" in demographic_features.columns:
        demographic_features["Expired_In_Hospital"] = demographic_features["hospital_expire_flag"]
        demographic_features = demographic_features.drop("hospital_expire_flag", axis=1)
    
    demographics_by_admission = demographic_features.groupby("hadm_id").first()
    print(f"Created demographic features with shape: {demographics_by_admission.shape}")

    print(f"Demographic features preview: \n{demographics_by_admission.head()}")

    #merge all features into one dataset
    all_features = pd.DataFrame(index = transactions_base["hadm_id"].unique())

    #list to keep track of dataframes with potential join issues

    empty_dfs = []  
    for name, df in [("Procedures", procedures_by_admission), ("Comorbidities", comorbidities_by_admission), ("Demographics", demographics_by_admission)]:
        if df.empty:
            print(f"WARNING: {name} dataframe is empty.")
            empty_dfs.append(name)
            continue
        before_rows = len(all_features)
        all_features = all_features.join(df, how='left')
        after_rows = len(all_features)

        if before_rows != after_rows:
            print(f"WARNING: {name} dataframe caused a join issue. Rows before: {before_rows}, Rows after: {after_rows}")
    
    if empty_dfs:
        print(f"WARNING: The following dataframes were empty and not included in the final dataset: {', '.join(empty_dfs)}")

    #fill missing values with 0
    all_features = all_features.fillna(0)

    
    # 8. Create diagnosis outcome features
    diagnosis_counts = transactions_base['primary_diagnosis'].value_counts()

    # We want at least 10 diagnoses, but respect max_features budget
    diagnosis_feature_limit = max(10, max_features_count - len(demographics_by_admission.columns) - 
                            (len(procedures_by_admission.columns) if hasattr(procedures_by_admission, 'columns') else 0) - 
                            (len(comorbidities_by_admission.columns) if hasattr(comorbidities_by_admission, 'columns') else 0))

    # Simply take the top N most frequent diagnoses
    common_diagnoses = diagnosis_counts.nlargest(min(diagnosis_feature_limit, len(diagnosis_counts))).index.tolist()

    print(f"Using {len(common_diagnoses)} most common diagnoses out of {len(diagnosis_counts)} total")
    diagnoses_filtered = transactions_base[transactions_base['primary_diagnosis'].isin(common_diagnoses)]
    print(f"Filtered diagnoses dataset shape: {diagnoses_filtered.shape}")
    print(f"Filtered diagnoses dataset preview: \n{diagnoses_filtered.head()}")

    # Check if we have diagnoses left after filtering
    if len(diagnoses_filtered) == 0:
        print("ERROR: No diagnoses meet the frequency threshold. Unable to create meaningful rules.")
        print("Please check your data or reduce the threshold further.")
        # Return a minimal dataframe to avoid errors
        return pd.DataFrame(columns=['no_features_available'])
    
    diagnosis_pivot = pd.get_dummies(diagnoses_filtered[['hadm_id', 'primary_diagnosis']], 
                                    columns=['primary_diagnosis'], 
                                    prefix='Diagnosis', 
                                    prefix_sep='_')
    diagnosis_by_admission = diagnosis_pivot.groupby('hadm_id').max()

    print(f"Created diagnosis features with shape: {diagnosis_by_admission.shape}")
    print(f"Diagnosis features preview: \n{diagnosis_by_admission.head()}")

    transactions_matrix = all_features.join(diagnosis_by_admission, how='inner')
    if transactions_matrix.empty:
        print("ERROR: Empty transaction matrix after joining features and outcomes.")
        print("Please check that hadm_ids are consistent across your datasets.")
        return pd.DataFrame(columns=['empty_transactions_matrix'])
    if (transactions_matrix.nunique() > 2).all():
        print("WARNING: No binary features found in transaction matrix. Check your data transformations.")

    # Check for excessive NaN values
    nan_percentage = transactions_matrix.isna().mean().mean() * 100
    if nan_percentage > 0:
        print(f"WARNING: Transaction matrix contains {nan_percentage:.2f}% NaN values")
        transactions_matrix = transactions_matrix.fillna(0)
    
    print(f"Final transaction matrix: {transactions_matrix.shape[0]} rows and {transactions_matrix.shape[1]} columns")
    print(f"Features include {len(demographics_by_admission.columns)} demographic features, " 
         f"{len(procedures_by_admission.columns) if hasattr(procedures_by_admission, 'columns') else 0} procedure features, "
         f"{len(comorbidities_by_admission.columns) if hasattr(comorbidities_by_admission, 'columns') else 0} comorbidity features, "
         f"and {len(diagnosis_by_admission.columns) if hasattr(diagnosis_by_admission, 'columns') else 0} diagnosis outcomes")
    
    # Save the transaction matrix
    os.makedirs('output', exist_ok=True)
    transactions_matrix.to_csv('output/transaction_matrix.csv')
    
    # For debug purposes, also save feature counts
    feature_counts = pd.Series({
        'demographic_features': len(demographics_by_admission.columns),
        'procedure_features': len(procedures_by_admission.columns) if hasattr(procedures_by_admission, 'columns') else 0,
        'comorbidity_features': len(comorbidities_by_admission.columns) if hasattr(comorbidities_by_admission, 'columns') else 0,
        'diagnosis_features': len(diagnosis_by_admission.columns) if hasattr(diagnosis_by_admission, 'columns') else 0,
        'total_features': transactions_matrix.shape[1],
        'total_transactions': transactions_matrix.shape[0]
    })
    feature_counts.to_csv('output/feature_counts.csv')
    
    return transactions_matrix
patients, admissions, diagnoses, d_icd_diagnoses, d_icd_procedures, procedures = load_data()
transactions_base, diagnoses_with_desc = preprocess_data(patients, admissions, diagnoses, d_icd_diagnoses)
transactions_matrix = engineer_features(transactions_base, procedures, d_icd_procedures, diagnoses_with_desc)