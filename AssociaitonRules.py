import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import ast
import os

def sample_data(data, fraction=0.1, seed=42):
    if data is None or data.empty:
        return data
        
    # For small datasets, use at least 1000 rows or the original size, whichever is smaller
    min_rows = min(1000, len(data))
    
    # Calculate how many rows to sample (at least min_rows)
    sample_size = max(min_rows, int(len(data) * fraction))
    
    # Sample the data with a fixed random seed for reproducibility
    return data.sample(n=sample_size, random_state=seed)
def load_data(sample_fraction=0.1):
    patients_path = 'Data/patients.csv'
    admissions_path = 'Data/admissions.csv'
    diagnoses_path = 'Data/diagnoses_icd.csv'
    lab_events_path = 'Data/labevents_sample.csv'
    d_icd_labs_path = 'Data/d_labitems.csv'
    d_icd_diagnoses_path = 'Data/d_icd_diagnoses.csv'
    d_icd_procedures_path = 'Data/d_icd_procedures.csv'
    procedures_path = 'Data/procedures_icd.csv'
    print(f"Loading data with sampling fraction: {sample_fraction}")
    patients = pd.read_csv(patients_path)
    admissions = pd.read_csv(admissions_path)
    diagnoses = pd.read_csv(diagnoses_path)
    d_icd_diagnoses = pd.read_csv(d_icd_diagnoses_path)
    d_icd_procedures = pd.read_csv(d_icd_procedures_path)
    procedures = pd.read_csv(procedures_path)
    lab_events = pd.read_csv(lab_events_path)
    d_icd_labs = pd.read_csv(d_icd_labs_path)

    print("Original dataset shapes:")
    print(f"Patients dataset shape: {patients.shape}")
    print(f"Admissions dataset shape: {admissions.shape}")
    print(f"Diagnoses dataset shape: {diagnoses.shape}")
    print(f"d_icd_diagnoses dataset shape: {d_icd_diagnoses.shape}")
    print(f"d_icd_procedures dataset shape: {d_icd_procedures.shape}")
    print(f"Procedures dataset shape: {procedures.shape}")
    print(f"Lab events dataset shape: {lab_events.shape}")
    print(f"d_icd_labs dataset shape: {d_icd_labs.shape}")

    # Sample the main data tables that contain patient-level information
    # We don't sample the reference tables (d_icd_*)
    if sample_fraction < 1.0:
        # First, sample patients
        patients_sampled = sample_data(patients, fraction=sample_fraction)
        
        # Then filter other tables to only include the sampled patients
        sampled_subject_ids = set(patients_sampled['subject_id'])
        
        admissions = admissions[admissions['subject_id'].isin(sampled_subject_ids)]
        diagnoses = diagnoses[diagnoses['subject_id'].isin(sampled_subject_ids)]
        procedures = procedures[procedures['subject_id'].isin(sampled_subject_ids)]
        lab_events = lab_events[lab_events['subject_id'].isin(sampled_subject_ids)]
        
        patients = patients_sampled
        
        print("\nSampled dataset shapes:")
        print(f"Patients dataset shape: {patients.shape}")
        print(f"Admissions dataset shape: {admissions.shape}")
        print(f"Diagnoses dataset shape: {diagnoses.shape}")
        print(f"Procedures dataset shape: {procedures.shape}")
        print(f"Lab events dataset shape: {lab_events.shape}")
    print("\nPatients data preview: \n", patients.head())
    print("Admissions data preview: \n", admissions.head())
    print("Diagnoses data preview: \n", diagnoses.head())
    print("d_icd_diagnoses data preview: \n", d_icd_diagnoses.head())
    print("d_icd_procedures data preview: \n", d_icd_procedures.head())
    print("Procedures data preview: \n", procedures.head())
    print("Lab events data preview: \n", lab_events.head())
    print("d_icd_labs data preview: \n", d_icd_labs.head())
    # Check for missing values
    print("Patients missing values: ", patients.isnull().sum())
    print("Admissions missing values: ", admissions.isnull().sum())
    print("Diagnoses missing values: ", diagnoses.isnull().sum())
    
    return patients, admissions, diagnoses, d_icd_diagnoses, d_icd_procedures, procedures, lab_events, d_icd_labs
def preprocess_data(patients, admissions, diagnoses, d_icd_diagnoses, lab_events, d_icd_labs):
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
    #Merge lab events with description
    lab_events_with_desc = pd.merge(lab_events, d_icd_labs, how='inner')

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
    return transactions_base, diagnoses_with_desc, lab_events_with_desc

def engineer_features(transactions_base, procedures, d_icd_procedures, diagnoses_with_desc, lab_events_with_desc):
    print("Starting feature engineering...")
    try:
        max_features_count = int(os.environ.get("MAX_FEATURES_COUNT", 1000))
        print(f"Max features count set to {max_features_count}")
    except (ValueError, TypeError):
        print("Invalid MAX_FEATURES_COUNT environment variable. Using default value of 1000.")
        max_features_count = 1000
    
    # ----- LAB RESULTS FEATURE ENGINEERING -----
    print("Generating lab result features...")
    lab_events_with_desc['lab_result'] = (
        lab_events_with_desc['itemid'].astype(str) + '_' + lab_events_with_desc['label'].fillna('')
    )
    lab_events_with_desc = lab_events_with_desc.dropna(subset=['hadm_id', 'flag'])  #hadm_id is not null and result is abnormal
    print(f"Lab events with description shape: {lab_events_with_desc.shape}")
    print(f"Lab events with description preview: \n{lab_events_with_desc.head()}")

    if lab_events_with_desc.empty:
        print("WARNING: No abnormal lab events found.")
        lab_summary = pd.DataFrame(index=transactions_base["hadm_id"].unique())
    else:
        # Categorize lab results as below or above range
        lab_events_with_desc['range_status'] = lab_events_with_desc.apply(
            lambda row: (
                'Below' if pd.notnull(row['valuenum']) and pd.notnull(row['ref_range_lower']) and row['valuenum'] < row['ref_range_lower']
                else 'Above' if pd.notnull(row['valuenum']) and pd.notnull(row['ref_range_upper']) and row['valuenum'] > row['ref_range_upper']
                else 'Unknown'
            ), axis=1
        )
        print(f"Range status distribution:\n{lab_events_with_desc['range_status'].value_counts()}")
        
        lab_events_with_desc['lab_result'] = (
            lab_events_with_desc['itemid'].astype(str) + '_' + 
            lab_events_with_desc['label'].fillna('') + '_' + 
            lab_events_with_desc['range_status']
        )
        lab_events_with_desc = lab_events_with_desc[lab_events_with_desc['range_status'] != 'Unknown']
        print(f"Lab events after filtering Unknown status: {lab_events_with_desc.shape}")
        
        lab_pivot = pd.get_dummies(
            lab_events_with_desc[['hadm_id', 'lab_result']],
            columns=['lab_result'],
            prefix='Lab',
            prefix_sep='_'
        )
        lab_summary = lab_pivot.groupby('hadm_id').max()
        print(f"Lab summary feature matrix shape: {lab_summary.shape}")
        print(f"Lab summary feature matrix preview: \n{lab_summary.head()}")
        
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
    min_procedure_freq = 25
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
    min_comorbidity_freq = 40
    common_comorbidities = comorbidity_counts[comorbidity_counts >= min_comorbidity_freq].index.tolist()
    print(f"Using {len(common_comorbidities)} common comorbidities for feature engineering out of {len(comorbidity_counts)} total comorbidities")
    comorbidities_filtered = comorbidities[comorbidities["long_title"].isin(common_comorbidities)]
    print(f"Filtered comorbidities dataset shape: {comorbidities_filtered.shape}")
    print(f"Filtered comorbidities dataset preview: \n{comorbidities_filtered.head()}")
    if len(comorbidities_filtered) == 0:
        print("WARNING: No comorbidities match the frequency threshold. Reducing threshold.")
        min_comorbidity_freq = 20
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
    for name, df in [("Procedures", procedures_by_admission), ("Comorbidities", comorbidities_by_admission), ("Demographics", demographics_by_admission), ("Lab", lab_summary)]:
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
        'lab_features': len(lab_summary.columns) if hasattr(lab_summary, 'columns') else 0,
        'total_features': transactions_matrix.shape[1],
        'total_transactions': transactions_matrix.shape[0]
    })
    feature_counts.to_csv('output/feature_counts.csv')
    
    return transactions_matrix
def mine_association_rules(transactions_matrix, min_support=0.01, min_confidence=0.5):
    print("Starting association rule mining...")
    
    # Convert the DataFrame to a one-hot encoded format
    transactions_matrix_bool = transactions_matrix.astype(bool)
    min_support_floor = min_support/10
    min_confidence_floor = min_confidence/2
    # Check if we should use a sample
    sample_size = 0
    try:
        sample_size = int(os.environ.get('SAMPLE_SIZE', '0'))
    except (ValueError, TypeError):
        sample_size = 0
    
    if sample_size > 0 and sample_size < transactions_matrix.shape[0]:
        print(f"Using a sample of {sample_size} transactions")
        transactions_matrix_bool = transactions_matrix_bool.sample(sample_size)
    
    # 1. Find frequent itemsets with adaptive support threshold
    # Try to find a reasonable number of itemsets
    frequent_itemsets = pd.DataFrame()
    
    try:
        frequent_itemsets = apriori(transactions_matrix_bool, 
                                min_support=min_support,
                                use_colnames=True,
                                max_len=4)  # Limit to combinations of at most 4 items
        
        # If we found too few itemsets, try with a lower threshold
        if len(frequent_itemsets) < 10:
            old_support = min_support
            min_support = max(min_support_floor, min_support / 2)
            print(f"Found too few itemsets ({len(frequent_itemsets)}). Reducing support from {old_support} to {min_support}")
            
            frequent_itemsets = apriori(transactions_matrix_bool, 
                                    min_support=min_support,
                                    use_colnames=True,
                                    max_len=4)
    except Exception as e:
        print(f"ERROR in apriori algorithm: {str(e)}")
        print("Trying with a smaller dataset...")
        
        # Sample the data if it's too large
        if transactions_matrix.shape[0] > 10000:
            sample_size = min(10000, int(transactions_matrix.shape[0] * 0.5))
            transactions_sample = transactions_matrix.sample(sample_size)
            try:
                frequent_itemsets = apriori(transactions_sample.astype(bool), 
                                        min_support=min_support,
                                        use_colnames=True,
                                        max_len=3)
                print(f"Successfully ran apriori on a sample of {sample_size} transactions")
            except Exception as e2:
                print(f"ERROR in apriori algorithm even with sampling: {str(e2)}")
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    if frequent_itemsets.empty:
        print("No frequent itemsets found. Cannot generate rules.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    print(f"Found {len(frequent_itemsets)} frequent itemsets with min_support={min_support}")
    
    # Save frequent itemsets
    os.makedirs('output', exist_ok=True)
    frequent_itemsets.to_csv('output/frequent_itemsets.csv')
    # 2. Generate association rules with adaptive confidence threshold
    rules = pd.DataFrame()
    
    try:
        rules = association_rules(frequent_itemsets, 
                                metric='confidence',
                                min_threshold=min_confidence)
        
        # If we found too few rules, try with a lower threshold
        if len(rules) < 10:
            old_confidence = min_confidence
            min_confidence = max(min_confidence_floor, min_confidence / 1.5)
            print(f"Found too few rules ({len(rules)}). Reducing confidence from {old_confidence} to {min_confidence}")
            
            rules = association_rules(frequent_itemsets, 
                                    metric='confidence',
                                    min_threshold=min_confidence)
    except Exception as e:
        print(f"ERROR in association_rules algorithm: {str(e)}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if rules.empty:
        print("No rules generated. Cannot proceed with rule filtering.")
        return frequent_itemsets, pd.DataFrame(), pd.DataFrame()
    
    print(f"Generated {len(rules)} rules with min_confidence={min_confidence}")
    
    # Save all rules
    rules.to_csv('output/all_rules.csv', index=False)
    
    # 3. Filter rules to focus on diagnoses
    diagnosis_cols = [col for col in transactions_matrix.columns if col.startswith('Diagnosis_')]
    if not diagnosis_cols:
        print("ERROR: No diagnosis columns found in transaction matrix")
        return frequent_itemsets, rules, pd.DataFrame()
    
    print(f"Found {len(diagnosis_cols)} diagnosis columns to use for rule filtering")
    # Convert string representations of sets to actual sets, if needed
    if isinstance(rules['antecedents'].iloc[0], str):
        rules['antecedents'] = rules['antecedents'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        rules['consequents'] = rules['consequents'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    # Filter for rules that predict diagnoses
    diagnosis_rules = rules[rules['consequents'].apply(
        lambda x: any(item in diagnosis_cols for item in x)
    )].copy()
    
    if diagnosis_rules.empty:
        print("WARNING: No rules found with diagnoses in the consequent")
        return frequent_itemsets, rules, pd.DataFrame()
    
    print(f"Found {len(diagnosis_rules)} rules with diagnoses in the consequent")
    
    # Additional filters to focus on more interesting rules
    if len(diagnosis_rules) > 1000:
        print(f"Too many rules ({len(diagnosis_rules)}). Filtering to more interesting ones...")
        
        # Filter by lift (stronger associations)
        high_lift_rules = diagnosis_rules[diagnosis_rules['lift'] > 1.5]
        if len(high_lift_rules) >= 100:
            diagnosis_rules = high_lift_rules
            print(f"Filtered to {len(diagnosis_rules)} rules with lift > 1.5")
    
    # Sort by lift and then confidence
    diagnosis_rules = diagnosis_rules.sort_values(['lift', 'confidence'], ascending=[False, False])
    
    # Save diagnosis rules
    diagnosis_rules.to_csv('output/diagnosis_rules.csv', index=False)
    
    return frequent_itemsets, rules, diagnosis_rules
sample_fraction = 0.01  # Use 10% of the data
patients, admissions, diagnoses, d_icd_diagnoses, d_icd_procedures, procedures, lab_events, d_icd_labs = load_data(sample_fraction)
transactions_base, diagnoses_with_desc, lab_events_with_desc = preprocess_data(patients, admissions, diagnoses, d_icd_diagnoses, lab_events, d_icd_labs)
#check if the transaction_matrix_csv is already existing
# if os.path.exists('output/transaction_matrix.csv'):
#     transactions_matrix = pd.read_csv('output/transaction_matrix.csv', index_col=0)
# else:
transactions_matrix = engineer_features(transactions_base, procedures, d_icd_procedures, diagnoses_with_desc, lab_events_with_desc)
frequent_itemsets, rules, diagnosis_rules = mine_association_rules(transactions_matrix, min_support=0.01, min_confidence=0.5)