# Main function remains largely the same, but with updated imports and function calls
def main(sample_fraction=0.1, min_support=0.0006, min_confidence=0.5, visualize=True, generate_report=True):
    """
    Main function to run the full pipeline.
    
    Args:
        sample_fraction (float): Fraction of data to sample
        min_support (float): Minimum support threshold for FP-Growth
        min_confidence (float): Minimum confidence threshold for rules
        visualize (bool): Whether to create visualizations
        generate_report (bool): Whether to generate HTML report
        
    Returns:
        tuple: (frequent_itemsets, rules, procedure_rules)
    """
    # 1. Load data
    patients, admissions, diagnoses, d_icd_diagnoses, d_icd_procedures, procedures = load_data(sample_fraction)
    
    # 2. Preprocess data
    transactions_base, diagnoses_with_desc = preprocess_data(patients, admissions, diagnoses, d_icd_diagnoses)
    
    # 3. Check if the transaction_matrix.csv already exists
    if os.path.exists('output/transaction_matrix.csv'):
        print("Loading existing transaction matrix...")
        transactions_matrix = pd.read_csv('output/transaction_matrix.csv', index_col=0)
        
        # Create human-readable versions if they don't exist yet
        if not os.path.exists('output/detailed_transaction_matrix.csv'):
            print("Creating human-readable transaction matrices...")
            create_readable_transaction_matrix(transactions_matrix)
            create_detailed_transaction_matrix(transactions_matrix, transactions_base, diagnoses_with_desc, procedures)
    else:
        # 4. Engineer features
        transactions_matrix = engineer_features(transactions_base, procedures, d_icd_procedures, diagnoses_with_desc)
    
    # 5. Mine association rules
    frequent_itemsets, rules, procedure_rules, diagnosis_to_procedure_rules = mine_association_rules(transactions_matrix, min_support, min_confidence)
    
    # 6. Create visualizations if requested and available
    if visualize and visualization_available:
        print("\nCreating visualizations...")
        
        # Visualize feature distribution in transaction matrix
        visualize_feature_distribution(transactions_matrix, save_path='output/feature_distribution.png')
        
        # Visualize rule metrics
        if not rules.empty:
            visualize_rule_metrics(rules, save_path='output/rule_metrics.png')
            visualize_rules_summary(rules, save_path='output/rules_summary.png')
        
        # Visualize procedure rules network
        if not procedure_rules.empty:
            visualize_rules_network(procedure_rules, max_rules=50, min_lift=1.0, 
                                   save_path='output/procedure_rules_network.png')
    
    # 7. Generate HTML report if requested
    if generate_report and visualization_available:
        print("\nGenerating HTML report...")
        report_path = create_html_report(transactions_matrix, rules, procedure_rules)
        if report_path:
            print(f"HTML report generated at: {report_path}")
    
    return frequent_itemsets, rules, procedure_rules

# Update main imports at the top
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import fpgrowth, association_rules  # Changed from apriori to fpgrowth
import matplotlib.pyplot as plt
import ast
import os
import pickle
# Import visualization functions
try:
    import networkx as nx
    from visualization import (
        visualize_rules_network, 
        visualize_rule_metrics, 
        visualize_feature_distribution, 
        visualize_rules_summary,
        create_html_report
    )
    visualization_available = True
except ImportError:
    print("Networkx not available. Network visualizations will be skipped.")
    visualization_available = False
    
    # Define empty visualization functions to avoid errors
    def visualize_rules_network(*args, **kwargs):
        print("Networkx not available. Skipping network visualization.")
    
    def visualize_rule_metrics(*args, **kwargs):
        print("Visualization functions not available. Skipping rule metrics visualization.")
    
    def visualize_feature_distribution(*args, **kwargs):
        print("Visualization functions not available. Skipping feature distribution visualization.")
    
    def visualize_rules_summary(*args, **kwargs):
        print("Visualization functions not available. Skipping rules summary visualization.")
        
    def create_html_report(*args, **kwargs):
        print("Visualization functions not available. Skipping HTML report generation.")
        return None
def sample_data(data, fraction=0.1, seed=42):
    if data is None or data.empty:
        return data
        
    # For small datasets, use at least 1000 rows or the original size, whichever is smaller
    min_rows = min(10000, len(data))
    
    # Calculate how many rows to sample (at least min_rows)
    sample_size = max(min_rows, int(len(data) * fraction))

    print("Sample size of the data:", sample_size)
    
    # Sample the data with a fixed random seed for reproducibility
    return data.sample(n=sample_size, random_state=seed)

def load_data(sample_fraction=0.1):
    patients_path = 'AR Creation/Data/patients.csv'
    admissions_path = 'AR Creation/Data/admissions.csv'
    diagnoses_path = 'AR Creation/Data/diagnoses_icd.csv'
    d_icd_diagnoses_path = 'AR Creation/Data/d_icd_diagnoses.csv'
    d_icd_procedures_path = 'AR Creation/Data/d_icd_procedures.csv'
    procedures_path = 'AR Creation/Data/procedures_icd.csv'

    print(f"Loading data with sampling fraction: {sample_fraction}")

    patients = pd.read_csv(patients_path)
    admissions = pd.read_csv(admissions_path)
    diagnoses = pd.read_csv(diagnoses_path)
    d_icd_diagnoses = pd.read_csv(d_icd_diagnoses_path)
    d_icd_procedures = pd.read_csv(d_icd_procedures_path)
    procedures = pd.read_csv(procedures_path)

    print("Original dataset shapes:")
    print(f"Patients dataset shape: {patients.shape}")
    print(f"Admissions dataset shape: {admissions.shape}")
    print(f"Diagnoses dataset shape: {diagnoses.shape}")
    print(f"d_icd_diagnoses dataset shape: {d_icd_diagnoses.shape}")
    print(f"d_icd_procedures dataset shape: {d_icd_procedures.shape}")
    print(f"Procedures dataset shape: {procedures.shape}")

    # Sample the main data tables that contain patient-level information
    # We don't sample the reference tables (d_icd_*)
    if sample_fraction < 1.0:
        # First, sample patients
        admissions_sampled = sample_data(admissions, fraction=sample_fraction)
        
        # Then filter other tables to only include the sampled patients
        sampled_subject_ids = set(admissions_sampled['subject_id'])
        sampled_hadm_ids = set(admissions_sampled['hadm_id'])        
        patients= patients[patients['subject_id'].isin(sampled_subject_ids)]
        diagnoses = diagnoses[diagnoses['hadm_id'].isin(sampled_hadm_ids)]
        procedures = procedures[procedures['hadm_id'].isin(sampled_hadm_ids)]
        
        admissions = admissions_sampled
        
        print("\nSampled dataset shapes:")
        print(f"Patients dataset shape: {patients.shape}")
        print(f"Admissions dataset shape: {admissions.shape}")
        print(f"Diagnoses dataset shape: {diagnoses.shape}")
        print(f"Procedures dataset shape: {procedures.shape}")

    print("\nPatients data preview: \n", patients.head())
    print("Admissions data preview: \n", admissions.head())
    print("Diagnoses data preview: \n", diagnoses.head())
    print("d_icd_diagnoses data preview: \n", d_icd_diagnoses.head())
    print("d_icd_procedures data preview: \n", d_icd_procedures.head())
    print("Procedures data preview: \n", procedures.head())

    # Check for missing values
    print("Patients missing values: ", patients.isnull().sum())
    print("Admissions missing values: ", admissions.isnull().sum())
    print("Diagnoses missing values: ", diagnoses.isnull().sum())
    
    return patients, admissions, diagnoses, d_icd_diagnoses, d_icd_procedures, procedures
def create_readable_rules(rules_df, feature_mappings=None):
    """
    Create a human-readable version of the rules.
    
    Args:
        rules_df (DataFrame): The rules DataFrame from association_rules function
        feature_mappings (dict, optional): Mappings of feature codes to descriptions
        
    Returns:
        DataFrame: A DataFrame with human-readable rules
    """
    print("Creating human-readable rules...")
    
    # Create a copy to avoid modifying the original
    readable_rules = rules_df.copy()
    
    # Function to format a single itemset to a readable string
    def format_itemset(itemset, feature_mappings=None):
        if isinstance(itemset, str):
            # Parse string representation if needed
            try:
                itemset = ast.literal_eval(itemset)
            except:
                return itemset
        
        items = []
        for item in itemset:
            if feature_mappings is not None:
                # Try to map to a readable description
                if item.startswith('Procedure_'):
                    # Extract the code from the feature name
                    proc_name = item.replace('Procedure_', '')
                    # Get description if available
                    if 'procedure_mapping' in feature_mappings:
                        for code, desc in feature_mappings['procedure_mapping'].items():
                            if desc == proc_name:
                                items.append(f"Procedure: {desc}")
                                break
                        else:
                            items.append(f"Procedure: {proc_name}")
                    else:
                        items.append(f"Procedure: {proc_name}")
                elif item.startswith('Diagnosis_'):
                    # Extract the code from the feature name
                    diag_name = item.replace('Diagnosis_', '')
                    # Get description if available
                    if 'diagnosis_mapping' in feature_mappings:
                        for code, desc in feature_mappings['diagnosis_mapping'].items():
                            if desc == diag_name:
                                items.append(f"Diagnosis: {desc}")
                                break
                        else:
                            items.append(f"Diagnosis: {diag_name}")
                    else:
                        items.append(f"Diagnosis: {diag_name}")
                elif item.startswith('Gender_'):
                    items.append(f"Gender: {item.replace('Gender_', '')}")
                elif item.startswith('Age_'):
                    items.append(f"Age Category: {item.replace('Age_', '')}")
                else:
                    items.append(item)
            else:
                # Just clean up the feature name a bit
                if item.startswith('Procedure_'):
                    items.append(f"Procedure: {item.replace('Procedure_', '')}")
                elif item.startswith('Diagnosis_'):
                    items.append(f"Diagnosis: {item.replace('Diagnosis_', '')}")
                elif item.startswith('Gender_'):
                    items.append(f"Gender: {item.replace('Gender_', '')}")
                elif item.startswith('Age_'):
                    items.append(f"Age Category: {item.replace('Age_', '')}")
                else:
                    items.append(item)
        
        return items
    
    # Format antecedents and consequents
    readable_rules['readable_antecedents'] = readable_rules['antecedents'].apply(
        lambda x: format_itemset(x, feature_mappings)
    )
    
    readable_rules['readable_consequents'] = readable_rules['consequents'].apply(
        lambda x: format_itemset(x, feature_mappings)
    )
    
    # Create rule strings
    readable_rules['rule_string'] = readable_rules.apply(
        lambda x: f"{x['readable_antecedents']} => {x['readable_consequents']} "
                 f"(Support: {x['support']:.3f}, Confidence: {x['confidence']:.3f}, Lift: {x['lift']:.3f})",
        axis=1
    )
    
    # Save to CSV
    os.makedirs('output', exist_ok=True)
    readable_rules[['rule_string', 'support', 'confidence', 'lift', 'readable_antecedents', 'readable_consequents']].to_csv('output/readable_rules.csv', index=False)
    
    print(f"Saved {len(readable_rules)} human-readable rules")
    
    return readable_rules
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
    essential_columns = ['hadm_id', 'subject_id']
    admissions_subset = admissions[essential_columns].copy()

    #admissions_subset["discharge_location"] = admissions_subset["discharge_location"].fillna("Unknown")

    #print(f"Missing values in discharge_location after handling: {admissions_subset['discharge_location'].isna().sum()}")

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
    print(f"Transactions base preview: \n{transactions_base.head()}")

    return transactions_base, diagnoses_with_desc

def create_readable_transaction_matrix(transactions_matrix):
    """
    Converts a one-hot encoded transaction matrix to a human-readable format.
    
    Args:
        transactions_matrix (DataFrame): The one-hot encoded transaction matrix
        
    Returns:
        DataFrame: A DataFrame where each row contains the hadm_id and a list of active features
    """
    print("Creating human-readable transaction matrix...")
    
    # Create a new DataFrame to store the results
    readable_matrix = pd.DataFrame(index=transactions_matrix.index)
    readable_matrix['hadm_id'] = readable_matrix.index
    readable_matrix['active_features'] = ''
    
    # For each row, collect the names of the columns where the value is 1
    for idx in transactions_matrix.index:
        # Get boolean series where True indicates a 1 in the original matrix
        active_cols = transactions_matrix.loc[idx] == 1
        
        # Get the names of active columns
        active_features = active_cols.index[active_cols].tolist()
        
        # Store in the new DataFrame
        readable_matrix.loc[idx, 'active_features'] = str(active_features)
    
    # Split the features by category for better readability
    readable_matrix['demographics'] = readable_matrix['active_features'].apply(
        lambda x: [f for f in eval(x) if f.startswith(('Gender_', 'Age_')) or f == 'anchor_age']
    )
    
    readable_matrix['procedures'] = readable_matrix['active_features'].apply(
        lambda x: [f for f in eval(x) if f.startswith('Procedure_')]
    )
    
    readable_matrix['diagnoses'] = readable_matrix['active_features'].apply(
        lambda x: [f for f in eval(x) if f.startswith('Diagnosis_')]
    )
    
    # Save to CSV
    os.makedirs('output', exist_ok=True)
    readable_matrix.to_csv('output/readable_transaction_matrix.csv', index=False)
    
    print(f"Saved human-readable transaction matrix with {len(readable_matrix)} rows")
    
    return readable_matrix

def create_detailed_transaction_matrix(transactions_matrix, transactions_base, diagnoses_with_desc, procedures_with_desc):
    """
    Creates a detailed, human-readable transaction matrix with decoded feature descriptions.
    
    Args:
        transactions_matrix (DataFrame): The one-hot encoded transaction matrix
        transactions_base (DataFrame): The base transactions dataframe with raw data
        diagnoses_with_desc (DataFrame): The diagnoses dataframe with descriptions
        procedures_with_desc (DataFrame): The procedures dataframe with descriptions
        
    Returns:
        DataFrame: A detailed, human-readable transaction matrix
    """
    print("Creating detailed human-readable transaction matrix...")
    
    # Create a mapping of hadm_id to patient info
    patient_info = transactions_base[['hadm_id', 'subject_id', 'anchor_age', 'gender', 'age_category', 'primary_diagnosis', 'primary_diagnosis_code']].drop_duplicates()
    patient_info_dict = patient_info.set_index('hadm_id').to_dict('index')
    
    # Create a new dataframe
    detailed_matrix = pd.DataFrame(index=transactions_matrix.index)
    detailed_matrix['hadm_id'] = detailed_matrix.index
    
    # Add patient demographic information
    detailed_matrix['subject_id'] = detailed_matrix['hadm_id'].map(lambda x: patient_info_dict.get(x, {}).get('subject_id', 'Unknown'))
    detailed_matrix['age'] = detailed_matrix['hadm_id'].map(lambda x: patient_info_dict.get(x, {}).get('anchor_age', 'Unknown'))
    detailed_matrix['gender'] = detailed_matrix['hadm_id'].map(lambda x: patient_info_dict.get(x, {}).get('gender', 'Unknown'))
    detailed_matrix['age_category'] = detailed_matrix['hadm_id'].map(lambda x: patient_info_dict.get(x, {}).get('age_category', 'Unknown'))
    detailed_matrix['primary_diagnosis'] = detailed_matrix['hadm_id'].map(lambda x: patient_info_dict.get(x, {}).get('primary_diagnosis', 'Unknown'))
    detailed_matrix['primary_diagnosis_code'] = detailed_matrix['hadm_id'].map(lambda x: patient_info_dict.get(x, {}).get('primary_diagnosis_code', 'Unknown'))
    
    # Create column for active procedures
    # Get procedure columns from transaction matrix
    procedure_columns = [col for col in transactions_matrix.columns if col.startswith('Procedure_')]
    
    # For each admission, find which procedures are active (value = 1)
    def get_active_procedures(hadm_id):
        # Check if hadm_id exists in transactions_matrix
        if hadm_id not in transactions_matrix.index:
            return []
        
        # Get active procedures
        row = transactions_matrix.loc[hadm_id]
        active_procs = [col for col in procedure_columns if row[col] == 1]
        
        # Extract procedure names from column names
        proc_names = [col.replace('Procedure_', '') for col in active_procs]
        
        return proc_names
    
    # Map procedure ICD codes to descriptions if available
    detailed_matrix['active_procedures'] = detailed_matrix['hadm_id'].apply(get_active_procedures)
    
    # Get procedure descriptions from procedures_with_desc
    if procedures_with_desc is not None and not procedures_with_desc.empty:
        proc_desc_dict = dict(zip(
            procedures_with_desc['long_title'],
            procedures_with_desc['long_title']
        ))
        
        def format_procedures(proc_list):
            if not proc_list:
                return []
            return [f"{proc}" for proc in proc_list]
        
        detailed_matrix['active_procedures'] = detailed_matrix['active_procedures'].apply(format_procedures)
    
    # Get diagnosis columns
    diagnosis_columns = [col for col in transactions_matrix.columns if col.startswith('Diagnosis_')]
    
    # For each admission, find which diagnoses are active
    def get_active_diagnoses(hadm_id):
        # Check if hadm_id exists in transactions_matrix
        if hadm_id not in transactions_matrix.index:
            return []
        
        # Get active diagnoses
        row = transactions_matrix.loc[hadm_id]
        active_diags = [col for col in diagnosis_columns if row[col] == 1]
        
        # Extract diagnosis names from column names
        diag_names = [col.replace('Diagnosis_', '') for col in active_diags]
        
        return diag_names
    
    detailed_matrix['active_diagnoses'] = detailed_matrix['hadm_id'].apply(get_active_diagnoses)
    
    # Count the number of active features in each category
    detailed_matrix['procedure_count'] = detailed_matrix['active_procedures'].apply(len)
    detailed_matrix['diagnosis_count'] = detailed_matrix['active_diagnoses'].apply(len)
    detailed_matrix['total_feature_count'] = detailed_matrix['procedure_count'] + detailed_matrix['diagnosis_count']
    
    # Save to CSV
    os.makedirs('output', exist_ok=True)
    detailed_matrix.to_csv('output/detailed_transaction_matrix.csv', index=False)
    
    print(f"Saved detailed human-readable transaction matrix with {len(detailed_matrix)} rows")
    
    return detailed_matrix

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

    # Create procedure presence feature
    procedure_counts = procedures_with_desc["long_title"].value_counts()
    min_procedure_freq = 2

    common_procedures = procedure_counts[procedure_counts >= min_procedure_freq].index.tolist()
    print(f"Using {len(common_procedures)} common procedures for feature engineering out of {len(procedure_counts)} total procedures")

    procedures_filtered = procedures_with_desc[procedures_with_desc["long_title"].isin(common_procedures)]

    print(f"Filtered procedures dataset shape: {procedures_filtered.shape}")
    print(f"Filtered procedures dataset preview: \n{procedures_filtered.head()}")

    # Checking if procedures are more than the frequency threshold
    if len(procedures_filtered) == 0:
        print("WARNING: No procedures match the frequency threshold. Reducing threshold.")
        min_procedure_freq = 10
        common_procedures = procedure_counts[procedure_counts >= min_procedure_freq].index.tolist()
        procedures_filtered = procedures_with_desc[procedures_with_desc['long_title'].isin(common_procedures)]
        print(f"Using {len(common_procedures)} procedures with reduced threshold")
    
    # Create mapping of procedure codes to descriptions for readability
    procedure_mapping = procedures_with_desc[['icd_code', 'long_title']].drop_duplicates().set_index('icd_code')['long_title'].to_dict()
    
    if len(procedures_filtered) > 0:
        procedures_pivot = pd.get_dummies(procedures_filtered[["hadm_id", "long_title"]], columns=["long_title"], prefix="Procedure", prefix_sep="_")
        procedures_by_admission= procedures_pivot.groupby("hadm_id").max()
        print(f"Created procedures with admissons with shape: {procedures_by_admission.shape}")
    else:
        print("WARNING: No procedures found after filtering. Skipping procedure feature engineering.")
        procedures_by_admission = pd.DataFrame(index = transactions_base["hadm_id"].unique())
    
    # Create demographic features - REMOVING anchor_age as requested
    demographic_cols = ["hadm_id", "gender", "age_category"]  # Removed anchor_age

    demographic_features = pd.get_dummies(transactions_base[demographic_cols], columns=["gender", "age_category"], prefix=["Gender", "Age"], prefix_sep="_")

    if "hospital_expire_flag" in demographic_features.columns:
        demographic_features["Expired_In_Hospital"] = demographic_features["hospital_expire_flag"]
        demographic_features = demographic_features.drop("hospital_expire_flag", axis=1)
    
    demographics_by_admission = demographic_features.groupby("hadm_id").first()
    print(f"Created demographic features with shape: {demographics_by_admission.shape}")

    print(f"Demographic features preview: \n{demographics_by_admission.head()}")

    # Merge all features into one dataset
    all_features = pd.DataFrame(index = transactions_base["hadm_id"].unique())

    # List to keep track of dataframes with potential join issues
    empty_dfs = []  
    for name, df in [("Procedures", procedures_by_admission), ("Demographics", demographics_by_admission)]:
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

    # Fill missing values with 0
    all_features = all_features.fillna(0)
    
    # 8. Create diagnosis outcome features
    diagnosis_counts = transactions_base['primary_diagnosis'].value_counts()

    # Create mapping of diagnosis codes to descriptions for readability
    diagnosis_mapping = transactions_base[['primary_diagnosis_code', 'primary_diagnosis']].drop_duplicates().set_index('primary_diagnosis_code')['primary_diagnosis'].to_dict()
    
    # We want at least 10 diagnoses, but respect max_features budget
    max_procedure_features = max_features_count // 2  # Reserve half for procedures, half for diagnoses
    diagnosis_feature_limit = max(10, max_features_count - len(demographics_by_admission.columns) - 
                            min(max_procedure_features, 
                                len(procedures_by_admission.columns) if hasattr(procedures_by_admission, 'columns') else 0))

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

    # Printing transaction matrix shape
    print("TRANSACTION MATRIX")
    print(f"Transaction matrix shape after joining features and outcomes: {transactions_matrix.shape}")
    print(f"Transaction matrix preview: \n{transactions_matrix.head()}")
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
         f"and {len(diagnosis_by_admission.columns) if hasattr(diagnosis_by_admission, 'columns') else 0} diagnosis outcomes")
    
    # Save the transaction matrix
    os.makedirs('output', exist_ok=True)
    transactions_matrix.to_csv('output/transaction_matrix.csv')
    
    # Save mappings for later use in readability
    mappings = {
        'procedure_mapping': procedure_mapping,
        'diagnosis_mapping': diagnosis_mapping
    }
    
    with open('output/feature_mappings.pkl', 'wb') as f:
        pickle.dump(mappings, f)
    
    # Create and save human-readable versions of the transaction matrix
    readable_matrix = create_readable_transaction_matrix(transactions_matrix)
    detailed_matrix = create_detailed_transaction_matrix(transactions_matrix, transactions_base, diagnoses_with_desc, procedures_with_desc)
    
    # For debug purposes, also save feature counts
    feature_counts = pd.Series({
        'demographic_features': len(demographics_by_admission.columns),
        'procedure_features': len(procedures_by_admission.columns) if hasattr(procedures_by_admission, 'columns') else 0,
        'diagnosis_features': len(diagnosis_by_admission.columns) if hasattr(diagnosis_by_admission, 'columns') else 0,
        'total_features': transactions_matrix.shape[1],
        'total_transactions': transactions_matrix.shape[0]
    })
    feature_counts.to_csv('output/feature_counts.csv')
    
    return transactions_matrix


def filter_diagnosis_to_procedure_demographic_rules(rules_df, transactions_matrix):
    """
    Filter rules where antecedents (LHS) are diagnoses and consequents (RHS) are 
    procedures or demographic features.
    
    Args:
        rules_df (DataFrame): The complete set of association rules
        transactions_matrix (DataFrame): The transaction matrix to identify feature types
        
    Returns:
        DataFrame: Filtered rules
    """
    if rules_df.empty:
        print("No rules to filter.")
        return pd.DataFrame()
    
    # Identify feature types from the transaction matrix
    diagnosis_cols = [col for col in transactions_matrix.columns if col.startswith('Diagnosis_')]
    procedure_cols = [col for col in transactions_matrix.columns if col.startswith('Procedure_')]
    # Modify demographic cols to exclude anchor_age
    demographic_cols = [col for col in transactions_matrix.columns 
                       if col.startswith(('Gender_', 'Age_'))]  # Removed anchor_age
    
    # Filter rules where:
    # 1. Antecedents (LHS) contain only diagnosis features
    # 2. Consequents (RHS) contain only procedure or demographic features
    filtered_rules = rules_df[rules_df.apply(
        lambda row: (
            # Check that all antecedents are diagnoses
            all(item in diagnosis_cols for item in row['antecedents'])
            and 
            # Check that all consequents are either procedures or demographics
            all(item in procedure_cols or item in demographic_cols for item in row['consequents'])
        ),
        axis=1
    )]
    
    # Check if we found any matching rules
    if filtered_rules.empty:
        print("No rules matching the Diagnosis → Procedure/Demographic pattern.")
        return pd.DataFrame()
    
    print(f"Found {len(filtered_rules)} rules where diagnoses predict procedures or demographics.")
    
    # Sort by lift for most interesting rules first
    filtered_rules = filtered_rules.sort_values('lift', ascending=False)
    
    # Save the filtered rules
    os.makedirs('output', exist_ok=True)
    filtered_rules.to_csv('output/diagnosis_to_proc_demo_rules.csv', index=False)
    
    # Create a human-readable version
    try:
        # Load the feature mappings if available
        if os.path.exists('output/feature_mappings.pkl'):
            with open('output/feature_mappings.pkl', 'rb') as f:
                feature_mappings = pickle.load(f)
            
            # Create a human-readable version of the filtered rules
            readable_filtered_rules = create_readable_rules(filtered_rules, feature_mappings)
            readable_filtered_rules.to_csv('output/readable_diagnosis_to_proc_demo_rules.csv', index=False)
    except Exception as e:
        print(f"WARNING: Could not create readable filtered rules: {str(e)}")
    
    return filtered_rules

def mine_association_rules(transactions_matrix, min_support=0.0006, min_confidence=0.5):
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
    
    # 1. Find frequent itemsets with adaptive support threshold using FP-Growth
    # Try to find a reasonable number of itemsets
    frequent_itemsets = pd.DataFrame()
    
    try:
        # Using fpgrowth instead of apriori
        frequent_itemsets = fpgrowth(transactions_matrix_bool, 
                                   min_support=min_support,
                                   use_colnames=True,
                                   max_len=4)  # Limit to combinations of at most 4 items
        
        # If we found too few itemsets, try with a lower threshold
        if len(frequent_itemsets) < 10:
            old_support = min_support
            min_support = max(min_support_floor, min_support / 2)
            print(f"Found too few itemsets ({len(frequent_itemsets)}). Reducing support from {old_support} to {min_support}")
            
            frequent_itemsets = fpgrowth(transactions_matrix_bool, 
                                       min_support=min_support,
                                       use_colnames=True,
                                       max_len=4)
    except Exception as e:
        print(f"ERROR in FP-Growth algorithm: {str(e)}")
        print("Trying with a smaller dataset...")
        
        # Sample the data if it's too large
        if transactions_matrix.shape[0] > 10000:
            sample_size = min(10000, int(transactions_matrix.shape[0] * 0.5))
            transactions_sample = transactions_matrix.sample(sample_size)
            try:
                frequent_itemsets = fpgrowth(transactions_sample.astype(bool), 
                                           min_support=min_support,
                                           use_colnames=True,
                                           max_len=3)
                print(f"Successfully ran FP-Growth on a sample of {sample_size} transactions")
            except Exception as e2:
                print(f"ERROR in FP-Growth algorithm even with sampling: {str(e2)}")
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    if frequent_itemsets.empty:
        print("No frequent itemsets found. Cannot generate rules.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
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
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    if rules.empty:
        print("No rules generated. Cannot proceed with rule filtering.")
        return frequent_itemsets, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    print(f"Generated {len(rules)} rules with min_confidence={min_confidence}")
    
    # Save all rules
    rules.to_csv('output/all_rules.csv', index=False)
    
    # 3. Filter rules to focus on procedures
    procedure_cols = [col for col in transactions_matrix.columns if col.startswith('Procedure_')]
    if not procedure_cols:
        print("ERROR: No procedures columns found in transaction matrix")
        return frequent_itemsets, rules, pd.DataFrame(), pd.DataFrame()
    
    print(f"Found {len(procedure_cols)} procedures columns to use for rule filtering")

    # Convert string representations of sets to actual sets, if needed
    if isinstance(rules['antecedents'].iloc[0], str):
        rules['antecedents'] = rules['antecedents'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        rules['consequents'] = rules['consequents'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    # Filter for rules that predict procedures
    procedure_rules = rules[rules['consequents'].apply(
        lambda x: any(item in procedure_cols for item in x)
    )].copy()
    
    if procedure_rules.empty:
        print("WARNING: No rules found with procedures in the consequent")
    else:
        print(f"Found {len(procedure_rules)} rules with procedures in the consequent")
        
        # Additional filters to focus on more interesting rules
        if len(procedure_rules) > 1000:
            print(f"Too many rules ({len(procedure_rules)}). Filtering to more interesting ones...")
            
            # Filter by lift (stronger associations)
            high_lift_rules = procedure_rules[procedure_rules['lift'] > 1.5]
            if len(high_lift_rules) >= 100:
                procedure_rules = high_lift_rules
                print(f"Filtered to {len(procedure_rules)} rules with lift > 1.5")
        
        # Sort by lift and then confidence
        procedure_rules = procedure_rules.sort_values(['lift', 'confidence'], ascending=[False, False])
        
        # Save procedure rules
        procedure_rules.to_csv('output/procedure_rules.csv', index=False)
    
    # 4. Filter for diagnosis -> procedure/demographic rules
    diagnosis_to_proc_demo_rules = filter_diagnosis_to_procedure_demographic_rules(rules, transactions_matrix)
    
    # 5. Create human-readable versions of the rules
    try:
        # Load the feature mappings if available
        if os.path.exists('output/feature_mappings.pkl'):
            with open('output/feature_mappings.pkl', 'rb') as f:
                feature_mappings = pickle.load(f)
            
            # Create human-readable versions of the rules
            if not procedure_rules.empty:
                readable_rules = create_readable_rules(procedure_rules, feature_mappings)
    except Exception as e:
        print(f"WARNING: Could not create readable rules: {str(e)}")
    
    return frequent_itemsets, rules, procedure_rules, diagnosis_to_proc_demo_rules



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run the association rule mining pipeline')
    parser.add_argument('--sample_fraction', type=float, default=0.01, help='Fraction of data to sample')
    parser.add_argument('--min_support', type=float, default=0.0006, help='Minimum support threshold')
    parser.add_argument('--min_confidence', type=float, default=0.5, help='Minimum confidence threshold')
    parser.add_argument('--skip_visualizations', action='store_true', help='Skip creating visualizations')
    parser.add_argument('--skip_report', action='store_true', help='Skip generating HTML report')
    parser.add_argument('--read_only', action='store_true', help='Only read existing data without reprocessing')
    
    args = parser.parse_args()
    
    print(f"Running pipeline with:")
    print(f"  - sample_fraction: {args.sample_fraction}")
    print(f"  - min_support: {args.min_support}")
    print(f"  - min_confidence: {args.min_confidence}")
    print(f"  - visualizations: {'Disabled' if args.skip_visualizations else 'Enabled'}")
    print(f"  - HTML report: {'Disabled' if args.skip_report else 'Enabled'}")
    print(f"  - mode: {'Read-only' if args.read_only else 'Full processing'}")
    
    # If we're in read-only mode, we'll just load existing files
    if args.read_only:
        print("\nRunning in read-only mode. Loading existing data...")
        
        if os.path.exists('output/transaction_matrix.csv'):
            transactions_matrix = pd.read_csv('output/transaction_matrix.csv', index_col=0)
            print(f"Loaded transaction matrix with shape {transactions_matrix.shape}")
            
            if os.path.exists('output/all_rules.csv'):
                rules = pd.read_csv('output/all_rules.csv')
                print(f"Loaded {len(rules)} rules")
                
                # Convert string representations of sets to actual sets for visualization
                if 'antecedents' in rules.columns and isinstance(rules['antecedents'].iloc[0], str):
                    rules['antecedents'] = rules['antecedents'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                    rules['consequents'] = rules['consequents'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            else:
                rules = pd.DataFrame()
                print("No rules file found.")
            
            if os.path.exists('output/procedure_rules.csv'):
                procedure_rules = pd.read_csv('output/procedure_rules.csv')
                print(f"Loaded {len(procedure_rules)} procedure rules")
                
                # Convert string representations of sets to actual sets for visualization
                if 'antecedents' in procedure_rules.columns and isinstance(procedure_rules['antecedents'].iloc[0], str):
                    procedure_rules['antecedents'] = procedure_rules['antecedents'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                    procedure_rules['consequents'] = procedure_rules['consequents'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            else:
                procedure_rules = pd.DataFrame()
                print("No procedure rules file found.")
            
            # Create visualizations if requested
            if not args.skip_visualizations and visualization_available:
                print("\nCreating visualizations from existing data...")
                
                # Visualize feature distribution in transaction matrix
                visualize_feature_distribution(transactions_matrix, save_path='output/feature_distribution.png')
                
                # Visualize rule metrics
                if not rules.empty:
                    visualize_rule_metrics(rules, save_path='output/rule_metrics.png')
                    visualize_rules_summary(rules, save_path='output/rules_summary.png')
                
                # Visualize procedure rules network
                if not procedure_rules.empty:
                    visualize_rules_network(procedure_rules, max_rules=50, min_lift=1.0, 
                                          save_path='output/procedure_rules_network.png')
        else:
            print("ERROR: No transaction matrix file found. Cannot proceed in read-only mode.")
    else:
        # Run the full pipeline
        frequent_itemsets, rules, procedure_rules = main(
            args.sample_fraction, 
            args.min_support, 
            args.min_confidence,
            not args.skip_visualizations,
            not args.skip_report
        )
        
        print("\nPipeline completed successfully!")
        
        if not procedure_rules.empty:
            print(f"\nTop 5 procedure rules by lift:")
            print(procedure_rules.sort_values('lift', ascending=False).head(5)[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
            
            print("\nCheck the output directory for detailed results and human-readable formats.")
        else:
            print("\nNo procedure rules were found. Try adjusting the parameters or check your data.")