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

load_data()