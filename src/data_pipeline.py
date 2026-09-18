import pandas as pd
import numpy as np
import os

def load_raw_data(data_dir: str = 'data/raw'):
    
    history = pd.read_csv('data/raw/clinical_history (1).csv')
    labs =  pd.read_csv('data/raw/laboratory_results (1).csv', parse_dates=['timestamp'])
    patients =   pd.read_csv('data/raw/patients (1).csv', parse_dates=['registration_date'])
    outcome =  pd.read_csv('data/raw/sepsis_outcomes (1).csv', parse_dates=['diagnosis_time'])
    vitals =  pd.read_csv('data/raw/vital_signs (1).csv', parse_dates=['timestamp'])

    return history, labs, patients, outcome, vitals

def clean_data(
        history: pd.DataFrame,
        labs: pd.DataFrame,
        patients: pd.DataFrame,
        outcome: pd.DataFrame,
        vitals: pd.DataFrame,
    ):
    history_clean = history.copy()
    labs_clean = labs.copy()
    patients_clean = patients.copy()
    outcome_clean =outcome.copy()
    vitals_clean = vitals.copy()

    vitals_clean = vitals_clean.drop_duplicates(subset=['patient_id','timestamp']).reset_index(drop=True)
    labs_clean = labs_clean.drop_duplicates(subset=['patient_id','timestamp']).reset_index(drop=True)
    

    vitals_clean['heart_rate'] = vitals_clean['heart_rate'].clip(30,220)
    vitals_clean['temperature'] = vitals_clean['temperature'].clip(32, 43)
    vitals_clean['oxygen_saturation'] = vitals_clean['oxygen_saturation'].clip(50,100)
    vitals_clean['respiratory_rate'] = vitals_clean['respiratory_rate'].clip(5,60)
    vitals_clean['blood_pressure'] = vitals_clean['blood_pressure'].clip(40,220)
    
    labs_clean['white_cell_count']= labs_clean['white_cell_count'].clip(0.1,50)
    labs_clean['crp'] = labs_clean['crp'].clip(0.0, 500.0)
    labs_clean['lactate'] = labs_clean['lactate'].clip(1.0,20.0)
    labs_clean['creatinine'] = labs_clean['creatinine'].clip(0.1,10.0)
    labs_clean['platelet_count'] = labs_clean['platelet_count'].clip(5.0,700.0)

    labs_cols = ['white_cell_count', 'crp',
            'lactate', 'creatinine', 'platelet_count']
    vital_cols = ['heart_rate','temperature', 'oxygen_saturation',
               'respiratory_rate','blood_pressure']

    vitals_clean[vital_cols] = vitals_clean.groupby('patient_id')[vital_cols].transform(
                 lambda x: x.ffill())
    vitals_clean[vital_cols] = vitals_clean[vital_cols].fillna(vitals_clean[vital_cols].median())

    labs_clean[labs_cols] = labs_clean.groupby('patient_id')[labs_cols].transform(lambda x: x.ffill())
    labs_clean[labs_cols] = labs_clean[labs_cols].fillna(labs_clean[labs_cols].median())

    return history_clean,labs_clean, patients_clean, outcome_clean, vitals_clean

if __name__ == '__main__':
    history, labs, patients, outcome, vitals, = load_raw_data()
    h_c, l_c, p_c, o_c, v_c = clean_data(history, labs, patients, outcome, vitals)
    