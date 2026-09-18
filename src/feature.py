import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.getcwd())

from src.data_pipeline import load_raw_data, clean_data


#Define feature columns
vital_cols = ['heart_rate',
       'temperature', 'oxygen_saturation', 'respiratory_rate',
       'blood_pressure']
labs_cols = ['white_cell_count', 'crp',
       'lactate', 'creatinine', 'platelet_count']

def get_prediction_time(outcome_df: pd.DataFrame, vitals_df: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    outcomes = outcome_df.copy()

    def calc_cutoff(row):
        if row['sepsis_event']:
                return row['diagnosis_time'] - pd.Timedelta(hours=9)
        pv = vitals_df[vitals_df['patient_id']== row['patient_id']]
        start, end = pv['timestamp'].min(), pv['timestamp'].max()
        span_hours = max((end - start).total_seconds() / 3600, 1)
        offset = rng.uniform(0.4, 0.9) * span_hours
        return start + pd.Timedelta(hours=offset)

    outcomes['prediction_time']= outcomes.apply(calc_cutoff, axis=1)
    return outcomes

def extract_vitals_features(vital_df: pd.DataFrame, patient_id: int, cutoff: pd.Timestamp, lookback_hours:
                            float = 6.0) -> dict:
    window = vital_df[
            (vital_df['patient_id'] == patient_id) &
            (vital_df['timestamp']<=cutoff)&
            (vital_df['timestamp'] >= cutoff - pd.Timedelta(hours= lookback_hours))
        ]
    if window.empty:
            window = vital_df[(vital_df['patient_id'] == patient_id) & (vital_df['timestamp'] <= cutoff)].tail(1)

    feats = {}
    for col in vital_cols:
            if window.empty or col not in window:
                 feats[f'{col}_mean'] = np.nan
                 feats[f'{col}_min'] = np.nan
                 feats[f'{col}_max'] = np.nan
                 feats[f'{col}_std'] =  0.0
                 feats[f'{col}_last'] = np.nan
                 feats[f'{col}_rate_per_hr'] = 0.0
            else:
                 vals = window[col].dropna()
                 if len(vals) == 0:
                      feats[f'{col}_mean'] = np.nan
                      feats[f'{col}_min'] = np.nan
                      feats[f'{col}_max'] = np.nan
                      feats[f'{col}_std'] =  0.0
                      feats[f'{col}_last'] = np.nan
                      feats[f'{col}_rate_per_hr'] = 0.0
                 else:
                    feats[f'{col}_mean'] = vals.mean()
                    feats[f'{col}_min'] = vals.min()
                    feats[f'{col}_max'] = vals.max()
                    feats[f'{col}_std'] = vals.std() if len(vals) >1 else 0.0
                    feats[f'{col}_last'] = vals.iloc[-1] 
                      
                    if len(window) > 1:
                                  hours = (window['timestamp'].iloc[-1] - window['timestamp'].iloc[0]).total_seconds()/ 3600
                                  feats[f'{col}_rate_per_hr'] = (vals.iloc[-1] - vals.iloc[0]) / hours if hours > 0 else 0.0
                    else:
                        feats[f'{col}_rate_per_hr'] = 0.0
                              
    return feats
                

def extract_lab_features(labs_df: pd.DataFrame, patient_id: int, cutoff: pd.Timestamp,
                         lookback_hours: float = 12.0) -> dict:
      window = labs_df[
              (labs_df['patient_id'] == patient_id) &
              (labs_df['timestamp']<=cutoff)&
              (labs_df['timestamp'] >= cutoff - pd.Timedelta(hours=lookback_hours))
          ]
      if window.empty:
              window = labs_df[(labs_df['patient_id']== patient_id) & (labs_df['timestamp'] <= cutoff)].tail(1)

      feats = {}
      for col in labs_cols:
              if window.empty or col not in window:
                 feats[f'{col}_mean'] = np.nan
                 feats[f'{col}_last'] = np.nan
              else:
                    vals = window[col].dropna()
                    if len(vals) == 0:
                          feats[f'{col}_mean'] = np.nan
                          feats[f'{col}_last'] = np.nan
                    else:
                          feats[f'{col}_mean'] = vals.mean()
                          feats[f'{col}_last'] = vals.iloc[-1]                         
      return feats

def build_feature_matrix(patient_df: pd.DataFrame,
            vitals_df: pd.DataFrame,
            labs_df: pd.DataFrame,
            outcomes_df: pd.DataFrame) -> pd.DataFrame: 
    
    outcomes_with_time = get_prediction_time(outcomes_df, vitals_df)

    #static features
    static = patient_df[['patient_id','age','gender','medical_conditions']].copy()
    static['comorbidity_count'] = static['medical_conditions'].apply(
                 lambda x: 0 if pd.isna(x) or str(x).strip().lower() in ('none','none reported')
                 else len(str(x).split(','))
                  )
    static['gender_Male'] = (static['gender'].astype(str).str.lower()== 'male').astype(int)
    static = static.drop(columns=['gender','medical_conditions'])

    #vital features
    vital_rows = []
    for pid, cutoff in zip(outcomes_with_time['patient_id'], outcomes_with_time['prediction_time']):
        vital_rows.append({'patient_id': pid, **extract_vitals_features(vitals_df, pid, cutoff)})
    vital_df = pd.DataFrame(vital_rows)

    #lab features
    lab_rows = []
    for pid, cutoff in zip(outcomes_with_time['patient_id'], outcomes_with_time['prediction_time']):
      lab_rows.append({'patient_id':pid, **extract_lab_features(labs_df, pid, cutoff)})
    lab_df = pd.DataFrame(lab_rows)

    #merge all tables
    features = (
    static
    .merge(vital_df, on='patient_id')
    .merge(lab_df, on='patient_id')
    .merge(outcomes_with_time[['patient_id','sepsis_event']], on='patient_id')
       )

    feature_cols = [
    c for c in features.columns
    if c not in ('patient_id','sepsis_event')]
    features[feature_cols] = features[feature_cols].fillna(features[feature_cols].median())
    features['sepsis_event'] = features['sepsis_event'].astype(int)

    return features

if __name__ == '__main__':

      from src.data_pipeline import load_raw_data, clean_data
      history, labs, patients, outcome, vitals = clean_data(*load_raw_data())
      feat_matrix = build_feature_matrix(patients,vitals,labs, outcome)
      print('Engineered feature matrix shape:', feat_matrix.shape)
      print('Class distribution:\n', feat_matrix['sepsis_event'].value_counts(normalize=True))      