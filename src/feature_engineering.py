import numpy as np
import pandas as pd
import os 


def load_clean_data(data_dir: str = 'data/processed'):
    
    history = pd.read_csv('../data/processed/history_clean.csv')
    labs =  pd.read_csv('../data/processed/laboratory_clean.csv', parse_dates=['timestamp'])
    patients =   pd.read_csv('../data/processed/patients_clean.csv', parse_dates=['registration_date'])
    outcome =  pd.read_csv('../data/processed/outcomes_clean.csv', parse_dates=['diagnosis_time'])
    vitals =  pd.read_csv('../data/processed/vitals_clean.csv', parse_dates=['timestamp'])

    return history, labs, patients, outcome, vitals

rng = np.random.default_rng(7)
def get_prediction_time(row, vitals_df):
    if row['sepsis_event']:
        return row['diagnosis_time'] - pd.Timedelta(hours=9)
    pv = vitals_df[vitals_df['patient_id']== row['patient_id']]
    start, end = pv['timestamp'].min(), pv['timestamp'].max()
    span_hours = max((end - start).total_seconds() / 3600, 1)
    offset = rng.uniform(0.4, 0.9) * span_hours
    return start + pd.Timedelta(hours=offset)

outcomes = outcome.copy()
outcomes['prediction_time'] = outcome.apply(get_prediction_time, vitals_df=vitals, axis=1)
    

LOOKBACK_HOURS = 6
def vital_features(pid, cutoff, df):
    window = df[
        (df['patient_id'] == pid) &
        (df['timestamp']<=cutoff)&
        (df['timestamp'] >= cutoff - pd.Timedelta(hours=LOOKBACK_HOURS))
    ]
    if window.empty:
        window = df[(df['patient_id'] == pid) & (df['timestamp'] <= cutoff)].tail(1)
    feats = {}
    for col in vital_cols:
        vals = window[col]
        feats[f'{col}_mean'] = vals.mean()
        feats[f'{col}_min'] = vals.min()
        feats[f'{col}_max'] = vals.max()
        feats[f'{col}_std'] = vals.std() if len(vals) >1 else 0.0
        feats[f'{col}_last'] = vals.iloc[-1] if len(vals) else np.nan

        if len(window) > 1:
            hours = (window['timestamp'].iloc[-1] - window['timestamp'].iloc[0]).total_seconds()/ 3600
            feats[f'{col}_rate_per_hr'] = (vals.iloc[0] - vals.iloc[0]) / hours if hours > 0 else 0.0
        else:
            feats[f'{col}_rate_per_hr'] = 0.0
        
    return feats


LAB_LOOKBACK_HOURS = 8
def lab_features(pid, cutoff, df):
    window = df[
        (df['patient_id'] == pid) &
        (df['timestamp']<=cutoff)&
        (df['timestamp'] >= cutoff - pd.Timedelta(hours=LAB_LOOKBACK_HOURS))
    ]
    if window.empty:
        window = df[(df['patient_id']==pid) & (df['timestamp'] <= cutoff)].tail(1)
    feats = {}
    for col in labs_cols:
        vals = window[col]
        feats[f'{col}_mean'] = vals.mean()
        feats[f'{col}_last'] = vals.iloc[-1] if len(vals) else np.nan

    return feats


#Patient demography
def pt_static(df):
    static = df[['patient_id','age','gender']].copy()

    static['comorbidity_count'] = patients['medical_conditions'].apply(
             lambda x: 0 if pd.isna(x) or x =='Not reported'
             else len(x.split(','))
              )
    static = pd.get_dummies(static, columns=['gender'], drop_first= True)
    return static


#sepsis features
def merged_df ( static: pd.DataFrame,
            vital_features_df: pd.DataFrame,
            lab_features_df: pd.DataFrame,
            outcome: pd.DataFrame):
    feature = (
    static
    .merge(vital_features_df, on='patient_id')
    .merge(lab_features_df, on='patient_id')
    .merge(outcome[['patient_id','sepsis_event']], on='patient_id')
     )

    feature_cols = [
    c for c in feature.columns
    if c not in ('patient_id','sepsis_event')
     ]

    numeric_cols = feature[feature_cols].select_dtypes(include='number').columns

    feature[numeric_cols]= feature[numeric_cols].fillna(
    feature[numeric_cols].median()
     )
    feature['sepsis_event'] = feature['sepsis_event'].astype(int)

    return feature

if __name__ == '__main__':

    #load cleaned datasets
    history, labs, patients, outcome, vitals, = load_clean_data()

    #Define feature columns
    vital_cols = ['heart_rate',
       'temperature', 'oxygen_saturation', 'respiratory_rate',
       'blood_pressure']
    labs_cols = ['white_cell_count', 'crp',
       'lactate', 'creatinine', 'platelet_count']

    #Create prediction times
    outcomes = outcome.copy()
    outcomes['prediction_time'] = outcome.apply(get_prediction_time, vitals_df=vitals, axis=1)

    # Generate vital features
    vital_features_rows = [
        {  'patient_id': pid, **vital_features(pid, cutoff, vitals)}

        for pid, cutoff in zip(
            outcomes['patient_id'], outcomes['prediction_time'] ) ]

    vital_features_df = pd.DataFrame(vital_features_rows)

    # Generate laboratory features
    lab_features_rows = [
        {'patient_id': pid, **lab_features(pid, cutoff, labs) }
        for pid, cutoff in zip(
            outcomes['patient_id'], outcomes['prediction_time'] )]

    lab_features_df = pd.DataFrame(lab_features_rows)

    # Generate static patient features
    static_features = pt_static(patients)

    # Merge all features into final modelling dataset
    sepsis_dataset = merged_df(
        static=static_features,
        vital_features_df=vital_features_df,
        lab_features_df=lab_features_df,
        outcome=outcomes
    )

    # Save final feature dataset
    sepsis_dataset.to_csv(
        '../data/processed/sepsis_model_features.csv',
        index=False
    )
    print("Feature engineering completed.")
    