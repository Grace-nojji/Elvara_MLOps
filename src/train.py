from pathlib import Path
import pandas as pd
import numpy as np
import os
import sys
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix,classification_report
)
sys.path.append(os.getcwd())

from src.data_pipeline import load_raw_data, clean_data
from src.feature import build_feature_matrix


def train_and_evaluate_model(
        models_dir : str =  'models',
        data_dir: str  = 'data/processed'
):
    models_dir = Path(models_dir)
    data_dir = Path(data_dir)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    print("------- Step 1: load & clean data ----")
    history, labs, patients, outcome, vitals = load_raw_data()
    h_c, l_c, p_c, o_c, v_c = clean_data(history, labs, patients, outcome, vitals)
    #history, labs, patients, outcome, vitals = clean_data(*load_raw_data())

    print("---- Step 2: Feature Engineering -----")
    df_features = build_feature_matrix(patient_df = p_c, 
                 vitals_df = v_c, 
                 labs_df = l_c, 
                 outcomes_df = o_c)
    processed_csv_path = data_dir / 'sepsis_features.csv'
    df_features.to_csv(processed_csv_path, index=False)
    print(f'saved to (processed_csv_path) and has dataframe shape df {df_features.shape}')

    X = df_features.drop(columns=['patient_id', 'sepsis_event'])
    Y = df_features['sepsis_event']
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size = 0.25, random_state=42, stratify = Y
    )

    #1 Baseline: Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    baseline_lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    baseline_lr.fit(X_train_scaled, y_train)

    lr_probs = baseline_lr.predict_proba(X_test_scaled)[:,1]
    lr_preds = (lr_probs >= 0.5).astype(int)

    lr_roc = roc_auc_score(y_test, lr_probs)
    lr_prec = precision_score(y_test, lr_preds, zero_division=0)
    lr_rec = recall_score(y_test, lr_preds, zero_division=0)
    lr_f1 = f1_score(y_test, lr_preds, zero_division = 0)

    print("\n============ Baseline Model (Logistic Regression) ==========")
    print(f'ROC-AUC: {lr_roc:.4f}')
    print(f'Precision: {lr_prec:.4f}')
    print(f'Recall: {lr_rec:.4f}')
    print(f'f1 score: {lr_f1:.4f}')

    #Primary model : HistGradientClassifier
    primary_hg = HistGradientBoostingClassifier(
                max_iter = 150,
                learning_rate= 0.05,
                max_depth=5,
                min_samples_leaf=10,
                class_weight='balanced',
                random_state = 42 )

    
    primary_hg.fit(X_train, y_train)
    
    hgc_probs = primary_hg.predict_proba(X_test)[:,1]
    hgc_preds = (hgc_probs >= 0.5).astype(int)
    
    hgc_roc = roc_auc_score(y_test, hgc_probs)
    hgc_prec = precision_score(y_test, hgc_preds, zero_division=0)
    hgc_rec = recall_score(y_test, hgc_preds, zero_division=0)
    hgc_f1 = f1_score(y_test, hgc_preds, zero_division = 0)

    print("\n============ Primary Model (HistGradientBoostingClassifier) ==========")
    print(f'ROC-AUC: {hgc_roc:.4f}')
    print(f'Precision: {hgc_prec:.4f}')
    print(f'Recall: {hgc_rec:.4f}')
    print(f'f1 score: {hgc_f1:.4f}')

    #Save artifacts

    model_payload = {
        'model': primary_hg,
        'feature_names': feature_names,
        'medians': X.median().to_dict(),
        'baseline_model': baseline_lr,
        'scaler': scaler
    }
    model_path = os.path.join(models_dir, 'sepsis_model.joblib')
    joblib.dump(model_payload, model_path)

    metadata = {
        'model_type': 'HistGradientBoostingClassifier',
        'features': feature_names,
        'primary_metrics': {
            'roc_auc': round(hgc_roc, 4),
            'precision': round(hgc_prec, 4),
            'recall':round(hgc_rec, 4),
            'f1_score':round(hgc_f1, 4)
        },
        'baseline_metrics': {
            'model_type': 'LogisticRegression',
            'roc_auc': round(lr_roc, 4),
            'precision': round(lr_prec, 4),
            'recall':round(lr_rec, 4),
            'f1_score':round(lr_f1, 4)
        },
        'training_samples': X_train.shape[0],
        'testing_sample': X_test.shape[0]
    }

    meta_path = os.path.join(models_dir, 'model_meta_data.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return metadata
if __name__== "__main__":
    train_and_evaluate_model()


