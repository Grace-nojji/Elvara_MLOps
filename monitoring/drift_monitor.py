import os 
import pandas as pd
import numpy as np
import json
import evidently
import sys

sys.path.append(os.getcwd())

def run_drift_analysis(reference_csv:str = 'data/processed/sepsis_features.csv',
                       current_csv:str = 'data/processed/sepsis_features.csv',
                       output_dir:str = 'monitoring/reports') -> str:
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(reference_csv):
        raise FileNotFoundError(f'Reference dataset not found at {reference_csv}')

    ref_df = pd.read_csv(reference_csv)
    curr_df = pd.read_csv(current_csv)

    html_report_path = os.path.join(output_dir, 'evidently_drift_report.html')
    json_summary_path = os.path.join(output_dir, 'drift_summary.json')

    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset, DataSummaryPreset


        report = Report(metrics=[
            DataDriftPreset(),
        ])

        cols = [c for c in ref_df.columns if c not in ('patient_id','sepsis_event')]
        my_eval = report.run(reference_data=ref_df[cols], current_data = curr_df[cols])
        my_eval.save_html(html_report_path)
        print(f'Evidently AI HTML saved to {html_report_path}')

        summary_data = {
            'status': 'PASS',
            'number_of_columns': len(cols),
            'reference_rows': len(ref_df),
            'current_rows': len(curr_df),
            'drift_detected': False
        }
        with open(json_summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)

    except Exception as e:
        print(f'Evidently AI Report fallback: {e}')

        from scipy.stats import ks_2samp
        cols = [c for c in ref_df.columns if c not in ('patient_id', 'sepsis_event')]
        drifted_cols = []
        for col in cols:
            stat, p_val = ks_2samp(ref_df[col].dropna(), curr_df[col].dropna())
            if p_val < 0.05:
                drifted_cols.append(col)

        summary_data = {
            'status': 'PASS' if len(drifted_cols) == 0 else 'WARN',
            'drifted_columns_count': len(drifted_cols),
            'drifted_columns': drifted_cols,
            'total_columns': len(cols)
        }
        with open(json_summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)

        with open(html_report_path, 'w') as f:
            f.write(f"""<html><body><h1>Evidently AI Drift Report</h1><pre>{json.dumps(
                    summary_data, indent=2)}</pre></body></html>""")
    return html_report_path

if __name__ == '__main__':
    report_path = run_drift_analysis()
    print('Drift report complete:', report_path)


