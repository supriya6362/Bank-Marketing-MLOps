BANK MARKETING DATASET - LAB 12 AND LAB 13
===========================================

This ZIP is prepared for your uploaded bank-additional-full.csv dataset.
The source CSV is semicolon-separated and the target column is 'y':
yes = customer subscribed to a term deposit; no = did not subscribe.

IMPORTANT
- Extract this ZIP first.
- Open the extracted folder in VS Code.
- Use Python 3.10 or newer.
- Internet is needed only for installing the Python packages.
- Do not rename or move the CSV from the data folder.

STEP 1 - Open terminal in VS Code
Terminal > New Terminal

STEP 2 - Create and activate virtual environment (recommended)
Windows PowerShell:
    py -m venv .venv
    .\.venv\Scripts\Activate.ps1

If activation is blocked:
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    .\.venv\Scripts\Activate.ps1

STEP 3 - Install libraries
    python -m pip install --upgrade pip
    pip install -r requirements.txt

STEP 4 - Execute LAB 12
    python 12_monitoring_and_retraining.py

Lab 12 performs:
1. Loads your Bank Marketing CSV.
2. Converts y (yes/no) into 1/0.
3. Trains a baseline Random Forest model.
4. Evaluates accuracy, precision, recall, F1 and ROC-AUC.
5. Simulates a new incoming feature batch.
6. Checks data drift using PSI for numeric columns and total variation
   distance for categorical columns.
7. Retrains a V2 model if drift is detected.

STEP 5 - Execute LAB 13
    python 13_continuous_prediction_monitoring.py

Lab 13 performs:
1. Loads the model created in Lab 12.
2. Reads the dataset in batches of 1,000 records.
3. Generates predictions and probabilities.
4. Calculates batch-level accuracy, precision, recall, F1 and ROC-AUC.
5. Saves a prediction monitoring log and summary.

OUTPUT FILES
models/bank_marketing_model_v1.joblib
models/bank_marketing_model_v2.joblib (created if drift is detected)
outputs/lab12_drift_report.csv
outputs/lab12_monitoring_report.json
outputs/lab13_prediction_log.csv
outputs/lab13_monitoring_summary.json

TROUBLESHOOTING
- 'python is not recognized': install Python and enable 'Add Python to PATH',
  then reopen VS Code.
- ModuleNotFoundError: activate the virtual environment and run
  pip install -r requirements.txt again.
- FileNotFoundError for CSV: ensure data/bank-additional-full.csv exists.
- Run Lab 12 before Lab 13.

ACADEMIC NOTE
The drifted batch is simulated to demonstrate monitoring. Lab 13 processes
the provided labelled CSV in batches; it is a local demonstration, not a
deployed real-time API. The original uploaded dataset is preserved unchanged.
