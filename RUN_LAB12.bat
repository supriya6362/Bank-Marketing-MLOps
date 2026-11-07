@echo off
cd /d "%~dp0"
python -m pip install -r requirements.txt
python 12_monitoring_and_retraining.py
pause
