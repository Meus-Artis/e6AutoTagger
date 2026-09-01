@echo off
title JTP³ Hydra
if not exist venv python -m venv venv
venv\Scripts\python -m pip install --upgrade pip
if not exist venv\Lib\site-packages\torch venv\Scripts\pip install -r requirements.txt
cls
venv\Scripts\python inference.py --service