@echo off
title JTP³ Hydra
if not exist venv python -m venv venv
venv\Scripts\python -m pip install --upgrade pip
venv\Scripts\pip install -r requirements.txt
venv\Scripts\python inference.py --service