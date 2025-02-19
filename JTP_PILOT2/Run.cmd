@echo off
title JTP Pilot²
if not exist venv python -m venv venv
venv\Scripts\python -m pip install --upgrade pip
venv\Scripts\pip install -r requirements.txt
venv\Scripts\python tagger_gui.py