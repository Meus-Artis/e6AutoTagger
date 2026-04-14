@echo off
if not exist venv python -m venv venv
venv\Scripts\python -m pip install --upgrade pip
set TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
AMDDetect.ps1
title JTP³ Hydra ROCm
venv\Scripts\pip install -r requirements.txt
cls
venv\Scripts\python inference.py --service