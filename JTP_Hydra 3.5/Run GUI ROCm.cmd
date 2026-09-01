@echo off
title JTP³•⁵ Hydra GUI ROCm
set TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
if not exist venv python -m venv venv
venv\Scripts\python -m pip install --upgrade pip
if not exist venv\Lib\site-packages\torch AMDDetect.ps1
if not exist venv\Lib\site-packages\uvicorn venv\Scripts\pip install -r requirements.txt
cls
venv\Scripts\pythonw gui.pyw