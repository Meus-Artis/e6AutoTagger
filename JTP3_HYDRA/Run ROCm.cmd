@echo off
set TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
title JTP³ Hydra ROCm
if not exist venv python -m venv venv
venv\Scripts\python -m pip install --upgrade pip
venv\Scripts\pip install --index-url https://rocm.nightlies.amd.com/v2/gfx110X-all/ --pre torch torchvision
venv\Scripts\pip install -r requirements.txt
venv\Scripts\python inference.py --service