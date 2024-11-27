---
title: Depth+
emoji: ➕
colorFrom: green
colorTo: purple
sdk: gradio
sdk_version: 4.20.0
app_file: app.py
pinned: false
license: apache-2.0
---

# Depth+

## Dependencies
- Python
- Cuda version 12.1
- SSH connection to Nilor Corp HuggingFace organization: https://huggingface.co/nilor-corp
- Git and LFS

## Installation
**Disclaimer:** These commands have only been tested using Powershell with **Administrative** privileges.

### Setup python environment
```
mkdir nilor-corp
cd nilor-corp
python -m venv venv
.\venv\Scripts\activate.ps1
```

### Install Depth+
In the `.\nilor-corp\` dir:
```
git clone git@hf.co:spaces/nilor-corp/depth-plus
```

### Install torch dependencies
In the `.\nilor-corp\` dir:
```
pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu121
```

- this step is liable to cause some pain to the user. before pulling out your own hair in a later step, return to this step and check:
  - the compatibility of your torch/torchvision/torchaudio packages with each other and with your CUDA Toolkit installation version (we are targeting 12.1)
  - if you're coming back here from an `fbgemm.dll` error, check this link and perform the manual copy and rename of `libiomp5md.dll` as stated: https://github.com/comfyanonymous/ComfyUI/issues/3703#issuecomment-2253349160

### Install other dependencies
In the `.\nilor-corp\` dir:
```
cd depth-plus
python -m pip install -r requirements.txt
```

### Directory Structure
After finishing installation, your directory structure should look like this:
- nilor-corp
  - depth-plus
  - venv

## Usage

### Run Zenerator
In the `.\nilor-corp\` dir:
```
.\venv\scripts\activate.ps1
cd depth-plus
gradio ./app.py
```

Gradio will likely throw some warnings at you that can be safely ignored.

Follow the Gradio UI instructions carefully, especially about the input dir if specifying input dir manually.

### Output
Generated output can be found in: `.\nilor-corp\depth-plus\output` dir.

### DepthAnyVideo Disclaimer
The DepthAnyVideo model can be quite resource intensive. In our testing with a RTX 4090, the model often runs at 100% GPU utilization, rendering the machine unusable during processing.

To avoid `Out of Memory (OOM)` errors while using DepthAnyVideo, we recommend that you first test it on a 1 second video of a small resolution (eg. 512x512) and work upward from there. 

For any video that you find can't be processed by DepthAnyVideo, use DepthAnythingV2 instead.
