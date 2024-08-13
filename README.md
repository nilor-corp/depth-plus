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

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Installing Depth+

## Dependencies:

- python
- cuda version 12.1
- ssh connection to nilor corp HF organization: https://huggingface.co/nilor-corp
- git lfs

## Installation

these commands have only been tested using adminsistrative powershell so far

### setup python environment

```
mkdir nilor-corp
cd nilor-corp
python -m venv venv
.\venv\Scripts\activate.ps1
```

### Install Depth+

```
git clone git@hf.co:spaces/nilor-corp/depth-plus
cd depth-plus
python -m pip install -r requirements.txt
```

### Install torch

`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`

- this step is liable to cause some pain to the user. before pulling out your own hair in a later step, return to this step and check:
  - the compatibility of your torch/torchvision/torchaudio packages with each other and with your CUDA Toolkit installation version (we are targeting 12.1)
  - if you're coming back here from an `fbgemm.dll` error, check this link and perform the manual copy and rename of `libiomp5md.dll` as stated: https://github.com/comfyanonymous/ComfyUI/issues/3703#issuecomment-2253349160


### Directory Structure

After finishing installation, your directory structure should look like this:

- nilor-corp
  - depth-plus
  - venv


### Run Depth+

From nilor-corp root:

```
.\venv\scripts\activate.ps1
cd depth-plus
gradio ./app.py
```
gradio will likely throw some warnings at you that can be ignored

Follow the gradio UI instructions carefully, especially about the input dir if specifying input dir manually

