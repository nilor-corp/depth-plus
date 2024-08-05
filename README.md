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
- cuda version 12.11
- ssh connection to nilor corp HF organization: https://huggingface.co/nilor-corp
- git lfs

## Installation

these commands have only been tested using adminsistrative powershell so far

### setup python environment

```
mkdir nilor-corp
cd nilor-corp
python -m venv venv
.\venv\Scripts\activate
```

### Install comfy-cli

```
pip install comfy-cli
```

### Install ComfyUI

```
comfy --workspace=ComfyUI install
comfy set-default ComfyUI
mkdir .\ComfyUI\output\WorkFlower
```

### Install Models

```py
cd ComfyUI
rm -rf models #if unix
rmdir models #if win. Answer yes if prompted
git clone git@hf.co:nilor-corp/depth-plus-models models
```

### Install Depth+

```
git clone git@hf.co:spaces/nilor-corp/zenerator
cd ..
------------------
cp .\zenerator\ComfyUI-Manager-Snapshots\2024-08-04_20-20-33_snapshot.json .\ComfyUI\custom_nodes\ComfyUI-Manager\snapshots\
-------------------
cd zenerator
python -m pip install -r requirements.txt
```

### Env Variables

- (in zenerator root, where you should be if you've been following the above commands) create a .env file and add:

```
NILOR_API_KEY=<API KEY>
NILOR_API_URI=https://api.nilor.cool/api
```

### Provision comfy UI

```py
cd .. # should be in nilor-corp dir root
comfy launch
```

- once launched, navigate to comfyUI in browser http://127.0.0.1:8188
- in the bottom right corner of the screen on the floating modal click the "manager" button
- a new window will appear, in the bottom left corner, under the "expiremental" section click "snapshot manager"
- Click "restore" on the snapshot and then press the "restart" button that will appear to restart comfyUI. This will download a lot packages which you should see in terminal

### Directory Structure

After finishing installation, your directory structure should look like this:

- nilor-corp
  - comfy-cli
  - ComfyUI
  - depth-plus
  - venv

## Run

You will need to run ComfyUI and Depth+ in seperate shells

### Run ComfyUI

From nilor-corp root:

```
.\venv\scripts\activate
comfy launch
```

- If you run into "import torch" error when trying to launch comfy for the first time, [see potential fix here](https://github.com/Comfy-Org/comfy-cli/issues/150)

### Run Depth+

From nilor-corp root:

```
.\venv\scripts\activate
cd depth-plus
gradio ./app.py
```

### Output

Generated output can be found in:

```
nilor-corp\ComfyUI\output\WorkFlower
```
