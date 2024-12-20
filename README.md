<!-- PROJECT SHIELDS -->
<!-- REF: https://github.com/othneildrew/Best-README-Template -->
[![Python][python-shield]][python-url]
<!-- TODO: gradio shield and url -->

<!-- GITHUB SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Apache-2.0 License][license-shield]][license-url]
<!-- TODO: github tag version shield and url https://shields.io/badges/git-hub-tag -->

<!-- SOCIAL SHIELDS -->
[![LinkedIn][linkedin-shield]][linkedin-url]
<!-- TODO: x.com shield and url https://shields.io/badges/x-formerly-twitter-url -->
<!-- TODO: instagram shield and url ? -->
<!-- TODO: discord server shield and url https://shields.io/badges/discord -->

<!-- TODO: add Depth+ banner -->

# Depth+
Depth+ is a tool for locally extracting depth, optical flow, and segmentation information from videos using the latest open-source AI models.

<!-- TODO: brag about how Depth+ has already been used to make production-quality work -->

## Dependencies
- [Python 3.12.7](https://www.python.org/downloads/release/python-3127/)
- [CUDA 12.11](https://developer.nvidia.com/cuda-12-1-1-download-archive)
- Git and LFS

## Installation
> [!WARNING]
> These setup instructions have only been tested on Windows using Powershell with Administrative privileges.

### Set up the Python Environment
* In a directory of your choosing, enter the following into your terminal:
  ```console
  mkdir nilor-corp
  cd nilor-corp
  python -m venv venv
  .\venv\Scripts\activate
  python.exe -m pip install --upgrade pip
  ```

### Clone Depth+ and Install Dependencies
* In the `.\nilor-corp\` directory:
  ```console
  git clone https://github.com/nilor-corp/depth-plus.git
  cd depth-plus
  python -m pip install -r requirements.txt
  ```

> [!WARNING]
> This step can potentially cause some pain to the user. Before pulling out your own hair in a later step, return to this step and check:
> - The compatibility of your torch/torchvision/torchaudio packages with each other and with your CUDA Toolkit installation version (we are targeting 12.1)
> - If you're coming back here from an `fbgemm.dll` error, check this [GitHub Issue comment](https://github.com/comfyanonymous/ComfyUI/issues/3703#issuecomment-2253349160) and perform either of the two stated fixes.

Congratulations, you are finished installing Depth+!

### Directory Structure
* If you have installed Depth+ correctly, your directory structure should look like this:
  ```
  nilor-corp
  ├── depth-plus
  └── venv
  ```


## Usage

### Run Depth+
* In the `.\nilor-corp\` directory:
  ```console
  .\venv\scripts\activate.ps1
  cd depth-plus
  gradio ./app.py
  ```
  
> [!NOTE]
> Gradio will likely throw some warnings at you that can be safely ignored.

> [!TIP]
> Generated outputs can be found in: `.\nilor-corp\depth-plus\output\` directory.

> [!WARNING]
> Follow the gradio UI instructions carefully, especially if specifying input directory manually.

> [!CAUTION]
> The DepthAnyVideo model can be quite resource intensive. In our testing with an RTX 4090, the model often runs at 100% GPU utilization, rendering the machine unusable during processing. To avoid `Out of Memory (OOM)` errors while using DepthAnyVideo, we recommend that you first test it on a 1 second video of a small resolution (eg. 512x512) and work upward from there. For any video that you find can't be processed by DepthAnyVideo, use DepthAnythingV2 instead.



<!-- MARKDOWN LINKS & IMAGES -->
<!-- REF: https://github.com/othneildrew/Best-README-Template -->
[contributors-shield]: https://img.shields.io/github/contributors/nilor-corp/depth-plus.svg?style=for-the-badge
[contributors-url]: https://github.com/nilor-corp/depth-plus/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/nilor-corp/depth-plus.svg?style=for-the-badge
[forks-url]: https://github.com/nilor-corp/depth-plus/network/members
[stars-shield]: https://img.shields.io/github/stars/nilor-corp/depth-plus.svg?style=for-the-badge
[stars-url]: https://github.com/nilor-corp/depth-plus/stargazers
[issues-shield]: https://img.shields.io/github/issues/nilor-corp/depth-plus.svg?style=for-the-badge
[issues-url]: https://github.com/nilor-corp/depth-plus/issues
[license-shield]: https://img.shields.io/github/license/nilor-corp/depth-plus.svg?style=for-the-badge
[license-url]: https://github.com/nilor-corp/depth-plus/blob/main/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/company/nilor-corp/
<!-- TODO: github tag version shield and url https://shields.io/badges/git-hub-tag -->
<!-- TODO: x.com shield and url https://shields.io/badges/x-formerly-twitter-url -->
<!-- TODO: discord server shield and url https://shields.io/badges/discord -->
[python-shield]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[python-url]: https://www.python.org/
<!-- TODO: gradio shield and url -->