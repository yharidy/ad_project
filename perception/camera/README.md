[![Run in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yharidy/ad_project/blob/main/camera_perception/colab_camera_perception.ipynb)
# Camera Perception Module

This module runs inference using BEVFormer on monocular camera images to generate a Bird's-Eye View (BEV) perception output.

## Features
- Runs BEVFormer inference using MMDetection3D
- Loads pretrained model and sample image
- Outputs detection visualizations
- Designed to plug into a full autonomous driving stack

## Requirements
- Python 3.8+
- CUDA-enabled environment (for real inference)
- See `requirements.txt` for full list

## Run Inference
```bash
python infer.py --image_path ./samples/sample.jpg