# VRMotion-Baselines
Official PyTorch implementation of the baseline models for the paper "VRMotion: A Large-Scale Dataset for Full-Body Motion Prediction in Ego-Vision VR Tasks".
# VRMotion-Baselines

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation of the baseline models for the paper: **"VRMotion: A Large-Scale Dataset for Full-Body Motion Prediction in Ego-Vision VR Tasks"**, accepted at ACM Multimedia 2026 (Rio de Janeiro, Brazil).

## 📖 Overview

This repository provides the reference implementations for the 10 visual-temporal baseline models evaluated in our paper. Our framework formulates full-body motion prediction as a cross-modal synthesis task, predicting future 3D skeletal trajectories based on immersive VR egocentric visual stimuli.

The baselines are constructed by pairing state-of-the-art Visual Encoders with different Temporal Modeling Heads:

* **Visual Encoders (5 variants):** * Large Vision-Language Models (LVLMs): **Qwen2.5-VL**, **OneVision** (with 8-bit quantization support)
  * Vision Transformers: **DINOv2**, **VideoMAE**
  * Convolutional Networks: **ResNet**
* **Temporal Heads (2 variants):** * **LSTM** (Implicit Sequence Decoder)
  * **ST-GCN** (Topology-aware Graph Network with a predefined 24-joint physical adjacency matrix)

## 📂 Repository Structure

* `data/` *(ignored in git)* - Directory for placing the downloaded VRMotion dataset.
* `common/` - Shared utilities, metrics computation (MPJPE, PA-MPJPE, Vel Error), and logging.
* `models/` - PyTorch implementations of the visual-temporal architectures.
* `scripts/` - Standardized training and evaluation pipelines.
* `plots/` - Visualization tools for rendering 3D skeletal motions and inter-joint trajectories.

## 🛠️ Installation

```bash
# Clone the repository
git clone [https://github.com/1530442592-hue/VRMotion-Baselines.git](https://github.com/1530442592-hue/VRMotion-Baselines.git)
cd VRMotion-Baselines

# Create a conda environment
conda create -n vrmotion python=3.10
conda activate vrmotion

# Install PyTorch and dependencies
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
# pip install -r requirements.txt
