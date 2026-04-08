# VRMotion-Baselines

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation of the baseline models for the paper: **"VRMotion: A Large-Scale Dataset for Full-Body Motion Prediction in Ego-Vision VR Tasks"**.

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

Dataset Preparation
Download the VRMotion dataset from [Link to Dataset - TBA].

Extract the contents and organize them into the data/ directory using the 16-frame historical to 16-frame future sliding window format as follows:
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
# pip install -r requirements.txt
VRMotion-Baselines/
├── data/
│   ├── train/
│   ├── val/
│   └── test/

## 🛠️ Installation
Training
To train a model from scratch, run the training script and specify the visual encoder and temporal head. For example, to train the best-performing Qwen2.5-VL + LSTM model:
python scripts/train.py --encoder qwen2.5-vl --head lstm --batch_size 16
To train the ResNet + ST-GCN baseline (utilizing a 5e-5 learning rate and Cosine Annealing):
python scripts/train.py --encoder resnet --head st-gcn --lr 5e-5
Evaluation
To evaluate a trained model and compute the core metrics (MPJPE, PA-MPJPE, and Velocity Error):
python scripts/evaluate.py --checkpoint path/to/your/checkpoint.pth
Visualization
To render predicted 3D skeletal trajectories or generate inter-joint velocity correlation matrices:
python plots/visualize_skeleton.py --prediction path/to/output.npy

Citation
If you find our dataset or baselines useful in your research, please consider citing our paper:
@inproceedings{zhang2026vrmotion,
  title={VRMotion: A Large-Scale Dataset for Full-Body Motion Prediction in Ego-Vision VR Tasks},
  author={Zhang, Dayou and Song, Yi and Lin, Shufang and Cao, Zijian and Zhang, Rongrong and Wang, Fangxin},
  booktitle={Submitted to ACM Multimedia},
  year={2026},
  address={Rio de Janeiro, Brazil}
}
