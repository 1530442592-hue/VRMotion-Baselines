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

* `common/` - Shared utilities, metrics computation (MPJPE, PA-MPJPE, Vel Error), and dataset loading.
* `models/` - PyTorch implementations of the 10 visual-temporal architectures (e.g., `qwen25_vl_stgcn.py`).
* `plots/saber_visuals_pro/` - Visualization tools for rendering predicted 3D skeletal motions.
* `scripts/` - Individual, standardized training pipelines for each specific model combination (e.g., `train_qwen25_vl_stgcn.py`).

## 🛠️ Installation

```bash
# Clone the repository
git clone [https://github.com/1530442592-hue/VRMotion-Baselines.git](https://github.com/1530442592-hue/VRMotion-Baselines.git)
cd VRMotion-Baselines
```

# Create a conda environment
```bash
conda create -n vrmotion python=3.10
conda activate vrmotion
```

# Install PyTorch and dependencies
```bash
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install -r requirements.txt
```

📊 Dataset Preparation
Download the VRMotion dataset from https://huggingface.co/datasets/strfysy/VRMotion.
Organize the extracted contents into a data/ directory (ignored in git) using the native structure:
VRMotion-Baselines/
├── data/
│   ├── beat_saber/
│   │   ├── motion_3d/      # Reconstructed 3D skeletal joint positions in JSON format
│   │   ├── raw_data/       # Raw motion capture data in FBX and CSV format
│   │   └── video_frames/   # Synchronized RGB video recordings

🚀 Usage
Training
To train a model from scratch, run the specific training script for your desired combination. For example, to train the best-performing Qwen2.5-VL + ST-GCN model:
```bash
python scripts/train_qwen25_vl_stgcn.py
```
To train the ResNet + ST-GCN baseline:
```bash
python scripts/train_res_stgcn.py
```

Evaluation & Visualization
We provide dedicated scripts for evaluation and visualization. To render predicted trajectories for the Qwen2.5-VL model:
```bash
python scripts/visualize_qwen25_vl_lstm.py
```
📝 Citation
If you find our dataset or baselines useful in your research, please consider citing our paper:

@inproceedings{zhang2026vrmotion,
  title={VRMotion: A Large-Scale Dataset for Full-Body Motion Prediction in Ego-Vision VR Tasks},
  author={Zhang, Dayou and Song, Yi and Lin, Shufang and Cao, Zijian and Zhang, Rongrong and Wang, Fangxin},
  booktitle={Submitted to ACM Multimedia},
  year={2026},
  address={Rio de Janeiro, Brazil}
}
