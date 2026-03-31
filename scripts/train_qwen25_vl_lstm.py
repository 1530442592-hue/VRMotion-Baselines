import sys
import os
import time
import random
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from transformers import AutoProcessor
from PIL import Image

from common.dataset import BeatSaberDataset
from common.metrics import compute_mpjpe, compute_pa_mpjpe
from common.utils import get_device, get_logger
from models.qwen25_vl_lstm import Qwen35VLLSTM 

# ================= 核心：全方位锁定种子 42 =================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
set_seed(42)

# ================= 指标计算 =================
def calculate_velocity_error(pred, gt):
    pred_v = pred[:, 1:] - pred[:, :-1]
    gt_v = gt[:, 1:] - gt[:, :-1]
    return torch.mean(torch.abs(pred_v - gt_v)).item() * 1000.0

def calculate_fps(inf_time, n_samples):
    return round(n_samples / inf_time, 2) if inf_time > 0 else 0

# ================= 配置区 =================
DEVICE = get_device()
# 更新模型名字为 3B 版本
MODEL_NAME = "qwen25_vl_3b_8bit_lstm_100ep" 
EPOCHS = 100   
BATCH_SIZE = 1 
LR = 1e-4
TEST_RATIO = 0.2

save_dir = os.path.join("outputs", MODEL_NAME) 
os.makedirs(save_dir, exist_ok=True)
os.makedirs(os.path.join(save_dir, "logs"), exist_ok=True)
logger = get_logger(save_dir)

# ================= 预处理 =================
logger.info("⚡ 初始化 Qwen 2.5-VL-3B 官方 Processor...")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", trust_remote_code=True)
resize_transform = T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC)

def vlm_transform(frames_np):
    pil_frames = [resize_transform(Image.fromarray(f)) for f in frames_np]
    fake_texts = ["<|vision_start|><|image_pad|><|vision_end|>"] * len(pil_frames)
    inputs = processor(images=pil_frames, text=fake_texts, return_tensors="pt", padding=True)
    return {"pixel_values": inputs.pixel_values, "image_grid_thw": inputs.image_grid_thw}

# ================= 数据准备 =================
logger.info("📂 正在初始化 BeatSaber 数据集...")
dataset = BeatSaberDataset(data_root="data/beat_saber", transform=vlm_transform)
train_size = int(len(dataset) * (1 - TEST_RATIO))
train_dataset, test_dataset = random_split(dataset, [train_size, len(dataset)-train_size])

train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False)

# ================= 模型初始化 =================
logger.info(f"🧠 正在加载 {MODEL_NAME} ...")
model = Qwen35VLLSTM(seq_len=16, pred_len=16, num_joints=24).to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

# 清理过往断点逻辑，从头开始
checkpoint_path = os.path.join(save_dir, f"{MODEL_NAME}_best.pth")
best_test_mpjpe = float('inf')
start_epoch = 0

# ================= 训练循环 =================
logger.info(f"🚀 任务启动: {MODEL_NAME} (目标 100 轮)...")

for epoch in range(start_epoch, EPOCHS):
    model.train()
    train_loss = train_mpjpe = train_vel = 0.0
    start_time = time.time()

    for video, past, gt in train_loader:
        v_in = {k: v.to(DEVICE) for k, v in video.items()}
        past, gt = past.to(DEVICE), gt.to(DEVICE)
        
        optimizer.zero_grad()
        pred = model(v_in, past)
        loss = criterion(pred, gt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()
        train_mpjpe += compute_mpjpe(pred, gt) * 1000.0
        train_vel += calculate_velocity_error(pred, gt)

    # 验证
    model.eval()
    test_mpjpe = test_pmpjpe = 0.0
    with torch.no_grad():
        for video, past, gt in test_loader:
            v_in = {k: v.to(DEVICE) for k, v in video.items()}
            pred = model(v_in, past.to(DEVICE))
            test_mpjpe += compute_mpjpe(pred, gt.to(DEVICE)) * 1000.0
            test_pmpjpe += compute_pa_mpjpe(pred, gt.to(DEVICE)) * 1000.0

    cur_mpjpe = test_mpjpe / len(test_loader)
    logger.info(f"Epoch {epoch+1:03d} | Train Loss: {train_loss/len(train_loader):.4f} | Test MPJPE: {cur_mpjpe:.2f} mm")

    if cur_mpjpe < best_test_mpjpe:
        best_test_mpjpe = cur_mpjpe
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f"🌟 新纪录! 最佳模型已保存 ({best_test_mpjpe:.2f} mm)")

logger.info("=" * 100)
logger.info(f"🎉 {MODEL_NAME} 训练结束！最佳成绩: {best_test_mpjpe:.2f} mm")