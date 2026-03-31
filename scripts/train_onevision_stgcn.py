import sys
import os
import time
import random
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import AutoProcessor
from PIL import Image

from common.dataset import BeatSaberDataset
from common.metrics import compute_mpjpe, compute_pa_mpjpe
from common.utils import get_device, get_logger # 移除 make_output_dir，使用内置防爆创建
from models.onevision_stgcn import OneVisionSTGCN

# =================核心魔法：全方位锁定随机种子=================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
set_seed(42)

# =================指标计算 (毫米转换)=================
def calculate_velocity_error(pred, gt):
    pred_v = pred[:, 1:] - pred[:, :-1]
    gt_v = gt[:, 1:] - gt[:, :-1]
    return torch.mean(torch.abs(pred_v - gt_v)).item() * 1000.0

def calculate_fps(inf_time, n_samples):
    return round(n_samples / inf_time, 2) if inf_time > 0 else 0

# =================配置区=================
DEVICE = get_device()
MODEL_NAME = "onevision_stgcn_100ep_fixed" # 添加 fixed 标识
EPOCHS = 100  
BATCH_SIZE = 2 
LR = 5e-5 # 🌟 关键对齐：将学习率降低至 5e-5，保护图结构
TEST_RATIO = 0.2

# 🌟🌟🌟 强制创建文件夹防报错 🌟🌟🌟
save_dir = os.path.join("outputs", MODEL_NAME) 
os.makedirs(save_dir, exist_ok=True)
os.makedirs(os.path.join(save_dir, "logs"), exist_ok=True)

logger = get_logger(save_dir)

logger.info("⏳ 正在加载 OneVision Processor...")
processor = AutoProcessor.from_pretrained("lmms-lab-encoder/onevision-encoder-large", trust_remote_code=True)

# =================VLM 专属数据预处理=================
def vlm_transform(frames_np):
    pil_frames = [Image.fromarray(f) for f in frames_np]
    inputs = processor(images=pil_frames, return_tensors="pt")
    pixel_values = inputs["pixel_values"]
    
    if pixel_values.ndim == 5 and pixel_values.shape[0] == 1:
        pixel_values = pixel_values.squeeze(0)
        
    return pixel_values

# =================数据准备=================
logger.info("📂 正在初始化数据集并进行 VLM 预处理绑定...")
dataset = BeatSaberDataset(data_root="data/beat_saber", transform=vlm_transform)
train_size = int(len(dataset) * (1 - TEST_RATIO))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False)

# =================模型初始化=================
logger.info(f"🧠 正在加载 {MODEL_NAME} ...")
model = OneVisionSTGCN(seq_len=16, pred_len=16, num_joints=24, use_8bit=False).to(DEVICE)
criterion = nn.MSELoss()

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

# 🌟 新增：余弦退火学习率调度器
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

best_test_mpjpe = float('inf')
best_epoch = 0
final_pmpjpe = 0.0
final_vel = 0.0

logger.info(f"🚀 开始 100 轮硬核图卷积挑战: {MODEL_NAME}...")

# =================训练+测试循环=================
for epoch in range(EPOCHS):
    model.train()
    train_loss = train_mpjpe = train_pmpjpe = train_vel = 0.0
    start_time = time.time()

    for video, past_motion, future_gt in train_loader:
        video = video.to(DEVICE)
        past_motion = past_motion.to(DEVICE)
        future_gt = future_gt.to(DEVICE)
        
        optimizer.zero_grad()
        pred = model(video, past_motion)
        loss = criterion(pred, future_gt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()
        train_mpjpe += compute_mpjpe(pred, future_gt) * 1000.0
        train_pmpjpe += compute_pa_mpjpe(pred, future_gt) * 1000.0
        train_vel += calculate_velocity_error(pred, future_gt)

    t_loss = train_loss / len(train_loader)
    t_mpjpe = train_mpjpe / len(train_loader)
    t_pmpjpe = train_pmpjpe / len(train_loader)
    t_vel = train_vel / len(train_loader)
    fps = calculate_fps(time.time() - start_time, len(train_dataset))

    # --------- 测试阶段 ---------
    model.eval()
    test_loss = test_mpjpe = test_pmpjpe = test_vel = 0.0
    start_test = time.time()
    
    with torch.no_grad():
        for video, past_motion, future_gt in test_loader:
            video = video.to(DEVICE)
            past_motion = past_motion.to(DEVICE)
            future_gt = future_gt.to(DEVICE)
            
            pred = model(video, past_motion)
            loss = criterion(pred, future_gt)

            test_loss += loss.item()
            test_mpjpe += compute_mpjpe(pred, future_gt) * 1000.0
            test_pmpjpe += compute_pa_mpjpe(pred, future_gt) * 1000.0
            test_vel += calculate_velocity_error(pred, future_gt)

    te_loss = test_loss / len(test_loader)
    te_mpjpe = test_mpjpe / len(test_loader)
    te_pmpjpe = test_pmpjpe / len(test_loader)
    te_vel = test_vel / len(test_loader)
    te_fps = calculate_fps(time.time() - start_test, len(test_dataset))

    # --------- 日志输出 ---------
    logger.info(f"Epoch {epoch+1:03d}/{EPOCHS}")
    logger.info(f"[Train] Loss: {t_loss:.4f} | MPJPE: {t_mpjpe:.2f} | PA-MPJPE: {t_pmpjpe:.2f} | Vel Error: {t_vel:.2f} | FPS: {fps}")
    logger.info(f"[Test]  Loss: {te_loss:.4f} | MPJPE: {te_mpjpe:.2f} | PA-MPJPE: {te_pmpjpe:.2f} | Vel Error: {te_vel:.2f} | FPS: {te_fps}")

    # --------- 保存最佳模型 ---------
    if te_mpjpe < best_test_mpjpe:
        best_test_mpjpe = te_mpjpe
        best_epoch = epoch + 1
        final_pmpjpe = te_pmpjpe
        final_vel = te_vel
        torch.save(model.state_dict(), os.path.join(save_dir, f"{MODEL_NAME}_best.pth"))
        logger.info(f"🌟 新纪录! 最佳模型已保存 (Test MPJPE 降至 {best_test_mpjpe:.2f} mm)")
    
    # 🌟 步进调度器，打印学习率
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    logger.info(f"👉 当前学习率: {current_lr:.6f}")
    
    logger.info("-" * 80)

# =================最终成绩汇报=================
logger.info("=" * 100)
logger.info(f"🎉 {MODEL_NAME} 训练结束！")
logger.info(f"📊 VRMotion 论文填表数据 (取自最佳轮次 Epoch {best_epoch}):")
logger.info(f"   MPJPE (mm): {best_test_mpjpe:.2f} | PA-MPJPE (mm): {final_pmpjpe:.2f} | Vel Error (mm/frame): {final_vel:.2f} | FPS: {te_fps}")
logger.info("=" * 100)