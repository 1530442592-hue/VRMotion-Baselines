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
from models.qwen25_vl_stgcn import Qwen25VLSTGCN 

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
MODEL_NAME = "qwen25_vl_7b_8bit_stgcn_100ep_fixed" 
EPOCHS = 100  
BATCH_SIZE = 1 
# 🌟 关键修改：降低基础学习率，保护脆弱的动态图矩阵
LR = 5e-5 
TEST_RATIO = 0.2

save_dir = os.path.join("outputs", MODEL_NAME) 
os.makedirs(save_dir, exist_ok=True)
os.makedirs(os.path.join(save_dir, "logs"), exist_ok=True)

logger = get_logger(save_dir)

# ================= Qwen 2.5 官方专属预处理 =================
logger.info("⚡ 初始化 Qwen 2.5-VL-7B 官方 Processor...")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True)
resize_transform = T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC)

def vlm_transform(frames_np):
    pil_frames = [resize_transform(Image.fromarray(f)) for f in frames_np]
    fake_texts = ["<|vision_start|><|image_pad|><|vision_end|>"] * len(pil_frames)
    inputs = processor(images=pil_frames, text=fake_texts, return_tensors="pt", padding=True)
    return {"pixel_values": inputs.pixel_values, "image_grid_thw": inputs.image_grid_thw}

# ================= 数据准备 =================
logger.info("📂 正在初始化数据集...")
dataset = BeatSaberDataset(data_root="data/beat_saber", transform=vlm_transform)
train_size = int(len(dataset) * (1 - TEST_RATIO))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False)

# ================= 模型与优化器初始化 =================
logger.info(f"🧠 正在加载 {MODEL_NAME} ...")
model = Qwen25VLSTGCN(seq_len=16, pred_len=16, num_joints=24).to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

# 🌟 新增：余弦退火学习率调度器，平滑收敛
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

# ================= 断点续传逻辑 =================
checkpoint_path = os.path.join(save_dir, f"{MODEL_NAME}_best.pth")
start_epoch = 0
best_test_mpjpe = float('inf')

if os.path.exists(checkpoint_path):
    logger.info(f"🔄 发现历史权重 {checkpoint_path}，正在尝试断点续传...")
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        logger.info(f"✅ 权重加载成功！")
    except Exception as e:
        logger.warning(f"❌ 加载失败，将重新开始。错误: {e}")

final_pmpjpe = 0.0
final_vel = 0.0

# ================= 训练循环 =================
logger.info(f"🚀 任务启动: {MODEL_NAME} (目标 100 轮)...")

for epoch in range(start_epoch, EPOCHS):
    model.train()
    train_loss = train_mpjpe = train_pmpjpe = train_vel = 0.0
    start_time = time.time()

    for video, past_motion, future_gt in train_loader:
        video_inputs = {k: v.to(DEVICE) for k, v in video.items()}
        past_motion, future_gt = past_motion.to(DEVICE), future_gt.to(DEVICE)
        
        optimizer.zero_grad()
        pred = model(video_inputs, past_motion)
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
    
    # --- 测试评估 ---
    model.eval()
    test_loss = test_mpjpe = test_pmpjpe = test_vel = 0.0
    start_test = time.time()
    with torch.no_grad():
        for video, past_motion, future_gt in test_loader:
            video_inputs = {k: v.to(DEVICE) for k, v in video.items()}
            past_motion, future_gt = past_motion.to(DEVICE), future_gt.to(DEVICE)
            pred = model(video_inputs, past_motion)
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

    logger.info(f"Epoch {epoch+1:03d}/{EPOCHS}")
    logger.info(f"[Train] Loss: {t_loss:.4f} | MPJPE: {t_mpjpe:.2f} | PA-MPJPE: {t_pmpjpe:.2f} | Vel Error: {t_vel:.2f} | FPS: {fps}")
    logger.info(f"[Test]  Loss: {te_loss:.4f} | MPJPE: {te_mpjpe:.2f} | PA-MPJPE: {te_pmpjpe:.2f} | Vel Error: {te_vel:.2f} | FPS: {te_fps}")

    # --- 权重保存 ---
    if te_mpjpe < best_test_mpjpe:
        best_test_mpjpe = te_mpjpe
        best_epoch = epoch + 1
        final_pmpjpe = te_pmpjpe
        final_vel = te_vel
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f"🌟 新纪录! 最佳模型已保存 (Test MPJPE: {best_test_mpjpe:.2f} mm)")
    
    # 🌟 新增：更新调度器并记录当前学习率
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    logger.info(f"👉 当前学习率: {current_lr:.6f}")
    
    logger.info("-" * 80)

logger.info("=" * 100)
logger.info(f"🎉 {MODEL_NAME} 训练结束！")
logger.info(f"📊 最佳轮次 Epoch {best_epoch} 成绩:")
logger.info(f"   MPJPE (mm): {best_test_mpjpe:.2f} | PA-MPJPE (mm): {final_pmpjpe:.2f} | Vel Error: {final_vel:.2f} | FPS: {te_fps}")
logger.info("=" * 100)