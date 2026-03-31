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
from PIL import Image

from common.dataset import BeatSaberDataset
from common.metrics import compute_mpjpe, compute_pa_mpjpe
from common.utils import get_device, make_output_dir, get_logger
from models.resnet_lstm import ResNetLSTM

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

# =================指标计算=================
def calculate_velocity_error(pred, gt):
    pred_v = pred[:, 1:] - pred[:, :-1]
    gt_v = gt[:, 1:] - gt[:, :-1]
    return torch.mean(torch.abs(pred_v - gt_v)).item() * 1000.0

def calculate_fps(inf_time, n_samples):
    return round(n_samples / inf_time, 2) if inf_time > 0 else 0

# =================配置区=================
DEVICE = get_device()
MODEL_NAME = "resnet_lstm_100ep"
EPOCHS = 100  
BATCH_SIZE = 32 # ResNet 较小，Batch 可以开大
LR = 1e-4
TEST_RATIO = 0.2

save_dir = make_output_dir(MODEL_NAME)
logger = get_logger(save_dir)

# =================标准预处理 (ImageNet)=================
logger.info("⚡ 使用标准 ResNet 视觉预处理管道...")
vlm_transform_pipeline = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def vlm_transform(frames_np):
    pil_frames = [Image.fromarray(f) for f in frames_np]
    pixel_values = torch.stack([vlm_transform_pipeline(img) for img in pil_frames])
    return pixel_values

# =================数据准备=================
logger.info("📂 正在初始化数据集...")
dataset = BeatSaberDataset(data_root="data/beat_saber", transform=vlm_transform)
train_size = int(len(dataset) * (1 - TEST_RATIO))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False)

# =================模型初始化=================
logger.info("🧠 正在加载 ResNet-50 + LSTM 经典模型架构...")
model = ResNetLSTM(seq_len=16, pred_len=16, num_joints=24).to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

best_test_mpjpe = float('inf')
best_epoch = 0
final_pmpjpe = 0.0
final_vel = 0.0

logger.info(f"🚀 开始 100 轮极限挑战: {MODEL_NAME}...")

for epoch in range(EPOCHS):
    # --------- 训练阶段 ---------
    model.train()
    train_loss = train_mpjpe = train_pmpjpe = train_vel = 0.0
    start_time = time.time()

    for video, past_motion, future_gt in train_loader:
        video, past_motion, future_gt = video.to(DEVICE), past_motion.to(DEVICE), future_gt.to(DEVICE)
        
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
            video, past_motion, future_gt = video.to(DEVICE), past_motion.to(DEVICE), future_gt.to(DEVICE)
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

    # --------- 日志输出 (强迫症对齐版) ---------
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
    
    logger.info("-" * 80)

# =================最终成绩汇报=================
logger.info("=" * 100)
logger.info(f"🎉 {MODEL_NAME} 训练结束！")
logger.info(f"📊 VRMotion 论文专用 ({MODEL_NAME} Epoch {best_epoch}):")
logger.info(f"   MPJPE (mm): {best_test_mpjpe:.2f} | PA-MPJPE (mm): {final_pmpjpe:.2f} | Vel Error (mm/frame): {final_vel:.2f} | FPS: {te_fps}")
logger.info("=" * 100)