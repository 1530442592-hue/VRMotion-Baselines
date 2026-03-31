import os
import torch
import matplotlib.pyplot as plt
import logging
from datetime import datetime

# 1. 自动配置设备（GPU/CPU）
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 创建输出文件夹（自动分模型保存）
def make_output_dir(model_name):
    save_dir = f"./outputs/{model_name}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{save_dir}/logs", exist_ok=True)
    os.makedirs(f"{save_dir}/plots", exist_ok=True)
    return save_dir

# 3. 保存模型权重
def save_model(model, save_dir, epoch):
    path = os.path.join(save_dir, "checkpoints", f"epoch_{epoch}.pth")
    torch.save(model.state_dict(), path)
    print(f"✅ 模型已保存: {path}")

# 4. 绘制损失/指标曲线
def plot_curve(train_loss, val_loss, save_dir):
    plt.figure(figsize=(10, 4))
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig(os.path.join(save_dir, "plots/loss.png"))
    plt.close()

# 5. 日志记录
def get_logger(save_dir):
    logger = logging.getLogger("BeatSaber")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    
    file_handler = logging.FileHandler(os.path.join(save_dir, "logs/train.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger