import sys  
import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import torchvision.transforms as T
from transformers import AutoProcessor

# 导入项目模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.qwen25_vl_lstm import Qwen35VLLSTM
from common.dataset import BeatSaberDataset
from common.utils import get_device

# ================= 配置区 =================
DEVICE = get_device()
MODEL_PATH = "outputs/qwen25_vl_7b_8bit_lstm_100ep/qwen25_vl_7b_8bit_lstm_100ep_best.pth"
SAVE_DIR = "plots/saber_visuals_pro" 
os.makedirs(SAVE_DIR, exist_ok=True)

# 关节名称按字典序排列后的索引映射
SKELETON_EDGES = [
    (2, 21), (21, 22), (22, 23), (23, 11), (11, 12), (12, 0), (0, 1),
    (23, 19), (19, 13), (13, 16), (16, 17),
    (23, 9), (9, 3), (3, 6), (6, 7),
    (2, 20), (20, 18), (18, 14), (14, 15),
    (2, 10), (10, 8), (8, 4), (4, 5)
]

def draw_skeleton(ax, pose):
    """
    绘制 3D 骨架，针对论文排版进行无死角放大与留白裁切
    """
    x = pose[:, 0]
    y = pose[:, 2]
    z = pose[:, 1]
    
    ax.scatter(x, y, z, color='red', s=40, zorder=5) 
    for i, j in SKELETON_EDGES:
        ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], color='blue', linewidth=2.5, zorder=4)

    ax.set_xlim([-1.0, 1.0])
    ax.set_ylim([-1.0, 1.0])
    ax.set_zlim([-1.0, 1.0])
    
    ticks = [-1.0, -0.5, 0.0, 0.5, 1.0]
    labels = ['-1.0', '', '0.0', '', '1.0']
    
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.set_zticks(ticks)
    ax.set_zticklabels(labels)
    
    # 刻度数字字号保持 20
    ax.tick_params(axis='x', labelsize=20, pad=5)
    ax.tick_params(axis='y', labelsize=20, pad=5)
    ax.tick_params(axis='z', labelsize=20, pad=5)

    # 坐标轴名称字号保持 22
    ax.set_xlabel('Right (X)', fontsize=22, labelpad=15, fontweight='normal')
    ax.set_ylabel('Forward (Z)', fontsize=22, labelpad=15, fontweight='normal')
    ax.set_zlabel('Up (Y)', fontsize=22, labelpad=15, fontweight='normal')

    ax.grid(True)
    ax.view_init(elev=20, azim=45)
    
    try:
        ax.set_box_aspect(None, zoom=1.25) 
    except AttributeError:
        pass 

# ================= 加载模型 =================
print("🧠 正在加载真正的 Qwen2.5-VL 模型权重...")
model = Qwen35VLLSTM(seq_len=16, pred_len=16, num_joints=24).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True)
resize_transform = T.Resize((224, 224))

def saber_transform(frames_np):
    pil_frames = [resize_transform(Image.fromarray(f)) for f in frames_np]
    fake_texts = ["<|vision_start|><|image_pad|><|vision_end|>"] * len(pil_frames)
    inputs = processor(images=pil_frames, text=fake_texts, return_tensors="pt", padding=True)
    return {"pixel_values": inputs.pixel_values, "image_grid_thw": inputs.image_grid_thw}, frames_np[-1]

# ================= 运行可视化 =================
dataset = BeatSaberDataset(data_root="data/beat_saber", transform=saber_transform)
samples_to_vis = [5, 18, 25, 35]

fig = plt.figure(figsize=(26, 14), dpi=200) 

for col, idx in enumerate(samples_to_vis):
    if idx >= len(dataset): continue
    
    (video_inputs, last_img), past_motion, future_gt = dataset[idx]
    
    with torch.no_grad():
        v_in = {k: v.unsqueeze(0).to(DEVICE) for k, v in video_inputs.items()}
        p_in = past_motion.unsqueeze(0).to(DEVICE)
        pred_pose = model(v_in, p_in).cpu().squeeze(0)

    target_f = 11
    gt_p = future_gt[target_f].numpy()
    pr_p = pred_pose[target_f].numpy()

    error_m = np.mean(np.linalg.norm(gt_p - pr_p, axis=1))
    error_mm = error_m * 1000.0

    # 第一行：Visual Input
    ax1 = fig.add_subplot(3, 4, col + 1)
    h = last_img.shape[0]
    last_img_cropped = last_img[int(h * 0.12):]
    ax1.imshow(last_img_cropped, extent=[-0.12, 1.12, -0.12, 1.12]) 
    ax1.axis('off')

    # 第二行
    ax2 = fig.add_subplot(3, 4, col + 5, projection='3d')
    draw_skeleton(ax2, pr_p)

    # 第三行
    ax3 = fig.add_subplot(3, 4, col + 9, projection='3d')
    draw_skeleton(ax3, gt_p)
    ax3.set_title(f"Error: {error_mm:.1f} mm", y=-0.80, fontsize=24, color='black', pad=15)

# 调整标题位置与内容
fig.text(0.065, 0.86, 'Visual Input', va='center', ha='center', rotation='vertical',
         fontsize=26, fontweight='bold')
fig.text(0.065, 0.60, 'Predicted', va='center', ha='center', rotation='vertical',
         fontsize=26, fontweight='bold')
fig.text(0.065, 0.34, 'Ground Truth', va='center', ha='center', rotation='vertical',
         fontsize=26, fontweight='bold')

# 核心布局优化
plt.subplots_adjust(
    left=0.04, right=0.98, top=0.96, bottom=0.25,
    wspace=-0.35, 
    hspace=0.35
)

save_path = os.path.join(SAVE_DIR, "final_paper_figure_3x4.png")
plt.savefig(save_path, bbox_inches='tight', dpi=200, pad_inches=0.1) 
print(f"✅ 完美的 3x4 论文对比图已保存至 {save_path}")

plt.close(fig)