import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VideoMAEModel

class VideoMAELSTM(nn.Module):
    def __init__(self, seq_len=16, pred_len=16, num_joints=24):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_joints = num_joints
        self.out_dim = num_joints * 3  # 72
        
        # 1. VideoMAE 视觉编码器
        self.videomae = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
        # 冻结 VideoMAE 权重以保证与 ResNet baseline 设定的公平性
        for param in self.videomae.parameters():
            param.requires_grad = False

        # 2. 🌟 核心修正一：空间适配器 (Spatial Adapter)
        # 替代原本灾难性的 mean(dim=2)。利用轻量级卷积保留局部空间拓扑，再提取显著特征。
        self.spatial_adapter = nn.Sequential(
            nn.Conv2d(in_channels=768, out_channels=512, kernel_size=3, padding=1, stride=2), # 14x14 -> 7x7
            nn.BatchNorm2d(512), # 缓解 Domain Gap
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, 1)), # 提取最强烈的姿态激活信号，而非平均模糊掉
            nn.Flatten() # 输出形状: 512
        )

        # 3. 视觉特征最终投影
        self.v_proj = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU() # 替换 ReLU 以获得更平滑的梯度
        )

        # 4. 动作路径编码
        self.motion_embed = nn.Sequential(
            nn.Linear(self.out_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )

        # 5. LSTM 层
        # 注意: 视觉(512) + 动作(256) = 768
        self.lstm = nn.LSTM(input_size=768, hidden_size=512, num_layers=2, batch_first=True)
        
        # 6. 输出层
        self.fc = nn.Linear(512, self.pred_len * self.out_dim)

    def forward(self, video_x, motion_x):
        """
        video_x: [B, 16, 3, 224, 224]
        motion_x: [B, 16, 24, 3]
        """
        B, T, C, H, W = video_x.shape
        
        # --- 提取视觉特征 ---
        with torch.no_grad():
            v_outputs = self.videomae(video_x)
            v_feat = v_outputs.last_hidden_state  # [B, 1568, 768]
        
        # 解析 VideoMAE 的 Tubelet 结构
        L = v_feat.shape[1]
        num_spatial_patches = 14 * 14  # 196 (14x14 的特征图)
        num_time_patches = L // num_spatial_patches  # 对于 16 帧，tubelet_size=2，得到 8
        
        # 🌟 核心修正一落地：空间维度的合理降采样
        # 重塑为 [B, Time, H, W, Channels] -> [B, 8, 14, 14, 768]
        v_feat = v_feat.reshape(B, num_time_patches, 14, 14, 768)
        # 转换维度以适应 Conv2d: [B*8, 768, 14, 14]
        v_feat_spatial = v_feat.permute(0, 1, 4, 2, 3).reshape(B * num_time_patches, 768, 14, 14)
        
        # 通过空间适配器提取特征: [B*8, 768, 14, 14] -> [B*8, 512]
        time_patch_features = self.spatial_adapter(v_feat_spatial)
        # 恢复时间维度: [B, 8, 512]
        time_patch_features = time_patch_features.reshape(B, num_time_patches, 512)
        
        # 🌟 核心修正二落地：时间维度的平滑对齐 (8 -> 16)
        # 转换维度以适应 1D 插值: [B, Channels, Time] -> [B, 512, 8]
        time_patch_features = time_patch_features.permute(0, 2, 1)
        # 使用线性插值平滑补全缺失帧，避免 repeat_interleave 造成的时序阶跃断层
        v_feat_per_frame = F.interpolate(
            time_patch_features, 
            size=T, 
            mode='linear', 
            align_corners=False
        )
        # 转换回序列格式: [B, 16, 512]
        v_feat_per_frame = v_feat_per_frame.permute(0, 2, 1)

        # 视觉特征归一化投影
        v_feat_proj = self.v_proj(v_feat_per_frame)  # [B, 16, 512]

        # --- 提取动作特征 ---
        m_feat = self.motion_embed(motion_x.reshape(B, T, -1))  # [B, 16, 256]

        # --- 多模态特征拼接 ---
        fused = torch.cat([v_feat_proj, m_feat], dim=-1)  # [B, 16, 768]

        # --- LSTM 时序处理 ---
        lstm_out, _ = self.lstm(fused)  # [B, 16, 512]
        
        # --- 预测未来 (取最后时间步) ---
        pred = self.fc(lstm_out[:, -1, :])  # [B, pred_len * 72]
        
        # 整理输出维度为 [B, pred_len, num_joints, 3]
        pred = pred.reshape(B, self.pred_len, self.num_joints, 3)
        return pred