import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VideoMAEModel

# ================= 🌟 终极修复：带准确物理拓扑先验的 STGCN 核心算子 🌟 =================
class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_joints=24, temporal_kernel=3):
        super().__init__()
        
        # 1. 植入 VRMotion (Beat Saber) 数据集的真实 SMPL 骨骼物理连接
        edges = [
            (0, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), # 脊柱与头部
            (11, 16), (16, 17), (17, 18), (18, 19), # 右臂
            (11, 20), (20, 21), (21, 22), (22, 23), # 左臂
            (0, 1), (1, 2), (2, 3), (3, 4), # 右腿
            (0, 5), (5, 6), (6, 7), (7, 8) # 左腿
        ]
        
        # 构建静态物理邻接矩阵 (A_physical)
        A_physical = torch.zeros(num_joints, num_joints)
        for i, j in edges:
            if i < num_joints and j < num_joints: 
                A_physical[i, j] = 1.0
                A_physical[j, i] = 1.0 
        
        for i in range(num_joints):
            A_physical[i, i] = 1.0
            
        D = torch.sum(A_physical, dim=1)
        D_inv_sqrt = torch.pow(D, -0.5)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0 
        D_mat = torch.diag(D_inv_sqrt)
        A_physical = D_mat @ A_physical @ D_mat
        
        self.register_buffer('A_physical', A_physical)
        
        # 2. 极弱的自适应矩阵
        self.A_adaptive = nn.Parameter(torch.zeros(num_joints, num_joints) + 1e-5)
        
        # ================= 时空卷积层 =================
        self.spatial_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.temporal_conv = nn.Conv2d(
            out_channels, out_channels, 
            kernel_size=(temporal_kernel, 1), 
            padding=(temporal_kernel // 2, 0)
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x 形状: [B, C, T, V]
        A_fused = self.A_physical + self.A_adaptive
        x = torch.einsum('bctv,vw->bctw', x, A_fused)
        x = self.spatial_conv(x)
        x = self.temporal_conv(x)
        x = self.bn(x)
        return self.relu(x)

# ================= 重写版 VideoMAE + STGCN =================
class VideoMAESTGCN(nn.Module):
    def __init__(self, seq_len=16, pred_len=16, num_joints=24):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_joints = num_joints
        
        # 1. VideoMAE 视觉编码器
        self.videomae = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
        # 冻结 VideoMAE 权重
        for param in self.videomae.parameters():
            param.requires_grad = False

        # 2. 空间适配器 (保留了你 LSTM 版本里优秀的时空处理逻辑)
        self.spatial_adapter = nn.Sequential(
            nn.Conv2d(in_channels=768, out_channels=512, kernel_size=3, padding=1, stride=2), # 14x14 -> 7x7
            nn.BatchNorm2d(512), 
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, 1)), 
            nn.Flatten() 
        )

        # 3. 视觉特征最终投影
        self.v_proj = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU() 
        )

        # 4. 🌟 节点级动作特征独立降维 (3 -> 64)
        self.motion_embed = nn.Sequential(
            nn.Linear(3, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )

        # 🌟🌟🌟 核心防爆盾：强制融合后特征归一化 (512视觉 + 64动作 = 576) 🌟🌟🌟
        self.fused_ln = nn.LayerNorm(576)

        # 5. STGCN 前的通道压缩
        self.pre_stgcn_conv = nn.Sequential(
            nn.Conv2d(576, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # 6. 真正的 STGCN 层构建
        self.stgcn_blocks = nn.Sequential(
            STGCNBlock(128, 128, num_joints=self.num_joints),
            STGCNBlock(128, 256, num_joints=self.num_joints)
        )
        
        # 7. 节点级预测头
        self.fc = nn.Linear(256, self.pred_len * 3)

    def forward(self, video_x, motion_x):
        """
        video_x: [B, 16, 3, 224, 224]
        motion_x: [B, 16, 24, 3]
        """
        B, T, C, H, W = video_x.shape
        
        # --- 1. 提取视觉特征 ---
        with torch.no_grad():
            v_outputs = self.videomae(video_x)
            v_feat = v_outputs.last_hidden_state  # [B, 1568, 768]
        
        # 解析 VideoMAE 的 Tubelet 结构
        L = v_feat.shape[1]
        num_spatial_patches = 14 * 14  # 196
        num_time_patches = L // num_spatial_patches  # 通常为 8
        
        # 重塑并提取空间特征
        v_feat = v_feat.reshape(B, num_time_patches, 14, 14, 768)
        v_feat_spatial = v_feat.permute(0, 1, 4, 2, 3).reshape(B * num_time_patches, 768, 14, 14)
        
        time_patch_features = self.spatial_adapter(v_feat_spatial) # [B*8, 512]
        time_patch_features = time_patch_features.reshape(B, num_time_patches, 512)
        
        # 时间维度平滑插值 (8 -> 16 帧)
        time_patch_features = time_patch_features.permute(0, 2, 1)
        v_feat_per_frame = F.interpolate(
            time_patch_features, 
            size=T, 
            mode='linear', 
            align_corners=False
        )
        v_feat_per_frame = v_feat_per_frame.permute(0, 2, 1) # [B, 16, 512]
        v_feat_proj = self.v_proj(v_feat_per_frame)  # [B, 16, 512]

        # --- 2. 提取节点级动作特征 ---
        # motion_x: [B, 16, 24, 3] -> [B, 16, 24, 64]
        m_feat = self.motion_embed(motion_x) 

        # --- 3. 🌟 全局特征广播到图节点 ---
        v_feat_expanded = v_feat_proj.unsqueeze(2).expand(-1, -1, self.num_joints, -1)
        fused = torch.cat([v_feat_expanded, m_feat], dim=-1) # [B, 16, 24, 576]

        # 🌟🌟🌟 应用防爆盾 🌟🌟🌟
        fused = self.fused_ln(fused)

        # --- 4. 标准 STGCN 数据流转换与建模 ---
        fused = fused.permute(0, 3, 1, 2) # [B, 576, T, 24]
        
        fused = self.pre_stgcn_conv(fused) # [B, 128, T, 24]
        stgcn_out = self.stgcn_blocks(fused) # [B, 256, T, 24]

        # --- 5. 预测 ---
        last_frame_feat = stgcn_out[:, :, -1, :] 
        last_frame_feat = last_frame_feat.permute(0, 2, 1) # [B, 24, 256]
        
        pred = self.fc(last_frame_feat) # [B, 24, pred_len * 3]
        
        # 记得加上 .contiguous() 防止报错
        return pred.view(B, self.num_joints, self.pred_len, 3).permute(0, 2, 1, 3).contiguous()