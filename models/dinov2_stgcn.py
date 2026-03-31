import torch
import torch.nn as nn
from transformers import AutoModel

# ================= 🌟 终极修复：带准确物理拓扑先验的 STGCN 核心算子 🌟 =================
class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_joints=24, temporal_kernel=3):
        super().__init__()
        
        # 1. 植入 VRMotion (Beat Saber) 数据集的真实 SMPL 骨骼物理连接 (24 关节)
        edges = [
            # 脊柱与头部
            (0, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15),
            # 右臂
            (11, 16), (16, 17), (17, 18), (18, 19),
            # 左臂
            (11, 20), (20, 21), (21, 22), (22, 23),
            # 右腿
            (0, 1), (1, 2), (2, 3), (3, 4),
            # 左腿
            (0, 5), (5, 6), (6, 7), (7, 8)
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
        
        # 2. 保留极弱的自适应矩阵
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
        A_fused = self.A_physical + self.A_adaptive
        x = torch.einsum('bctv,vw->bctw', x, A_fused)
        x = self.spatial_conv(x)
        x = self.temporal_conv(x)
        x = self.bn(x)
        return self.relu(x)

# ================= 重写版 DINOv2 + STGCN =================
class DINOv2STGCN(nn.Module):
    def __init__(self, seq_len=16, pred_len=16, num_joints=24, use_8bit=False):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_joints = num_joints
        
        # 1. 加载 Meta DINOv2-Giant 
        model_id = "facebook/dinov2-giant"
        self.vlm = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16)
        
        for param in self.vlm.parameters():
            param.requires_grad = False

        # 2. 视觉特征降维 (1536 -> 512)
        self.v_proj = nn.Sequential(
            nn.Linear(1536, 512), 
            nn.LayerNorm(512),
            nn.ReLU()
        )

        # 3. 节点级动作特征降维 (3 -> 64)
        self.motion_embed = nn.Sequential(
            nn.Linear(3, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )

        # 🌟🌟🌟 核心防爆盾：强制融合后特征归一化 (512+64=576) 🌟🌟🌟
        self.fused_ln = nn.LayerNorm(576)

        # 4. STGCN 前的通道压缩
        self.pre_stgcn_conv = nn.Sequential(
            nn.Conv2d(576, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # 5. 真正的 STGCN 层构建
        self.stgcn_blocks = nn.Sequential(
            STGCNBlock(128, 128, num_joints=self.num_joints),
            STGCNBlock(128, 256, num_joints=self.num_joints)
        )
        
        # 6. 节点级预测头 
        self.fc = nn.Linear(256, self.pred_len * 3)

    def forward(self, visual_x, motion_x):
        B = motion_x.shape[0]
        T = self.seq_len
        
        with torch.no_grad():
            v_x = visual_x.view(B * T, *visual_x.shape[2:]).to(self.vlm.dtype)
            outputs = self.vlm(pixel_values=v_x)
            v_feat = outputs.last_hidden_state[:, 0, :]
                
        v_feat = v_feat.view(B, T, -1).to(motion_x.dtype)
        v_feat = self.v_proj(v_feat) # [B, T, 512]

        m_feat = self.motion_embed(motion_x) # [B, T, 24, 64]

        # 🌟 全局特征广播
        v_feat_expanded = v_feat.unsqueeze(2).expand(-1, -1, self.num_joints, -1)
        fused = torch.cat([v_feat_expanded, m_feat], dim=-1) # [B, T, 24, 576]

        # 🌟🌟🌟 应用防爆盾 🌟🌟🌟
        fused = self.fused_ln(fused)

        # ================= STGCN 数据流转换 =================
        fused = fused.permute(0, 3, 1, 2) # [B, 576, T, 24]
        
        fused = self.pre_stgcn_conv(fused) # [B, 128, T, 24]
        stgcn_out = self.stgcn_blocks(fused) # [B, 256, T, 24]
        
        last_frame_feat = stgcn_out[:, :, -1, :] 
        last_frame_feat = last_frame_feat.permute(0, 2, 1) 
        
        pred = self.fc(last_frame_feat) 
        return pred.view(B, self.num_joints, self.pred_len, 3).permute(0, 2, 1, 3)