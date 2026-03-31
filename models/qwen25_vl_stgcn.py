import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig

# ================= 🌟 终极修复：带准确物理拓扑先验的 STGCN 核心算子 🌟 =================
class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_joints=24, temporal_kernel=3):
        super().__init__()
        
        # 1. 植入 VRMotion (Beat Saber) 数据集的真实 SMPL 骨骼物理连接 (24 关节)
        # 这是之前我们一起调试通过的绝对正确的拓扑结构
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
            if i < num_joints and j < num_joints: # 防御性编程：确保索引不越界
                A_physical[i, j] = 1.0
                A_physical[j, i] = 1.0 # 无向图，双向连接
        
        # 加上自连接 (节点自身的信息必须保留)
        for i in range(num_joints):
            A_physical[i, i] = 1.0
            
        # 归一化处理：防止度数高的节点（如节点 11 或 0）特征在卷积后爆炸
        D = torch.sum(A_physical, dim=1)
        D_inv_sqrt = torch.pow(D, -0.5)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0 # 防御除零错
        D_mat = torch.diag(D_inv_sqrt)
        A_physical = D_mat @ A_physical @ D_mat
        
        # 注册为 buffer，作为常数矩阵不参与反向传播，随模型 state_dict 保存
        self.register_buffer('A_physical', A_physical)
        
        # 2. 保留极弱的自适应矩阵 (A_adaptive)
        # 初始化为接近 0，让模型前期 100% 依赖物理骨架，后期再微调非物理的隐式关联
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
        # 将静态物理图与动态学习图融合
        A_fused = self.A_physical + self.A_adaptive
        
        # 执行图卷积聚合
        x = torch.einsum('bctv,vw->bctw', x, A_fused)
        x = self.spatial_conv(x)
        x = self.temporal_conv(x)
        x = self.bn(x)
        return self.relu(x)

# ================= 修复版 Qwen 2.5-VL + STGCN =================
class Qwen25VLSTGCN(nn.Module):
    def __init__(self, seq_len=16, pred_len=16, num_joints=24):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_joints = num_joints
        
        # 1. 8-bit 量化配置
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # 2. 加载 Qwen 2.5-VL-7B
        model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
        self.qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, 
            device_map="cuda", 
            quantization_config=bnb_config,
            trust_remote_code=True
        )
        
        # 3. 定位并冻结视觉塔
        if hasattr(self.qwen, 'visual'):
            self.visual_tower = self.qwen.visual
        elif hasattr(self.qwen.model, 'visual'):
            self.visual_tower = self.qwen.model.visual
        else:
            self.visual_tower = self.qwen.transformer.visual
            
        for param in self.visual_tower.parameters():
            param.requires_grad = False

        # 4. 视觉特征降维
        self.vlm_hidden_dim = 3584
        self.v_proj = nn.Sequential(
            nn.Linear(self.vlm_hidden_dim, 512), 
            nn.LayerNorm(512),
            nn.ReLU()
        )

        # 5. 节点级动作特征降维
        self.motion_embed = nn.Sequential(
            nn.Linear(3, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )

        # 🌟🌟🌟 核心防爆盾：强制融合后特征归一化 🌟🌟🌟
        self.fused_ln = nn.LayerNorm(576)

        # 6. STGCN 前的通道压缩 (576 -> 128)
        self.pre_stgcn_conv = nn.Sequential(
            nn.Conv2d(576, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # 7. STGCN 层构建
        self.stgcn_blocks = nn.Sequential(
            STGCNBlock(128, 128, num_joints=self.num_joints),
            STGCNBlock(128, 256, num_joints=self.num_joints)
        )
        
        # 8. 节点级预测头
        self.fc = nn.Linear(256, self.pred_len * 3)

    def forward(self, visual_inputs, motion_x):
        B = motion_x.shape[0]
        T = self.seq_len
        
        # --- 提取 VLM 全局视觉特征 ---
        pixel_values = visual_inputs['pixel_values'].to(self.qwen.dtype)
        image_grid_thw = visual_inputs['image_grid_thw']
        
        with torch.no_grad():
            flat_pixel_values = pixel_values.view(-1, pixel_values.shape[-1])
            flat_grid_thw = image_grid_thw.view(-1, 3)
            visual_outputs = self.visual_tower(flat_pixel_values, grid_thw=flat_grid_thw)
            
            if hasattr(visual_outputs, 'last_hidden_state'):
                v_feat = visual_outputs.last_hidden_state
            else:
                v_feat = visual_outputs[0]
            
            tokens_per_image = (flat_grid_thw[:, 1] * flat_grid_thw[:, 2]).tolist()
            v_feats_split = torch.split(v_feat, tokens_per_image)
            pooled_feats = [feat.mean(dim=0) for feat in v_feats_split]
            v_feat = torch.stack(pooled_feats) 
                
        if v_feat.shape[-1] != self.vlm_hidden_dim:
            self.vlm_hidden_dim = v_feat.shape[-1]
            self.v_proj[0] = nn.Linear(self.vlm_hidden_dim, 512).to(v_feat.device)

        v_feat = v_feat.view(B, T, -1).to(motion_x.dtype)
        v_feat = self.v_proj(v_feat) # [B, T, 512]

        # --- 提取节点级动作特征 ---
        m_feat = self.motion_embed(motion_x) # [B, T, 24, 64]

        # --- 全局特征广播到图节点 ---
        v_feat_expanded = v_feat.unsqueeze(2).expand(-1, -1, self.num_joints, -1)
        
        # 拼接: [B, T, 24, 576]
        fused = torch.cat([v_feat_expanded, m_feat], dim=-1)

        # 🌟🌟🌟 应用防爆盾 🌟🌟🌟
        fused = self.fused_ln(fused)

        # ================= STGCN 数据流转换 =================
        fused = fused.permute(0, 3, 1, 2) # [B, 576, T, 24]
        
        fused = self.pre_stgcn_conv(fused) # [B, 128, T, 24]
        stgcn_out = self.stgcn_blocks(fused) # [B, 256, T, 24]
        
        # --- 预测未来 ---
        last_frame_feat = stgcn_out[:, :, -1, :] 
        last_frame_feat = last_frame_feat.permute(0, 2, 1)
        
        pred = self.fc(last_frame_feat) 
        pred = pred.view(B, self.num_joints, self.pred_len, 3).permute(0, 2, 1, 3)
        return pred