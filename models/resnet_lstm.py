import torch
import torch.nn as nn
import torchvision.models as models

class ResNetLSTM(nn.Module):
    def __init__(self, seq_len=16, pred_len=16, num_joints=24):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_joints = num_joints
        self.out_dim = num_joints * 3 
        
        # 1. 加载经典 ResNet-50 (作为基础 Baseline)
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # 剥离最后的全连接层，保留特征提取部分 (输出 2048 维)
        self.backbone = nn.Sequential(*(list(resnet.children())[:-1]))
        
        # 冻结 ResNet 权重
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 2. 视觉特征投影 (2048 -> 512)
        self.v_proj = nn.Sequential(
            nn.Linear(2048, 512), 
            nn.LayerNorm(512),
            nn.ReLU()
        )

        # 3. 动作路径编码
        self.motion_embed = nn.Sequential(
            nn.Linear(self.out_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )

        # 4. LSTM 层
        self.lstm = nn.LSTM(input_size=768, hidden_size=512, num_layers=2, batch_first=True)
        self.fc = nn.Linear(512, self.pred_len * self.out_dim)

    def forward(self, visual_x, motion_x):
        B, T = visual_x.shape[:2]
        
        with torch.no_grad():
            v_x = visual_x.view(B * T, *visual_x.shape[2:]) # 初始形状可能是 [B*T, 3, 224, 224] 或 [B*T, 224, 224, 3]
            
            # 🌟 修复一：通道对齐保险
            # 如果通道在最后 (H, W, C)，将其转置为 ResNet 需要的 (C, H, W)
            if v_x.shape[-1] == 3:
                v_x = v_x.permute(0, 3, 1, 2).contiguous()
                
            # 🌟 修复二：类型对齐保险
            # 强制将输入数据转换为与 backbone 权重相同的数据类型 (防 float16 报错)
            v_x = v_x.to(next(self.backbone.parameters()).dtype)
            
            v_feat = self.backbone(v_x) # [B*T, 2048, 1, 1]
            v_feat = v_feat.view(B*T, -1) # [B*T, 2048]
                
        v_feat = v_feat.view(B, T, -1).to(motion_x.dtype)
        v_feat = self.v_proj(v_feat) 

        m_feat = self.motion_embed(motion_x.reshape(B, T, -1))
        fused = torch.cat([v_feat, m_feat], dim=-1) 
        lstm_out, _ = self.lstm(fused)
        pred = self.fc(lstm_out[:, -1, :])
        
        return pred.reshape(B, self.pred_len, self.num_joints, 3)