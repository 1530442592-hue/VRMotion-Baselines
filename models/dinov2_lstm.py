import torch
import torch.nn as nn
from transformers import AutoModel

class DINOv2LSTM(nn.Module):
    def __init__(self, seq_len=16, pred_len=16, num_joints=24, use_8bit=False):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_joints = num_joints
        self.out_dim = num_joints * 3 
        
        # 1. 加载 Meta DINOv2-Giant (11亿参数纯视觉巨兽)
        model_id = "facebook/dinov2-giant"
        self.vlm = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16)
        
        # 冻结 1B 底座参数
        for param in self.vlm.parameters():
            param.requires_grad = False

        # 2. 视觉特征投影层 (DINOv2-Giant 维度为 1536)
        self.v_proj = nn.Sequential(
            nn.Linear(1536, 512), 
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
            v_x = visual_x.view(B * T, *visual_x.shape[2:]).to(self.vlm.dtype)
            
            # DINOv2 前向传播，提取 CLS token 特征
            outputs = self.vlm(pixel_values=v_x)
            # 取 sequence 的第一个 token (CLS token) 作为整图特征
            v_feat = outputs.last_hidden_state[:, 0, :]
                
        v_feat = v_feat.view(B, T, -1).to(motion_x.dtype)
        v_feat = self.v_proj(v_feat) 

        m_feat = self.motion_embed(motion_x.reshape(B, T, -1))
        fused = torch.cat([v_feat, m_feat], dim=-1) 
        lstm_out, _ = self.lstm(fused)
        pred = self.fc(lstm_out[:, -1, :])
        return pred.reshape(B, self.pred_len, self.num_joints, 3)