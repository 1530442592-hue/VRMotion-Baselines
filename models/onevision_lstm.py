import torch
import torch.nn as nn
from transformers import AutoModel, BitsAndBytesConfig

class OneVisionLSTM(nn.Module):
    # 🌟 核心修改：为了实验公平性，默认关闭 8-bit 量化 (对齐 DINOv2 等模型)
    def __init__(self, seq_len=16, pred_len=16, num_joints=24, use_8bit=False):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_joints = num_joints
        self.out_dim = num_joints * 3  # 72
        
        # ==========================================
        # 1. 核心：加载 OneVision 大视觉编码器
        # ==========================================
        model_id = "lmms-lab-encoder/onevision-encoder-large"
        
        if use_8bit:
            # 开启 8-bit 量化，极大地降低显存占用 (仅在显存不足或跑特定消融实验时开启)
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
            self.vlm = AutoModel.from_pretrained(model_id, quantization_config=bnb_config, trust_remote_code=True)
        else:
            # 正常 FP16 加载 (为了横向对比实验的公平性)
            self.vlm = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16, trust_remote_code=True)
            
        # 严格冻结大模型的所有参数！我们只用它的“眼睛”，不更新它
        for param in self.vlm.parameters():
            param.requires_grad = False
            
        # 假设 large 版本的隐层维度通常是 1024
        # 我们加一个 Projector (投影层)，把大维度压缩，防止 LSTM 参数爆炸
        self.vlm_hidden_dim = 1024 
        self.v_proj = nn.Sequential(
            nn.Linear(self.vlm_hidden_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU()
        )

        # ==========================================
        # 2. 运动路径编码 (与之前保持一致，加固稳定性)
        # ==========================================
        self.motion_embed = nn.Sequential(
            nn.Linear(self.out_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )

        # ==========================================
        # 3. 时序预测后端 (LSTM)
        # ==========================================
        # 融合后的维度：视觉 512 + 动作 256 = 768
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # 4. 预测头
        self.fc = nn.Linear(512, self.pred_len * self.out_dim)

    def forward(self, visual_x, motion_x):
        # visual_x: [B, T, C, H, W]  (VLM 处理好的 Tensor)
        # motion_x: [B, T, J, 3]
        B, T = visual_x.shape[:2]

        # --- 1. 提取 VLM 视觉特征 ---
        # 必须使用 no_grad 确保显存不崩
        with torch.no_grad():
            # 将 B 和 T 合并，送入视觉编码器
            v_x = visual_x.view(B * T, *visual_x.shape[2:])
            v_x = v_x.to(self.vlm.dtype)
            outputs = self.vlm(v_x)
            
            # 兼容性池化：有些 Encoder 返回 pooler_output，有些返回 last_hidden_state
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                v_feat = outputs.pooler_output
            else:
                # 如果没有现成的 pooler，就对所有图像补丁(Patches)做一个全局平均池化
                v_feat = outputs.last_hidden_state.mean(dim=1) 
                
        # 恢复 [B, T, D] 形状，并进行降维
        v_feat = v_feat.view(B, T, -1).to(motion_x.dtype) # 统一数据类型
        v_feat = self.v_proj(v_feat) # [B, T, 512]

        # --- 2. 提取动作特征 ---
        m_x = motion_x.reshape(B, T, -1)
        m_feat = self.motion_embed(m_x) # [B, T, 256]

        # --- 3. 多模态融合与时序建模 ---
        fused = torch.cat([v_feat, m_feat], dim=-1) # [B, T, 768]
        
        lstm_out, _ = self.lstm(fused) # [B, T, 512]
        
        # 取 LSTM 最后一帧的隐藏状态作为包含全部时序信息的“动作意图”
        last_hidden = lstm_out[:, -1, :] # [B, 512]

        # --- 4. 最终坐标预测 ---
        pred = self.fc(last_hidden)
        pred = pred.reshape(B, self.pred_len, self.num_joints, 3)

        return pred