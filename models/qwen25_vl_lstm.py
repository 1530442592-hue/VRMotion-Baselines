import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig

class Qwen35VLLSTM(nn.Module):
    def __init__(self, seq_len=16, pred_len=16, num_joints=24):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_joints = num_joints
        self.out_dim = num_joints * 3 
        
        # 1. 8-bit 量化配置
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # 2. 架构：确实是 Qwen2.5-VL-7B
        model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
        self.qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, 
            device_map="cuda", 
            quantization_config=bnb_config,
            trust_remote_code=True
        )
        
        # 3. 稳健地获取视觉塔 (兼容不同 transformers 版本)
        if hasattr(self.qwen, 'visual'):
            self.visual_tower = self.qwen.visual
        elif hasattr(self.qwen.model, 'visual'):
            self.visual_tower = self.qwen.model.visual
        else:
            self.visual_tower = self.qwen.transformer.visual
            
        for param in self.visual_tower.parameters():
            param.requires_grad = False

        # 4. 视觉层特征：7B 的独立视觉塔输出维度为 1280
        self.vlm_hidden_dim = 1280
        self.v_proj = nn.Sequential(
            nn.Linear(self.vlm_hidden_dim, 512), 
            nn.LayerNorm(512),
            nn.ReLU()
        )

        # 5. 动作特征编码
        self.motion_embed = nn.Sequential(
            nn.Linear(self.out_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )

        # 6. 🔥 终极修正：LSTM 隐藏层必须是 512，完美对齐 Checkpoint！
        self.lstm = nn.LSTM(input_size=768, hidden_size=512, num_layers=2, batch_first=True)
        
        # 7. 🔥 终极修正：全连接层输入必须是 512，完美对齐 Checkpoint！
        self.fc = nn.Linear(512, self.pred_len * self.out_dim)

    def forward(self, visual_inputs, motion_x):
        B = motion_x.shape[0]
        T = self.seq_len
        
        pixel_values = visual_inputs['pixel_values'].to(self.qwen.dtype)
        image_grid_thw = visual_inputs['image_grid_thw']
        
        with torch.no_grad():
            flat_pixel_values = pixel_values.view(-1, pixel_values.shape[-1])
            flat_grid_thw = image_grid_thw.view(-1, 3)
            
            # 提取视觉特征
            visual_outputs = self.visual_tower(flat_pixel_values, grid_thw=flat_grid_thw)
            
            # 自动解包输出
            if hasattr(visual_outputs, 'last_hidden_state'):
                v_feat = visual_outputs.last_hidden_state
            elif isinstance(visual_outputs, (list, tuple)):
                v_feat = visual_outputs[0]
            else:
                v_feat = visual_outputs
            
            # 均值池化 tokens
            tokens_per_image = (flat_grid_thw[:, 1] * flat_grid_thw[:, 2]).tolist()
            v_feats_split = torch.split(v_feat, tokens_per_image)
            pooled_feats = [feat.mean(dim=0) for feat in v_feats_split]
            v_feat = torch.stack(pooled_feats) 
                
        # 特征映射与融合
        v_feat = v_feat.view(B, T, -1).to(motion_x.dtype)
        v_feat = self.v_proj(v_feat) 

        m_feat = self.motion_embed(motion_x.reshape(B, T, -1))
        fused = torch.cat([v_feat, m_feat], dim=-1) 
        
        # 时序建模与预测
        lstm_out, _ = self.lstm(fused)
        last_state = lstm_out[:, -1, :] 
        pred = self.fc(last_state)
        
        return pred.view(B, self.pred_len, self.num_joints, 3)