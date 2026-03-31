import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig

class VLMFeatureExtractor(nn.Module):
    def __init__(self, model_type="onevision"):
        super().__init__()
        
        # 8-bit 量化配置
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )

        if model_type == "onevision":
            model_id = "lmms-lab-encoder/onevision-encoder-large"
            self.model = AutoModel.from_pretrained(model_id, quantization_config=bnb_config, trust_remote_code=True)
            self.hidden_dim = 1024 # 根据具体模型实际维度调整
        elif model_type == "gigabrain":
            model_id = "open-gigaai/GigaBrain-0.1-3.5B-Base"
            self.model = AutoModel.from_pretrained(model_id, quantization_config=bnb_config, trust_remote_code=True)
            self.hidden_dim = 1536 # 假设维度
            
        # 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        # x: [B, T, 3, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        
        with torch.no_grad():
            # 提取隐层特征 (不同模型取法略有不同)
            outputs = self.model(x, output_hidden_states=True)
            # 取最后一层或倒数第二层的池化特征
            feat = outputs.hidden_states[-1].mean(dim=1) 
            
        return feat.view(B, T, -1) # [B, T, hidden_dim]