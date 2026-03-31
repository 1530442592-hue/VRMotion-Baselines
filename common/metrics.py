import torch

def compute_mpjpe(pred, target):
    """
    Mean Per Joint Position Error (MPJPE)
    pred, target: (B, T, J, 3) 
    返回标量均值
    """
    error = torch.norm(pred - target, dim=-1) # (B, T, J)
    return torch.mean(error).item()

def compute_pa_mpjpe(pred, target):
    """
    Procrustes-Aligned MPJPE (PA-MPJPE) 
    (PyTorch 全向量化版本，速度极快)
    pred, target: (B, T, J, 3)
    """
    B, T, J, C = pred.shape
    pred = pred.view(-1, J, C)     # (B*T, J, 3)
    target = target.view(-1, J, C) # (B*T, J, 3)
    
    # 1. 均值中心化
    muX = torch.mean(pred, dim=1, keepdim=True)
    muY = torch.mean(target, dim=1, keepdim=True)
    X0 = pred - muX
    Y0 = target - muY
    
    # 2. 缩放归一化
    normX = torch.sqrt(torch.sum(X0**2, dim=(1, 2), keepdim=True))
    normY = torch.sqrt(torch.sum(Y0**2, dim=(1, 2), keepdim=True))
    X0 = X0 / (normX + 1e-8)
    Y0 = Y0 / (normY + 1e-8)
    
    # 3. 计算最优旋转矩阵 (SVD)
    H = torch.bmm(X0.transpose(1, 2), Y0)
    U, S, V = torch.svd(H)
    R = torch.bmm(V, U.transpose(1, 2))
    
    # 处理反射避免镜像翻转
    det = torch.det(R)
    V[:, :, -1] *= torch.sign(det).unsqueeze(-1)
    R = torch.bmm(V, U.transpose(1, 2))
    
    # 4. 应用旋转和缩放
    pred_aligned = torch.bmm(X0, R.transpose(1, 2)) * normY + muY
    
    # 5. 计算对齐后的误差
    error = torch.norm(pred_aligned - target, dim=-1)
    return torch.mean(error).item()