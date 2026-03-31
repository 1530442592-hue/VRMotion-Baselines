import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class BeatSaberDataset(Dataset):
    def __init__(self, data_root="data/beat_saber", seq_len=16, pred_len=16, transform=None):
        """
        从 MP4 和 JSON 文件中实时读取并对齐 VRMotion 数据。
        :param transform: 用于 VLM 或其他自定义模型的视频预处理函数 (传入 List[np.ndarray], 返回 Tensor)
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.transform = transform
        self.video_dir = os.path.join(data_root, "video_frames")
        self.motion_dir = os.path.join(data_root, "motion_3d")
        
        self.samples = []
        
        if not os.path.exists(self.video_dir) or not os.path.exists(self.motion_dir):
            print(f"[错误] 找不到目录: {self.video_dir} 或 {self.motion_dir}")
            return
            
        video_files = sorted([f for f in os.listdir(self.video_dir) if f.endswith('.mp4')])
        
        for v_file in video_files:
            base_name = os.path.splitext(v_file)[0]
            json_file = base_name + '.json'
            json_path = os.path.join(self.motion_dir, json_file)
            
            if os.path.exists(json_path):
                self.samples.append({
                    'video_path': os.path.join(self.video_dir, v_file),
                    'json_path': json_path
                })
                
        if len(self.samples) == 0:
            print(f"[警告] 没有找到匹配的 mp4 和 json 文件对！")
        else:
            print(f"[*] 成功找到 {len(self.samples)} 个视频-骨骼数据对！")

    def __len__(self):
        return len(self.samples)

    def _normalize_hip(self, motion_data, hip_index=0):
        """
        Hip-joint 归一化：将所有关节坐标减去骨盆坐标
        """
        hip_pos = motion_data[:, hip_index:hip_index+1, :] 
        return motion_data - hip_pos

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # ==========================================
        # 1. 解析 MP4 视频 (解耦预处理)
        # ==========================================
        cap = cv2.VideoCapture(sample['video_path'])
        frames = []
        for _ in range(self.seq_len):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        
        # 补齐不足的帧数
        while len(frames) < self.seq_len:
            frames.append(np.zeros_like(frames[-1]) if len(frames)>0 else np.zeros((224, 224, 3), dtype=np.uint8))

        # 【核心修改点】灵活的预处理分发
        if self.transform is not None:
            # 如果是 VLM，交给外部传入的 processor 处理
            video = self.transform(frames)
        else:
            # 兼容旧版本 (ResNet / VideoMAE) 的默认处理逻辑
            frames_resized = [cv2.resize(f, (224, 224)) for f in frames]
            video = torch.from_numpy(np.array(frames_resized)).permute(0, 3, 1, 2).float() / 255.0

        # ==========================================
        # 2. 智能解析 JSON 骨骼动作
        # ==========================================
        with open(sample['json_path'], 'r') as f:
            raw_data = json.load(f)
            
        motion_data = []
        if isinstance(raw_data, list):
            motion_data = raw_data
        elif isinstance(raw_data, dict):
            # 情况1：字典里包裹着列表
            for key, val in raw_data.items():
                if isinstance(val, list):
                    motion_data = val
                    break
            # 情况2：直接用数字作 key
            if len(motion_data) == 0:
                motion_data = [raw_data[k] for k in sorted(raw_data.keys()) if isinstance(raw_data[k], dict) and "joints" in raw_data[k]]

        frames_list = []
        if len(motion_data) > 0 and "joints" in motion_data[0]:
            joint_names = sorted(motion_data[0]["joints"].keys())
            hip_index = joint_names.index("Hips") if "Hips" in joint_names else 0
        else:
            joint_names = []
            hip_index = 0

        for frame_data in motion_data:
            if "joints" in frame_data:
                joints_dict = frame_data["joints"]
                frame_joints = [joints_dict.get(name, [0,0,0]) for name in joint_names]
                frames_list.append(frame_joints)
            
        motion_np = np.array(frames_list, dtype=np.float32)
        actual_joints = motion_np.shape[1] if motion_np.ndim > 1 else 0
            
        # 补齐帧数
        required_len = self.seq_len + self.pred_len
        if len(motion_np) < required_len:
            pad_len = required_len - len(motion_np)
            padding = np.repeat(motion_np[-1:], pad_len, axis=0) if len(motion_np) > 0 else np.zeros((pad_len, actual_joints, 3))
            motion_np = np.concatenate([motion_np, padding], axis=0)

        # ==========================================
        # 3. 截取并归一化
        # ==========================================
        past_motion = torch.from_numpy(motion_np[:self.seq_len])
        past_motion = self._normalize_hip(past_motion, hip_index)

        future_gt = torch.from_numpy(motion_np[self.seq_len : self.seq_len + self.pred_len])
        future_gt = self._normalize_hip(future_gt, hip_index)

        return video, past_motion, future_gt