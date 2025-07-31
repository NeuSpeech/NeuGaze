# =============================================================================
# NeuSpeech Institute, NeuGaze Project
# Copyright (c) 2024 Yiqian Yang
#
# This code is part of the NeuGaze project developed at NeuSpeech Institute.
# Author: Yiqian Yang
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 
# International License. To view a copy of this license, visit:
# http://creativecommons.org/licenses/by-nc/4.0/
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler  # 添加这行
from pathlib import Path
import os
import json
import numpy as np
from collections import defaultdict
import torch.nn.functional as F
from torch.amp import autocast, GradScaler  # 更新导入方式

class ImprovedGazeNet(nn.Module):
    def __init__(self, screen_width=3072, screen_height=1920):
        super().__init__()
        self.screen_width = screen_width
        self.screen_height = screen_height
        # 关键点定义
        self.left_iris = [468, 469, 470, 471, 472]
        self.right_iris = [473, 474, 475, 476, 477]
        self.left_eye_corners = [33, 133]
        self.right_eye_corners = [362, 263]
        self.left_eye_top = [159, 145]
        self.right_eye_top = [386, 374]
        
        # 增强特征提取网络
        self.abs_eye_net = nn.Sequential(
            nn.Linear(5, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
        )
        
        self.rel_eye_net = nn.Sequential(
            nn.Linear(5, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
        )
        
        # 多头注意力机制
        self.attention = nn.MultiheadAttention(embed_dim=32, num_heads=4)
        
        # 融合网络
        fusion_dim = 128
        self.fusion_net = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ) for _ in range(3)
        ])
        
        # 预测头
        self.final_net = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
        )
        
        self.x_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        self.y_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def normalize_landmarks(self, landmarks):
        face_center = landmarks.mean(dim=1, keepdim=True)
        rel_landmarks = landmarks - face_center
        face_scale = torch.max(torch.abs(rel_landmarks), dim=1, keepdim=True)[0]
        rel_landmarks = rel_landmarks / (face_scale + 1e-7)
        return rel_landmarks

    def compute_eye_features(self, landmarks, iris_points, corner_points, top_points, use_relative=True):
        if use_relative:
            landmarks = self.normalize_landmarks(landmarks)
        
        iris_center = landmarks[:, iris_points].mean(dim=1)
        left_corner = landmarks[:, corner_points[0]]
        right_corner = landmarks[:, corner_points[1]]
        eye_center = (left_corner + right_corner) / 2
        eye_vector = right_corner - left_corner
        eye_vector = eye_vector / (torch.norm(eye_vector, dim=1, keepdim=True) + 1e-7)
        iris_offset = iris_center - eye_center
        eye_width = torch.norm(right_corner - left_corner, dim=1, keepdim=True)
        iris_offset = iris_offset / (eye_width + 1e-7)
        top_point = landmarks[:, top_points[0]]
        bottom_point = landmarks[:, top_points[1]]
        eye_height = torch.norm(top_point - bottom_point, dim=1, keepdim=True)
        eye_aspect_ratio = eye_height / (eye_width + 1e-7)
        
        return torch.cat([
            eye_vector[:, :2],
            iris_offset[:, :2],
            eye_aspect_ratio,
        ], dim=1)
    
    def forward(self, x):
        left_abs_features = self.compute_eye_features(
            x, self.left_iris, self.left_eye_corners, self.left_eye_top, use_relative=False
        )
        left_abs_features = self.abs_eye_net(left_abs_features)
        
        right_abs_features = self.compute_eye_features(
            x, self.right_iris, self.right_eye_corners, self.right_eye_top, use_relative=False
        )
        right_abs_features = self.abs_eye_net(right_abs_features)
        
        left_rel_features = self.compute_eye_features(
            x, self.left_iris, self.left_eye_corners, self.left_eye_top, use_relative=True
        )
        left_rel_features = self.rel_eye_net(left_rel_features)
        
        right_rel_features = self.compute_eye_features(
            x, self.right_iris, self.right_eye_corners, self.right_eye_top, use_relative=True
        )
        right_rel_features = self.rel_eye_net(right_rel_features)
        
        left_att, _ = self.attention(left_abs_features.unsqueeze(0), left_abs_features.unsqueeze(0), left_abs_features.unsqueeze(0))
        right_att, _ = self.attention(right_abs_features.unsqueeze(0), right_abs_features.unsqueeze(0), right_abs_features.unsqueeze(0))
        left_rel_att, _ = self.attention(left_rel_features.unsqueeze(0), left_rel_features.unsqueeze(0), left_rel_features.unsqueeze(0))
        right_rel_att, _ = self.attention(right_rel_features.unsqueeze(0), right_rel_features.unsqueeze(0), right_rel_features.unsqueeze(0))
        
        features = torch.cat([
            left_abs_features * left_att.squeeze(0),
            right_abs_features * right_att.squeeze(0),
            left_rel_features * left_rel_att.squeeze(0),
            right_rel_features * right_rel_att.squeeze(0)
        ], dim=1)
        
        for fusion_layer in self.fusion_net:
            features = features + fusion_layer(features)
        
        features = self.final_net(features)
        x_pred = self.x_head(features) * self.screen_width
        y_pred = self.y_head(features) * self.screen_height
        
        return torch.cat([x_pred, y_pred], dim=1)
    
    def fit(self, X, y, epochs=100, batch_size=32, validation_split=0.2):
        """训练模型

        参数:
        X : ndarray, shape (n_samples, n_features)
            训练数据
        y : ndarray, shape (n_samples,) or (n_samples, n_targets)
            目标值
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        
        # 转换数据为PyTorch张量
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y)
        
        # 划分训练集和验证集
        val_size = int(len(X) * validation_split)
        indices = torch.randperm(len(X))
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        
        train_X, train_y = X[train_indices], y[train_indices]
        val_X, val_y = X[val_indices], y[val_indices]
        
        # 初始化优化器和损失函数
        optimizer = optim.AdamW(self.parameters(), lr=0.0005, weight_decay=0.01)
        criterion = F.mse_loss
        
        best_val_loss = float('inf')
        best_state_dict = None
        patience = 20
        patience_counter = 0
        
        # 训练循环
        for epoch in range(epochs):
            # 训练阶段
            self.train()
            train_loss = 0
            for i in range(0, len(train_X), batch_size):
                batch_X = train_X[i:i+batch_size].to(device)
                batch_y = train_y[i:i+batch_size].to(device)
                
                optimizer.zero_grad()
                outputs = self(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # 验证阶段
            self.eval()
            val_loss = 0
            with torch.no_grad():
                for i in range(0, len(val_X), batch_size):
                    batch_X = val_X[i:i+batch_size].to(device)
                    batch_y = val_y[i:i+batch_size].to(device)
                    outputs = self(batch_X)
                    val_loss += criterion(outputs, batch_y).item()
            
            # 计算平均损失
            avg_train_loss = train_loss / (len(train_X) // batch_size)
            avg_val_loss = val_loss / (len(val_X) // batch_size)
            
            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state_dict = self.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
                
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], '
                      f'Train Loss: {avg_train_loss:.4f}, '
                      f'Val Loss: {avg_val_loss:.4f}')
        
        # 加载最佳模型权重
        if best_state_dict is not None:
            self.load_state_dict(best_state_dict)
            print(f'Loaded best model with validation loss: {best_val_loss:.4f}')
        
        return self

    def predict(self, X):
        """使用模型进行预测

        参数:
        X : array-like, shape (n_samples, n_features)
            输入数据

        返回:
        y : array-like, shape (n_samples, n_outputs)
            预测结果
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        self.eval()
        
        # 转换输入数据为PyTorch张量
        X = torch.FloatTensor(X)
        
        # 分批次预测以节省内存
        batch_size = 32
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i+batch_size].to(device)
                batch_pred = self(batch_X)
                predictions.append(batch_pred.cpu().numpy())
        
        return np.vstack(predictions)
    
    

# 数据加载和处理


def load_data(root_dir='calibration'):
    folders = defaultdict(list)
    all_dirs = os.listdir(root_dir)
    # 选择指定时间后的数据
    # all_dirs = [x for x in all_dirs if int(x.split('_')[0])>=20241125]
    
    for _dir in all_dirs:
        folder_path = os.path.join(root_dir, _dir)
        jsonl_path = os.path.join(folder_path, 'train_data.jsonl')
        if os.path.exists(jsonl_path):
            folders[folder_path].append(jsonl_path)
    
    # 按文件夹划分训练集和验证集
    all_folders = list(folders.keys())
    np.random.shuffle(all_folders)
    train_folders = all_folders[:-len(all_folders)//5]
    val_folders = all_folders[-len(all_folders)//5:]
    
    def process_folder_data(folder_list):
        landmarks_list = []
        gaze_points_list = []
        
        for folder in folder_list:
            for path in folders[folder]:
                with open(path, 'r') as f:
                    for line in f:
                        data = json.loads(line)
                        if 'mediapipe_results' in data:
                            landmarks = np.array(data['mediapipe_results'])
                            if landmarks.shape == (478, 3):  # 只检查形状，不做任何预处理
                                landmarks_list.append(landmarks)
                                gaze_points_list.append(np.array(data['label']))
        
        return (torch.tensor(landmarks_list, dtype=torch.float32),
                torch.tensor(gaze_points_list, dtype=torch.float32))
    
    return process_folder_data(train_folders), process_folder_data(val_folders)

def main():
    # 设置设备和保存目录
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = Path('model_weights/only_mediapipe/')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    train_data, val_data = load_data()
    print(f"训练集大小: {len(train_data[0])}, 验证集大小: {len(val_data[0])}")
    
    # 定义evaluate函数
    def evaluate(data):
        model.eval()
        landmarks, gaze_points = data
        with torch.no_grad():
            landmarks = landmarks.to(device)
            gaze_points = gaze_points.to(device)
            outputs = model(landmarks)
            
            # 分别计算x和y的误差
            errors = torch.abs(outputs - gaze_points)
            x_mae = errors[:, 0].mean().item()
            y_mae = errors[:, 1].mean().item()
            total_mae = errors.mean().item()
            
            print(f"\n误差统计:")
            print(f"X轴平均误差: {x_mae:.2f}")
            print(f"Y轴平均误差: {y_mae:.2f}")
            
            return total_mae
    
    # 定义超参数
    epochs = 5000
    batch_size = 256
    accumulation_steps = 4
    patience = 500
    min_delta = 0.5
    max_grad_norm = 0.5
    
    # 初始化模型
    model = ImprovedGazeNet().to(device)
    
    # 初始优化器和学习率调度器
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # 初始化其他训练组件
    scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')  # 更新初始化方式
    best_val_mae = float('inf')
    best_models = []
    best_scores = []
    max_models = 5
    
    # 定义损失函数
    def criterion(pred, target):
        # 基础L1损失
        l1_loss = F.l1_loss(pred, target, reduction='none')
        
        # 计算预测点和目标点之间的角度
        pred_angle = torch.atan2(pred[:, 1], pred[:, 0])
        target_angle = torch.atan2(target[:, 1], target[:, 0])
        angle_loss = 1 - torch.cos(pred_angle - target_angle)
        
        # 计算距离损失
        pred_dist = torch.norm(pred, dim=1)
        target_dist = torch.norm(target, dim=1)
        dist_loss = F.smooth_l1_loss(pred_dist, target_dist)
        
        return l1_loss.mean() + 0.1 * angle_loss.mean() + 0.1 * dist_loss
    
    criterion = criterion
    
    # 训练循环
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        for i in range(0, len(train_data[0]), batch_size):
            batch_landmarks = train_data[0][i:i+batch_size].to(device)
            batch_gaze = train_data[1][i:i+batch_size].to(device)
            
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):  # 更新使用方式
                outputs = model(batch_landmarks)
                loss = criterion(outputs, batch_gaze)
                loss = loss / accumulation_steps
            
            total_loss += loss.item() * accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (i // batch_size + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
        
        # 每个epoch都验证
        val_mae = evaluate(val_data)
        print(f'Epoch {epoch+1}, Loss: {total_loss:.4f}, Val MAE: {val_mae:.4f}')
        
        # 更新最佳验证MAE和早停逻辑
        if val_mae < best_val_mae:
            if val_mae < best_val_mae - min_delta:
                patience_counter = 0  # 显著改善，重置计数器
            else:
                patience_counter += 0.5  # 轻微改善，缓慢增加计数器
            best_val_mae = val_mae
            
            # 保存当前最佳模型
            torch.save(model.state_dict(), save_dir / f'best_model_epoch_{epoch+1}.pth')
            print(f"模型在epoch {epoch+1}时保存，验证MAE: {val_mae:.4f}")
        else:
            patience_counter += 1
        
        # 动态调整min_delta
        if patience_counter >= patience * 0.5:
            min_delta *= 0.75
            print(f"Adjusting min_delta to {min_delta}")
        
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
        
        # 更新学习率
        scheduler.step()
    
    # 保存评估结果
    eval_results = {
        'best_models': [
            {'path': model_path, 'val_mae': score}
            for model_path, score in zip(best_models, best_scores)
        ]
    }
    
    with open(save_dir / 'eval_results.json', 'w') as f:
        json.dump(eval_results, f, indent=4)
    
    print("训练完成！最佳验证MAE:", best_val_mae)

if __name__ == "__main__":
    main()
