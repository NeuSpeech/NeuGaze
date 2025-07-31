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

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
import jsonlines
import argparse
import pickle
from pathlib import Path
import cv2
from sklearn.linear_model import Ridge, Lasso, MultiTaskLassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from filterpy.kalman import KalmanFilter


class GazeKalmanFilter:
    def __init__(self, dt, std_measurement, Q_coef=0.03):
        # 定义卡尔曼滤波器的初始参数
        self.dt = dt
        self.std_measurement = std_measurement
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array([[1, 0, self.dt, 0.5 * self.dt ** 2],
                              [0, 1, 0, self.dt],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]])
        self.kf.P *= 1000  # 初始化状态协方差矩阵
        self.kf.Q = np.eye(4) * Q_coef  # 过程噪声协方差矩阵 这个系数越小越平滑（慢）
        self.kf.R *= std_measurement ** 2  # 观测噪声协方差矩阵 这个系数越大越平滑
        self.kf.x = np.array([0, 0, 0, 0])  # 初始状态

    def smooth_position(self, pos):
        # 使用卡尔曼滤波器进行状态估计
        kf = self.kf
        kf.predict()
        kf.update(pos)
        filtered_pos = kf.x.copy()[:2]

        # 返回平滑后的位置
        return filtered_pos


class ErrorMapCalculator:
    def __init__(self, screen_width_px=3072, screen_height_px=1920, 
                 screen_width_mm=344.6, screen_height_mm=215.4,
                 regression_model_type='lassocv', feature_type='basic'):
        """
        初始化误差映射计算器
        
        Args:
            screen_width_px: 屏幕宽度像素
            screen_height_px: 屏幕高度像素  
            screen_width_mm: 屏幕宽度毫米
            screen_height_mm: 屏幕高度毫米
            regression_model_type: 回归模型类型 ('ridge', 'lasso', 'lassocv', 'random_forest', 'gradient_boosting', 'mlp')
            feature_type: 特征类型 ('basic', 'mediapipe', 'combined')
        """
        self.screen_width_px = screen_width_px
        self.screen_height_px = screen_height_px
        self.screen_width_mm = screen_width_mm
        self.screen_height_mm = screen_height_mm
        self.regression_model_type = regression_model_type
        self.feature_type = feature_type
        
        # 计算像素到毫米的转换比例
        self.px_to_mm_x = screen_width_mm / screen_width_px
        self.px_to_mm_y = screen_height_mm / screen_height_px
        
        self.regression_model = self._create_regression_model()
        self.is_trained = False
        
    def _create_regression_model(self):
        """创建回归模型"""
        if self.regression_model_type == 'ridge':
            return Ridge(alpha=1)
        elif self.regression_model_type == 'lasso':
            return Lasso(alpha=1)
        elif self.regression_model_type == 'lassocv':
            return MultiTaskLassoCV()
        elif self.regression_model_type == 'random_forest':
            return MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
        elif self.regression_model_type == 'gradient_boosting':
            return MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, random_state=42))
        elif self.regression_model_type == 'mlp':
            return MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        else:
            raise ValueError(f"Unsupported regression model type: {self.regression_model_type}")
    
    def load_model(self, model_path):
        """加载预训练模型 (可选)"""
        with open(model_path, 'rb') as f:
            self.regression_model = pickle.load(f)
            self.is_trained = True
    
    def load_calibration_data(self, data_folders):
        """
        从多个文件夹加载校准数据
        
        Args:
            data_folders: 校准数据文件夹路径列表
            
        Returns:
            all_data: 所有校准数据的列表
        """
        all_data = []
        
        for folder in data_folders:
            folder_path = Path(folder)
            
            # 查找train_data.jsonl文件
            jsonl_file = folder_path / 'train_data.jsonl'
            if jsonl_file.exists():
                print(f"Loading data from {jsonl_file}")
                with jsonlines.open(jsonl_file) as reader:
                    for item in reader:
                        all_data.append(item)
            else:
                print(f"Warning: {jsonl_file} not found")
                
        print(f"Total loaded {len(all_data)} data points")
        return all_data
    
    def data_dict_list_preparation_for_evaluation(self, data_dict_list):
        """
        准备用于评估的数据
        
        Args:
            data_dict_list: 数据字典列表
            
        Returns:
            X: 特征矩阵
        """
        if self.feature_type == 'basic':
            # 只使用基本特征: box, pitch, yaw
            X = [[*data['box'], data['pitch'], data['yaw']] for data in data_dict_list]
        elif self.feature_type == 'mediapipe':
            # 只使用mediapipe特征
            X = [[*np.array(data['mediapipe_results']).flatten()] for data in data_dict_list]
        elif self.feature_type == 'combined':
            # 使用所有特征: box, pitch, yaw, mediapipe
            X = [[*data['box'], data['pitch'], data['yaw'], *np.array(data['mediapipe_results']).flatten()] 
                 for data in data_dict_list]
        else:
            raise ValueError(f"Unsupported feature type: {self.feature_type}")
            
        return np.array(X)
    
    def train_model(self, train_data, test_data=None):
        """
        训练回归模型
        
        Args:
            train_data: 训练数据列表
            test_data: 测试数据列表 (可选，如果提供则计算测试得分)
            
        Returns:
            train_score, test_score: 训练集和测试集得分
        """
        # 准备训练数据
        X_train = self.data_dict_list_preparation_for_evaluation(train_data)
        y_train = np.array([[*data['label']] for data in train_data])
        
        print(f"Training data - Feature shape: {X_train.shape}, Target shape: {y_train.shape}")
        print(f"Using {self.feature_type} features with {self.regression_model_type} model")
        
        # 训练模型
        print("Training model...")
        self.regression_model.fit(X_train, y_train)
        self.is_trained = True
        
        # 计算训练得分
        train_score = self.regression_model.score(X_train, y_train)
        print(f"Training R² score: {train_score:.4f}")
        
        # 计算测试得分（如果提供测试数据）
        test_score = None
        if test_data is not None:
            X_test = self.data_dict_list_preparation_for_evaluation(test_data)
            y_test = np.array([[*data['label']] for data in test_data])
            test_score = self.regression_model.score(X_test, y_test)
            print(f"Testing data - Feature shape: {X_test.shape}, Target shape: {y_test.shape}")
            print(f"Testing R² score: {test_score:.4f}")
        
        return train_score, test_score

    def calculate_errors(self, calibration_data):
        """
        计算预测误差
        
        Args:
            calibration_data: 校准数据列表
            
        Returns:
            error_results: 包含误差统计的字典
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Please train the model first.")
        
        # 准备数据
        X = self.data_dict_list_preparation_for_evaluation(calibration_data)
        true_points = np.array([[*data['label']] for data in calibration_data])
        
        # 进行预测
        pred_points = self.regression_model.predict(X)
        pred_points[:,0]=np.clip(pred_points[:,0],0,self.screen_width_px)
        pred_points[:,1]=np.clip(pred_points[:,1],0,self.screen_height_px)
        # 计算像素误差
        errors_px = np.sqrt(np.sum((true_points - pred_points) ** 2, axis=1))
        mae_x_px = np.mean(np.abs(true_points[:, 0] - pred_points[:, 0]))
        mae_y_px = np.mean(np.abs(true_points[:, 1] - pred_points[:, 1]))
        med_px = np.mean(errors_px)
        
        # 计算毫米误差 - 先转换坐标再计算
        true_points_mm = np.array([[point[0] * self.px_to_mm_x, point[1] * self.px_to_mm_y] for point in true_points])
        pred_points_mm = np.array([[point[0] * self.px_to_mm_x, point[1] * self.px_to_mm_y] for point in pred_points])
        
        errors_mm = np.sqrt(np.sum((true_points_mm - pred_points_mm) ** 2, axis=1))
        mae_x_mm = np.mean(np.abs(true_points_mm[:, 0] - pred_points_mm[:, 0]))
        mae_y_mm = np.mean(np.abs(true_points_mm[:, 1] - pred_points_mm[:, 1]))
        med_mm = np.mean(errors_mm)
        
        # 获取唯一校准点的误差
        unique_true_points = np.unique(true_points, axis=0)
        point_errors_px = []
        point_errors_mm = []
        
        # 调试信息：检查数据范围
        print(f"Debug: True points range - X: [{np.min(true_points[:, 0]):.1f}, {np.max(true_points[:, 0]):.1f}], Y: [{np.min(true_points[:, 1]):.1f}, {np.max(true_points[:, 1]):.1f}]")
        print(f"Debug: Pred points range - X: [{np.min(pred_points[:, 0]):.1f}, {np.max(pred_points[:, 0]):.1f}], Y: [{np.min(pred_points[:, 1]):.1f}, {np.max(pred_points[:, 1]):.1f}]")
        print(f"Debug: Found {len(unique_true_points)} unique calibration points")
        
        for i, true_point in enumerate(unique_true_points):
            mask = np.all(true_points == true_point, axis=1)
            corresponding_preds = pred_points[mask]
            
            # 像素误差 - 欧几里得距离
            distances_px = np.sqrt(np.sum((corresponding_preds - true_point) ** 2, axis=1))
            point_med_px = np.mean(distances_px)
            
            # 毫米误差 - 先转换坐标再计算距离（正确做法）
            true_point_mm = np.array([true_point[0] * self.px_to_mm_x, true_point[1] * self.px_to_mm_y])
            corresponding_preds_mm = np.array([[pred[0] * self.px_to_mm_x, pred[1] * self.px_to_mm_y] 
                                             for pred in corresponding_preds])
            distances_mm = np.sqrt(np.sum((corresponding_preds_mm - true_point_mm) ** 2, axis=1))
            point_med_mm = np.mean(distances_mm)
            
            # 检查前几个点的详细信息
            if i < 3:
                print(f"Debug Point {i}: True={true_point}, Num_predictions={len(corresponding_preds)}")
                print(f"  Sample predictions: {corresponding_preds[:2] if len(corresponding_preds) >= 2 else corresponding_preds}")
                print(f"  Distances (px): {distances_px[:3] if len(distances_px) >= 3 else distances_px}")
                print(f"  Mean distance (px): {point_med_px:.3f}")
                print(f"  Distances (mm): {distances_mm[:3] if len(distances_mm) >= 3 else distances_mm}")
                print(f"  Mean distance (mm): {point_med_mm:.3f}")
            
            assert point_med_px >= 0, f"Pixel error is negative: {point_med_px}"
            assert point_med_mm >= 0, f"MM error is negative: {point_med_mm}"
            
            point_errors_px.append((true_point[0], true_point[1], point_med_px))
            point_errors_mm.append((true_point[0], true_point[1], point_med_mm))
        
        # 验证所有计算出的误差
        px_errors_array = np.array(point_errors_px)
        mm_errors_array = np.array(point_errors_mm)
        print(f"Debug: Final px errors range: [{np.min(px_errors_array[:, 2]):.3f}, {np.max(px_errors_array[:, 2]):.3f}]")
        print(f"Debug: Final mm errors range: [{np.min(mm_errors_array[:, 2]):.3f}, {np.max(mm_errors_array[:, 2]):.3f}]")
        
        # 检查是否有异常值
        if np.any(px_errors_array[:, 2] < 0):
            print("ERROR: Found negative pixel errors!")
            negative_indices = np.where(px_errors_array[:, 2] < 0)[0]
            for idx in negative_indices:
                print(f"  Negative error at point {px_errors_array[idx]}")
        
        if np.any(mm_errors_array[:, 2] < 0):
            print("ERROR: Found negative mm errors!")
            negative_indices = np.where(mm_errors_array[:, 2] < 0)[0]
            for idx in negative_indices:
                print(f"  Negative error at point {mm_errors_array[idx]}")
        
        return {
            'statistics_px': {
                'mae_x': float(mae_x_px),
                'mae_y': float(mae_y_px),
                'med': float(med_px),
                'std': float(np.std(errors_px)),
                'max': float(np.max(errors_px)),
                'min': float(np.min(errors_px))
            },
            'statistics_mm': {
                'mae_x': float(mae_x_mm),
                'mae_y': float(mae_y_mm),
                'med': float(med_mm),
                'std': float(np.std(errors_mm)),
                'max': float(np.max(errors_mm)),
                'min': float(np.min(errors_mm))
            },
            'point_data': {
                'true_points': true_points.tolist(),
                'pred_points': pred_points.tolist(),
            },
            'point_errors_px': np.array(point_errors_px),
            'point_errors_mm': np.array(point_errors_mm),
            'px_to_mm_x': self.px_to_mm_x,
            'px_to_mm_y': self.px_to_mm_y
        }
    
    def create_error_map(self, error_results, output_dir, unit='px'):
        """
        创建误差分布图
        
        Args:
            error_results: 误差结果字典
            output_dir: 输出目录
            unit: 'px' 或 'mm'
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if unit == 'px':
            point_errors = error_results['point_errors_px']
            statistics = error_results['statistics_px']
            unit_label = 'Error (pixels)'
            xlabel = 'Screen X Position (pixels)'
            ylabel = 'Screen Y Position (pixels)'
            title = f'Gaze Prediction Error Distribution (Pixels)\nMean Error: {statistics["med"]:.2f}px'
        else:
            point_errors = error_results['point_errors_mm']
            statistics = error_results['statistics_mm']
            unit_label = 'Error (mm)'
            xlabel = 'Screen X Position (mm)'
            ylabel = 'Screen Y Position (mm)'
            title = f'Gaze Prediction Error Distribution (Millimeters)\nMean Error: {statistics["med"]:.2f}mm'
        
        # 创建可视化
        plt.figure(figsize=(15, 10))
        
        # 创建网格点进行插值
        grid_x, grid_y = np.mgrid[0:self.screen_width_px:100j, 0:self.screen_height_px:100j]
        
        # 使用径向基函数进行插值，设置smooth=0确保在原始点上精确匹配
        rbf = Rbf(point_errors[:, 0], point_errors[:, 1], point_errors[:, 2], 
                 function='multiquadric', smooth=0)
        grid_z = rbf(grid_x, grid_y)
        
        # 验证插值在原始点上的精度
        interpolated_at_points = rbf(point_errors[:, 0], point_errors[:, 1])
        max_interpolation_error = np.max(np.abs(interpolated_at_points - point_errors[:, 2]))
        print(f"Debug: Max interpolation error at original points: {max_interpolation_error:.6f}")
        
        # 确保误差值非负（处理插值可能产生的小负值）
        grid_z = np.maximum(grid_z, 0)
        
        # 绘制填充的彩色等高线图
        contourf = plt.contourf(grid_x, grid_y, grid_z,
                              levels=15,
                              cmap='YlOrRd',
                              alpha=0.7)
        
        # 添加颜色条
        plt.colorbar(contourf, label=unit_label)
        
        # 叠加等高线
        contour = plt.contour(grid_x, grid_y, grid_z, 
                            levels=10, 
                            colors='k',
                            alpha=0.3,
                            linewidths=0.8)
        plt.clabel(contour, inline=True, fontsize=8, fmt='%.1f')
        
        # 添加校准点
        plt.scatter(point_errors[:, 0], point_errors[:, 1], 
                   c='red', s=50, alpha=0.8, marker='o', 
                   label='Calibration Points')
        
        # plt.title(title, fontsize=14, pad=20)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.xlim(0, self.screen_width_px)
        plt.ylim(0, self.screen_height_px)
        plt.grid(True, alpha=0.2)
        plt.legend()
        
        # 保存图像
        plt.savefig(output_dir / f'gaze_error_distribution_{unit}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 创建统计条形图
        self.create_statistics_plot(statistics, output_dir, unit)
    
    def create_statistics_plot(self, statistics, output_dir, unit):
        """创建统计数据条形图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 误差统计
        metrics = ['mae_x', 'mae_y', 'med', 'std', 'max', 'min']
        values = [statistics[m] for m in metrics]
        
        ax1.bar(metrics, values, color=['skyblue', 'lightcoral', 'gold', 'lightgreen', 'salmon', 'plum'])
        ax1.set_title(f'Error Statistics ({unit})', fontsize=14)
        ax1.set_ylabel(f'Error ({unit})', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for i, v in enumerate(values):
            ax1.text(i, v + max(values) * 0.01, f'{v:.2f}', ha='center', va='bottom')
        
        # 屏幕分辨率信息
        screen_info = [
            f'Screen Size: {self.screen_width_px}×{self.screen_height_px}px',
            f'Physical Size: {self.screen_width_mm:.1f}×{self.screen_height_mm:.1f}mm',
            f'Pixel Density: {self.screen_width_px/self.screen_width_mm*25.4:.1f} PPI',
            f'Conversion: {self.px_to_mm_x:.4f}mm/px (X)',
            f'Conversion: {self.px_to_mm_y:.4f}mm/px (Y)'
        ]
        
        ax2.text(0.1, 0.9, '\n'.join(screen_info), transform=ax2.transAxes, 
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        ax2.set_title('Screen Information', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'error_statistics_{unit}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, error_results, output_dir, train_scores=None):
        """保存误差结果到JSON文件"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 准备保存的数据
        save_data = {
            'model_info': {
                'regression_model_type': self.regression_model_type,
                'feature_type': self.feature_type,
                'train_score': train_scores[0] if train_scores else None,
                'test_score': train_scores[1] if train_scores else None,
            },
            'screen_info': {
                'width_px': self.screen_width_px,
                'height_px': self.screen_height_px,
                'width_mm': self.screen_width_mm,
                'height_mm': self.screen_height_mm,
                'px_to_mm_x': self.px_to_mm_x,
                'px_to_mm_y': self.px_to_mm_y
            },
            'statistics_px': error_results['statistics_px'],
            'statistics_mm': error_results['statistics_mm'],
            'point_data': error_results['point_data'],
            'point_errors_px': error_results['point_errors_px'].tolist(),
            'point_errors_mm': error_results['point_errors_mm'].tolist()
        }
        
        # 保存到JSON文件
        with open(output_dir / 'error_analysis_results.json', 'w') as f:
            json.dump(save_data, f, indent=4)
            
        # 保存训练好的模型
        if self.is_trained:
            with open(output_dir / 'trained_model.pkl', 'wb') as f:
                pickle.dump(self.regression_model, f)
        
        print(f"Results saved to {output_dir}")
    
    def analyze_kalman_filtering(self, calibration_data, output_dir, 
                                dt=0.04, kalman_filter_std_measurement=2.0, Q_coef=0.005,
                                num_predictions_per_point=15):
        """
        分析卡尔曼滤波对每个校准点的影响
        
        Args:
            calibration_data: 校准数据列表
            output_dir: 输出目录
            dt: 时间间隔
            kalman_filter_std_measurement: 卡尔曼滤波测量噪声标准差
            Q_coef: 过程噪声系数
            num_predictions_per_point: 每个点的预测次数
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Please train the model first.")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("Analyzing Kalman filtering effects...")
        
        # 准备数据
        X = self.data_dict_list_preparation_for_evaluation(calibration_data)
        true_points = np.array([[*data['label']] for data in calibration_data])
        
        # 获取唯一的校准点
        unique_true_points = np.unique(true_points, axis=0)
        
        # 存储所有点的卡尔曼滤波轨迹和误差
        all_kalman_trajectories = {}
        all_error_sequences = []
        
        print(f"Analyzing {len(unique_true_points)} unique calibration points...")
        
        for point_idx, true_point in enumerate(unique_true_points):
            # 找到属于这个点的所有数据
            mask = np.all(true_points == true_point, axis=1)
            point_data_indices = np.where(mask)[0]
            
            
            # 初始化卡尔曼滤波器
            kalman_filter = GazeKalmanFilter(dt, kalman_filter_std_measurement, Q_coef)
            
            # 存储这个点的轨迹
            raw_predictions = []
            filtered_predictions = []
            errors_raw = []
            errors_filtered = []
            
            for pred_idx, data_idx in enumerate(point_data_indices):
                # 进行原始预测
                raw_pred = self.regression_model.predict(X[data_idx:data_idx+1])[0]
                raw_pred = np.clip(raw_pred, [0, 0], [self.screen_width_px, self.screen_height_px])
                
                # 应用卡尔曼滤波
                filtered_pred = kalman_filter.smooth_position(raw_pred)
                
                # 计算误差
                error_raw = np.sqrt(np.sum((true_point - raw_pred) ** 2))
                error_filtered = np.sqrt(np.sum((true_point - filtered_pred) ** 2))
                
                raw_predictions.append(raw_pred)
                filtered_predictions.append(filtered_pred)
                errors_raw.append(error_raw)
                errors_filtered.append(error_filtered)
            
            all_kalman_trajectories[point_idx] = {
                'true_point': true_point,
                'raw_predictions': np.array(raw_predictions),
                'filtered_predictions': np.array(filtered_predictions),
                'errors_raw': np.array(errors_raw),
                'errors_filtered': np.array(errors_filtered)
            }
            
            all_error_sequences.append({
                'raw_errors': errors_raw,
                'filtered_errors': errors_filtered
            })
        
        # 创建可视化
        self.create_kalman_visualization(all_kalman_trajectories, all_error_sequences, 
                                       output_dir, num_predictions_per_point)
        
        # 保存分析结果
        self.save_kalman_analysis(all_kalman_trajectories, all_error_sequences, output_dir)
        
        print(f"Kalman filtering analysis saved to {output_dir}")
    
    def create_kalman_visualization(self, trajectories, error_sequences, output_dir, num_predictions):
        """创建卡尔曼滤波可视化图表"""
        
        # 图1: 所有点的卡尔曼滤波轨迹在同一画布上
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        fig.suptitle('Kalman Filter Trajectories for All Calibration Points', fontsize=16, fontweight='bold')
        
        # 绘制所有真实点
        for point_idx, (idx, trajectory) in enumerate(trajectories.items()):
            true_point = trajectory['true_point']
            raw_preds = trajectory['raw_predictions']
            filtered_preds = trajectory['filtered_predictions']
            
            # 绘制真实点（绿色）
            if point_idx == 0:
                ax.plot(true_point[0], true_point[1], 'go', markersize=10, 
                       label='True Points', markeredgecolor='black', markeredgewidth=1)
            else:
                ax.plot(true_point[0], true_point[1], 'go', markersize=10, 
                       markeredgecolor='black', markeredgewidth=1)
            
            # 绘制原始预测点（红色，透明度递增）
            for i, pred in enumerate(raw_preds):
                alpha = 0.1 + (i / len(raw_preds)) * 0.8  # 透明度从0.1到0.9
                if point_idx == 0 and i == 0:
                    ax.plot(pred[0], pred[1], 'ro', alpha=alpha, markersize=4, label='Raw Predictions')
                else:
                    ax.plot(pred[0], pred[1], 'ro', alpha=alpha, markersize=4)
            
            # 绘制轨迹（蓝色线条）
            # if point_idx == 0:
            #     ax.plot(filtered_preds[:, 0], filtered_preds[:, 1], 'b-', alpha=0.1, linewidth=1, label='Prediction Trajectories')
            # else:
            ax.plot(filtered_preds[:, 0], filtered_preds[:, 1], 'b-', alpha=0.1, linewidth=1)
        
        ax.set_xlabel('Screen X Position (pixels)', fontsize=12)
        ax.set_ylabel('Screen Y Position (pixels)', fontsize=12)
        ax.set_xlim(0, self.screen_width_px)
        ax.set_ylim(0, self.screen_height_px)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        ax.invert_yaxis()  # 反转Y轴以匹配屏幕坐标系
        
        plt.tight_layout()
        plt.savefig(output_dir / 'kalman_trajectories.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 图2: 误差随时间变化统计（以毫米为单位）
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle('Error Evolution with Kalman Filtering', fontsize=16, fontweight='bold')
        
        # 计算所有点的平均误差（转换为毫米）
        time_steps = list(range(1, num_predictions + 1))
        avg_raw_errors_mm = []
        avg_filtered_errors_mm = []
        std_raw_errors_mm = []
        std_filtered_errors_mm = []
        
        for step in range(num_predictions):
            raw_errors_at_step = [seq['raw_errors'][step] for seq in error_sequences]
            filtered_errors_at_step = [seq['filtered_errors'][step] for seq in error_sequences]
            
            # 转换为毫米
            raw_errors_mm = [err * self.px_to_mm_x for err in raw_errors_at_step]  # 使用x方向的转换率
            filtered_errors_mm = [err * self.px_to_mm_x for err in filtered_errors_at_step]
            
            avg_raw_errors_mm.append(np.mean(raw_errors_mm))
            avg_filtered_errors_mm.append(np.mean(filtered_errors_mm))
            std_raw_errors_mm.append(np.std(raw_errors_mm))
            std_filtered_errors_mm.append(np.std(filtered_errors_mm))
        
        # 绘制平均误差
        ax1.plot(time_steps, avg_raw_errors_mm, 'r-', linewidth=2, label='Raw Predictions', marker='o')
        ax1.plot(time_steps, avg_filtered_errors_mm, 'b-', linewidth=2, label='Kalman Filtered', marker='s')
        
        # 添加误差带
        ax1.fill_between(time_steps, 
                        np.array(avg_raw_errors_mm) - np.array(std_raw_errors_mm),
                        np.array(avg_raw_errors_mm) + np.array(std_raw_errors_mm),
                        alpha=0.2, color='red')
        ax1.fill_between(time_steps,
                        np.array(avg_filtered_errors_mm) - np.array(std_filtered_errors_mm),
                        np.array(avg_filtered_errors_mm) + np.array(std_filtered_errors_mm),
                        alpha=0.2, color='blue')
        
        ax1.set_xlabel('Prediction Number')
        ax1.set_ylabel('Average Error (mm)')
        ax1.set_title('Average Error Evolution Across All Points')
        ax1.set_xticks(time_steps)  # 确保横坐标都是整数
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 绘制误差改善百分比
        improvement_percent = [(raw - filt) / raw * 100 for raw, filt in zip(avg_raw_errors_mm, avg_filtered_errors_mm)]
        
        ax2.plot(time_steps, improvement_percent, 'g-', linewidth=2, marker='^')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Prediction Number')
        ax2.set_ylabel('Error Improvement (%)')
        ax2.set_title('Kalman Filter Error Improvement Over Time')
        ax2.set_xticks(time_steps)  # 确保横坐标都是整数
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'kalman_error_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 图3: 累积误差分布对比
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Error Distribution Comparison', fontsize=16, fontweight='bold')
        
        # 收集所有误差
        all_raw_errors = []
        all_filtered_errors = []
        for seq in error_sequences:
            all_raw_errors.extend(seq['raw_errors'])
            all_filtered_errors.extend(seq['filtered_errors'])
        
        # 绘制误差直方图
        ax1.hist(all_raw_errors, bins=30, alpha=0.7, color='red', label='Raw Predictions', density=True)
        ax1.hist(all_filtered_errors, bins=30, alpha=0.7, color='blue', label='Kalman Filtered', density=True)
        ax1.set_xlabel('Error (pixels)')
        ax1.set_ylabel('Density')
        ax1.set_title('Error Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 绘制累积分布函数
        sorted_raw = np.sort(all_raw_errors)
        sorted_filtered = np.sort(all_filtered_errors)
        y_raw = np.arange(1, len(sorted_raw) + 1) / len(sorted_raw)
        y_filtered = np.arange(1, len(sorted_filtered) + 1) / len(sorted_filtered)
        
        ax2.plot(sorted_raw, y_raw, 'r-', linewidth=2, label='Raw Predictions')
        ax2.plot(sorted_filtered, y_filtered, 'b-', linewidth=2, label='Kalman Filtered')
        ax2.set_xlabel('Error (pixels)')
        ax2.set_ylabel('Cumulative Probability')
        ax2.set_title('Cumulative Error Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'kalman_error_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_kalman_analysis(self, trajectories, error_sequences, output_dir):
        """保存卡尔曼滤波分析结果"""
        # 计算统计信息
        all_raw_errors = []
        all_filtered_errors = []
        
        for seq in error_sequences:
            all_raw_errors.extend(seq['raw_errors'])
            all_filtered_errors.extend(seq['filtered_errors'])
        
        analysis_results = {
            'kalman_config': {
                'dt': 0.04,
                'kalman_filter_std_measurement': 4.0,
                'Q_coef': 0.01
            },
            'statistics': {
                'raw_predictions': {
                    'mean_error': float(np.mean(all_raw_errors)),
                    'std_error': float(np.std(all_raw_errors)),
                    'median_error': float(np.median(all_raw_errors)),
                    'max_error': float(np.max(all_raw_errors)),
                    'min_error': float(np.min(all_raw_errors))
                },
                'filtered_predictions': {
                    'mean_error': float(np.mean(all_filtered_errors)),
                    'std_error': float(np.std(all_filtered_errors)),
                    'median_error': float(np.median(all_filtered_errors)),
                    'max_error': float(np.max(all_filtered_errors)),
                    'min_error': float(np.min(all_filtered_errors))
                },
                'improvement': {
                    'mean_error_reduction': float(np.mean(all_raw_errors) - np.mean(all_filtered_errors)),
                    'improvement_percentage': float((np.mean(all_raw_errors) - np.mean(all_filtered_errors)) / np.mean(all_raw_errors) * 100)
                }
            },
            'num_points': len(trajectories),
            'num_predictions_per_point': len(error_sequences[0]['raw_errors']) if error_sequences else 0
        }
        
        # 保存到JSON文件
        with open(output_dir / 'kalman_analysis_results.json', 'w') as f:
            json.dump(analysis_results, f, indent=4)
        
        # 打印摘要
        print("\n" + "="*60)
        print("KALMAN FILTERING ANALYSIS SUMMARY")
        print("="*60)
        print(f"Number of calibration points: {analysis_results['num_points']}")
        print(f"Predictions per point: {analysis_results['num_predictions_per_point']}")
        print(f"\nRaw Predictions:")
        print(f"  Mean Error: {analysis_results['statistics']['raw_predictions']['mean_error']:.2f}px")
        print(f"  Std Error: {analysis_results['statistics']['raw_predictions']['std_error']:.2f}px")
        print(f"\nKalman Filtered:")
        print(f"  Mean Error: {analysis_results['statistics']['filtered_predictions']['mean_error']:.2f}px")
        print(f"  Std Error: {analysis_results['statistics']['filtered_predictions']['std_error']:.2f}px")
        print(f"\nImprovement:")
        print(f"  Error Reduction: {analysis_results['statistics']['improvement']['mean_error_reduction']:.2f}px")
        print(f"  Improvement: {analysis_results['statistics']['improvement']['improvement_percentage']:.1f}%")
        print("="*60)
    
    def print_summary(self, error_results, train_scores=None, evaluation_type="training"):
        """打印误差统计摘要"""
        stats_px = error_results['statistics_px']
        stats_mm = error_results['statistics_mm']
        
        print("\n" + "="*60)
        print("GAZE TRACKING ERROR ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"\nModel Configuration:")
        print(f"  Regression Model: {self.regression_model_type}")
        print(f"  Feature Type: {self.feature_type}")
        if train_scores:
            print(f"  Training R² Score: {train_scores[0]:.4f}")
            if train_scores[1] is not None:
                print(f"  Cross-session Testing R² Score: {train_scores[1]:.4f}")
        
        print(f"\nEvaluation Setup:")
        print(f"  Error calculated on: {evaluation_type} data")
        if evaluation_type == "test":
            print("  ✓ Cross-session evaluation (training on one session, testing on another)")
        else:
            print("  ⚠ Same-session evaluation (training and testing on same data)")
        
        print(f"\nScreen Configuration:")
        print(f"  Resolution: {self.screen_width_px} × {self.screen_height_px} pixels")
        print(f"  Physical Size: {self.screen_width_mm:.1f} × {self.screen_height_mm:.1f} mm")
        print(f"  Pixel Density: {self.screen_width_px/self.screen_width_mm*25.4:.1f} PPI")
        
        print(f"\nPixel Errors ({evaluation_type} data):")
        print(f"  Mean Error (MED): {stats_px['med']:.2f} px")
        print(f"  X-axis MAE: {stats_px['mae_x']:.2f} px")
        print(f"  Y-axis MAE: {stats_px['mae_y']:.2f} px")
        print(f"  Standard Deviation: {stats_px['std']:.2f} px")
        print(f"  Max Error: {stats_px['max']:.2f} px")
        print(f"  Min Error: {stats_px['min']:.2f} px")
        
        print(f"\nMillimeter Errors ({evaluation_type} data):")
        print(f"  Mean Error (MED): {stats_mm['med']:.2f} mm")
        print(f"  X-axis MAE: {stats_mm['mae_x']:.2f} mm")
        print(f"  Y-axis MAE: {stats_mm['mae_y']:.2f} mm")
        print(f"  Standard Deviation: {stats_mm['std']:.2f} mm")
        print(f"  Max Error: {stats_mm['max']:.2f} mm")
        print(f"  Min Error: {stats_mm['min']:.2f} mm")
        
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Calculate gaze tracking error maps')
    parser.add_argument('--train_folders', nargs='+', required=True,
                       help='Paths to training calibration data folders')
    parser.add_argument('--test_folders', nargs='*', default=None,
                       help='Paths to testing calibration data folders (optional, if not provided will evaluate on training data)')
    parser.add_argument('--output_dir', required=True,
                       help='Directory to save results and plots')
    parser.add_argument('--regression_model', type=str, default='lassocv',
                       choices=['ridge', 'lasso', 'lassocv', 'random_forest', 'gradient_boosting', 'mlp'],
                       help='Regression model type (default: lassocv)')
    parser.add_argument('--feature_type', type=str, default='basic',
                       choices=['basic', 'mediapipe', 'combined'],
                       help='Feature type: basic (box+pitch+yaw), mediapipe (landmark points), combined (all) (default: basic)')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to pre-trained model (optional, if not provided will train new model)')
    parser.add_argument('--screen_width_px', type=int, default=3072,
                       help='Screen width in pixels (default: 3072)')
    parser.add_argument('--screen_height_px', type=int, default=1920,
                       help='Screen height in pixels (default: 1920)')
    parser.add_argument('--screen_width_mm', type=float, default=344.6,
                       help='Screen width in millimeters (default: 344.6)')
    parser.add_argument('--screen_height_mm', type=float, default=215.4,
                       help='Screen height in millimeters (default: 215.4)')
    
    args = parser.parse_args()
    
    # 创建误差计算器
    calculator = ErrorMapCalculator(
        screen_width_px=args.screen_width_px,
        screen_height_px=args.screen_height_px,
        screen_width_mm=args.screen_width_mm,
        screen_height_mm=args.screen_height_mm,
        regression_model_type=args.regression_model,
        feature_type=args.feature_type
    )
    
    # 加载训练数据
    print("Loading training data...")
    train_data = calculator.load_calibration_data(args.train_folders)
    
    if not train_data:
        print("No training data found!")
        return
    
    # 加载测试数据（如果提供）
    test_data = None
    evaluation_data = train_data  # 默认在训练数据上评估误差
    
    if args.test_folders:
        print("Loading testing data...")
        test_data = calculator.load_calibration_data(args.test_folders)
        if test_data:
            evaluation_data = test_data  # 如果有测试数据，在测试数据上评估误差
            print(f"Will evaluate errors on {len(test_data)} test samples")
        else:
            print("Warning: No test data found, will evaluate on training data")
    else:
        print("No test folders provided, will evaluate on training data")
    
    # 训练或加载模型
    train_scores = None
    if args.model_path:
        print(f"Loading pre-trained model from {args.model_path}")
        calculator.load_model(args.model_path)
    else:
        print("Training new model from data...")
        train_scores = calculator.train_model(train_data, test_data)
    
    # 计算误差（在评估数据上）
    print(f"Calculating errors on {'test' if test_data else 'training'} data...")
    error_results = calculator.calculate_errors(evaluation_data)
    
    # 创建误差分布图（像素和毫米）
    print("Creating error maps...")
    calculator.create_error_map(error_results, args.output_dir, unit='px')
    calculator.create_error_map(error_results, args.output_dir, unit='mm')
    
    # 保存结果
    calculator.save_results(error_results, args.output_dir, train_scores)
    
    # 打印摘要
    evaluation_type = "test" if test_data else "training"
    calculator.print_summary(error_results, train_scores, evaluation_type)
    
    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
