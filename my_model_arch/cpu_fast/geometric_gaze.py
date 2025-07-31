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

#!/usr/bin/env python3
"""
几何凝视映射系统 - 完整参数估计实现
基于双眼到Cyclopean eye的参数化建模

核心特点:
1. Cyclopean eye位置参数估计（不使用固定权重）
2. 屏幕-摄像头非平行配置支持
3. 像素纵横比处理
4. 数值稳定的参数优化
5. 避免病态矩阵设计
6. 物理意义明确的参数

作者: AI Assistant
日期: 2024年
"""

import numpy as np
import scipy.optimize
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass, asdict
import warnings
import json

@dataclass
class GeometricParameters:
    """几何模型参数 - 每个参数都有明确物理意义"""
    
    # Cyclopean eye权重参数
    w_dominant: float = 0.5      # 眼部权重 [0.0=左眼主导, 1.0=右眼主导, 0.5=平均]
    
    # 深度缩放参数
    alpha: float = 1.0           # MediaPipe深度缩放因子
    
    # 角度校正参数
    delta_yaw: float = 0.0       # yaw角系统偏差 (弧度)
    delta_pitch: float = 0.0     # pitch角系统偏差 (弧度)
    k_yaw: float = 1.0           # yaw角线性校正系数
    k_pitch: float = 1.0         # pitch角线性校正系数
    
    # 眼球位置偏移参数
    offset_x: float = 0.0        # X方向偏移 (mm)
    offset_y: float = 0.0        # Y方向偏移 (mm) 
    offset_z: float = 0.0        # Z方向偏移 (mm)
    
    # 屏幕-摄像头相对位置和角度参数
    screen_distance: float = 600.0   # 屏幕距离 (mm)
    pixel_per_mm_x: float = 3.8      # X方向像素密度 (像素/mm)
    pixel_per_mm_y: float = 3.8      # Y方向像素密度 (像素/mm)
    screen_tilt_x: float = 0.0       # 屏幕绕X轴倾斜角度 (弧度)
    screen_tilt_y: float = 0.0       # 屏幕绕Y轴倾斜角度 (弧度)
    screen_offset_x: float = 0.0     # 屏幕中心X偏移 (mm)
    screen_offset_y: float = 0.0     # 屏幕中心Y偏移 (mm)
    
    def to_normalized_vector(self) -> Tuple[np.ndarray, Dict[str, Tuple[float, float, float]]]:
        """
        转换为归一化优化向量，避免数值优化问题
        
        Returns:
            normalized_vector: 归一化参数向量 [所有值在-1到1之间]
            scaling_info: 缩放信息字典 {param_name: (min_val, max_val, current_val)}
        """
        # 定义每个参数的合理范围和当前值
        param_ranges = {
            'w_dominant': (0.0, 1.0, self.w_dominant),          # 0=左眼, 1=右眼
            'alpha': (0.5, 2.0, self.alpha),
            'delta_yaw': (-0.3, 0.3, self.delta_yaw),
            'delta_pitch': (-0.3, 0.3, self.delta_pitch),
            'k_yaw': (0.7, 1.3, self.k_yaw),
            'k_pitch': (0.7, 1.3, self.k_pitch),
            'offset_x': (-100.0, 100.0, self.offset_x),
            'offset_y': (-100.0, 100.0, self.offset_y),
            'offset_z': (-200.0, 200.0, self.offset_z),
            'screen_distance': (400.0, 1000.0, self.screen_distance),
            'pixel_per_mm_x': (1.0, 10.0, self.pixel_per_mm_x), # 像素密度范围
            'pixel_per_mm_y': (1.0, 10.0, self.pixel_per_mm_y),
            'screen_tilt_x': (-0.5, 0.5, self.screen_tilt_x),
            'screen_tilt_y': (-0.5, 0.5, self.screen_tilt_y),
            'screen_offset_x': (-200.0, 200.0, self.screen_offset_x),
            'screen_offset_y': (-200.0, 200.0, self.screen_offset_y),
        }
        
        normalized_vector = []
        scaling_info = {}
        
        for param_name, (min_val, max_val, current_val) in param_ranges.items():
            # 归一化到 [-1, 1] 区间
            normalized_val = 2.0 * (current_val - min_val) / (max_val - min_val) - 1.0
            normalized_vector.append(normalized_val)
            scaling_info[param_name] = (min_val, max_val, current_val)
        
        return np.array(normalized_vector), scaling_info
    
    @classmethod
    def from_normalized_vector(cls, normalized_vec: np.ndarray, scaling_info: Dict[str, Tuple[float, float, float]]) -> 'GeometricParameters':
        """
        从归一化向量恢复参数对象
        
        Args:
            normalized_vec: 归一化参数向量
            scaling_info: 缩放信息
        
        Returns:
            params: 参数对象
        """
        params = cls()
        param_names = [
            'w_dominant', 'alpha', 'delta_yaw', 'delta_pitch', 
            'k_yaw', 'k_pitch', 'offset_x', 'offset_y', 'offset_z',
            'screen_distance', 'pixel_per_mm_x', 'pixel_per_mm_y',
            'screen_tilt_x', 'screen_tilt_y', 'screen_offset_x', 'screen_offset_y'
        ]
        
        for i, param_name in enumerate(param_names):
            min_val, max_val, _ = scaling_info[param_name]
            # 从 [-1, 1] 恢复到原始范围
            normalized_val = normalized_vec[i]
            original_val = min_val + (max_val - min_val) * (normalized_val + 1.0) / 2.0
            setattr(params, param_name, original_val)
        
        return params
    
    # 保留旧方法以兼容性
    def to_vector(self) -> np.ndarray:
        """转换为优化向量 (保留兼容性)"""
        normalized_vec, _ = self.to_normalized_vector()
        return normalized_vec
    
    @classmethod
    def from_vector(cls, vec: np.ndarray, keep_screen_size: bool = True) -> 'GeometricParameters':
        """从优化向量创建参数对象 (保留兼容性)"""
        # 这个方法现在需要scaling_info，但为了兼容性暂时保留
        # 实际使用中应该用 from_normalized_vector
        params = cls()
        if len(vec) >= 16:  # 更新参数数量
            params.w_dominant = vec[0]
            params.alpha = vec[1]
            params.delta_yaw = vec[2]
            params.delta_pitch = vec[3]
            params.k_yaw = vec[4]
            params.k_pitch = vec[5]
            params.offset_x = vec[6]
            params.offset_y = vec[7]
            params.offset_z = vec[8]
            params.screen_distance = vec[9]
            params.pixel_per_mm_x = vec[10]
            params.pixel_per_mm_y = vec[11]
            params.screen_tilt_x = vec[12]
            params.screen_tilt_y = vec[13]
            params.screen_offset_x = vec[14]
            params.screen_offset_y = vec[15]
        return params

class GeometricGazeMapper:
    """几何凝视映射器"""
    
    def __init__(self, screen_width_px: int = 1920, screen_height_px: int = 1080):
        self.screen_width_px = screen_width_px
        self.screen_height_px = screen_height_px
        self.pixel_aspect_ratio = screen_width_px / screen_height_px  # 像素纵横比
        self.params = GeometricParameters()
        self.is_calibrated = False
        self.scaling_info = None  # 存储参数缩放信息
        
        # MediaPipe眼部特征点索引 (468个特征点中的眼部点)
        self.left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # 归一化参数约束（所有参数都在 [-1, 1] 范围内）
        self.normalized_bounds = [(-1.0, 1.0)] * 16  # 16个参数，都在[-1,1]范围
    
    def extract_eye_landmarks_from_mediapipe(self, mediapipe_results: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        从MediaPipe结果中提取左右眼特征点
        
        Args:
            mediapipe_results: MediaPipe面部特征点结果 (468*3 = 1404个值的扁平列表)
        
        Returns:
            left_eye_center, right_eye_center: 左右眼中心坐标 [x, y, z] (归一化)
        """
        # 重塑为 (468, 3) 格式
        landmarks = np.array(mediapipe_results).reshape(-1, 3)
        
        # 提取左右眼特征点
        left_eye_points = landmarks[self.left_eye_indices]
        right_eye_points = landmarks[self.right_eye_indices]
        
        # 计算眼部中心（取平均值）
        left_eye_center = np.mean(left_eye_points, axis=0)
        right_eye_center = np.mean(right_eye_points, axis=0)
        
        return left_eye_center, right_eye_center
    
    def compute_cyclopean_eye_position(self, 
                                     left_eye_center: np.ndarray,
                                     right_eye_center: np.ndarray) -> np.ndarray:
        """
        计算Cyclopean eye 3D位置
        
        Args:
            left_eye_center: 左眼中心MediaPipe坐标 [x, y, z] (归一化)
            right_eye_center: 右眼中心MediaPipe坐标 [x, y, z] (归一化)
        
        Returns:
            cyclopean_pos: Cyclopean eye 3D位置 (mm)
        """
        # 1. 转换到3D空间 (mm)
        left_3d = self.params.alpha * left_eye_center * 1000
        right_3d = self.params.alpha * right_eye_center * 1000
        
        # 2. 计算加权平均Cyclopean位置
        # w_dominant: 0.0=完全左眼, 1.0=完全右眼, 0.5=平均
        w_right = self.params.w_dominant  # 右眼权重
        w_left = 1.0 - w_right            # 左眼权重
        
        cyclopean_pos = w_left * left_3d + w_right * right_3d
        
        # 3. 应用位置偏移校正
        offset = np.array([self.params.offset_x, 
                          self.params.offset_y, 
                          self.params.offset_z])
        
        return cyclopean_pos + offset
    
    def correct_gaze_angles(self, gaze_yaw: float, gaze_pitch: float) -> Tuple[float, float]:
        """
        校正凝视角度
        
        Args:
            gaze_yaw: 原始yaw角 (弧度)
            gaze_pitch: 原始pitch角 (弧度)
        
        Returns:
            corrected_yaw, corrected_pitch: 校正后角度 (弧度)
        """
        # 线性校正 + 偏差校正
        corrected_yaw = self.params.k_yaw * gaze_yaw + self.params.delta_yaw
        corrected_pitch = self.params.k_pitch * gaze_pitch + self.params.delta_pitch
        
        return corrected_yaw, corrected_pitch
    
    def get_screen_transform_matrix(self) -> np.ndarray:
        """
        获取屏幕坐标系变换矩阵
        考虑屏幕相对摄像头的位置和角度
        
        Returns:
            transform_matrix: 4x4变换矩阵
        """
        # 1. 平移矩阵（屏幕中心位置）
        T = np.eye(4)
        T[0, 3] = self.params.screen_offset_x
        T[1, 3] = self.params.screen_offset_y
        T[2, 3] = -self.params.screen_distance
        
        # 2. 旋转矩阵（屏幕倾斜）
        # 绕X轴旋转
        Rx = np.eye(4)
        cos_x, sin_x = np.cos(self.params.screen_tilt_x), np.sin(self.params.screen_tilt_x)
        Rx[1:3, 1:3] = [[cos_x, -sin_x], [sin_x, cos_x]]
        
        # 绕Y轴旋转
        Ry = np.eye(4)
        cos_y, sin_y = np.cos(self.params.screen_tilt_y), np.sin(self.params.screen_tilt_y)
        Ry[0, 0], Ry[0, 2] = cos_y, sin_y
        Ry[2, 0], Ry[2, 2] = -sin_y, cos_y
        
        # 组合变换
        return T @ Ry @ Rx
    
    def gaze_to_screen_coordinates(self,
                                 cyclopean_pos: np.ndarray,
                                 gaze_yaw: float,
                                 gaze_pitch: float) -> Optional[Tuple[float, float]]:
        """
        将凝视角度转换为屏幕像素坐标
        
        Args:
            cyclopean_pos: Cyclopean eye 3D位置 [x, y, z] (mm)
            gaze_yaw: 凝视yaw角 (弧度) 
            gaze_pitch: 凝视pitch角 (弧度)
        
        Returns:
            (screen_x_px, screen_y_px): 屏幕像素坐标 或 None
        """
        # 1. 校正角度
        corr_yaw, corr_pitch = self.correct_gaze_angles(gaze_yaw, gaze_pitch)
        
        # 2. 构建单位视线向量 (Camera坐标系)
        gaze_dir = np.array([
            np.sin(corr_yaw) * np.cos(corr_pitch),    # X (右)
            np.sin(corr_pitch),                        # Y (上) 
            np.cos(corr_yaw) * np.cos(corr_pitch)      # Z (前)
        ])
        
        # 3. 获取屏幕变换矩阵
        screen_transform = self.get_screen_transform_matrix()
        
        # 4. 计算视线与屏幕平面的交点
        # 屏幕法向量（考虑倾斜）
        screen_normal = (screen_transform[:3, :3] @ np.array([0, 0, -1]))
        screen_center = screen_transform[:3, 3]
        
        # 视线与平面相交
        denominator = np.dot(gaze_dir, screen_normal)
        if abs(denominator) < 1e-6:
            return None  # 视线平行于屏幕
        
        t = np.dot(screen_center - cyclopean_pos, screen_normal) / denominator
        if t < 0:
            return None  # 视线朝后
        
        # 交点在世界坐标系
        intersection_world = cyclopean_pos + t * gaze_dir
        
        # 5. 转换到屏幕局部坐标系
        # 逆变换到屏幕坐标系
        screen_transform_inv = np.linalg.inv(screen_transform)
        intersection_homo = np.append(intersection_world, 1.0)
        intersection_screen = (screen_transform_inv @ intersection_homo)[:3]
        
        # 6. 转换为像素坐标
        # 屏幕局部坐标(mm) -> 像素坐标
        # intersection_screen[0,1] 是相对屏幕中心的物理坐标(mm)
        
        # 转换为相对屏幕左上角的物理坐标
        screen_width_mm = self.screen_width_px / self.params.pixel_per_mm_x
        screen_height_mm = self.screen_height_px / self.params.pixel_per_mm_y
        
        physical_x = intersection_screen[0] + screen_width_mm / 2   # 相对左上角
        physical_y = intersection_screen[1] + screen_height_mm / 2  # 相对左上角
        
        # 物理坐标转像素坐标
        pixel_x = physical_x * self.params.pixel_per_mm_x
        pixel_y = physical_y * self.params.pixel_per_mm_y
        
        # 7. 检查是否在屏幕范围内
        if 0 <= pixel_x <= self.screen_width_px and 0 <= pixel_y <= self.screen_height_px:
            return (pixel_x, pixel_y)
        else:
            return None
    
    def predict_screen_gaze(self,
                          mediapipe_results: List[float],
                          gaze_yaw: float,
                          gaze_pitch: float) -> Optional[Tuple[float, float]]:
        """
        完整的凝视预测流程
        
        Args:
            mediapipe_results: MediaPipe面部特征点结果
            gaze_yaw: 凝视yaw角 (弧度)
            gaze_pitch: 凝视pitch角 (弧度)
        
        Returns:
            (screen_x_px, screen_y_px): 屏幕像素坐标 或 None
        """
        # 1. 提取眼部特征点
        left_eye_center, right_eye_center = self.extract_eye_landmarks_from_mediapipe(mediapipe_results)
        
        # 2. 计算Cyclopean eye位置
        cyclopean_pos = self.compute_cyclopean_eye_position(left_eye_center, right_eye_center)
        
        # 3. 转换为屏幕坐标
        return self.gaze_to_screen_coordinates(cyclopean_pos, gaze_yaw, gaze_pitch)
    
    def data_dict_to_training_format(self, data_dict: Dict) -> np.ndarray:
        """
        将单个数据字典转换为训练格式
        类似于 data_dict_list_preparation_for_training_and_evaluation
        
        Args:
            data_dict: 包含 'pitch', 'yaw', 'box', 'mediapipe_results' 的字典
        
        Returns:
            feature_vector: 特征向量
        """
        # 提取基础特征
        pitch = data_dict['pitch']
        yaw = data_dict['yaw']
        box = data_dict['box']  # [x1/w, y1/w, x2/h, y2/h]
        mediapipe_results = data_dict['mediapipe_results']
        
        # 提取眼部特征点
        left_eye_center, right_eye_center = self.extract_eye_landmarks_from_mediapipe(mediapipe_results)
        
        # 组合特征向量：[box特征, 角度, 眼部特征]
        feature_vector = np.concatenate([
            box,                              # 4维：人脸边界框
            [pitch, yaw],                     # 2维：凝视角度
            left_eye_center,                  # 3维：左眼中心
            right_eye_center,                 # 3维：右眼中心
        ])
        
        return feature_vector
    
    def data_dict_list_preparation_for_geometric_calibration(self, data_dict_list: List[Dict], include_labels: bool = False):
        """
        为几何校准准备数据列表
        类似于原始的 data_dict_list_preparation_for_training_and_evaluation 函数
        
        Args:
            data_dict_list: 数据字典列表
            include_labels: 是否包含标签
        
        Returns:
            calibration_data: 几何校准格式的数据列表
        """
        calibration_data = []
        
        for data_dict in data_dict_list:
            # 提取基础数据
            pitch = data_dict['pitch']
            yaw = data_dict['yaw']
            mediapipe_results = data_dict['mediapipe_results']
            
            # 提取眼部特征点
            left_eye_center, right_eye_center = self.extract_eye_landmarks_from_mediapipe(mediapipe_results)
            
            calibration_point = {
                'left_eye_center': left_eye_center,
                'right_eye_center': right_eye_center,
                'gaze_yaw': yaw,
                'gaze_pitch': pitch,
            }
            
            # 如果包含标签，添加目标屏幕坐标
            if include_labels and 'label' in data_dict:
                label = data_dict['label']
                calibration_point.update({
                    'target_screen_x_px': label[0],  # 像素坐标
                    'target_screen_y_px': label[1],
                })
            
            calibration_data.append(calibration_point)
        
        return calibration_data
    
    def calibration_objective(self, normalized_param_vector: np.ndarray, 
                            calibration_data: List[Dict], 
                            scaling_info: Dict) -> float:
        """
        校准目标函数 - 使用归一化参数，避免数值问题
        
        Args:
            normalized_param_vector: 归一化参数向量 [-1, 1]
            calibration_data: 校准数据列表
            scaling_info: 参数缩放信息
        
        Returns:
            rmse: 均方根误差 (像素)
        """
        try:
            # 恢复原始参数
            temp_params = GeometricParameters.from_normalized_vector(normalized_param_vector, scaling_info)
            original_params = self.params
            self.params = temp_params
            
            errors = []
            valid_predictions = 0
            
            for data_point in calibration_data:
                # 提取数据
                left_eye_center = data_point['left_eye_center']
                right_eye_center = data_point['right_eye_center']
                gaze_yaw = data_point['gaze_yaw']
                gaze_pitch = data_point['gaze_pitch']
                target_x_px = data_point['target_screen_x_px']
                target_y_px = data_point['target_screen_y_px']
                
                # 计算Cyclopean eye位置
                cyclopean_pos = self.compute_cyclopean_eye_position(left_eye_center, right_eye_center)
                
                # 预测屏幕坐标
                prediction = self.gaze_to_screen_coordinates(cyclopean_pos, gaze_yaw, gaze_pitch)
                
                if prediction is not None:
                    pred_x_px, pred_y_px = prediction
                    error_x = pred_x_px - target_x_px
                    error_y = pred_y_px - target_y_px
                    pixel_error = np.sqrt(error_x**2 + error_y**2)
                    errors.append(pixel_error)
                    valid_predictions += 1
                else:
                    # 惩罚无效预测
                    errors.append(500.0)  # 大误差（像素）
            
            # 恢复原参数
            self.params = original_params
            
            if valid_predictions == 0:
                return 10000.0  # 巨大惩罚
            
            # 均方根误差 + 有效预测比例惩罚
            rmse = np.sqrt(np.mean([e**2 for e in errors]))
            validity_penalty = (1.0 - valid_predictions / len(calibration_data)) * 200
            
            return rmse + validity_penalty
            
        except Exception as e:
            # 数值错误时返回大惩罚
            return 10000.0
    
    def calibrate(self, data_dict_list: List[Dict], 
                 max_iterations: int = 2000) -> Dict[str, Any]:
        """
        执行参数校准 - 使用归一化参数优化
        
        Args:
            data_dict_list: 数据字典列表（包含label）
            max_iterations: 最大迭代次数
        
        Returns:
            calibration_result: 校准结果字典
        """
        print(f"开始几何参数校准，数据点数: {len(data_dict_list)}")
        
        # 准备校准数据
        calibration_data = self.data_dict_list_preparation_for_geometric_calibration(data_dict_list, include_labels=True)
        
        # 获取初始归一化参数和缩放信息
        initial_normalized_params, scaling_info = self.params.to_normalized_vector()
        self.scaling_info = scaling_info  # 保存缩放信息
        
        print("参数初始值:")
        for i, (param_name, (min_val, max_val, current_val)) in enumerate(scaling_info.items()):
            normalized_val = initial_normalized_params[i]
            print(f"  {param_name}: {current_val:.3f} (范围: {min_val:.1f} ~ {max_val:.1f}, 归一化: {normalized_val:.3f})")
        
        # 使用专门针对归一化参数的优化器
        best_result = None
        best_error = float('inf')
        
        # 由于所有参数都归一化了，可以使用更积极的优化策略
        optimizers = [
            ('L-BFGS-B', {'ftol': 1e-6, 'gtol': 1e-6}),
            ('TNC', {'ftol': 1e-6, 'gtol': 1e-6}),
            ('SLSQP', {'ftol': 1e-6})
        ]
        
        for optimizer_name, options in optimizers:
            try:
                print(f"尝试优化器: {optimizer_name}")
                
                result = scipy.optimize.minimize(
                    fun=self.calibration_objective,
                    x0=initial_normalized_params,
                    args=(calibration_data, scaling_info),
                    method=optimizer_name,
                    bounds=self.normalized_bounds,  # 所有参数都在[-1,1]
                    options={'maxiter': max_iterations, **options}
                )
                
                if result.success and result.fun < best_error:
                    best_result = result
                    best_error = result.fun
                    print(f"  → 成功，误差: {result.fun:.2f} 像素")
                else:
                    print(f"  → 失败或误差较大: {result.fun:.2f} 像素")
                    
            except Exception as e:
                print(f"  → 优化器 {optimizer_name} 异常: {e}")
                continue
        
        if best_result is None:
            print("警告: 所有优化器都失败，使用初始参数")
            calibration_result = {
                'success': False,
                'final_error_px': float('inf'),
                'iterations': 0,
                'message': '优化失败'
            }
        else:
            # 更新最优参数
            self.params = GeometricParameters.from_normalized_vector(best_result.x, scaling_info)
            self.is_calibrated = True
            
            calibration_result = {
                'success': True,
                'final_error_px': best_result.fun,
                'iterations': best_result.nit,
                'message': best_result.message,
                'optimized_parameters': asdict(self.params)
            }
            
            print(f"校准完成! 最终RMSE: {best_result.fun:.2f} 像素")
            print(f"关键参数: Cyclopean权重={self.params.w_dominant:.3f}, "
                  f"深度缩放={self.params.alpha:.3f}")
            print(f"屏幕倾斜: X={np.degrees(self.params.screen_tilt_x):.1f}°, "
                  f"Y={np.degrees(self.params.screen_tilt_y):.1f}°")
            
            # 显示参数变化
            print("\n参数优化结果:")
            final_normalized, _ = self.params.to_normalized_vector()
            for i, (param_name, (min_val, max_val, initial_val)) in enumerate(scaling_info.items()):
                final_val = getattr(self.params, param_name)
                change = final_val - initial_val
                print(f"  {param_name}: {initial_val:.3f} → {final_val:.3f} (变化: {change:+.3f})")
        
        return calibration_result
    
    def evaluate_calibration(self, data_dict_list: List[Dict]) -> Dict[str, float]:
        """
        评估校准效果
        
        Args:
            data_dict_list: 测试数据字典列表
        
        Returns:
            metrics: 评估指标
        """
        test_data = self.data_dict_list_preparation_for_geometric_calibration(data_dict_list, include_labels=True)
        
        errors = []
        valid_count = 0
        
        for data_point in test_data:
            left_eye_center = data_point['left_eye_center']
            right_eye_center = data_point['right_eye_center']
            gaze_yaw = data_point['gaze_yaw']
            gaze_pitch = data_point['gaze_pitch']
            target_x_px = data_point['target_screen_x_px']
            target_y_px = data_point['target_screen_y_px']
            
            cyclopean_pos = self.compute_cyclopean_eye_position(left_eye_center, right_eye_center)
            prediction = self.gaze_to_screen_coordinates(cyclopean_pos, gaze_yaw, gaze_pitch)
            
            if prediction is not None:
                pred_x_px, pred_y_px = prediction
                pixel_error = np.sqrt((pred_x_px - target_x_px)**2 + (pred_y_px - target_y_px)**2)
                errors.append(pixel_error)
                valid_count += 1
        
        if valid_count == 0:
            return {'rmse_px': float('inf'), 'mean_error_px': float('inf'), 'validity_rate': 0.0}
        
        metrics = {
            'rmse_px': np.sqrt(np.mean([e**2 for e in errors])),
            'mean_error_px': np.mean(errors),
            'max_error_px': np.max(errors),
            'validity_rate': valid_count / len(test_data),
            'valid_predictions': valid_count,
            'total_samples': len(test_data)
        }
        
        return metrics
    
    def save_parameters(self, filepath: str):
        """保存校准参数"""
        save_data = {
            'parameters': asdict(self.params),
            'scaling_info': self.scaling_info,
            'screen_size': [self.screen_width_px, self.screen_height_px]
        }
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"参数已保存到: {filepath}")
    
    def load_parameters(self, filepath: str):
        """加载校准参数"""
        try:
            with open(filepath, 'r') as f:
                save_data = json.load(f)
            
            param_dict = save_data.get('parameters', {})
            for key, value in param_dict.items():
                if hasattr(self.params, key):
                    setattr(self.params, key, value)
            
            self.scaling_info = save_data.get('scaling_info', None)
            self.is_calibrated = True
            print(f"参数已加载: {filepath}")
            
        except Exception as e:
            print(f"参数加载失败: {e}")

# 使用示例和测试函数
def create_test_data_dict_list(n_points: int = 16) -> List[Dict]:
    """
    创建测试数据，模拟原始pipeline的数据格式
    """
    data_dict_list = []
    
    # 16点网格
    for i in range(4):
        for j in range(4):
            screen_x_px = j * 640  # 像素坐标
            screen_y_px = i * 360
            
            # 模拟数据
            fake_mediapipe_results = np.random.random(468 * 3).tolist()  # 468个3D点
            fake_pitch = (screen_y_px / 1080 - 0.5) * 0.6
            fake_yaw = (screen_x_px / 1920 - 0.5) * 0.8
            fake_box = [0.2, 0.2, 0.3, 0.3]  # 归一化边界框
            
            data_dict = {
                'pitch': fake_pitch,
                'yaw': fake_yaw,
                'box': fake_box,
                'mediapipe_results': fake_mediapipe_results,
                'label': [screen_x_px, screen_y_px]  # 像素坐标标签
            }
            
            data_dict_list.append(data_dict)
    
    return data_dict_list

if __name__ == "__main__":
    # 测试代码
    print("=== 几何凝视映射系统测试 ===")
    print("主要改进:")
    print("1. w_dominant: 0.0=左眼主导, 1.0=右眼主导, 0.5=双眼平均")
    print("2. 使用像素密度参数而非物理尺寸")
    print("3. 参数归一化避免数值优化问题")
    print("4. 16个优化参数，所有参数值域统一到[-1,1]")
    
    # 创建映射器
    mapper = GeometricGazeMapper(screen_width_px=1920, screen_height_px=1080)
    
    print(f"\n初始像素密度设置:")
    print(f"  X方向: {mapper.params.pixel_per_mm_x:.2f} 像素/mm")
    print(f"  Y方向: {mapper.params.pixel_per_mm_y:.2f} 像素/mm")
    print(f"  对应屏幕物理尺寸: {1920/3.8:.1f}mm × {1080/3.8:.1f}mm")
    
    # 创建测试数据
    test_data = create_test_data_dict_list(16)
    print(f"\n创建了 {len(test_data)} 个测试校准点")
    
    # 执行校准
    result = mapper.calibrate(test_data)
    
    if result['success']:
        print("\n=== 校准参数 ===")
        for key, value in result['optimized_parameters'].items():
            if 'tilt' in key:
                print(f"{key}: {np.degrees(value):.2f}°")
            elif key == 'w_dominant':
                if value < 0.4:
                    dominance = "左眼主导"
                elif value > 0.6:
                    dominance = "右眼主导"
                else:
                    dominance = "双眼平衡"
                print(f"{key}: {value:.3f} ({dominance})")
            elif 'pixel_per_mm' in key:
                print(f"{key}: {value:.2f} 像素/mm")
            else:
                print(f"{key}: {value:.4f}")
        
        # 评估效果
        metrics = mapper.evaluate_calibration(test_data)
        print(f"\n=== 评估结果 ===")
        for key, value in metrics.items():
            print(f"{key}: {value:.2f}")
    
    print("\n测试完成!")
