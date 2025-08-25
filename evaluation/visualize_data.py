#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
校准数据可视化脚本
读取train_data.jsonl文件，可视化每个数据项，并生成30FPS的视频
"""

import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional
import os

class CalibrationDataVisualizer:
    def __init__(self, data_dir: str):
        """
        初始化可视化器
        
        Args:
            data_dir: 校准数据目录路径
        """
        self.data_dir = Path(data_dir)
        self.train_data_path = self.data_dir / "train_data.jsonl"
        self.images_dir = self.data_dir / "images"
        
        # 检查路径
        if not self.train_data_path.exists():
            raise FileNotFoundError(f"训练数据文件不存在: {self.train_data_path}")
        if not self.images_dir.exists():
            raise FileNotFoundError(f"图片目录不存在: {self.images_dir}")
        
        # 读取数据
        self.data_items = self._load_data()
        print(f"加载了 {len(self.data_items)} 个数据项")
    
    def _load_data(self) -> List[Dict]:
        """加载train_data.jsonl文件"""
        data_items = []
        with open(self.train_data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    data_items.append(item)
                except json.JSONDecodeError as e:
                    print(f"第{line_num}行JSON解析错误: {e}")
                    continue
        return data_items
    
    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        """加载图片"""
        image_name=os.path.basename(image_path)
        full_path = self.images_dir / image_name
        if not full_path.exists():
            print(f"图片文件不存在: {full_path}")
            return None
        
        img = cv2.imread(str(full_path))
        if img is None:
            print(f"无法加载图片: {full_path}")
            return None
        
        # 转换为RGB格式
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def _draw_face_visualization(self, img: np.ndarray, data: Dict) -> np.ndarray:
        """
        在图片上绘制面部信息可视化
        
        Args:
            img: 原始图片
            data: 数据项
            
        Returns:
            绘制后的图片
        """
        img_copy = img.copy()
        height, width = img_copy.shape[:2]
        
        # 1. 绘制pitch和yaw信息并画出凝视方向箭头
        pitch = data.get('pitch', 0)
        yaw = data.get('yaw', 0)
        
        # 找到面部中心位置
        face_center_x, face_center_y = width // 2, height // 2
        
        # 从box获取面部中心
        box = data.get('box', [])
        if len(box) == 4:
            # box格式是[x1, y1, x2, y2]，已经是归一化坐标
            x1, y1, x2, y2 = box
            face_center_x = int((x1 + x2) / 2 * width)
            face_center_y = int((y1 + y2) / 2 * height)
        
        
        # 计算凝视方向箭头的终点
        # pitch控制垂直方向，yaw控制水平方向
        arrow_length = 80
        end_x = int(face_center_x + yaw * arrow_length * 10)  # yaw影响水平方向
        end_y = int(face_center_y + pitch * arrow_length * 10)  # pitch影响垂直方向
        
        # 确保箭头终点在图像范围内
        end_x = max(0, min(width - 1, end_x))
        end_y = max(0, min(height - 1, end_y))
        
        # 绘制凝视方向箭头
        cv2.arrowedLine(img_copy, (face_center_x, face_center_y), (end_x, end_y), 
                       (0, 255, 255), 3, cv2.LINE_AA, tipLength=0.3)
        
        # 在箭头旁边显示数值
        cv2.putText(img_copy, f"Pitch: {pitch:.3f}", 
                   (face_center_x + 20, face_center_y - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(img_copy, f"Yaw: {yaw:.3f}", 
                   (face_center_x + 20, face_center_y - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 2. 绘制头部角度
        head_angles = data.get('head_angles', [0, 0, 0])
        if head_angles:
            cv2.putText(img_copy, f"Head: [{head_angles[0]:.2f}, {head_angles[1]:.2f}, {head_angles[2]:.2f}]", 
                       (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 3. 绘制边界框
        box = data.get('box', [])
        if len(box) == 4:
            # box格式是[x1, y1, x2, y2]，已经是归一化坐标
            x1, y1, x2, y2 = box
            
            # 转换为像素坐标
            x1_px = int(x1 * width)
            y1_px = int(y1 * height)
            x2_px = int(x2 * width)
            y2_px = int(y2 * height)
            
            # 确保坐标在图像范围内
            x1_px = max(0, min(width - 1, x1_px))
            y1_px = max(0, min(height - 1, y1_px))
            x2_px = max(0, min(width, x2_px))
            y2_px = max(0, min(height, y2_px))
            
            # 绘制边界框
            cv2.rectangle(img_copy, (x1_px, y1_px), (x2_px, y2_px), (0, 255, 0), 2)
            cv2.putText(img_copy, "Face Box", (x1_px, max(10, y1_px - 10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 4. 绘制MediaPipe特征点
        mediapipe_results = data.get('mediapipe_results', [])
        if mediapipe_results:
            for i, point in enumerate(mediapipe_results):
                if point and len(point) >= 2 and point[0] is not None and point[1] is not None:
                    # 转换为像素坐标
                    px = int(point[0] * width)
                    py = int(point[1] * height)
                    
                    # 绘制特征点
                    cv2.circle(img_copy, (px, py), 2, (255, 0, 0), -1)
                    
                    # 每10个点标注一个序号
                    if i % 10 == 0:
                        cv2.putText(img_copy, str(i), (px + 5, py - 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        
        return img_copy
    
    def _draw_gaze_visualization(self, img: np.ndarray, data: Dict) -> np.ndarray:
        """
        在白布背景上绘制凝视信息可视化
        
        Args:
            img: 原始图片
            data: 数据项
            
        Returns:
            绘制后的图片
        """
        height, width = img.shape[:2]
        
        # 创建白色背景
        white_canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # 1. 绘制目标点(label)
        label = data.get('label', [])
        if label and len(label) == 2 and label[0] is not None and label[1] is not None:
            target_x, target_y = int(label[0]), int(label[1])
            
            # 确保坐标在画布范围内
            if 0 <= target_x < width and 0 <= target_y < height:
                # 绘制目标点
                cv2.circle(white_canvas, (target_x, target_y), 15, (0, 255, 0), -1)
                cv2.circle(white_canvas, (target_x, target_y), 15, (0, 0, 0), 2)
                cv2.putText(white_canvas, "Target", (target_x + 20, target_y + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 0), 2)
        
        # 2. 绘制预测的凝视点(gaze_position)
        gaze_position = data.get('gaze_position', [])
        if gaze_position and len(gaze_position) == 2 and gaze_position[0] is not None and gaze_position[1] is not None:
            gaze_x, gaze_y = int(gaze_position[0]), int(gaze_position[1])
            
            # 确保坐标在画布范围内
            if 0 <= gaze_x < width and 0 <= gaze_y < height:
                # 绘制凝视点
                cv2.circle(white_canvas, (gaze_x, gaze_y), 12, (255, 0, 0), -1)
                cv2.circle(white_canvas, (gaze_x, gaze_y), 12, (0, 0, 0), 2)
                cv2.putText(white_canvas, "Gaze", (gaze_x + 20, gaze_y + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 0, 0), 2)
                
                # 如果目标点和凝视点都存在，绘制连线
                if label and len(label) == 2 and label[0] is not None and label[1] is not None:
                    target_x, target_y = int(label[0]), int(label[1])
                    if 0 <= target_x < width and 0 <= target_y < height:
                        cv2.line(white_canvas, (target_x, target_y), (gaze_x, gaze_y), 
                                (255, 165, 0), 3, cv2.LINE_AA)
                        
                        # 计算并显示误差距离
                        distance = np.sqrt((target_x - gaze_x)**2 + (target_y - gaze_y)**2)
                        mid_x = (target_x + gaze_x) // 2
                        mid_y = (target_y + gaze_y) // 2
                        cv2.putText(white_canvas, f"Error: {distance:.1f}px", 
                                   (mid_x, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 2)
        
        # 3. 绘制质量分数和其他信息
        quality_score = data.get('quality_score', 0)
        cv2.putText(white_canvas, f"Quality: {quality_score:.2f}", 
                   (20, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # 显示pitch和yaw数值
        pitch = data.get('pitch', 0)
        yaw = data.get('yaw', 0)
        cv2.putText(white_canvas, f"Pitch: {pitch:.3f}, Yaw: {yaw:.3f}", 
                   (20, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return white_canvas
    
    def visualize_single_item(self, data: Dict, save_path: Optional[str] = None) -> np.ndarray:
        """
        可视化单个数据项
        
        Args:
            data: 数据项
            save_path: 保存路径（可选）
            
        Returns:
            可视化结果图片
        """
        # 加载图片
        image_path = data.get('image_path', '')
        if not image_path:
            print("数据项中没有image_path")
            return None
        
        img = self._load_image(image_path)
        if img is None:
            return None
        
        # 创建两个子图
        height, width = img.shape[:2]
        
        # 上图：面部信息可视化
        face_img = self._draw_face_visualization(img, data)
        
        # 下图：凝视信息可视化
        gaze_img = self._draw_gaze_visualization(img, data)
        
        # 合并两个图片
        combined_img = np.vstack([face_img, gaze_img])
        
        # 添加分隔线
        cv2.line(combined_img, (0, height), (width, height), (128, 128, 128), 3)
        
        # 添加标题
        cv2.putText(combined_img, "Face Visualization", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(combined_img, "Gaze Visualization", (20, height + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # 保存图片（如果指定了路径）
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR))
        
        return combined_img
    
    def create_video(self, output_path: str, fps: int = 30):
        """
        创建可视化视频
        
        Args:
            output_path: 输出视频路径
            fps: 视频帧率
        """
        if not self.data_items:
            print("没有数据项可以可视化")
            return
        
        # 获取第一张图片的尺寸
        first_img = self.visualize_single_item(self.data_items[0])
        if first_img is None:
            print("无法加载第一张图片")
            return
        
        height, width = first_img.shape[:2]
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not video_writer.isOpened():
            print("无法创建视频写入器")
            return
        
        print(f"开始创建视频: {output_path}")
        print(f"视频尺寸: {width}x{height}, 帧率: {fps}")
        
        # 逐帧写入
        for i, data_item in enumerate(self.data_items):
            print(f"处理第 {i+1}/{len(self.data_items)} 帧...")
            
            # 可视化当前数据项
            vis_img = self.visualize_single_item(data_item)
            if vis_img is not None:
                # 转换为BGR格式并写入视频
                bgr_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
                video_writer.write(bgr_img)
        
        # 释放资源
        video_writer.release()
        print(f"视频创建完成: {output_path}")
    
    def visualize_all_items(self, output_dir: str):
        """
        可视化所有数据项并保存为图片
        
        Args:
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"开始可视化所有数据项，输出到: {output_path}")
        
        for i, data_item in enumerate(self.data_items):
            print(f"处理第 {i+1}/{len(self.data_items)} 项...")
            
            # 可视化当前数据项
            vis_img = self.visualize_single_item(data_item)
            if vis_img is not None:
                # 保存图片
                save_path = output_path / f"item_{i:04d}.png"
                cv2.imwrite(str(save_path), cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
        
        print(f"所有图片已保存到: {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="校准数据可视化工具")
    parser.add_argument("--data_dir", help="校准数据目录路径")
    parser.add_argument("--output", "-o", default=None, help="输出目录/文件路径")
    parser.add_argument("--mode", "-m", choices=["video", "images"], default="video", 
                       help="输出模式: video(视频) 或 images(图片序列)")
    parser.add_argument("--fps", "-f", type=int, default=30, help="视频帧率")
    
    args = parser.parse_args()
    
    try:
        # 创建可视化器
        visualizer = CalibrationDataVisualizer(args.data_dir)
        if args.output is None:
            args.output = args.data_dir + "/output"
        if args.mode == "video":
            # 创建视频
            output_path = args.output if args.output.endswith(('.mp4', '.avi')) else f"{args.output}.mp4"
            visualizer.create_video(output_path, args.fps)
        else:
            # 创建图片序列
            visualizer.visualize_all_items(args.output)
            
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
