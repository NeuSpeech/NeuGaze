#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试新的校准点生成算法
可视化连续运动轨迹和覆盖范围
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import random
import math

def generate_continuous_motion_calibration_points(screen_size, num_points, start_ratio=0.1, end_ratio=0.9, 
                                                motion_smoothness=0.8, coverage_ensurance=True):
    """
    生成连续随机运动的校准点序列，确保覆盖屏幕所有重要位置
    
    Args:
        screen_size: (width, height) 屏幕尺寸
        num_points: 校准点数量
        start_ratio: 起始比例 (避免边缘)
        end_ratio: 结束比例 (避免边缘)
        motion_smoothness: 运动平滑度 (0-1, 1为最平滑)
        coverage_ensurance: 是否确保覆盖重要区域
    
    Returns:
        calibration_points: 连续运动的校准点列表
    """
    screen_width, screen_height = screen_size
    
    # 定义屏幕重要区域 (9宫格布局)
    important_regions = [
        # 左上、中上、右上
        (start_ratio, start_ratio, 0.4, 0.4),
        (0.4, start_ratio, 0.6, 0.4), 
        (0.6, start_ratio, end_ratio, 0.4),
        # 左中、中心、右中
        (start_ratio, 0.4, 0.4, 0.6),
        (0.4, 0.4, 0.6, 0.6),
        (0.6, 0.4, end_ratio, 0.6),
        # 左下、中下、右下
        (start_ratio, 0.6, 0.4, end_ratio),
        (0.4, 0.6, 0.6, end_ratio),
        (0.6, 0.6, end_ratio, end_ratio)
    ]
    
    # 计算每个区域的中心点
    region_centers = []
    for x1, y1, x2, y2 in important_regions:
        center_x = int((x1 + x2) / 2 * screen_width)
        center_y = int((y1 + y2) / 2 * screen_height)
        region_centers.append((center_x, center_y))
    
    # 如果点数少于区域数，确保至少覆盖中心区域
    if num_points < len(important_regions):
        # 优先选择中心区域和四个角落
        priority_indices = [4, 0, 2, 6, 8]  # 中心 + 四个角落
        selected_regions = [region_centers[i] for i in priority_indices[:num_points]]
        calibration_points = selected_regions
    else:
        # 确保覆盖所有重要区域
        calibration_points = region_centers.copy()
        
        # 剩余点数用于生成连续轨迹
        remaining_points = num_points - len(important_regions)
        
        if remaining_points > 0:
            # 生成连续运动轨迹
            trajectory_points = generate_continuous_trajectory(
                screen_size, remaining_points, start_ratio, end_ratio, 
                motion_smoothness, region_centers
            )
            calibration_points.extend(trajectory_points)
    
    # 随机化顺序，但保持空间连续性
    if len(calibration_points) > 1:
        calibration_points = optimize_trajectory_order(calibration_points)
    
    return calibration_points

def generate_continuous_trajectory(screen_size, num_points, start_ratio, end_ratio, 
                                 smoothness, existing_points):
    """
    生成连续运动轨迹
    """
    screen_width, screen_height = screen_size
    
    # 从现有点开始，或者从屏幕中心开始
    if existing_points:
        start_point = existing_points[-1]
    else:
        start_point = (int(screen_width * 0.5), int(screen_height * 0.5))
    
    trajectory = []
    current_point = start_point
    
    for i in range(num_points):
        # 计算目标区域 (确保覆盖不同区域)
        target_region = i % 4  # 4个象限
        if target_region == 0:  # 左上
            target_x = random.uniform(start_ratio, 0.5) * screen_width
            target_y = random.uniform(start_ratio, 0.5) * screen_height
        elif target_region == 1:  # 右上
            target_x = random.uniform(0.5, end_ratio) * screen_width
            target_y = random.uniform(start_ratio, 0.5) * screen_height
        elif target_region == 2:  # 左下
            target_x = random.uniform(start_ratio, 0.5) * screen_width
            target_y = random.uniform(0.5, end_ratio) * screen_height
        else:  # 右下
            target_x = random.uniform(0.5, end_ratio) * screen_width
            target_y = random.uniform(0.5, end_ratio) * screen_height
        
        # 生成到目标点的平滑路径
        path_points = generate_smooth_path(current_point, (target_x, target_y), smoothness)
        trajectory.extend(path_points)
        
        # 更新当前点
        current_point = (int(target_x), int(target_y))
        
        # 如果轨迹点太多，提前结束
        if len(trajectory) >= num_points:
            break
    
    # 截取需要的点数
    return trajectory[:num_points]

def generate_smooth_path(start_point, end_point, smoothness, max_steps=5):
    """
    生成两点之间的平滑路径
    """
    path = []
    
    # 计算距离
    distance = np.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)
    
    if distance < 50:  # 距离太近，直接返回
        return [start_point]
    
    # 根据平滑度决定中间点数量
    num_steps = max(2, int(distance / 100 * (1 - smoothness) * max_steps))
    
    for i in range(1, num_steps):
        t = i / num_steps
        
        # 线性插值
        x = start_point[0] + t * (end_point[0] - start_point[0])
        y = start_point[1] + t * (end_point[1] - start_point[1])
        
        # 添加随机扰动 (根据平滑度调整)
        if smoothness < 0.9:
            noise_scale = (1 - smoothness) * 20
            x += random.uniform(-noise_scale, noise_scale)
            y += random.uniform(-noise_scale, noise_scale)
        
        path.append((int(x), int(y)))
    
    return path

def optimize_trajectory_order(points):
    """
    优化轨迹顺序，减少不必要的跳跃
    """
    if len(points) <= 1:
        return points
    
    # 使用最近邻算法优化顺序
    optimized = [points[0]]
    remaining = points[1:].copy()
    
    while remaining:
        current = optimized[-1]
        
        # 找到最近的点
        min_dist = float('inf')
        best_idx = 0
        
        for i, point in enumerate(remaining):
            dist = np.sqrt((point[0] - current[0])**2 + (point[1] - current[1])**2)
            if dist < min_dist:
                min_dist = dist
                best_idx = i
        
        # 添加最近的点
        optimized.append(remaining.pop(best_idx))
    
    return optimized

def generate_spiral_calibration_points(screen_size, num_points, start_ratio=0.1, end_ratio=0.9):
    """
    生成螺旋形校准点，确保覆盖整个屏幕
    """
    screen_width, screen_height = screen_size
    
    # 计算螺旋参数
    center_x = screen_width * 0.5
    center_y = screen_height * 0.5
    max_radius = min(screen_width, screen_height) * (end_ratio - start_ratio) * 0.4
    
    points = []
    for i in range(num_points):
        # 螺旋参数
        angle = i * 2 * np.pi / num_points * 3  # 3圈螺旋
        radius = max_radius * (i / num_points) ** 0.5  # 半径逐渐增加
        
        # 计算坐标
        x = int(center_x + radius * np.cos(angle))
        y = int(center_y + radius * np.sin(angle))
        
        # 确保在屏幕范围内
        x = max(int(start_ratio * screen_width), min(int(end_ratio * screen_width), x))
        y = max(int(start_ratio * screen_height), min(int(end_ratio * screen_height), y))
        
        points.append((x, y))
    
    return points

def generate_adaptive_calibration_points(screen_size, num_points, start_ratio=0.1, end_ratio=0.9):
    """
    自适应校准点生成：根据屏幕尺寸和点数自动选择最佳策略
    """
    screen_width, screen_height = screen_size
    aspect_ratio = screen_width / screen_height
    
    if num_points <= 9:
        # 点数少时，使用重要区域覆盖
        return generate_continuous_motion_calibration_points(
            screen_size, num_points, start_ratio, end_ratio, 
            motion_smoothness=0.9, coverage_ensurance=True
        )
    elif num_points <= 20:
        # 中等点数，使用连续运动轨迹
        return generate_continuous_motion_calibration_points(
            screen_size, num_points, start_ratio, end_ratio, 
            motion_smoothness=0.7, coverage_ensurance=True
        )
    else:
        # 高点数，使用螺旋轨迹确保均匀覆盖
        return generate_spiral_calibration_points(screen_size, num_points, start_ratio, end_ratio)

def generate_dragon_trajectory_calibration_points(screen_size, num_points, start_ratio=0.05, end_ratio=0.95, max_distance_cm=0.5):
    """
    基于贝塞尔曲线生成龙形轨迹校准点，确保覆盖屏幕上几乎所有地方，且相邻点距离不超过指定值
    
    Args:
        screen_size: (width, height) 屏幕尺寸
        num_points: 校准点数量
        start_ratio: 起始比例 (几乎到边缘)
        end_ratio: 结束比例 (几乎到边缘)
        max_distance_cm: 相邻点之间的最大距离（厘米）
    
    Returns:
        calibration_points: 龙形轨迹的校准点列表
    """
    screen_width, screen_height = screen_size
    
    # 计算最大距离对应的像素数
    pixels_per_cm = 96 / 2.54  # 约37.8像素/厘米
    max_distance_pixels = max_distance_cm * pixels_per_cm
    
    print(f"Screen: {screen_width}x{screen_height}, Max distance: {max_distance_cm}cm = {max_distance_pixels:.1f} pixels")
    print(f"Target points: {num_points}")
    
    # 确保点数足够覆盖整个屏幕
    num_points = max(25, num_points)
    
    # 1. 计算贝塞尔曲线的控制点
    control_points = calculate_bezier_control_points(screen_size, num_points, start_ratio, end_ratio)
    
    # 2. 生成多段贝塞尔曲线
    bezier_segments = generate_bezier_segments(control_points)
    
    # 3. 在曲线上采样点，满足距离约束
    trajectory_points = sample_bezier_curve_with_distance_constraint(
        bezier_segments, num_points, max_distance_pixels
    )
    
    print(f"Generated {len(trajectory_points)} points using Bezier curves")
    return trajectory_points

def calculate_bezier_control_points(screen_size, num_points, start_ratio, end_ratio):
    """计算贝塞尔曲线的控制点，完全避开中心区域，优先覆盖边缘和角落"""
    screen_width, screen_height = screen_size
    
    # 计算有效区域
    effective_width = int((end_ratio - start_ratio) * screen_width)
    effective_height = int((end_ratio - start_ratio) * screen_height)
    
    # 根据点数确定控制点数量
    if num_points <= 50:
        num_control_points = 24  # 增加控制点
    elif num_points <= 200:
        num_control_points = 36
    else:
        num_control_points = 48
    
    print(f"Using {num_control_points} control points for {num_points} trajectory points")
    
    control_points = []
    
    # 1. 四个角落控制点（确保边角覆盖）
    corner_points = [
        (int(start_ratio * screen_width), int(start_ratio * screen_height)),      # 左上
        (int(end_ratio * screen_width), int(start_ratio * screen_height)),        # 右上
        (int(end_ratio * screen_width), int(end_ratio * screen_height)),          # 右下
        (int(start_ratio * screen_width), int(end_ratio * screen_height))         # 左下
    ]
    control_points.extend(corner_points)
    
    # 2. 边缘密集控制点（确保边缘覆盖，避开中心）
    edge_density = max(6, num_control_points // 8)  # 每条边至少6个点
    
    # 上边缘（避开中心区域）
    for i in range(1, edge_density):
        # 分段生成，避开中心
        if i <= edge_density // 3:  # 左段
            x = int(start_ratio * screen_width + (i / (edge_density // 3)) * (0.4 - start_ratio) * screen_width)
        elif i >= 2 * edge_density // 3:  # 右段
            x = int(0.6 * screen_width + ((i - 2 * edge_density // 3) / (edge_density // 3)) * (end_ratio - 0.6) * screen_width)
        else:  # 跳过中心段
            continue
        y = int(start_ratio * screen_height)
        control_points.append((x, y))
    
    # 右边缘（避开中心区域）
    for i in range(1, edge_density):
        if i <= edge_density // 3:  # 上段
            y = int(start_ratio * screen_height + (i / (edge_density // 3)) * (0.4 - start_ratio) * screen_height)
        elif i >= 2 * edge_density // 3:  # 下段
            y = int(0.6 * screen_height + ((i - 2 * edge_density // 3) / (edge_density // 3)) * (end_ratio - 0.6) * screen_height)
        else:  # 跳过中心段
            continue
        x = int(end_ratio * screen_width)
        control_points.append((x, y))
    
    # 下边缘（避开中心区域）
    for i in range(1, edge_density):
        if i <= edge_density // 3:  # 右段
            x = int(end_ratio * screen_width - (i / (edge_density // 3)) * (end_ratio - 0.6) * screen_width)
        elif i >= 2 * edge_density // 3:  # 左段
            x = int(0.4 * screen_width - ((i - 2 * edge_density // 3) / (edge_density // 3)) * (0.4 - start_ratio) * screen_width)
        else:  # 跳过中心段
            continue
        y = int(end_ratio * screen_height)
        control_points.append((x, y))
    
    # 左边缘（避开中心区域）
    for i in range(1, edge_density):
        if i <= edge_density // 3:  # 下段
            y = int(end_ratio * screen_height - (i / (edge_density // 3)) * (end_ratio - 0.6) * screen_height)
        elif i >= 2 * edge_density // 3:  # 上段
            y = int(0.4 * screen_height - ((i - 2 * edge_density // 3) / (edge_density // 3)) * (0.4 - start_ratio) * screen_height)
        else:  # 跳过中心段
            continue
        x = int(start_ratio * screen_width)
        control_points.append((x, y))
    
    # 3. 使用完全避开中心的内部控制点策略
    remaining_points = num_control_points - len(control_points)
    if remaining_points > 0:
        internal_points = generate_completely_anti_center_points(
            screen_size, remaining_points, start_ratio, end_ratio
        )
        control_points.extend(internal_points)
    
    print(f"Generated {len(control_points)} control points")
    return control_points

def generate_completely_anti_center_points(screen_size, num_points, start_ratio, end_ratio):
    """生成完全避开中心的内部控制点，使用分层环形分布"""
    screen_width, screen_height = screen_size
    
    internal_points = []
    
    # 1. 使用多层环形分布，完全避开中心
    # 定义环形区域（避开中心40%的区域）
    ring_regions = [
        (0.4, 0.5),   # 内环：40%-50%
        (0.5, 0.6),   # 中环：50%-60% 
        (0.6, 0.7),   # 外环：60%-70%
        (0.7, 0.8),   # 最外环：70%-80%
    ]
    
    points_per_ring = num_points // len(ring_regions)
    
    for ring_idx, (inner_radius, outer_radius) in enumerate(ring_regions):
        ring_points = []
        
        # 在环形区域内生成点
        for i in range(points_per_ring):
            # 使用黄金角分布，确保均匀
            angle = i * 2 * math.pi * 0.618
            
            # 在环形区域内随机选择半径
            radius_ratio = inner_radius + (outer_radius - inner_radius) * (i / points_per_ring)
            
            x = int((0.5 + radius_ratio * math.cos(angle)) * screen_width)
            y = int((0.5 + radius_ratio * math.sin(angle)) * screen_height)
            
            # 确保在有效区域内
            x = max(int(start_ratio * screen_width), min(int(end_ratio * screen_width), x))
            y = max(int(start_ratio * screen_height), min(int(end_ratio * screen_height), y))
            
            # 严格检查：确保不在中心区域
            center_x, center_y = screen_width * 0.5, screen_height * 0.5
            dist_to_center = math.sqrt((x - center_x)**2 + (y - center_y)**2)
            min_dist_from_center = 0.4 * min(screen_width, screen_height)  # 40%的最小距离
            
            if dist_to_center >= min_dist_from_center:
                ring_points.append((x, y))
        
        internal_points.extend(ring_points)
    
    # 2. 如果还有剩余点数，在四个角落区域添加
    remaining_points = num_points - len(internal_points)
    if remaining_points > 0:
        corner_points = []
        
        # 定义四个角落区域（完全避开中心）
        corner_regions = [
            (start_ratio, start_ratio, 0.35, 0.35),           # 左上
            (0.65, start_ratio, end_ratio, 0.35),             # 右上
            (start_ratio, 0.65, 0.35, end_ratio),             # 左下
            (0.65, 0.65, end_ratio, end_ratio)                # 右下
        ]
        
        points_per_corner = remaining_points // 4
        
        for x1, y1, x2, y2 in corner_regions:
            corner_list = []
            
            # 在角落区域生成网格点
            grid_spacing = min(screen_width, screen_height) * 0.03  # 3%的网格间距
            
            for x in range(int(x1 * screen_width), int(x2 * screen_width), int(grid_spacing)):
                for y in range(int(y1 * screen_height), int(y2 * screen_height), int(grid_spacing)):
                    if len(corner_list) >= points_per_corner:
                        break
                    
                    # 检查是否在有效区域内
                    if (start_ratio <= x/screen_width <= end_ratio and 
                        start_ratio <= y/screen_height <= end_ratio):
                        corner_list.append((x, y))
            
            corner_points.extend(corner_list)
        
        internal_points.extend(corner_points)
    
    return internal_points

def generate_bezier_segments(control_points):
    """生成多段贝塞尔曲线段"""
    if len(control_points) < 4:
        return [control_points]
    
    segments = []
    
    # 将控制点分组，每组4个点形成一段三次贝塞尔曲线
    for i in range(0, len(control_points) - 3, 3):
        segment = control_points[i:i+4]
        segments.append(segment)
    
    # 如果最后一段不足4个点，用前面的点补充
    if len(control_points) % 3 != 0:
        last_segment = control_points[-4:] if len(control_points) >= 4 else control_points
        segments.append(last_segment)
    
    print(f"Generated {len(segments)} Bezier curve segments")
    return segments

def sample_bezier_curve_with_distance_constraint(bezier_segments, num_points, max_distance_pixels):
    """在贝塞尔曲线上采样点，满足距离约束，确保边角覆盖"""
    all_points = []
    
    for segment in bezier_segments:
        if len(segment) == 4:
            # 三次贝塞尔曲线
            segment_points = sample_cubic_bezier(segment, max_distance_pixels)
        else:
            # 线性插值
            segment_points = sample_linear_segment(segment, max_distance_pixels)
        
        all_points.extend(segment_points)
    
    # 如果点数不够，在曲线上添加更多点
    if len(all_points) < num_points:
        all_points = add_more_points_on_curve(all_points, bezier_segments, num_points, max_distance_pixels)
    
    # 如果点数太多，进行采样
    if len(all_points) > num_points:
        all_points = sample_points_uniformly(all_points, num_points)
    
    # 确保边角区域有足够的点
    all_points = ensure_corner_coverage(all_points, num_points, max_distance_pixels)
    
    print(f"Sampled {len(all_points)} points from Bezier curves")
    return all_points

def ensure_corner_coverage(points, target_points, max_distance_pixels):
    """确保边角区域有足够的覆盖点，使用更智能的分布策略"""
    if len(points) >= target_points:
        return points[:target_points]
    
    # 定义边角区域（屏幕的四个角落）
    screen_width, screen_height = 1920, 1080  # 假设标准尺寸
    corner_regions = [
        (0, 0, screen_width * 0.25, screen_height * 0.25),           # 左上角
        (screen_width * 0.75, 0, screen_width, screen_height * 0.25), # 右上角
        (0, screen_height * 0.75, screen_width * 0.25, screen_height), # 左下角
        (screen_width * 0.75, screen_height * 0.75, screen_width, screen_height) # 右下角
    ]
    
    # 分析现有点的分布，找出空白区域
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    
    # 将屏幕划分为网格，找出未覆盖的区域
    grid_size = 8
    grid_width = screen_width / grid_size
    grid_height = screen_height / grid_size
    
    covered_grids = set()
    for point in points:
        grid_x = int(point[0] / grid_width)
        grid_y = int(point[1] / grid_height)
        covered_grids.add((grid_x, grid_y))
    
    # 找出未覆盖的网格
    uncovered_grids = []
    for x in range(grid_size):
        for y in range(grid_size):
            if (x, y) not in covered_grids:
                # 计算网格中心
                center_x = int((x + 0.5) * grid_width)
                center_y = int((y + 0.5) * grid_height)
                uncovered_grids.append((center_x, center_y))
    
    # 优先在未覆盖的网格中添加点
    additional_points = []
    points_to_add = target_points - len(points)
    
    # 首先在未覆盖的网格中添加点
    for i, (x, y) in enumerate(uncovered_grids):
        if len(additional_points) >= points_to_add:
            break
        
        # 检查是否与现有点太近
        too_close = False
        for existing_point in points + additional_points:
            dist = math.sqrt((x - existing_point[0])**2 + (y - existing_point[1])**2)
            if dist < max_distance_pixels * 0.6:
                too_close = True
                break
        
        if not too_close:
            additional_points.append((x, y))
    
    # 如果还不够，在边角区域添加点
    if len(additional_points) < points_to_add:
        remaining = points_to_add - len(additional_points)
        points_per_corner = max(1, remaining // 4)
        
        for x1, y1, x2, y2 in corner_regions:
            corner_points = []
            
            # 在边角区域生成网格点
            grid_spacing = max_distance_pixels * 0.7
            for x in range(int(x1), int(x2), int(grid_spacing)):
                for y in range(int(y1), int(y2), int(grid_spacing)):
                    # 检查是否与现有点太近
                    too_close = False
                    for existing_point in points + additional_points:
                        dist = math.sqrt((x - existing_point[0])**2 + (y - existing_point[1])**2)
                        if dist < max_distance_pixels * 0.5:
                            too_close = True
                            break
                    
                    if not too_close and len(corner_points) < points_per_corner:
                        corner_points.append((x, y))
            
            additional_points.extend(corner_points)
    
    # 合并所有点
    result = points + additional_points
    
    # 如果还是不够，使用分层分布填充
    if len(result) < target_points:
        remaining = target_points - len(result)
        for i in range(remaining):
            # 使用同心圆分布，避免中间集中
            radius_ratio = 0.2 + 0.3 * (i / remaining)  # 从20%到50%的半径
            angle = i * 2 * math.pi * 0.618
            
            x = int((0.5 + radius_ratio * math.cos(angle)) * screen_width)
            y = int((0.5 + radius_ratio * math.sin(angle)) * screen_height)
            
            # 确保在屏幕范围内
            x = max(50, min(screen_width - 50, x))
            y = max(50, min(screen_height - 50, y))
            
            result.append((x, y))
    
    return result[:target_points]

def sample_cubic_bezier(control_points, max_distance_pixels):
    """采样三次贝塞尔曲线，完全避开中心区域"""
    if len(control_points) != 4:
        return control_points
    
    P0, P1, P2, P3 = control_points
    
    # 计算曲线长度估计
    curve_length = estimate_bezier_curve_length(P0, P1, P2, P3)
    
    # 根据长度和距离约束计算采样步数
    num_steps = max(3, int(curve_length / max_distance_pixels))
    
    points = []
    
    # 使用完全避开中心的采样策略
    for i in range(num_steps + 1):
        # 使用U形分布，在两端采样密集，中间完全不采样
        if i <= num_steps // 3:  # 前1/3段
            t = (i / (num_steps // 3)) ** 1.5 * 0.33
        elif i >= 2 * num_steps // 3:  # 后1/3段
            t = 0.67 + ((i - 2 * num_steps // 3) / (num_steps // 3)) ** 1.5 * 0.33
        else:  # 中间1/3段，完全不采样
            continue
        
        point = cubic_bezier_point(P0, P1, P2, P3, t)
        
        # 额外检查：确保采样点不在中心区域
        center_x, center_y = 1920 * 0.5, 1080 * 0.5  # 假设标准屏幕尺寸
        dist_to_center = math.sqrt((point[0] - center_x)**2 + (point[1] - center_y)**2)
        min_dist_from_center = 0.4 * min(1920, 1080)  # 40%的最小距离
        
        if dist_to_center >= min_dist_from_center:
            points.append((int(point[0]), int(point[1])))
    
    return points

def cubic_bezier_point(P0, P1, P2, P3, t):
    """计算三次贝塞尔曲线上的点"""
    # 三次贝塞尔曲线公式: B(t) = (1-t)³P0 + 3(1-t)²tP1 + 3(1-t)t²P2 + t³P3
    x = (1-t)**3 * P0[0] + 3*(1-t)**2*t * P1[0] + 3*(1-t)*t**2 * P2[0] + t**3 * P3[0]
    y = (1-t)**3 * P0[1] + 3*(1-t)**2*t * P1[1] + 3*(1-t)*t**2 * P2[1] + t**3 * P3[1]
    return (x, y)

def estimate_bezier_curve_length(P0, P1, P2, P3):
    """估计贝塞尔曲线的长度"""
    # 使用控制多边形的长度作为估计
    length = 0
    points = [P0, P1, P2, P3]
    
    for i in range(len(points) - 1):
        dx = points[i+1][0] - points[i][0]
        dy = points[i+1][1] - points[i][1]
        length += math.sqrt(dx*dx + dy*dy)
    
    return length

def sample_linear_segment(points, max_distance_pixels):
    """采样线性段"""
    if len(points) < 2:
        return points
    
    result = []
    for i in range(len(points) - 1):
        start = points[i]
        end = points[i + 1]
        
        # 计算距离
        distance = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        
        # 根据距离决定采样点数
        num_steps = max(1, int(distance / max_distance_pixels))
        
        for j in range(num_steps + 1):
            t = j / num_steps
            x = start[0] + t * (end[0] - start[0])
            y = start[1] + t * (end[1] - start[1])
            result.append((int(x), int(y)))
    
    return result

def add_more_points_on_curve(existing_points, bezier_segments, target_points, max_distance_pixels):
    """在曲线上添加更多点"""
    additional_points = target_points - len(existing_points)
    
    if additional_points <= 0:
        return existing_points
    
    # 在现有点之间插入中间点
    enhanced_points = []
    
    for i in range(len(existing_points) - 1):
        enhanced_points.append(existing_points[i])
        
        # 检查是否需要插入中间点
        current = existing_points[i]
        next_point = existing_points[i + 1]
        
        distance = math.sqrt((next_point[0] - current[0])**2 + (next_point[1] - current[1])**2)
        
        if distance > max_distance_pixels * 1.2:
            # 插入中间点
            mid_x = int((current[0] + next_point[0]) / 2)
            mid_y = int((current[1] + next_point[1]) / 2)
            enhanced_points.append((mid_x, mid_y))
    
    enhanced_points.append(existing_points[-1])
    
    return enhanced_points

def sample_points_uniformly(points, num_points):
    """均匀采样点"""
    if len(points) <= num_points:
        return points
    
    step = len(points) / num_points
    sampled = []
    
    for i in range(num_points):
        idx = int(i * step)
        sampled.append(points[min(idx, len(points) - 1)])
    
    return sampled

def optimize_path_distance(path, max_distance_pixels):
    """优化路径，确保相邻点距离不超过限制"""
    if len(path) <= 1:
        return path
    
    optimized = [path[0]]
    
    for i in range(1, len(path)):
        current = path[i]
        prev = optimized[-1]
        
        distance = math.sqrt((current[0] - prev[0])**2 + (current[1] - prev[1])**2)
        
        if distance > max_distance_pixels:
            # 在两点之间插入中间点
            num_intermediate = int(distance / max_distance_pixels) + 1
            
            for j in range(1, num_intermediate):
                t = j / num_intermediate
                x = int(prev[0] + t * (current[0] - prev[0]))
                y = int(prev[1] + t * (current[1] - prev[1]))
                optimized.append((x, y))
        
        optimized.append(current)
    
    return optimized

def smart_fill_path(path, screen_size, target_points, max_distance_pixels):
    """智能填充路径到目标点数，确保点分布均匀"""
    if len(path) >= target_points:
        return path
    
    screen_width, screen_height = screen_size
    additional_points = target_points - len(path)
    
    print(f"Need to add {additional_points} more points")
    
    # 分析现有路径的分布
    x_coords = [p[0] for p in path]
    y_coords = [p[1] for p in path]
    
    # 计算覆盖区域
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # 计算网格间距
    grid_spacing = max_distance_pixels * 0.8
    
    # 创建填充网格
    fill_points = []
    
    # 在覆盖区域内生成均匀的填充点
    for x in range(x_min, x_max + 1, int(grid_spacing)):
        for y in range(y_min, y_max + 1, int(grid_spacing)):
            # 检查是否与现有点太近
            too_close = False
            for existing_point in path:
                dist = math.sqrt((x - existing_point[0])**2 + (y - existing_point[1])**2)
                if dist < max_distance_pixels * 0.5:
                    too_close = True
                    break
            
            if not too_close:
                fill_points.append((x, y))
    
    # 如果填充点不够，在边缘区域添加
    if len(fill_points) < additional_points:
        edge_points = generate_edge_points(screen_size, additional_points - len(fill_points), max_distance_pixels)
        fill_points.extend(edge_points)
    
    # 随机选择需要的填充点数量
    if len(fill_points) > additional_points:
        fill_points = random.sample(fill_points, additional_points)
    
    # 合并路径和填充点
    result_path = path + fill_points
    
    print(f"Final path has {len(result_path)} points")
    return result_path

def generate_edge_points(screen_size, num_points, max_distance_pixels):
    """在屏幕边缘生成点"""
    screen_width, screen_height = screen_size
    edge_points = []
    
    # 在四个边缘生成点
    edges = [
        # 上边缘
        [(x, 50) for x in range(50, screen_width - 50, int(max_distance_pixels * 1.2))],
        # 下边缘
        [(x, screen_height - 50) for x in range(50, screen_width - 50, int(max_distance_pixels * 1.2))],
        # 左边缘
        [(50, y) for y in range(50, screen_height - 50, int(max_distance_pixels * 1.2))],
        # 右边缘
        [(screen_width - 50, y) for y in range(50, screen_height - 50, int(max_distance_pixels * 1.2))]
    ]
    
    for edge in edges:
        edge_points.extend(edge)
    
    # 随机选择需要的点数
    if len(edge_points) > num_points:
        edge_points = random.sample(edge_points, num_points)
    
    return edge_points

def generate_filling_spiral(screen_size, num_points, start_ratio, end_ratio):
    """生成填充螺旋，确保覆盖所有角落"""
    screen_width, screen_height = screen_size
    points = []
    
    # 创建多个螺旋中心，确保覆盖
    centers = [
        (start_ratio + 0.1, start_ratio + 0.1),  # 左上
        (end_ratio - 0.1, start_ratio + 0.1),    # 右上
        (start_ratio + 0.1, end_ratio - 0.1),    # 左下
        (end_ratio - 0.1, end_ratio - 0.1),      # 右下
        (0.5, 0.5),                              # 中心
    ]
    
    points_per_spiral = max(1, num_points // len(centers))
    
    for center_ratio_x, center_ratio_y in centers:
        center_x = int(center_ratio_x * screen_width)
        center_y = int(center_ratio_y * screen_height)
        
        # 生成从中心向外的小螺旋
        for i in range(points_per_spiral):
            angle = i * 2 * math.pi / points_per_spiral * 2
            radius = (i / points_per_spiral) ** 0.7 * min(screen_width, screen_height) * 0.2
            
            x = int(center_x + radius * math.cos(angle))
            y = int(center_y + radius * math.sin(angle))
            
            # 确保在屏幕范围内
            x = max(int(start_ratio * screen_width), min(int(end_ratio * screen_width), x))
            y = max(int(start_ratio * screen_height), min(int(end_ratio * screen_height), y))
            
            points.append((x, y))
    
    return points

def generate_smooth_dragon_path(points, num_points):
    """生成平滑的龙形路径"""
    if len(points) <= 1:
        return points
    
    # 使用样条插值生成平滑路径
    smooth_points = []
    
    for i in range(len(points) - 1):
        start = points[i]
        end = points[i + 1]
        
        # 添加起点
        smooth_points.append(start)
        
        # 计算中间点数量（基于距离）
        distance = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        num_intermediate = max(1, int(distance / 100))
        
        # 生成中间点
        for j in range(1, num_intermediate):
            t = j / num_intermediate
            # 使用贝塞尔曲线样式的插值
            x = int(start[0] + t * (end[0] - start[0]))
            y = int(start[1] + t * (end[1] - start[1]))
            smooth_points.append((x, y))
    
    # 添加最后一个点
    smooth_points.append(points[-1])
    
    # 如果点数太多，进行采样
    if len(smooth_points) > num_points:
        step = len(smooth_points) / num_points
        sampled_points = []
        for i in range(num_points):
            idx = int(i * step)
            sampled_points.append(smooth_points[min(idx, len(smooth_points) - 1)])
        return sampled_points
    
    return smooth_points

def add_coverage_enhancement(points, screen_size, start_ratio, end_ratio):
    """添加覆盖增强，确保没有遗漏的区域"""
    screen_width, screen_height = screen_size
    enhanced_points = points.copy()
    
    # 检查是否有遗漏的区域
    covered_regions = set()
    for point in points:
        # 将屏幕划分为网格，记录覆盖的区域
        grid_x = int((point[0] - start_ratio * screen_width) / (screen_width * (end_ratio - start_ratio) / 10))
        grid_y = int((point[1] - start_ratio * screen_height) / (screen_height * (end_ratio - start_ratio) / 10))
        covered_regions.add((grid_x, grid_y))
    
    # 寻找未覆盖的区域
    missing_regions = []
    for x in range(10):
        for y in range(10):
            if (x, y) not in covered_regions:
                # 计算该区域的中心点
                region_center_x = int((start_ratio + (x + 0.5) * (end_ratio - start_ratio) / 10) * screen_width)
                region_center_y = int((start_ratio + (y + 0.5) * (end_ratio - start_ratio) / 10) * screen_height)
                missing_regions.append((region_center_x, region_center_y))
    
    # 添加遗漏区域的点
    if missing_regions:
        # 选择一些重要的遗漏区域
        num_to_add = min(len(missing_regions), 5)
        selected_missing = random.sample(missing_regions, num_to_add)
        enhanced_points.extend(selected_missing)
    
    return enhanced_points


def visualize_calibration_points(screen_size=(1920, 1080), num_points=20):
    """可视化校准点的分布和轨迹"""
    
    # 生成不同类型的校准点
    continuous_points = generate_continuous_motion_calibration_points(
        screen_size, num_points, motion_smoothness=0.8
    )
    
    spiral_points = generate_spiral_calibration_points(screen_size, num_points)
    
    adaptive_points = generate_adaptive_calibration_points(screen_size, num_points)
    
    # 生成龙形轨迹
    dragon_points = generate_dragon_trajectory_calibration_points(screen_size, num_points)
    
    # 创建图形 - 现在有4个子图
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle(f'Calibration Points Comparison (Screen: {screen_size[0]}x{screen_size[1]}, Points: {num_points})', 
                 fontsize=16)
    
    # 绘制连续运动轨迹
    ax1 = axes[0, 0]
    ax1.set_title('Continuous Motion Trajectory')
    ax1.set_xlim(0, screen_size[0])
    ax1.set_ylim(0, screen_size[1])
    ax1.set_aspect('equal')
    
    # 绘制轨迹线
    for i in range(len(continuous_points) - 1):
        p1 = continuous_points[i]
        p2 = continuous_points[i + 1]
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', alpha=0.6, linewidth=1)
    
    # 绘制点
    for i, point in enumerate(continuous_points):
        ax1.plot(point[0], point[1], 'bo', markersize=8)
        ax1.annotate(str(i+1), (point[0], point[1]), xytext=(5, 5), 
                     textcoords='offset points', fontsize=8)
    
    # 绘制螺旋轨迹
    ax2 = axes[0, 1]
    ax2.set_title('Spiral Trajectory')
    ax2.set_xlim(0, screen_size[0])
    ax2.set_ylim(0, screen_size[1])
    ax2.set_aspect('equal')
    
    # 绘制轨迹线
    for i in range(len(spiral_points) - 1):
        p1 = spiral_points[i]
        p2 = spiral_points[i + 1]
        ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', alpha=0.6, linewidth=1)
    
    # 绘制点
    for i, point in enumerate(spiral_points):
        ax2.plot(point[0], point[1], 'ro', markersize=8)
        ax2.annotate(str(i+1), (point[0], point[1]), xytext=(5, 5), 
                     textcoords='offset points', fontsize=8)
    
    # 绘制自适应轨迹
    ax3 = axes[0, 1]
    ax3.set_title('Adaptive Trajectory')
    ax3.set_xlim(0, screen_size[0])
    ax3.set_ylim(0, screen_size[1])
    ax3.set_aspect('equal')
    
    # 绘制轨迹线
    for i in range(len(adaptive_points) - 1):
        p1 = adaptive_points[i]
        p2 = adaptive_points[i + 1]
        ax3.plot([p1[0], p2[0]], [p1[1], p2[1]], 'g-', alpha=0.6, linewidth=1)
    
    # 绘制点
    for i, point in enumerate(adaptive_points):
        ax3.plot(point[0], point[1], 'go', markersize=8)
        ax3.annotate(str(i+1), (point[0], point[1]), xytext=(5, 5), 
                     textcoords='offset points', fontsize=8)
    
    # 绘制龙形轨迹
    ax4 = axes[1, 0]
    ax4.set_title('Dragon Trajectory (High Coverage)')
    ax4.set_xlim(0, screen_size[0])
    ax4.set_ylim(0, screen_size[1])
    ax4.set_aspect('equal')
    
    # 绘制轨迹线
    for i in range(len(dragon_points) - 1):
        p1 = dragon_points[i]
        p2 = dragon_points[i + 1]
        ax4.plot([p1[0], p2[0]], [p1[1], p2[1]], 'm-', alpha=0.8, linewidth=2)
    
    # 绘制点
    for i, point in enumerate(dragon_points):
        ax4.plot(point[0], point[1], 'mo', markersize=10)
        ax4.annotate(str(i+1), (point[0], point[1]), xytext=(5, 5), 
                     textcoords='offset points', fontsize=8)
    
    # 添加网格和标签
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
    
    plt.tight_layout()
    plt.show()
    
    # 打印统计信息
    print("\n=== Calibration Points Statistics ===")
    print(f"Screen size: {screen_size[0]}x{screen_size[1]}")
    print(f"Number of points: {num_points}")
    
    # 计算覆盖范围
    def calculate_coverage(points):
        if not points:
            return 0, 0, 0, 0
        
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)
        x_center = (max(x_coords) + min(x_coords)) / 2
        y_center = (max(y_coords) + min(y_coords)) / 2
        
        return x_range, y_range, x_center, y_center
    
    for name, points in [("Continuous", continuous_points), 
                         ("Spiral", spiral_points), 
                         ("Adaptive", adaptive_points),
                         ("Dragon", dragon_points)]:
        x_range, y_range, x_center, y_center = calculate_coverage(points)
        print(f"\n{name} trajectory:")
        print(f"  X range: {x_range:.0f} pixels ({x_range/screen_size[0]*100:.1f}% of screen width)")
        print(f"  Y range: {y_range:.0f} pixels ({y_range/screen_size[1]*100:.1f}% of screen height)")
        print(f"  Center: ({x_center:.0f}, {y_center:.0f})")
        print(f"  Screen center: ({screen_size[0]/2:.0f}, {screen_size[1]/2:.0f})")

def test_different_screen_sizes():
    """测试不同屏幕尺寸下的校准点生成"""
    screen_sizes = [
        (1920, 1080),  # 16:9 Full HD
        (2560, 1440),  # 16:9 2K
        (3840, 2160),  # 16:9 4K
        (1366, 768),   # 16:9 HD
        (1440, 900),   # 16:10
    ]
    
    num_points = 25  # 增加点数以确保龙形轨迹有足够覆盖
    
    for screen_size in screen_sizes:
        print(f"\n{'='*50}")
        print(f"Testing screen size: {screen_size[0]}x{screen_size[1]}")
        print(f"{'='*50}")
        
        try:
            # 测试自适应生成
            points = generate_adaptive_calibration_points(screen_size, num_points)
            
            # 测试龙形轨迹
            dragon_points = generate_dragon_trajectory_calibration_points(screen_size, num_points, max_distance_cm=0.5)
            
            # 计算覆盖范围
            for name, test_points in [("Adaptive", points), ("Dragon", dragon_points)]:
                x_coords = [p[0] for p in test_points]
                y_coords = [p[1] for p in test_points]
                
                x_range = max(x_coords) - min(x_coords)
                y_range = max(y_coords) - min(y_coords)
                
                print(f"\n{name} trajectory:")
                print(f"  Generated {len(test_points)} points")
                print(f"  X coverage: {x_range/screen_size[0]*100:.1f}%")
                print(f"  Y coverage: {y_range/screen_size[1]*100:.1f}%")
                print(f"  First 5 points: {test_points[:5]}")
            
        except Exception as e:
            print(f"Error: {e}")

def test_dragon_trajectory_coverage():
    """专门测试龙形轨迹的覆盖效果"""
    print("\n" + "="*60)
    print("Testing Dragon Trajectory Coverage (Max Distance: 0.5cm)")
    print("="*60)
    
    screen_size = (1920, 1080)
    
    
    # 额外显示一个高分辨率的龙形轨迹
    print("\n" + "="*60)
    print("Generating High-Resolution Dragon Trajectory Visualization")
    print("="*60)
    
    high_res_points = generate_dragon_trajectory_calibration_points(screen_size, 600, max_distance_cm=0.5)
    
    # 验证距离限制
    distance_stats = analyze_point_distances(high_res_points)
    print(f"Distance validation: max={distance_stats['max_cm']:.2f}cm, min={distance_stats['min_cm']:.2f}cm, avg={distance_stats['avg_cm']:.2f}cm")
    
    # 创建高分辨率可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    
    # 左图：轨迹线
    ax1.set_title(f'Dragon Trajectory Path (60 points)\nMax Distance: {distance_stats["max_cm"]:.2f}cm', fontsize=14)
    ax1.set_xlim(0, screen_size[0])
    ax1.set_ylim(0, screen_size[1])
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # 绘制轨迹线
    for i in range(len(high_res_points) - 1):
        p1 = high_res_points[i]
        p2 = high_res_points[i + 1]
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], 'm-', alpha=0.8, linewidth=2)
    
    # 绘制关键点
    for i, point in enumerate(high_res_points):
        if i % 5 == 0:  # 每5个点标注一个
            ax1.plot(point[0], point[1], 'mo', markersize=10)
            ax1.annotate(str(i+1), (point[0], point[1]), xytext=(5, 5), 
                       textcoords='offset points', fontsize=10, weight='bold')
        else:
            ax1.plot(point[0], point[1], 'mo', markersize=6)
    
    # 右图：点分布热力图
    ax2.set_title('Point Distribution Heatmap', fontsize=14)
    ax2.set_xlim(0, screen_size[0])
    ax2.set_ylim(0, screen_size[1])
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # 创建点分布热力图
    x_coords = [p[0] for p in high_res_points]
    y_coords = [p[1] for p in high_res_points]
    
    # 绘制所有点
    ax2.scatter(x_coords, y_coords, c=range(len(high_res_points)), 
                cmap='viridis', s=50, alpha=0.7)
    
    # 添加颜色条
    scatter = ax2.scatter(x_coords, y_coords, c=range(len(high_res_points)), 
                          cmap='viridis', s=50, alpha=0.7)
    plt.colorbar(scatter, ax=ax2, label='Point Order')
    
    # 添加屏幕边界
    for ax in [ax1, ax2]:
        ax.plot([0, screen_size[0], screen_size[0], 0, 0], 
               [0, 0, screen_size[1], screen_size[1], 0], 'k--', alpha=0.5, linewidth=2)
    
    plt.tight_layout()
    plt.show()

def analyze_point_distances(points):
    """分析相邻点之间的距离统计"""
    if len(points) <= 1:
        return {'max_cm': 0, 'min_cm': 0, 'avg_cm': 0, 'max_pixels': 0, 'min_pixels': 0, 'avg_pixels': 0}
    
    distances = []
    pixels_per_cm = 96 / 2.54  # 约37.8像素/厘米
    
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]
        
        distance_pixels = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        distance_cm = distance_pixels / pixels_per_cm
        
        distances.append(distance_cm)
    
    return {
        'max_cm': max(distances),
        'min_cm': min(distances),
        'avg_cm': sum(distances) / len(distances),
        'max_pixels': max(distances) * pixels_per_cm,
        'min_pixels': min(distances) * pixels_per_cm,
        'avg_pixels': (sum(distances) / len(distances)) * pixels_per_cm
    }

if __name__ == "__main__":
    print("Testing new calibration point generation algorithms...")
    
    # # 测试主要功能
    # print("\n1. Testing main visualization...")
    # visualize_calibration_points(screen_size=(1920, 1080), num_points=30)
    
    # # 测试不同屏幕尺寸
    # print("\n2. Testing different screen sizes...")
    # test_different_screen_sizes()
    
    # 专门测试龙形轨迹覆盖效果
    print("\n3. Testing dragon trajectory coverage...")
    test_dragon_trajectory_coverage()
    
    print("\nTest completed!") 