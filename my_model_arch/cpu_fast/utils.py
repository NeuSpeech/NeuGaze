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

import numpy as np
import random
import matplotlib.pyplot as plt
import cv2
import torchvision
from filterpy.kalman import KalmanFilter
from .model import L2CS
import jsonlines
import math
import time
import copy


from .models import (
    resnet18,
    resnet34,
    resnet50,
    mobilenet_v2,
    mobileone_s0,
    mobileone_s1,
    mobileone_s2,
    mobileone_s3,
    mobileone_s4
)

def getArch(arch, bins, pretrained=False, inference_mode=True):
    # Base network structure
    if arch == 'mresnet18':
        model = resnet18(pretrained=pretrained, num_classes=bins)
    elif arch == 'mresnet34':
        model = resnet34(pretrained=pretrained, num_classes=bins)
    elif arch == 'mresnet50':
        model = resnet50(pretrained=pretrained, num_classes=bins)
    elif arch == "mobilenetv2":
        model = mobilenet_v2(pretrained=pretrained, num_classes=bins)
    elif arch == "mobileone_s0":
        model = mobileone_s0(pretrained=pretrained, num_classes=bins, inference_mode=inference_mode)
    elif arch == "mobileone_s1":
        model = mobileone_s1(pretrained=pretrained, num_classes=bins, inference_mode=inference_mode)
    elif arch == "mobileone_s2":
        model = mobileone_s2(pretrained=pretrained, num_classes=bins, inference_mode=inference_mode)
    elif arch == "mobileone_s3":
        model = mobileone_s3(pretrained=pretrained, num_classes=bins, inference_mode=inference_mode)
    elif arch == "mobileone_s4":
        model = mobileone_s4(pretrained=pretrained, num_classes=bins, inference_mode=inference_mode)
    elif arch == 'ResNet18':
        model = L2CS(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], bins)
    elif arch == 'ResNet34':
        model = L2CS(torchvision.models.resnet.BasicBlock, [3, 4, 6, 3], bins)
    elif arch == 'ResNet101':
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], bins)
    elif arch == 'ResNet152':
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 8, 36, 3], bins)
    else:
        if arch != 'ResNet50':
            print('Invalid value for architecture is passed! '
                  'The default value of ResNet50 will be used instead!')
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], bins)
    return model


def read_jsonlines(file_path):
    json_dicts = []
    with jsonlines.open(file_path, mode='r') as reader:
        for json_obj in reader:
            json_dicts.append(json_obj)
    return json_dicts


class GazeKalmanFilter:
    def __init__(self, dt, std_measurement,Q_coef=0.03):
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
    
    def update_config(self, dt,std_measurement,Q_coef):
        self.__init__(dt, std_measurement,Q_coef)


def plot_points(outputs, batch_labels):
    # 创建一个新的图形
    plt.figure(figsize=(10, 6))

    # 为每个 batch 中的样本绘制预测和真实坐标，并用线连接它们
    for i in range(outputs.shape[0]):
        # 提取当前样本的预测和真实坐标
        pred_coords = outputs[i]
        true_coords = batch_labels[i]

        # 绘制预测坐标（蓝色，圆圈）
        plt.scatter(pred_coords[0], pred_coords[1], color='blue', marker='o')

        # 绘制真实坐标（红色，方形）
        plt.scatter(true_coords[0], true_coords[1], color='red', marker='s')

        # 连接预测和真实坐标
        plt.plot([pred_coords[0], true_coords[0]], [pred_coords[1], true_coords[1]], color='green', linestyle='--')

    # 添加标题和坐标轴标签
    plt.title('Predicted vs True Coordinates')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend(['Predicted', 'True'])
    # 显示图形
    plt.show()


def crop_center_rectangle(image, rectangle_width, rectangle_height):
    # 获取图像的尺寸
    image_height, image_width = image.shape[:2]

    # 计算长方形在图像中的位置
    x = (image_width - rectangle_width) // 2
    y = (image_height - rectangle_height) // 2

    # 截取长方形
    rectangle = image[y:y + rectangle_height, x:x + rectangle_width]

    return rectangle


def generate_random_calibration_points(screen_size, num_points, start_ratio=0.1, end_ratio=0.9):
    screen_width, screen_height = screen_size
    calibration_points = []
    for _ in range(num_points):
        x = random.randint(int(start_ratio * screen_width), int(end_ratio * screen_width))
        y = random.randint(int(start_ratio * screen_height), int(end_ratio * screen_height))
        calibration_points.append((x, y))
    return calibration_points


def generate_calibration_points(screen_size, num_points, start_ratio=0.1, end_ratio=0.9):
    screen_width, screen_height = screen_size

    # 计算网格的行数和列数
    num_rows = int(np.sqrt(num_points))
    num_cols = int(np.ceil(num_points / num_rows))

    # 计算每个格子的宽度和高度
    cell_width = (end_ratio - start_ratio) * screen_width / (num_cols - 1)
    cell_height = (end_ratio - start_ratio) * screen_height / (num_rows - 1)

    calibration_points = []
    for row in range(num_rows):
        row_points = []
        for col in range(num_cols):
            x = int(start_ratio * screen_width + col * cell_width)
            y = int(start_ratio * screen_height + row * cell_height)
            row_points.append((x, y))

        # 如果是偶数行，反转顺序以创建蛇形模式
        if row % 2 == 1:
            row_points.reverse()

        calibration_points.extend(row_points)

    # 如果生成的点多于需要的点，删除多余的点
    calibration_points = calibration_points[:num_points]
    return calibration_points


def rotation_matrix_to_angles(rotation_matrix):
    """
    Calculate Euler angles from rotation matrix.
    :param rotation_matrix: A 3*3 matrix with the following structure
    [Cosz*Cosy  Cosz*Siny*Sinx - Sinz*Cosx  Cosz*Siny*Cosx + Sinz*Sinx]
    [Sinz*Cosy  Sinz*Siny*Sinx + Sinz*Cosx  Sinz*Siny*Cosx - Cosz*Sinx]
    [  -Siny             CosySinx                   Cosy*Cosx         ]
    :return: Angles in degrees for each axis
    """
    x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    y = math.atan2(-rotation_matrix[2, 0], math.sqrt(rotation_matrix[0, 0] ** 2 +
                                                     rotation_matrix[1, 0] ** 2))
    z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    return np.array([x, y, z]) * 180. / math.pi


class StateRecordDict:
    def __init__(self, state_dict=None):
        if state_dict is None:
            state_dict = {}
        self.state_dict = state_dict
        # every key record current value, change value, last time start True,
        # last time start False
        # cp for change pattern

    def update(self, k, v):
        if k not in self.state_dict:
            self.state_dict[k] = {'v': v, 'last_v': None,
                                  'time_True': None,
                                  'time_False': None,
                                  'diff': None,
                                  'cp': None,  # cp for change pattern
                                  }
        else:
            self.state_dict[k]['last_v'] = self.state_dict[k]['v']
            self.state_dict[k]['v'] = v
            self.state_dict[k]['diff'] = self.state_dict[k]['last_v'] != self.state_dict[k]['v']
            if self.state_dict[k]['diff']:
                if self.state_dict[k]['v']:
                    self.state_dict[k]['time_True'] = time.time()
                else:
                    self.state_dict[k]['time_False'] = time.time()
            if self.state_dict[k]['last_v']:
                if self.state_dict[k]['v']:
                    self.state_dict[k]['cp'] = 'TT'
                else:
                    self.state_dict[k]['cp'] = 'TF'
            else:
                if self.state_dict[k]['v']:
                    self.state_dict[k]['cp'] = 'FT'
                else:
                    self.state_dict[k]['cp'] = 'FF'

    def update_dict(self, state_dict):
        for k, v in state_dict.items():
            self.update(k, v)


def combine_dicts(dict1, dict2):
    nd = copy.deepcopy(dict1)
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = v
    return nd



# 以下提供一些控制鼠标移动的函数

# 检测鼠标可见状态
import ctypes
from ctypes import wintypes

# 加载 kernel32 和 user32 DLL
kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
user32 = ctypes.WinDLL('user32', use_last_error=True)

# 定义 GetLastError 函数原型
GetLastError = kernel32.GetLastError
GetLastError.restype = wintypes.DWORD

class CURSORINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("hCursor", wintypes.HANDLE),  # 使用 HANDLE 代替 HCURSOR
        ("ptScreenPos", wintypes.POINT),
    ]

# 设置函数参数类型和返回类型
GetCursorInfo = user32.GetCursorInfo
GetCursorInfo.argtypes = [ctypes.POINTER(CURSORINFO)]
GetCursorInfo.restype = wintypes.BOOL


def is_cursor_visible_func():

    # 创建 CURSORINFO 实例并初始化 cbSize
    cursor_info = CURSORINFO()
    cursor_info.cbSize = ctypes.sizeof(CURSORINFO)  # 确保这是正确的结构大小
    # 调用 GetCursorInfo 函数
    if not GetCursorInfo(ctypes.byref(cursor_info)):
        error_code = GetLastError()
        print(f"Failed to get cursor info. Error code: {error_code}")
        return True
    else:
        # 如果 flags 为 0，则光标被隐藏；非 0 则光标可见
        is_cursor_visible = cursor_info.flags != 0
        # print(f"Is cursor visible: {is_cursor_visible}")
        # print(f"Cursor position: ({cursor_info.ptScreenPos.x}, {cursor_info.ptScreenPos.y})")
        return is_cursor_visible

# 这个函数输入是屏幕尺寸，鼠标移动参考位置，视点估计位置，控制不动区大小，移动倍数。
