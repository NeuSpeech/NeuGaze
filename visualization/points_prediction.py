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

import matplotlib.pyplot as plt
import numpy as np

# 假设我们有一个 batch_size = 3 的示例
batch_size = 3

# 初始化 outputs 和 batch_labels
# 这里我们随机生成一些数据作为示例
outputs = np.random.rand(batch_size, 2)
batch_labels = np.random.rand(batch_size, 2)

def plot_points(outputs,batch_labels):
    # 创建一个新的图形
    plt.figure(figsize=(10, 6))

    # 为每个 batch 中的样本绘制预测和真实坐标，并用线连接它们
    for i in range(batch_size):
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
