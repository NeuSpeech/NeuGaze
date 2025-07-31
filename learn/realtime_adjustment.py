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
import cv2
import numpy as np
import random
from my_model_arch.model import MyL2CS
import time
# import pyautogui  # 用于获取屏幕信息

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# def get_screen_info():
#     # 获取所有屏幕的信息
#     screens = pyautogui.screenshot().screens
#     return screens
#
#
# def create_fullscreen_window(window_name, screen_index=0):
#     screens = get_screen_info()
#     if screen_index < 0 or screen_index >= len(screens):
#         print(f"Invalid screen index. Using primary screen.")
#         screen_index = 0
#
#     screen = screens[screen_index]
#     cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
#     cv2.moveWindow(window_name, screen.left, screen.top)
#     cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
#     return screen.width, screen.height


def get_screen_resolution():
    screen = cv2.getWindowImageRect('Gaze Tracking')
    return screen[2], screen[3]


def calibrate(cap, num_points=20):
    screen_width, screen_height = get_screen_resolution()
    calibration_points = []
    images = []
    labels = []

    # 生成随机校准点
    for _ in range(num_points):
        x = random.randint(int(0.1 * screen_width), int(0.9 * screen_width))
        y = random.randint(int(0.1 * screen_height), int(0.9 * screen_height))
        calibration_points.append((x, y))

    for point in calibration_points:
        img = np.zeros((screen_height, screen_width, 3), np.uint8)
        cv2.putText(img, "正在校准", (int(screen_width/2)-100, int(screen_height/2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.circle(img, point, 20, (0, 255, 0), -1)
        cv2.imshow('Gaze Tracking', img)
        cv2.waitKey(1000)  # 等待2秒

        ret, frame = cap.read()
        if ret:
            images.append(frame)
            labels.append(point)

    return images, labels


def train_model(model, images, labels):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(100):  # 训练100轮
        for img, label in zip(images, labels):
            img = cv2.resize(img, (640, 480))
            img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            img = img.to(device)
            label = torch.tensor(label).float().to(device)

            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), 'gaze_model.pth')


def predict_gaze_position(frame, model):
    frame = cv2.resize(frame, (640, 480))
    frame = torch.from_numpy(frame).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    frame = frame.to(device)
    with torch.no_grad():
        predicted_position = model(frame)
    return predicted_position[0].cpu().numpy()


def display_gaze(true_position, predicted_position):
    screen_width, screen_height = get_screen_resolution()
    img = np.zeros((screen_height, screen_width, 3), np.uint8)

    # 显示真实注视点（绿色）
    cv2.circle(img, (int(true_position[0]), int(true_position[1])), 20, (0, 255, 0), -1)

    # 显示预测注视点（红色）
    cv2.circle(img, (int(predicted_position[0]), int(predicted_position[1])), 20, (0, 0, 255), -1)

    cv2.imshow('Gaze Tracking', img)


def main():
    # 指定要使用的屏幕索引（0 表示主屏幕，1 表示第二个屏幕，以此类推）
    # screen_index = 0  # 修改这个值来选择不同的屏幕

    # window_name = 'Gaze Tracking'
    # screen_width, screen_height = create_fullscreen_window(window_name, screen_index)

    model = SimpleGazeModel().to(device)

    cv2.namedWindow('Gaze Tracking', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Gaze Tracking', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    true_position = (0, 0)
    is_calibrating = True
    is_training = False

    while True:
        if is_calibrating:
            print("start calibrating")
            cv2.waitKey(2000)
            images, labels = calibrate(cap, num_points=30)  # 您可以根据需要调整点的数量
            is_calibrating = False
            is_training = True
            continue

        if is_training:
            print("校准完成，开始训练模型...")
            train_model(model, images, labels)
            print("模型训练完成，开始预测...")
            is_training = False
            true_position = labels[-1]  # 使用最后一个校准点作为初始真实位置
            continue

        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头帧")
            break

        predicted_position = predict_gaze_position(frame, model)
        display_gaze(true_position, predicted_position)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            print("重新校准...")
            is_calibrating = True
        elif key == ord('n'):
            print("移动到下一个真实位置...")
            true_position = (np.random.randint(0, get_screen_resolution()[0]),
                             np.random.randint(0, get_screen_resolution()[1]))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()