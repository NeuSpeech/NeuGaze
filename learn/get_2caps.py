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


def crop_center_rectangle(image, rectangle_width, rectangle_height):

    # 获取图像的尺寸
    image_height, image_width = image.shape[:2]

    # 计算长方形在图像中的位置
    x = (image_width - rectangle_width) // 2
    y = (image_height - rectangle_height) // 2

    # 截取长方形
    rectangle = image[y:y+rectangle_height, x:x+rectangle_width]

    return rectangle


def main():
    cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080*2)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720*2)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"默认帧尺寸: 宽度 = {frame_width}, 高度 = {frame_height}")
    capF = cv2.VideoCapture(4)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头帧")
            break
        print(frame.shape)
        # frame=crop_center_rectangle(frame,frame.shape[1]//2,frame.shape[0]//2)
        frame=cv2.flip(frame,1,0)
        cv2.imshow('cap', frame)
        ret, frame = capF.read()
        cv2.imshow('capF', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()