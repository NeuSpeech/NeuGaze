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

import time

from l2cs import Pipeline, render
from my_model_arch.my_l2cs import MultiEuroFilter
import cv2
import torch
from pathlib import Path
import numpy as np


class SimpleFilter:
    def __init__(self, alpha=0.8):
        self.last_x = None
        self.alpha = alpha

    def __call__(self, x):
        if self.last_x is None:
            self.last_x = x
        x = self.alpha * x + (1 - self.alpha) * self.last_x
        return x

CWD=Path(__file__).parent
gaze_pipeline = Pipeline(
    weights=CWD / 'models' / 'L2CSNet_gaze360.pkl',
    arch='ResNet50',
    device=torch.device('cuda')  # or 'gpu'
)

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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4000)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 4000)
    # config = {
    #     'freq': 50,  # Hz
    #     'mincutoff': 1.0,  # Hz
    #     'beta': 0.1,
    #     'dcutoff': 1.0
    # }
    # multifilter=MultiEuroFilter(config)

    simple_filter=SimpleFilter(alpha=0.8)
    while True:
        _, frame = cap.read()

        # Process frame and visualize
        results = gaze_pipeline.step(frame)

        # one dollar filter
        pitch=results.pitch[0]
        yaw=results.yaw[0]
        bboxes=results.bboxes[0]
        landmarks=results.landmarks[0]
        landmarks=landmarks.reshape([-1,])
        bboxes=bboxes.tolist()
        landmarks=landmarks.tolist()
        X=[pitch,yaw,*bboxes,*landmarks]
        X=simple_filter(np.array(X))
        # X=multifilter(X,time.time())
        results.pitch=np.array([X[0]])
        results.yaw=np.array([X[1]])
        results.bboxes=np.array([X[2:6]])
        results.landmarks=np.array(X[6:]).reshape([1,-1,2])

        # print(landmarks)
        # print('*'*100)
        # cv2.drawContours(frame, [landmarks], -1, (0, 255, 0), 2)
        # print(results.yaw,results.pitch,np.tan(results.yaw),np.tan(results.pitch))
        landmarks=results.landmarks[0]
        frame = render(frame, results)
        for i in range(landmarks.shape[0]):
            cv2.circle(frame, (int(landmarks[i, 0]), int(landmarks[i, 1])), 5, (0, 0, 255), -1)
        frame=crop_center_rectangle(frame,frame.shape[0]//2,frame.shape[1]//2)
        frame=cv2.flip(frame,1,0)
        cv2.imshow('摄像头', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()