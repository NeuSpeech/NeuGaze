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

import cv2
import numpy as np
from .results import GazeResultContainer
import tkinter as tk
import time

def draw_gaze(a,b,c,d,image_in, pitchyaw, thickness=2, color=(255, 255, 0),sclae=2.0):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    (h, w) = image_in.shape[:2]
    length = c
    pos = (int(a+c / 2.0), int(b+d / 2.0))
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[0]) * np.cos(pitchyaw[1])
    dy = -length * np.sin(pitchyaw[1])
    cv2.arrowedLine(image_out, tuple(np.round(pos).astype(np.int32)),
                   tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), color,
                   thickness, cv2.LINE_AA, tipLength=0.18)
    return image_out

def draw_bbox(frame: np.ndarray, bbox: np.ndarray):
    
    x_min=int(bbox[0])
    if x_min < 0:
        x_min = 0
    y_min=int(bbox[1])
    if y_min < 0:
        y_min = 0
    x_max=int(bbox[2])
    y_max=int(bbox[3])

    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 1)

    return frame

def render(frame: np.ndarray, results):

    # Draw bounding boxes
    for bbox in results.bboxes:
        frame = draw_bbox(frame, bbox)

    # Draw Gaze
    for i in range(results.pitch.shape[0]):

        bbox = results.bboxes[i]
        pitch = results.pitch[i]
        yaw = results.yaw[i]
        
        # Extract safe min and max of x,y
        x_min=int(bbox[0])
        if x_min < 0:
            x_min = 0
        y_min=int(bbox[1])
        if y_min < 0:
            y_min = 0
        x_max=int(bbox[2])
        y_max=int(bbox[3])

        # Compute sizes
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min

        draw_gaze(x_min,y_min,bbox_width, bbox_height,frame,(pitch,yaw),color=(0,0,255))

    return frame


def popout_fading_window(text="NumLock is ON",duration=0.5,window_width=200,window_height=100, fontsize=20):
    def fade_out(window, duration=1):
        alpha = 1.0
        step = alpha / (duration * 100)
        while alpha > 0:
            alpha -= step
            window.attributes('-alpha', alpha)
            window.update()
            time.sleep(0.01)

    # 创建一个tkinter窗口
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口

    # 弹出消息框
    messagebox = tk.Toplevel(root)
    messagebox.overrideredirect(True)  # 移除边框和标题栏
    messagebox.attributes('-topmost', True)  # 确保窗口始终在最前面
    # 设置窗口位置在屏幕中心
    screen_width = messagebox.winfo_screenwidth()
    screen_height = messagebox.winfo_screenheight()
    x_cordinate = int((screen_width / 2) - (window_width / 2))
    y_cordinate = int((screen_height / 2) - (window_height / 2))
    messagebox.geometry(f"{window_width}x{window_height}+{x_cordinate}+{y_cordinate}")

    # 添加一个标签
    label = tk.Label(messagebox, text=text, font=('Song', fontsize,'bold'),foreground='black',)
    label.pack(expand=True)

    # 显示消息框
    messagebox.deiconify()
    messagebox.update()

    fade_out(messagebox,duration)

    # 关闭窗口
    root.destroy()



