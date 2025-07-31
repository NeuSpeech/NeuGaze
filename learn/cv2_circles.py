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
import time
import numpy as np

window_name = 'track'

cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.moveWindow(window_name, x=0, y=0)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def get_screen_resolution():
    from win32.lib import win32con
    from win32 import win32gui, win32print

    hDC = win32gui.GetDC(0)
    width = win32print.GetDeviceCaps(hDC, win32con.DESKTOPHORZRES)
    height = win32print.GetDeviceCaps(hDC, win32con.DESKTOPVERTRES)
    return width, height


time.sleep(5)
cam_id = 0
screen_size = get_screen_resolution()
mid_point = (screen_size[0] // 2, screen_size[1] // 2)

frame = np.zeros((screen_size[1], screen_size[0], 3), np.uint8)
cv2.circle(frame, (mid_point[0], mid_point[1]), 20, (0, 0, 255), -1)
cv2.imshow(window_name, frame)
cap = cv2.VideoCapture(cam_id)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    t_start = time.time()
    time.sleep(0.1)
    frame = np.zeros((screen_size[1], screen_size[0], 3), np.uint8)
    x = np.random.randint(screen_size[0], size=1)[0]
    y = np.random.randint(screen_size[1], size=1)[0]
    cv2.circle(frame, (x, y), 20, (0, 255, 255), -1)
    cv2.imshow(window_name, frame)

    # 添加这一行来处理窗口事件和刷新显示
    # key = cv2.waitKey(1)
    # # 按ESC键退出
    # if key == 27:
    #     break

cap.release()
cv2.destroyAllWindows()

# 一定要用cv2.waitKey来更新显示