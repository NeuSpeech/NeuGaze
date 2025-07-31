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

import win32api
import win32con
import time
import threading
from queue import Queue
import numpy as np
import pyautogui as pg
from win32gui import GetCursorInfo
import ctypes
from ctypes import wintypes

# 加载 user32 DLL
user32 = ctypes.WinDLL('user32', use_last_error=True)
# 定义 CURSORINFO 结构
class CURSORINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("hCursor", wintypes.HANDLE),
        ("ptScreenPos", wintypes.POINT)
    ]

GetCursorInfo = user32.GetCursorInfo
GetCursorInfo.argtypes = [ctypes.POINTER(CURSORINFO)]
GetCursorInfo.restype = wintypes.BOOL

class GazeMouseController:
    def __init__(self, observer, screen_width=1920, screen_height=1080,
                 use_head_control_mouse=True, select_wheel_using_head=True,
                 dead_zone=250, max_speed=10, smoothing=0.5, 
                 lock_head_duration=2,
                 y_speed_coef=1.3,
                 head_coef=10,
                 wheel_head_coef=100,
                 ):
        """
        observer: RealAction实例，用于获取状态和配置
        screen_width, screen_height: 屏幕分辨率
        dead_zone: 中心区域死区大小（像素）
        max_speed: 最大移动速度（像素/步）
        smoothing: 平滑系数 (0-1)
        """
        self.observer = observer
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen_center = (screen_width // 2, screen_height // 2)
        self.use_head_control_mouse = use_head_control_mouse
        self.select_wheel_using_head = select_wheel_using_head
        self.dead_zone = dead_zone
        self.max_speed = max_speed
        self.smoothing = smoothing
        self.y_speed_coef = y_speed_coef
        self.head_coef = head_coef
        self.lock_head_duration = lock_head_duration
        self.wheel_head_coef = wheel_head_coef
        # 状态控制
        self.running = False
        self.gaze_queue = Queue()
        self.last_move = (0, 0)
        
        self.old_pos = None
        
        # 锁定控制
        self.lock_until_time = time.time()
        self.lock_with_head_until_time = time.time()
    
    def update_screen_size(self, width, height):
        """更新屏幕分辨率"""
        self.screen_width = width
        self.screen_height = height
        self.screen_center = (width // 2, height // 2)
    def start(self):
        """启动控制器"""
        if not self.running:
            self.running = True
            self.control_thread = threading.Thread(target=self._control_loop)
            self.control_thread.daemon = True
            self.control_thread.start()
    
    def stop(self):
        """停止控制器"""
        self.running = False
        if hasattr(self, 'control_thread'):
            self.control_thread.join()
    
    def update_gaze(self, gaze_x, gaze_y):
        """更新凝视点位置"""
        # print(f'gaze_x:{gaze_x} gaze_y:{gaze_y}')
        self.gaze_queue.put((gaze_x, gaze_y))
    
    def _calculate_move(self, gaze_x, gaze_y):
        """计算需要的移动量"""
        # 计算与屏幕中心的距离
        dx = gaze_x - self.screen_center[0]
        dy = gaze_y - self.screen_center[1]
        
        # 使用椭圆方程判断死区
        # 水平半轴用 dead_zone，垂直半轴根据屏幕比例调整
        dead_zone_y = self.dead_zone * (self.screen_height / self.screen_width)
        ellipse_distance = (dx/self.dead_zone)**2 + (dy/dead_zone_y)**2
        
        # 如果在椭圆死区内，不移动
        if ellipse_distance < 1:
            return 0, 0
        
        # 计算实际距离用于速度计算
        distance = np.sqrt(dx**2 + dy**2)
        
        # 计算移动速度
        speed = min((distance - self.dead_zone) / self.screen_width * self.max_speed, 
                self.max_speed)
        
        # 标准化方向向量
        if distance > 0:
            dx = dx / distance * speed
            dy = dy / distance * speed
        
        # 应用平滑
        dx = dx * (1 - self.smoothing) + self.last_move[0] * self.smoothing
        dy = dy * (1 - self.smoothing) + self.last_move[1] * self.smoothing
        
        # 给y轴乘以一个系数，因为y轴的距离比x轴的距离要短。
        dy = dy * self.y_speed_coef

        return dx, dy
    
    def _move_mouse(self, dx, dy):
        """执行实际的鼠标移动"""
        if abs(dx) > 1 or abs(dy) > 1:  # 移动量太小就忽略
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, 
                               int(dx), int(dy), 0, 0)


    def is_cursor_visible(self):
        """检查鼠标光标是否可见"""
        cursor_info = CURSORINFO()
        cursor_info.cbSize = ctypes.sizeof(CURSORINFO)
        if GetCursorInfo(ctypes.byref(cursor_info)):
            # 检查是否是标准光标（表示可见）
            handle = cursor_info.hCursor
            if handle is None:
                return False
            else:
                return True
            # print(handle)
            # return handle < 331737755  # 标准光标的句柄值通常很小，这个好像也不一定。黑悟空里面可能很大的句柄。
        return False
        
    def _handle_visible_cursor(self, x, y):
        """处理可见光标的情况"""
        if self.observer.wheel.is_hidden:
            # 使用头部控制
            if self.use_head_control_mouse:
                print(f'using head control mouse')
                y_dir = int(self.observer.head_dict.state_dict['head_up']['v']) - \
                       int(self.observer.head_dict.state_dict['head_down']['v'])
                x_dir = int(self.observer.head_dict.state_dict['head_right']['v']) - \
                       int(self.observer.head_dict.state_dict['head_left']['v'])
                yaw_residual=abs(self.observer.head_angles['yaw'])-self.observer.head_angles_scale['yaw']
                pitch_residual=abs(self.observer.head_angles['pitch'])-self.observer.head_angles_scale['pitch']
                rel = self.head_coef
                rel_x = rel * x_dir* yaw_residual
                rel_y = rel * y_dir* pitch_residual
                if rel_x != 0 or rel_y != 0:
                    mx, my = pg.position()
                    new_x = np.clip(mx + rel_x, 0, self.screen_width)
                    new_y = np.clip(my + rel_y, 0, self.screen_height)
                    threading.Thread(target=pg.moveTo, args=(int(new_x), int(new_y)),
                                   kwargs={"_pause": False}).start()
                    self.lock_with_head_until_time = time.time() + self.lock_head_duration
            # 使用凝视点控制
            # else:
            print(f'using gaze control mouse')
            if time.time() > self.lock_with_head_until_time and time.time() > self.lock_until_time:
                x = np.clip(x, 0, self.screen_width)
                y = np.clip(y, 0, self.screen_height)
                if self.old_pos != (x, y):
                    threading.Thread(target=pg.moveTo, args=(int(x), int(y))).start()
                    self.old_pos = (x, y)
        else:
            # 轮盘显示时有两种控制方式都要写，这样在配置文件里面可以控制。
            if self.select_wheel_using_head:
                # 轮盘显示时用头部角度控制
                pitch = self.observer.head_angles['pitch']
                yaw = self.observer.head_angles['yaw']
                scale = self.wheel_head_coef
                y = self.screen_center[1] + scale * pitch
                x = self.screen_center[0] - scale * yaw
            else:
                # 轮盘显示时用凝视点控制
                pass 
            x = np.clip(x, 0, self.screen_width)
            y = np.clip(y, 0, self.screen_height)
            if self.old_pos != (x, y):
                # threading.Thread(target=pg.moveTo, args=(int(x), int(y))).start()
                # 这里需要改observer里面的 op_xy , 因为之后是用这个来驱动wheel选择的，2025年之后的版本在wheel上面就不用移动鼠标了。
                self.observer.op_xy=x,y
                self.old_pos = (x, y)
                
    def _handle_invisible_cursor(self, x, y):
        """处理不可见光标的情况（FPS模式）"""
        # edge_length = self.dead_zone
        # mx, my = self.screen_center
        # bbox_xxyy = [mx - edge_length // 2, mx + edge_length // 2,
        #             my - edge_length // 2, my + edge_length // 2]
        # bbox_xxyy = np.array(bbox_xxyy)
        # # print(f' x:{x} y:{y}')
        # screen_edge = 20
        # bbox_xxyy[:2] = np.clip(bbox_xxyy[:2], screen_edge, self.screen_width - screen_edge)
        # bbox_xxyy[2:] = np.clip(bbox_xxyy[2:], screen_edge, self.screen_height - screen_edge)
        
        # rel_x = (np.sign(x - bbox_xxyy[0]) + np.sign(x - bbox_xxyy[1])) / 2 * (
        #     np.min([np.abs(x - bbox_xxyy[0]), np.abs(x - bbox_xxyy[1])]))
        # rel_y = (np.sign(y - bbox_xxyy[2]) + np.sign(y - bbox_xxyy[3])) / 2 * (
        #     np.min([np.abs(y - bbox_xxyy[2]), np.abs(y - bbox_xxyy[3])]))
            
        # rel_x = rel_x // 20
        # rel_y = rel_y // 20
        
        # if np.abs(rel_x) > 5 or np.abs(rel_y) > 2:
        #     max_mag = 20
        #     rel_x = np.clip(rel_x, -max_mag, max_mag)
        #     rel_y = np.clip(rel_y, -max_mag, max_mag)
        rel_x,rel_y=self._calculate_move(x,y)
        self._move_mouse(rel_x, rel_y)
    
    def _control_loop(self):
        """控制循环"""
        while self.running:
            if self.observer.mouse_control:
                try:
                    gaze_x, gaze_y = self.gaze_queue.get_nowait()
                    while not self.gaze_queue.empty():
                        gaze_x, gaze_y = self.gaze_queue.get_nowait()
                except:
                    time.sleep(0.01)
                    continue
                    
                if time.time() <= self.lock_until_time:
                    continue
                print(f'control loop:gaze_x:{gaze_x} gaze_y:{gaze_y}')
                is_visible = self.is_cursor_visible()
                # print(f'is_visible:{is_visible}')
                if is_visible:
                    self._handle_visible_cursor(gaze_x, gaze_y)
                else:
                    self._handle_invisible_cursor(gaze_x, gaze_y)
                
            time.sleep(0.01)
            
    def lock_control(self, duration):
        """锁定眼动控制一段时间"""
        self.lock_until_time = time.time() + duration
