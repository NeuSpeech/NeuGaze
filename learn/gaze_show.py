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

import win32gui
import win32con
import win32api
import threading
import time
import ctypes
from ctypes import wintypes
import math
from collections import deque
from datetime import datetime, timedelta
import random

# 定义 PAINTSTRUCT 结构
class PAINTSTRUCT(ctypes.Structure):
    _fields_ = [
        ('hdc', wintypes.HDC),
        ('fErase', wintypes.BOOL),
        ('rcPaint', wintypes.RECT),
        ('fRestore', wintypes.BOOL),
        ('fIncUpdate', wintypes.BOOL),
        ('rgbReserved', wintypes.BYTE * 32)
    ]

# 直接使用 Windows API
BeginPaint = ctypes.windll.user32.BeginPaint
BeginPaint.argtypes = [wintypes.HWND, ctypes.POINTER(PAINTSTRUCT)]
BeginPaint.restype = wintypes.HDC

EndPaint = ctypes.windll.user32.EndPaint
EndPaint.argtypes = [wintypes.HWND, ctypes.POINTER(PAINTSTRUCT)]
EndPaint.restype = wintypes.BOOL

class BLENDFUNCTION(ctypes.Structure):
    _fields_ = [
        ('BlendOp', ctypes.c_ubyte),
        ('BlendFlags', ctypes.c_ubyte),
        ('SourceConstantAlpha', ctypes.c_ubyte),
        ('AlphaFormat', ctypes.c_ubyte)
    ]

# 定义 AlphaBlend 函数
AlphaBlend = ctypes.windll.msimg32.AlphaBlend
AlphaBlend.argtypes = [
    wintypes.HDC,    # hdcDest
    ctypes.c_int,    # xoriginDest
    ctypes.c_int,    # yoriginDest
    ctypes.c_int,    # wDest
    ctypes.c_int,    # hDest
    wintypes.HDC,    # hdcSrc
    ctypes.c_int,    # xoriginSrc
    ctypes.c_int,    # yoriginSrc
    ctypes.c_int,    # wSrc
    ctypes.c_int,    # hSrc
    BLENDFUNCTION    # ftn
]
AlphaBlend.restype = wintypes.BOOL

# 定义常量
AC_SRC_OVER = 0x00
AC_SRC_ALPHA = 0x01

class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ('biSize', wintypes.DWORD),
        ('biWidth', wintypes.LONG),
        ('biHeight', wintypes.LONG),
        ('biPlanes', wintypes.WORD),
        ('biBitCount', wintypes.WORD),
        ('biCompression', wintypes.DWORD),
        ('biSizeImage', wintypes.DWORD),
        ('biXPelsPerMeter', wintypes.LONG),
        ('biYPelsPerMeter', wintypes.LONG),
        ('biClrUsed', wintypes.DWORD),
        ('biClrImportant', wintypes.DWORD)
    ]

class BITMAPINFO(ctypes.Structure):
    _fields_ = [
        ('bmiHeader', BITMAPINFOHEADER),
        ('bmiColors', wintypes.DWORD * 3)
    ]

# 定义 CreateDIBSection 函数
CreateDIBSection = ctypes.windll.gdi32.CreateDIBSection
CreateDIBSection.argtypes = [
    wintypes.HDC,
    ctypes.POINTER(BITMAPINFO),
    wintypes.UINT,
    ctypes.POINTER(ctypes.c_void_p),
    wintypes.HANDLE,
    wintypes.DWORD
]
CreateDIBSection.restype = wintypes.HANDLE

class GazePoint:
    def __init__(self, x, y, timestamp=None):
        self.x = x
        self.y = y
        self.timestamp = timestamp or datetime.now()

class GazeOverlay:
    def __init__(self, 
                 history_duration=0.15,      # 历史记录保持时间（秒）
                 update_interval=0.016,      # 更新间隔（秒，约60fps）
                 point_radius=12,            # 点的半径
                 point_alpha=100,            # 点的基础透明度
                 clear_radius=55,            # 清除区域的半径
                 color_r=230,               # 点的红色分量
                 color_g=20,               # 点的绿色分量
                 color_b=20,               # 点的蓝色分量
                 gaussian_sigma_ratio=1.0,   # 高斯函数的 sigma 比例
                 window_alpha=100):          # 窗口整体透明度
        self.is_running = False
        self.history_duration = history_duration
        self.update_interval = update_interval
        self.point_radius = point_radius
        self.point_alpha = point_alpha
        self.clear_radius = clear_radius
        self.color_r = color_r
        self.color_g = color_g
        self.color_b = color_b
        self.gaussian_sigma_ratio = gaussian_sigma_ratio
        self.window_alpha = window_alpha
        
        self.gaze_history = deque()
        self.window_class = self._register_window_class()
        self.hwnd = None
        
    def _register_window_class(self):
        """注册窗口类"""
        wc = win32gui.WNDCLASS()
        wc.lpszClassName = "GazeOverlay"
        wc.hbrBackground = win32gui.GetStockObject(win32con.NULL_BRUSH)
        wc.style = win32con.CS_HREDRAW | win32con.CS_VREDRAW
        wc.lpfnWndProc = self._wnd_proc
        return win32gui.RegisterClass(wc)
        
    def _calculate_gaze_area(self):
        """计算凝视区域"""
        if not self.gaze_history:
            return None
            
        # 使用所有当前点（因为已经在update时清理过了）
        points = [(p.x, p.y) for p in self.gaze_history]
        xs, ys = zip(*points)
        
        # 计算中心点和标准差
        center_x = sum(xs) / len(xs)
        center_y = sum(ys) / len(ys)
        std_x = math.sqrt(sum((x - center_x) ** 2 for x in xs) / len(xs))
        std_y = math.sqrt(sum((y - center_y) ** 2 for y in ys) / len(ys))
        
        return (int(center_x), int(center_y), int(std_x * 2), int(std_y * 2))
        
    def _clean_old_points(self):
        """清理过期的点"""
        now = datetime.now()
        cutoff_time = now - timedelta(seconds=self.history_duration)
        
        # 记录清理前的点数
        before_count = len(self.gaze_history)
        
        # 清理所有过期的点
        while self.gaze_history and self.gaze_history[0].timestamp < cutoff_time:
            oldest_point = self.gaze_history[0]
            age = (now - oldest_point.timestamp).total_seconds()
            print(f"删除点: ({oldest_point.x}, {oldest_point.y}), 年龄: {age:.3f}秒")
            self.gaze_history.popleft()
            
        # 记录清理后的点数
        after_count = len(self.gaze_history)
        if before_count != after_count:
            print(f"清理了 {before_count - after_count} 个点，剩余 {after_count} 个点")

    def _create_gradient_brush(self, hdc, center_x, center_y, radius, alpha):
        """创建径向渐变画刷"""
        size = radius * 2
        
        # 创建32位位图（包含 alpha 通道）
        bmi = BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = size
        bmi.bmiHeader.biHeight = -size  # 负值表示自上而下
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32
        bmi.bmiHeader.biCompression = win32con.BI_RGB
        bmi.bmiHeader.biSizeImage = size * size * 4
        bmi.bmiHeader.biXPelsPerMeter = 0
        bmi.bmiHeader.biYPelsPerMeter = 0
        bmi.bmiHeader.biClrUsed = 0
        bmi.bmiHeader.biClrImportant = 0
        
        # 创建 DIB section
        hdc_mem = win32gui.CreateCompatibleDC(hdc)
        bits = ctypes.c_void_p()
        bitmap = CreateDIBSection(
            hdc_mem, 
            ctypes.byref(bmi), 
            win32con.DIB_RGB_COLORS,
            ctypes.byref(bits), 
            None, 
            0
        )
        
        if not bitmap:
            print(f"CreateDIBSection failed: {ctypes.get_last_error()}")
            return
        
        old_bitmap = win32gui.SelectObject(hdc_mem, bitmap)
        
        # 清空位图内容为完全透明
        bitmap_size = size * size * 4
        bitmap_array = (ctypes.c_ubyte * bitmap_size).from_address(bits.value)
        ctypes.memset(bits, 0, bitmap_size)
        
        # 填充渐变（使用高斯分布）
        for y in range(size):
            for x in range(size):
                dx = x - radius
                dy = y - radius
                distance = math.sqrt(dx*dx + dy*dy)
                if distance <= radius:
                    # 使用更陡峭的高斯函数
                    sigma = radius / self.gaussian_sigma_ratio
                    intensity = math.exp(-(distance * distance) / (2 * sigma * sigma))
                    intensity = intensity * intensity  # 平方使衰减更快
                    
                    # 使用配置的颜色
                    a = int(200 * intensity)
                    r = int(self.color_r * intensity)
                    g = int(self.color_g * intensity)
                    b = int(self.color_b * intensity)
                    
                    index = (y * size + x) * 4
                    bitmap_array[index + 0] = b
                    bitmap_array[index + 1] = g
                    bitmap_array[index + 2] = r
                    bitmap_array[index + 3] = a

        # 创建混合函数
        blend_function = BLENDFUNCTION()
        blend_function.BlendOp = AC_SRC_OVER
        blend_function.BlendFlags = 0
        blend_function.SourceConstantAlpha = alpha
        blend_function.AlphaFormat = AC_SRC_ALPHA
        
        # 应用渐变
        AlphaBlend(
            hdc, 
            center_x - radius,
            center_y - radius, 
            size, 
            size,
            hdc_mem, 
            0, 
            0, 
            size, 
            size,
            blend_function
        )
        
        # 清理资源
        win32gui.SelectObject(hdc_mem, old_bitmap)
        win32gui.DeleteObject(bitmap)
        win32gui.DeleteDC(hdc_mem)

    def _wnd_proc(self, hwnd, msg, wparam, lparam):
        """窗口消息处理函数"""
        try:
            if msg == win32con.WM_PAINT:
                ps = PAINTSTRUCT()
                hdc = BeginPaint(hwnd, ctypes.byref(ps))
                
                # 获取需要重绘的区域并转换为元组
                paint_rect = (
                    ps.rcPaint.left,
                    ps.rcPaint.top,
                    ps.rcPaint.right,
                    ps.rcPaint.bottom
                )
                
                # 使用完全透明的黑色清除重绘区域
                brush = win32gui.CreateSolidBrush(win32api.RGB(0, 0, 0))
                old_brush = win32gui.SelectObject(hdc, brush)
                win32gui.FillRect(hdc, paint_rect, brush)
                win32gui.SelectObject(hdc, old_brush)
                win32gui.DeleteObject(brush)
                
                # 只绘制最新的点
                if self.gaze_history:
                    point = self.gaze_history[-1]
                    self._create_gradient_brush(hdc, point.x, point.y, 
                                             self.point_radius, self.point_alpha)
                
                EndPaint(hwnd, ctypes.byref(ps))
                return 0
                
            return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)
        except Exception as e:
            print(f"窗口处理错误: {e}")
            return 0
    
    def update_gaze_position(self, x, y):
        """更新凝视点位置"""
        if not self.is_running:
            return
            
        now = datetime.now()
        if (self.gaze_history and 
            (now - self.gaze_history[-1].timestamp).total_seconds() < self.update_interval):
            return
            
        # 如果有旧点，先清除旧点的区域
        if self.gaze_history:
            old_point = self.gaze_history[-1]
            rect = (old_point.x - self.clear_radius, 
                   old_point.y - self.clear_radius,
                   old_point.x + self.clear_radius, 
                   old_point.y + self.clear_radius)
            win32gui.InvalidateRect(self.hwnd, rect, True)
        
        self.gaze_history.append(GazePoint(int(x), int(y)))
        
        if self.hwnd:
            rect = (x - self.clear_radius, 
                   y - self.clear_radius,
                   x + self.clear_radius, 
                   y + self.clear_radius)
            win32gui.InvalidateRect(self.hwnd, rect, True)
            win32gui.UpdateWindow(self.hwnd)

    def start(self):
        """启动凝视点显示"""
        if self.is_running:
            return
            
        # 获取屏幕尺寸
        screen_width = win32api.GetSystemMetrics(0)
        screen_height = win32api.GetSystemMetrics(1)
        
        # 创建透明窗口，添加更多的样式标志
        ex_style = (
            win32con.WS_EX_LAYERED |      # 分层窗口
            win32con.WS_EX_TRANSPARENT |   # 鼠标事件穿透
            win32con.WS_EX_TOPMOST |      # 总在最前
            win32con.WS_EX_TOOLWINDOW |   # 工具窗口（不在任务栏显示）
            win32con.WS_EX_NOACTIVATE     # 不激活窗口
        )
        
        self.hwnd = win32gui.CreateWindowEx(
            ex_style,
            "GazeOverlay",
            "Gaze Point",
            win32con.WS_POPUP,  # 无边框弹出窗口
            0, 0, screen_width, screen_height,
            0, 0, 0, None
        )
        
        # 设置窗口完全透明（仅显示内容）
        win32gui.SetLayeredWindowAttributes(
            self.hwnd, 
            win32api.RGB(0, 0, 0),
            self.window_alpha,
            win32con.LWA_COLORKEY | win32con.LWA_ALPHA
        )
        
        # 设置窗口扩展样式
        style = win32gui.GetWindowLong(self.hwnd, win32con.GWL_EXSTYLE)
        style |= win32con.WS_EX_TRANSPARENT | win32con.WS_EX_LAYERED
        win32gui.SetWindowLong(self.hwnd, win32con.GWL_EXSTYLE, style)
        
        # 显示窗口
        win32gui.ShowWindow(self.hwnd, win32con.SW_SHOW)
        win32gui.UpdateWindow(self.hwnd)
        
        self.is_running = True
        
        # 启动消息循环
        def message_loop():
            try:
                while self.is_running:
                    win32gui.PumpWaitingMessages()
                    time.sleep(0.001)
            except Exception as e:
                print(f"消息循环错误: {e}")
            finally:
                print("消息循环结束")
                
        self.message_thread = threading.Thread(target=message_loop)
        self.message_thread.daemon = True
        self.message_thread.start()
    
    def stop(self):
        """停止凝视点显示"""
        if self.is_running:
            self.is_running = False
            if self.hwnd:
                win32gui.DestroyWindow(self.hwnd)
                self.hwnd = None

# 测试代码
if __name__ == "__main__":
    try:
        print("开始测试凝视点显示...")
        overlay = GazeOverlay()
        overlay.start()
        
        time.sleep(0.5)
        
        print("开始移动凝视点...")
        screen_width = win32api.GetSystemMetrics(0)
        screen_height = win32api.GetSystemMetrics(1)
        
        # 使用更平滑的运动算法
        x = screen_width // 2
        y = screen_height // 2
        vx = 0
        vy = 0
        target_x = x
        target_y = y
        
        for i in range(400):
            # 每隔一段时间更新目标位置
            if i % 30 == 0:  # 大约每秒更新一次目标
                target_x = x + random.gauss(0, 200)  # 标准差200像素
                target_y = y + random.gauss(0, 200)
                # 确保目标在屏幕内
                target_x = max(100, min(target_x, screen_width - 100))
                target_y = max(100, min(target_y, screen_height - 100))
            
            # 计算当前位置到目标的向量
            dx = target_x - x
            dy = target_y - y
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance > 1:
                # 添加柔和的加速度朝向目标
                ax = dx * 0.02
                ay = dy * 0.02
                
                # 更新速度（带阻尼）
                vx = vx * 0.95 + ax
                vy = vy * 0.95 + ay
                
                # 限制最大速度
                speed = math.sqrt(vx*vx + vy*vy)
                if speed > 15:
                    vx = vx / speed * 15
                    vy = vy / speed * 15
                
                # 更新位置
                x += vx
                y += vy
            
            # 添加微小的自然抖动
            jitter_x = x + random.gauss(0, 1)
            jitter_y = y + random.gauss(0, 1)
            
            overlay.update_gaze_position(int(jitter_x), int(jitter_y))
            time.sleep(0.03)
            
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        print("停止显示...")
        overlay.stop()
        print("程序退出") 