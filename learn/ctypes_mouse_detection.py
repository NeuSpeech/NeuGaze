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

import ctypes
from ctypes import wintypes
import time
import win32gui
import win32con
import win32api

# 加载 user32 DLL
user32 = ctypes.WinDLL('user32', use_last_error=True)

# 定义更多的光标相关常量
IDC_ARROW = 32512
IDC_IBEAM = 32513
IDC_WAIT = 32514
IDC_CROSS = 32515
IDC_UPARROW = 32516
IDC_SIZE = 32640
IDC_ICON = 32641
IDC_SIZENWSE = 32642
IDC_SIZENESW = 32643
IDC_SIZEWE = 32644
IDC_SIZENS = 32645
IDC_SIZEALL = 32646
IDC_NO = 32648
IDC_HAND = 32649
IDC_APPSTARTING = 32650
IDC_HELP = 32651

CURSOR_TYPES = {
    IDC_ARROW: "ARROW",
    IDC_IBEAM: "IBEAM",
    IDC_WAIT: "WAIT",
    IDC_CROSS: "CROSS",
    IDC_UPARROW: "UPARROW",
    IDC_SIZE: "SIZE",
    IDC_ICON: "ICON",
    IDC_SIZENWSE: "SIZENWSE",
    IDC_SIZENESW: "SIZENESW",
    IDC_SIZEWE: "SIZEWE",
    IDC_SIZENS: "SIZENS",
    IDC_SIZEALL: "SIZEALL",
    IDC_NO: "NO",
    IDC_HAND: "HAND",
    IDC_APPSTARTING: "APPSTARTING",
    IDC_HELP: "HELP"
}

# 定义更多的结构体
class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long),
                ("y", ctypes.c_long)]

class CURSORINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("hCursor", wintypes.HANDLE),
        ("ptScreenPos", POINT)
    ]

class ICONINFO(ctypes.Structure):
    _fields_ = [
        ("fIcon", wintypes.BOOL),
        ("xHotspot", wintypes.DWORD),
        ("yHotspot", wintypes.DWORD),
        ("hbmMask", wintypes.HANDLE),
        ("hbmColor", wintypes.HANDLE)
    ]

# 定义更多的函数
GetCursor = user32.GetCursor
GetCursor.restype = wintypes.HANDLE

GetIconInfo = user32.GetIconInfo
GetIconInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(ICONINFO)]
GetIconInfo.restype = wintypes.BOOL

GetCursorInfo = user32.GetCursorInfo
GetCursorInfo.argtypes = [ctypes.POINTER(CURSORINFO)]
GetCursorInfo.restype = wintypes.BOOL

LoadCursor = user32.LoadCursorW
LoadCursor.argtypes = [wintypes.HINSTANCE, wintypes.LPCWSTR]
LoadCursor.restype = wintypes.HANDLE

def get_detailed_cursor_info():
    """获取详细的光标信息"""
    # 获取当前光标句柄
    current_cursor = GetCursor()
    
    # 获取光标信息
    cursor_info = CURSORINFO()
    cursor_info.cbSize = ctypes.sizeof(CURSORINFO)
    GetCursorInfo(ctypes.byref(cursor_info))
    
    # 获取系统标准光标句柄用于比较
    standard_cursors = {}
    for cursor_id, name in CURSOR_TYPES.items():
        handle = win32gui.LoadCursor(0, cursor_id)
        standard_cursors[name] = handle
    
    # 尝试获取图标信息
    icon_info = ICONINFO()
    icon_info_success = False
    if cursor_info.hCursor:
        icon_info_success = GetIconInfo(cursor_info.hCursor, ctypes.byref(icon_info))
    
    # 获取窗口信息
    foreground_window = win32gui.GetForegroundWindow()
    window_class = win32gui.GetClassName(foreground_window)
    window_text = win32gui.GetWindowText(foreground_window)
    window_rect = win32gui.GetWindowRect(foreground_window)
    window_style = win32gui.GetWindowLong(foreground_window, win32con.GWL_STYLE)
    
    details = {
        'cursor': {
            'current_handle': hex(current_cursor) if current_cursor else None,
            'info_handle': hex(cursor_info.hCursor) if cursor_info.hCursor else None,
            'flags': cursor_info.flags,
            'flags_hex': hex(cursor_info.flags),
            'screen_pos': (cursor_info.ptScreenPos.x, cursor_info.ptScreenPos.y),
            'is_standard': cursor_info.hCursor in [h for h in standard_cursors.values()],
            'matches_standard': [name for name, handle in standard_cursors.items() 
                               if handle == cursor_info.hCursor],
        },
        'icon_info': {
            'success': icon_info_success,
            'is_icon': icon_info.fIcon if icon_info_success else None,
            'hotspot': (icon_info.xHotspot, icon_info.yHotspot) if icon_info_success else None,
            'has_mask': bool(icon_info.hbmMask) if icon_info_success else None,
            'has_color': bool(icon_info.hbmColor) if icon_info_success else None,
        },
        'window': {
            'handle': foreground_window,
            'class': window_class,
            'title': window_text,
            'rect': window_rect,
            'style': hex(window_style),
            'is_fullscreen': not (window_style & win32con.WS_BORDER),
        },
        'system': {
            'screen_size': (win32api.GetSystemMetrics(0), win32api.GetSystemMetrics(1)),
            'cursor_showing': win32api.GetSystemMetrics(win32con.SM_MOUSEPRESENT),
        }
    }
    
    return details

if __name__ == "__main__":
    print("开始测试鼠标光标检测...")
    
    while True:
        details = get_detailed_cursor_info()
        print("\n" + "="*50)
        print("详细光标信息:")
        for category, info in details.items():
            print(f"\n{category.upper()}:")
            for key, value in info.items():
                print(f"  {key}: {value}")
        print("="*50 + "\n")
        time.sleep(1)

