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

"""
键盘控制工具模块 (Keyboard Control Utilities)

这个模块提供了一套完整的键盘控制接口，用于模拟键盘输入和检测键盘状态。
使用 Windows API (win32api) 实现，支持所有标准键盘按键。

主要功能：
1. 模拟键盘按键的按下(keydown)
2. 模拟键盘按键的释放(keyup)
3. 模拟完整的按键过程(keypress)
4. 检测按键的当前状态(is_key_down)

支持的按键类型：
- 字母键 (a-z)
- 数字键 (0-9)
- 功能键 (F1-F12)
- 特殊键 (Shift, Ctrl, Alt, Space, Enter等)
- 数字小键盘 (Numpad 0-9及运算符)
- 方向键 (上下左右)
- 编辑和导航键 (Insert, Delete, Home, End等)
- 符号键 (-, =, [, ], \, ;, ', ,, ., /)
- 多媒体键 (Print Screen, Scroll Lock, Pause)
- Windows特殊键 (Win, Menu)

已测试功能：
1. 所有按键的 keydown/keyup 操作
2. 组合键操作（如Shift+A）
3. 按键状态检测
4. 特殊符号键的映射
5. 持续时间可控的按键操作

使用示例：
    # 按下并释放A键
    keypress('a')
    
    # 按住Shift键
    keydown('shift')
    
    # 检查某个键是否被按下
    if is_key_down('a'):
        print('A键被按下')
    
    # 释放之前按下的键
    keyup('shift')

注意事项：
1. 所有按键名称都是小写
2. 对于Shift等有左右之分的按键，默认使用左侧按键
3. 按键持续时间可以通过duration参数调整
4. 未知按键会抛出ValueError异常
"""

import win32api
import win32con
import time
from dataclasses import dataclass
from enum import Enum, auto
from ctypes import windll
import ctypes
from ctypes import wintypes, windll, Structure, POINTER, WINFUNCTYPE
import win32gui
import win32process
import threading
import pythoncom
import keyboard as kb

# 定义缺失的 Windows 类型
LRESULT = ctypes.c_long
HMODULE = wintypes.HANDLE
HHOOK = wintypes.HANDLE
ULONG_PTR = ctypes.c_ulong

class OpType(Enum):
    """键盘操作类型"""
    KEYDOWN = auto()      # 按下按键
    KEYDOWN_SAFE = auto() # 安全按下按键
    KEYUP = auto()        # 释放按键
    KEYPRESS = auto()     # 完整的按下释放过程
    KEYUP_SAFE = auto()   # 安全释放按键
    NONE = auto()         # 不执行任何操作


@dataclass
class Action:
    """
    键盘动作数据类
    
    属性:
        keyname: str - 按键名称，必须是 KEY_MAP 中定义的键名
        op_type: OpType - 操作类型，定义在 OpType 枚举中
        duration: float = 0.01 - 按键持续时间（秒），仅在 KEYPRESS 类型中使用
        ensure_release: bool = True - 是否确保按键释放，仅在 KEYUP_SAFE 类型中使用
        timeout: float = 1.0 - 确保释放的超时时间，仅在 KEYUP_SAFE 类型中使用
        check_interval: float = 0.01 - 检查释放状态的时间间隔，仅在 KEYUP_SAFE 类型中使用
    """
    keyname: str
    op_type: OpType
    duration: float = 0.01
    ensure_release: bool = True
    timeout: float = 0.1
    check_interval: float = 0.01

    # def __post_init__(self):
    #     """验证按键名称的有效性"""
        # if self.keyname.lower() not in KEY_MAP:
        #     raise ValueError(f"未知的按键: {self.keyname}")
        # self.keyname = self.keyname.lower()

    def execute(self):
        """执行键盘动作"""
        if self.op_type == OpType.KEYDOWN:
            keydown(self.keyname)
        elif self.op_type == OpType.KEYDOWN_SAFE:
            return keydown_safe(self.keyname)  # 返回是否成功按下
        elif self.op_type == OpType.KEYUP:
            keyup(self.keyname)
        elif self.op_type == OpType.KEYPRESS:
            keypress(self.keyname, self.duration)
        elif self.op_type == OpType.KEYUP_SAFE:
            keyup_safe(self.keyname, self.ensure_release, self.timeout, self.check_interval)


# 键盘按键射字典，包含虚拟键码和扫描码
KEY_MAP = {
    # 字母键
    'a': {'vk': ord('A'), 'scan': 0x1E},
    'b': {'vk': ord('B'), 'scan': 0x30},
    'c': {'vk': ord('C'), 'scan': 0x2E},
    'd': {'vk': ord('D'), 'scan': 0x20},
    'e': {'vk': ord('E'), 'scan': 0x12},
    'f': {'vk': ord('F'), 'scan': 0x21},
    'g': {'vk': ord('G'), 'scan': 0x22},
    'h': {'vk': ord('H'), 'scan': 0x23},
    'i': {'vk': ord('I'), 'scan': 0x17},
    'j': {'vk': ord('J'), 'scan': 0x24},
    'k': {'vk': ord('K'), 'scan': 0x25},
    'l': {'vk': ord('L'), 'scan': 0x26},
    'm': {'vk': ord('M'), 'scan': 0x32},
    'n': {'vk': ord('N'), 'scan': 0x31},
    'o': {'vk': ord('O'), 'scan': 0x18},
    'p': {'vk': ord('P'), 'scan': 0x19},
    'q': {'vk': ord('Q'), 'scan': 0x10},
    'r': {'vk': ord('R'), 'scan': 0x13},
    's': {'vk': ord('S'), 'scan': 0x1F},
    't': {'vk': ord('T'), 'scan': 0x14},
    'u': {'vk': ord('U'), 'scan': 0x16},
    'v': {'vk': ord('V'), 'scan': 0x2F},
    'w': {'vk': ord('W'), 'scan': 0x11},
    'x': {'vk': ord('X'), 'scan': 0x2D},
    'y': {'vk': ord('Y'), 'scan': 0x15},
    'z': {'vk': ord('Z'), 'scan': 0x2C},
    
    # 数字键
    '0': {'vk': ord('0'), 'scan': 0x0B},
    '1': {'vk': ord('1'), 'scan': 0x02},
    '2': {'vk': ord('2'), 'scan': 0x03},
    '3': {'vk': ord('3'), 'scan': 0x04},
    '4': {'vk': ord('4'), 'scan': 0x05},
    '5': {'vk': ord('5'), 'scan': 0x06},
    '6': {'vk': ord('6'), 'scan': 0x07},
    '7': {'vk': ord('7'), 'scan': 0x08},
    '8': {'vk': ord('8'), 'scan': 0x09},
    '9': {'vk': ord('9'), 'scan': 0x0A},
    
    # 功能键
    'f1': {'vk': win32con.VK_F1, 'scan': 0x3B},
    'f2': {'vk': win32con.VK_F2, 'scan': 0x3C},
    'f3': {'vk': win32con.VK_F3, 'scan': 0x3D},
    'f4': {'vk': win32con.VK_F4, 'scan': 0x3E},
    'f5': {'vk': win32con.VK_F5, 'scan': 0x3F},
    'f6': {'vk': win32con.VK_F6, 'scan': 0x40},
    'f7': {'vk': win32con.VK_F7, 'scan': 0x41},
    'f8': {'vk': win32con.VK_F8, 'scan': 0x42},
    'f9': {'vk': win32con.VK_F9, 'scan': 0x43},
    'f10': {'vk': win32con.VK_F10, 'scan': 0x44},
    'f11': {'vk': win32con.VK_F11, 'scan': 0x57},
    'f12': {'vk': win32con.VK_F12, 'scan': 0x58},
    
    # 特殊键
    'shift': {'vk': win32con.VK_LSHIFT, 'scan': 0x2A},
    'ctrl': {'vk': win32con.VK_LCONTROL, 'scan': 0x1D},
    'alt': {'vk': win32con.VK_LMENU, 'scan': 0x38},
    'space': {'vk': win32con.VK_SPACE, 'scan': 0x39},
    'enter': {'vk': win32con.VK_RETURN, 'scan': 0x1C},
    'backspace': {'vk': win32con.VK_BACK, 'scan': 0x0E},
    'tab': {'vk': win32con.VK_TAB, 'scan': 0x0F},
    'esc': {'vk': win32con.VK_ESCAPE, 'scan': 0x01},
    'caps_lock': {'vk': win32con.VK_CAPITAL, 'scan': 0x3A},

    # 数字小键盘
    'numpad0': {'vk': win32con.VK_NUMPAD0, 'scan': 0x52},
    'numpad1': {'vk': win32con.VK_NUMPAD1, 'scan': 0x4F},
    'numpad2': {'vk': win32con.VK_NUMPAD2, 'scan': 0x50},
    'numpad3': {'vk': win32con.VK_NUMPAD3, 'scan': 0x51},
    'numpad4': {'vk': win32con.VK_NUMPAD4, 'scan': 0x4B},
    'numpad5': {'vk': win32con.VK_NUMPAD5, 'scan': 0x4C},
    'numpad6': {'vk': win32con.VK_NUMPAD6, 'scan': 0x4D},
    'numpad7': {'vk': win32con.VK_NUMPAD7, 'scan': 0x47},
    'numpad8': {'vk': win32con.VK_NUMPAD8, 'scan': 0x48},
    'numpad9': {'vk': win32con.VK_NUMPAD9, 'scan': 0x49},
    'numpad_multiply': {'vk': win32con.VK_MULTIPLY, 'scan': 0x37},
    'numpad_add': {'vk': win32con.VK_ADD, 'scan': 0x4E},
    'numpad_subtract': {'vk': win32con.VK_SUBTRACT, 'scan': 0x4A},
    'numpad_decimal': {'vk': win32con.VK_DECIMAL, 'scan': 0x53},
    'numpad_divide': {'vk': win32con.VK_DIVIDE, 'scan': 0xB5},
    'num_lock': {'vk': win32con.VK_NUMLOCK, 'scan': 0x45},

    # 方向键
    'left': {'vk': win32con.VK_LEFT, 'scan': 0xCB},
    'right': {'vk': win32con.VK_RIGHT, 'scan': 0xCD},
    'up': {'vk': win32con.VK_UP, 'scan': 0xC8},
    'down': {'vk': win32con.VK_DOWN, 'scan': 0xD0},

    # 编辑和导航键
    'insert': {'vk': win32con.VK_INSERT, 'scan': 0xD2},
    'delete': {'vk': win32con.VK_DELETE, 'scan': 0xD3},
    'home': {'vk': win32con.VK_HOME, 'scan': 0xC7},
    'end': {'vk': win32con.VK_END, 'scan': 0xCF},
    'page_up': {'vk': win32con.VK_PRIOR, 'scan': 0xC9},
    'page_down': {'vk': win32con.VK_NEXT, 'scan': 0xD1},

    # 符号键
    '-': {'vk': 0xBD, 'scan': 0x0C},     # - (连字符)
    '=': {'vk': 0xBB, 'scan': 0x0D},      # = (等号)
    ',': {'vk': 0xBC, 'scan': 0x33},     # ,
    '.': {'vk': 0xBE, 'scan': 0x34},    # .
    ';': {'vk': 0xBA, 'scan': 0x27}, # ;
    '/': {'vk': 0xBF, 'scan': 0x35},     # /
    '`': {'vk': 0xC0, 'scan': 0x29},  # `
    '[': {'vk': 0xDB, 'scan': 0x1A},  # [
    '\\': {'vk': 0xDC, 'scan': 0x2B},    # \
    ']': {'vk': 0xDD, 'scan': 0x1B}, # ]
    "'": {'vk': 0xDE, 'scan': 0x28},        # '

    # 多媒体键
    'print_screen': {'vk': win32con.VK_PRINT, 'scan': 0x37},
    'scroll_lock': {'vk': win32con.VK_SCROLL, 'scan': 0x46},
    'pause': {'vk': win32con.VK_PAUSE, 'scan': 0xC5},

    # 右侧控制键
    'right_shift': {'vk': win32con.VK_RSHIFT, 'scan': 0x36},
    'right_ctrl': {'vk': win32con.VK_RCONTROL, 'scan': 0x1D},
    'right_alt': {'vk': win32con.VK_RMENU, 'scan': 0x38},

    # Windows特殊键
    'win': {'vk': win32con.VK_LWIN, 'scan': 0xDB},
    'right_win': {'vk': win32con.VK_RWIN, 'scan': 0xDC},
    'apps': {'vk': win32con.VK_APPS, 'scan': 0xDD},

    # 鼠标按键
    'mouse_left': {
        'vk': win32con.VK_LBUTTON,
        'scan': 0,
        'is_mouse': True,
        'down_flag': win32con.MOUSEEVENTF_LEFTDOWN,
        'up_flag': win32con.MOUSEEVENTF_LEFTUP
    },
    'mouse_right': {
        'vk': win32con.VK_RBUTTON,
        'scan': 0,
        'is_mouse': True,
        'down_flag': win32con.MOUSEEVENTF_RIGHTDOWN,
        'up_flag': win32con.MOUSEEVENTF_RIGHTUP
    },
    'mouse_middle': {
        'vk': win32con.VK_MBUTTON,
        'scan': 0,
        'is_mouse': True,
        'down_flag': win32con.MOUSEEVENTF_MIDDLEDOWN,
        'up_flag': win32con.MOUSEEVENTF_MIDDLEUP
    },
    'mouse_x1': {
        'vk': 0x05,  # VK_XBUTTON1
        'scan': 0,
        'is_mouse': True,
        'down_flag': win32con.MOUSEEVENTF_XDOWN,
        'up_flag': win32con.MOUSEEVENTF_XUP,
        'x_flag': 0x0001  # XBUTTON1
    },
    'mouse_x2': {
        'vk': 0x06,  # VK_XBUTTON2
        'scan': 0,
        'is_mouse': True,
        'down_flag': win32con.MOUSEEVENTF_XDOWN,
        'up_flag': win32con.MOUSEEVENTF_XUP,
        'x_flag': 0x0002  # XBUTTON2
    }
}

def keydown(key):
    """
    模拟按键或鼠标按下
    :param key: 按键名称（小写），如'a', 'shift', 'mouse_left'等
    """
    print(f'type:{type(key)}')
    if key.lower() not in KEY_MAP:
        raise ValueError(f"未知的按键: {key}")
    
    key_info = KEY_MAP[key.lower()]
    if key_info.get('is_mouse', False):
        # 鼠标按键
        if 'x_flag' in key_info:  # X1 或 X2 按钮
            win32api.mouse_event(key_info['down_flag'], 0, 0, key_info['x_flag'], 0)
        else:
            win32api.mouse_event(key_info['down_flag'], 0, 0, 0, 0)
    else:
        # 键盘按键
        win32api.keybd_event(key_info['vk'], key_info['scan'], 0, 0)

def keyup(key):
    """
    模拟按键或鼠标释放
    :param key: 按键名称（小写），如'a', 'shift', 'mouse_left'等
    """
    if key.lower() not in KEY_MAP:
        raise ValueError(f"未知的按键: {key}")
    
    key_info = KEY_MAP[key.lower()]
    if key_info.get('is_mouse', False):
        # 鼠标按键
        if 'x_flag' in key_info:  # X1 或 X2 按钮
            win32api.mouse_event(key_info['up_flag'], 0, 0, key_info['x_flag'], 0)
        else:
            win32api.mouse_event(key_info['up_flag'], 0, 0, 0, 0)
    else:
        # 键盘按键
        win32api.keybd_event(key_info['vk'], key_info['scan'], win32con.KEYEVENTF_KEYUP, 0)

def keypress(key, duration=0.01):
    """
    模拟完整的按键过程（按下然后释放）
    :param key: 按键名称（小写），如'a', 'shift', 'f1'等
    :param duration: 按键持续时间，单位为秒
    """
    keydown(key)
    time.sleep(duration)
    keyup(key)

def is_key_down(key):
    """
    检查按键或鼠标按键是否处于按下��态
    :param key: 按键名称（小写），如'a', 'shift', 'mouse_left'等
    :return: True 如果按键被按下，False 如果按键未被按下
    """
    if key.lower() not in KEY_MAP:
        raise ValueError(f"未知的按键: {key}")
    
    key_info = KEY_MAP[key.lower()]
    state = win32api.GetAsyncKeyState(key_info['vk'])
    return bool(state & 0x8000)

def keyup_safe(key, ensure_release=True, timeout=1.0, check_interval=0.01):
    """
    安全的按键释放函数，可以确保按键被完全释放
    
    :param key: 按键名称（小写），如'a', 'shift', 'f1'等
    :param ensure_release: 是否确保按键被释放，如果为False则等同于普通的keyup
    :param timeout: 确保释放的最大等待时间（秒），超过这个时间会抛出异常
    :param check_interval: 检查按键状态的时间间隔（秒）
    :raises: ValueError: 当按键未知时
            TimeoutError: 当超过timeout时间按键仍未释放时
    """
    if key.lower() not in KEY_MAP:
        raise ValueError(f"未知的按键: {key}")
    
    # 首先执行正常的keyup操作
    keyup(key)
    
    # 如果不需要确保释放，直接返回
    if not ensure_release:
        return
    
    # 确保按键被释放
    start_time = time.time()
    while is_key_down(key):
        # 如果超时，抛出异常
        if time.time() - start_time > timeout:
            raise TimeoutError(f"按键 {key} 在 {timeout} 秒内未能成功释放")
        
        # 再次尝试释放按键
        keyup(key)
        time.sleep(check_interval)

def keydown_safe(key):
    """
    安全的按键按下函数，只在按键未被按下时执行按下操作
    
    :param key: 按键名称（小写），如'a', 'shift', 'f1'等
    :return: bool - True 如果成功按下，False 如果按键已经处于按下状态
    :raises: ValueError: 当按键未知时
    """
    if key.lower() not in KEY_MAP:
        raise ValueError(f"未知的按键: {key}")
    
    # 检查按键当前状态
    if is_key_down(key):
        return False  # 按键已经处于按下状态，不执行操作
    
    # 确认按键未被按下，执行按下操作
    keydown(key)
    return True

# Raw Input 相关常量
RIDEV_INPUTSINK = 0x00000100
RID_INPUT = 0x10000003
RIM_TYPEKEYBOARD = 1
WM_INPUT = 0x00FF

# Raw Input 结构体定义
class RAWINPUTDEVICE(Structure):
    _fields_ = [
        ("usUsagePage", wintypes.USHORT),
        ("usUsage", wintypes.USHORT),
        ("dwFlags", wintypes.DWORD),
        ("hwndTarget", wintypes.HWND)
    ]

class RAWINPUTHEADER(Structure):
    _fields_ = [
        ("dwType", wintypes.DWORD),
        ("dwSize", wintypes.DWORD),
        ("hDevice", wintypes.HANDLE),
        ("wParam", wintypes.WPARAM)
    ]

class RAWKEYBOARD(Structure):
    _fields_ = [
        ("MakeCode", wintypes.USHORT),
        ("Flags", wintypes.USHORT),
        ("Reserved", wintypes.USHORT),
        ("VKey", wintypes.USHORT),
        ("Message", wintypes.UINT),
        ("ExtraInformation", wintypes.ULONG)
    ]

class RAWINPUT(Structure):
    class _U1(Structure):
        _fields_ = [
            ("keyboard", RAWKEYBOARD)
        ]
    _anonymous_ = ("u1",)
    _fields_ = [
        ("header", RAWINPUTHEADER),
        ("u1", _U1)
    ]

class KeyboardHook:
    def __init__(self):
        self.physical_keys = set()
        self.is_running = False
        self.window_class = self._register_window_class()
        self.hwnd = None
        self.message_thread = None
        
    def _register_window_class(self):
        """注册窗口类"""
        wc = win32gui.WNDCLASS()
        wc.lpfnWndProc = self._window_proc
        wc.lpszClassName = "RawInputWindow"
        return win32gui.RegisterClass(wc)
        
    def _window_proc(self, hwnd, msg, wparam, lparam):
        """窗口消息处理函数"""
        try:
            if msg == WM_INPUT:
                size = ctypes.c_uint()
                result = windll.user32.GetRawInputData(
                    lparam, RID_INPUT, None, 
                    ctypes.byref(size), 
                    ctypes.sizeof(RAWINPUTHEADER)
                )
                print(f"GetRawInputData size result: {result}, size: {size.value}")
                
                raw_buffer = (ctypes.c_ubyte * size.value)()
                result = windll.user32.GetRawInputData(
                    lparam, RID_INPUT, raw_buffer,
                    ctypes.byref(size), 
                    ctypes.sizeof(RAWINPUTHEADER)
                )
                print(f"GetRawInputData buffer result: {result}")
                
                if result == size.value:
                    raw_input = RAWINPUT.from_buffer(raw_buffer)
                    
                    if raw_input.header.dwType == RIM_TYPEKEYBOARD:
                        key_code = raw_input.keyboard.VKey
                        flags = raw_input.keyboard.Flags
                        scan_code = raw_input.keyboard.MakeCode
                        
                        print(f"Raw Input: key_code={key_code}, flags={flags}, scan_code={scan_code}")
                        
                        # 检查是否是物理按键事件
                        if flags & 0x01:  # Key Up
                            print(f"物理按键释放: {key_code} (0x{key_code:02X})")
                            self.physical_keys.discard(key_code)
                        else:  # Key Down
                            print(f"物理按键按下: {key_code} (0x{key_code:02X})")
                            self.physical_keys.add(key_code)
                            
            return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)
        except Exception as e:
            print(f"窗口处理错误: {e}")
            return 0
            
    def start(self):
        if self.is_running:
            return
            
        # 创建隐藏窗口
        self.hwnd = win32gui.CreateWindow(
            self.window_class,
            "RawInput Window",
            0, 0, 0, 0, 0,
            win32con.HWND_MESSAGE,
            None, None, None
        )
        
        # 注册 Raw Input 设备
        rid = RAWINPUTDEVICE(
            0x01,  # UsagePage (Generic Desktop)
            0x06,  # Usage (Keyboard)
            RIDEV_INPUTSINK,  # Flags
            self.hwnd  # Target Window
        )
        
        if not windll.user32.RegisterRawInputDevices(
            ctypes.byref(rid),
            1,
            ctypes.sizeof(RAWINPUTDEVICE)
        ):
            raise RuntimeError("无法注册Raw Input设备")
            
        self.is_running = True
        
        # 启动消息循环
        def message_loop():
            try:
                while self.is_running:
                    win32gui.PumpWaitingMessages()
                    threading.Event().wait(0.001)  # 短暂休眠以减少CPU使用
            except Exception as e:
                print(f"消息循环错误: {e}")
            finally:
                print("消息循环结束")
                
        self.message_thread = threading.Thread(target=message_loop)
        self.message_thread.daemon = True
        self.message_thread.start()
    
    def stop(self):
        """安全停止钩子"""
        if self.is_running:
            self.is_running = False
            # 等待消息循环线程结束
            if self.message_thread and self.message_thread.is_alive():
                self.message_thread.join(timeout=1.0)  # 最多等待1秒
            if self.hwnd:
                try:
                    win32gui.DestroyWindow(self.hwnd)
                except:
                    pass  # 忽略可能的错误
                self.hwnd = None
            self.physical_keys.clear()
            
    def is_physically_pressed(self, key):
        """检查按键是否被物理按下"""
        if not self.is_running:
            return False
            
        if key.lower() not in KEY_MAP:
            raise ValueError(f"未知的按键: {key}")
            
        key_info = KEY_MAP[key.lower()]
        vk = key_info['vk']
        print(f"检查按键 {key}: vk=0x{vk:02X}, physical_keys={[hex(k) for k in self.physical_keys]}")
        return vk in self.physical_keys

# 创建全局钩子实例
keyboard_hook = KeyboardHook()

def is_key_physically_pressed(key):
    """
    检查按键是否是物理按下的（不是通过程序模拟）
    
    :param key: 按键名称（小写），如'a', 'shift', 'f1'等
    :return: bool - True 如果是物理按下，False 如果是模拟按下或未按下
    """
    global keyboard_hook
    if not keyboard_hook.is_running:
        keyboard_hook.start()
    return keyboard_hook.is_physically_pressed(key)

if __name__ == "__main__":
    try:
        print("开始监测键盘...")
        
        # 打印系统信息
        print(f"Python Bits: {ctypes.sizeof(ctypes.c_void_p) * 8}")
        print(f"Current Module Handle: {windll.kernel32.GetModuleHandleW(None)}")
        
        keyboard_hook.start()
        time.sleep(2)
        
        # print("\n模拟按下Q键")
        # print(f"Q键的虚拟键码: 0x{KEY_MAP['q']['vk']:02X}")
        keydown_safe('w')
        
        # print("\n请按下物理Q键退出...")
        # while True:
        #     if is_key_physically_pressed('q'):
        #         print('检测到Q键被物理按下')
        #         break
        #     time.sleep(0.1)
            
    except (KeyboardInterrupt, Exception) as e:
        if isinstance(e, KeyboardInterrupt):
            print("\n程序被用户中断")
        else:
            print(f"发生错误: {e}")
    finally:
        keyboard_hook.stop()
        print("程序退出")
