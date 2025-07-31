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

import tkinter as tk
import time


def popout_fading_window(text="NumLock is ON",duration=1, fontsize=20,window_width=200,window_height=100):
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


if __name__ == "__main__":
    popout_fading_window()
    state=True
    for i in range(10):
        state=not state
        print(state)
