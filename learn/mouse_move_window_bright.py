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
import math


def get_angle_from_mouse_position(canvas, event):
    # 计算鼠标位置与圆心之间的角度
    x = event.x - canvas.winfo_width() // 2
    y = canvas.winfo_height() // 2 - event.y
    angle = (math.degrees(math.atan2(y, x))) % 360
    print(angle,event.x**2+event.y**2)
    if x**2+y**2 > radius**2:
        return None
    return angle


def draw_sectors(canvas, highlighted_sector=None):
    # 绘制所有扇形区域，并根据需要突出显示一个扇形
    for i in range(num_sectors):
        fill_color = 'lightgray' if i == highlighted_sector else 'white'
        outline_color = 'white'  # 使用相同的颜色作为轮廓以避免缝隙
        canvas.create_arc(
            0, 0,  # 使用Canvas的左上角作为起点
            2*radius, 2*radius,  # 使用Canvas的宽度和高度作为终点
            start=i * (360 // num_sectors), extent=360 // num_sectors,
            fill=fill_color, outline=outline_color, tags=f'sector{i}',
            width=0  # 设置arc的边框宽度为0
        )


def on_mouse_move(event):
    # 获取鼠标位置对应的角度
    angle = get_angle_from_mouse_position(canvas, event)
    # 计算当前鼠标位置对应的扇形区域
    if angle is not None:
        sector = angle // (360 // num_sectors)
    else:
        sector = None
    # 当鼠标位置不在这个圆形里面，sector就是None
    draw_sectors(canvas, highlighted_sector=sector)

# 初始化参数
num_sectors = 3  # 扇形区域的数量
radius = 100  # 圆的半径

# 创建一个tkinter窗口
root = tk.Tk()
root.withdraw()  # 隐藏主窗口

# 弹出消息框
messagebox = tk.Toplevel(root)
messagebox.overrideredirect(True)  # 移除边框和标题栏
messagebox.attributes('-topmost', True)  # 确保窗口始终在最前面
messagebox.attributes('-alpha', 0.5)  # 设置窗口的透明度
screen_width = messagebox.winfo_screenwidth()
screen_height = messagebox.winfo_screenheight()
x_cordinate = int((screen_width / 2) - radius)
y_cordinate = int((screen_height / 2) - radius)
messagebox.geometry(f"{2*radius}x{2*radius}+{x_cordinate}+{y_cordinate}")

# 创建一个Canvas组件
canvas = tk.Canvas(messagebox, width=2*radius, height=2*radius, bg='white', highlightthickness=0)
canvas.pack()

# 绘制扇形区域
draw_sectors(canvas)

# 绑定鼠标移动事件
canvas.bind('<Motion>', on_mouse_move)

# 显示消息框
messagebox.deiconify()
messagebox.update()

# 运行tkinter事件循环
messagebox.mainloop()
