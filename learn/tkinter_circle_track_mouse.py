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

class CircleTrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Circle Tracker App")

        # 创建一个Canvas来绘制圆
        self.canvas = tk.Canvas(self.root, width=400, height=400)
        self.canvas.pack()

        # 圆的半径
        self.radius = 20

        # 在Canvas上绘制圆
        self.circle = self.canvas.create_oval(0, 0, self.radius * 2, self.radius * 2, fill='red', outline='gray')

        # 绑定鼠标移动事件
        self.root.bind('<Motion>', self.update_circle_position)

    def update_circle_position(self, event):
        # 更新Canvas上圆的位置来追踪鼠标
        self.canvas.coords(self.circle, event.x - self.radius, event.y - self.radius,
                           event.x + self.radius, event.y + self.radius)

# 创建主窗口
root = tk.Tk()
app = CircleTrackerApp(root)

# 运行主循环
root.mainloop()
