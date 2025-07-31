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
import time
from threading import Thread, Lock

class SectorWheel:
    def __init__(self, messagebox, radius=100):
        self.messagebox = messagebox
        self.radius = radius
        self.selected_sector = None
        self.layout_type = 'circle'

        self.canvas = tk.Canvas(self.messagebox, width=2 * self.radius, height=2 * self.radius, bg='white',
                                highlightthickness=0)
        self.canvas.pack()

        self.canvas.bind('<Motion>', self.on_mouse_move)

    def update_categories(self, categories, layout_type='circle'):
        self.categories = categories
        self.num_sectors = len(categories)
        self.layout_type = layout_type
        if layout_type == 'circle':
            self.draw_sectors()
        elif layout_type == 'square':
            self.draw_square()
        self.messagebox.deiconify()

    def draw_sectors(self, highlighted_sector=None):
        self.selected_sector = self.categories[highlighted_sector] if highlighted_sector is not None else None
        self.canvas.delete("all")
        for i in range(self.num_sectors):
            fill_color = 'lightgray' if i == highlighted_sector else 'white'
            outline_color = 'white'
            start_angle = i * (360 / self.num_sectors) + 90
            self.canvas.create_arc(
                0, 0, 2 * self.radius, 2 * self.radius,
                start=start_angle, extent=360 / self.num_sectors,
                fill=fill_color, outline=outline_color, tags=f'sector{i}',
                width=0
            )
            text_angle = math.radians(start_angle + 180 / self.num_sectors)
            text_x = self.radius + 0.7 * self.radius * math.cos(text_angle)
            text_y = self.radius - 0.7 * self.radius * math.sin(text_angle)
            self.canvas.create_text(text_x, text_y, text=self.categories[i], font=("Arial", 14, "bold"))

    def draw_square(self, highlighted_sector=None):
        self.selected_sector = self.categories[highlighted_sector] if highlighted_sector is not None else None
        self.canvas.delete("all")
        rows, cols = 5, 6
        square_size = 2 * self.radius / max(rows, cols)
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                if idx < self.num_sectors:
                    x1 = j * square_size
                    y1 = i * square_size
                    x2 = x1 + square_size
                    y2 = y1 + square_size
                    fill_color = 'lightgray' if idx == highlighted_sector else 'white'
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill_color, outline='black')
                    self.canvas.create_text((x1 + x2) / 2, (y1 + y2) / 2, text=self.categories[idx], font=("Arial", 14, "bold"))

    def on_mouse_move(self, event):
        if self.layout_type == 'circle':
            angle = self.get_angle_from_mouse_position(event)
            if angle is not None:
                sector = int(angle // (360 / self.num_sectors))
            else:
                sector = None
            self.draw_sectors(highlighted_sector=sector)
        elif self.layout_type == 'square':
            sector = self.get_square_from_mouse_position(event)
            self.draw_square(highlighted_sector=sector)

    def get_square_from_mouse_position(self, event):
        rows, cols = 5, 6
        square_size = 2 * self.radius / max(rows, cols)
        col = int(event.x // square_size)
        row = int(event.y // square_size)
        if 0 <= col < cols and 0 <= row < rows:
            idx = row * cols + col
            if idx < self.num_sectors:
                return idx
        return None

    def get_angle_from_mouse_position(self, event):
        x = event.x - self.canvas.winfo_width() // 2
        y = self.canvas.winfo_height() // 2 - event.y
        angle = (math.degrees(math.atan2(y, x)) - 90) % 360
        if x ** 2 + y ** 2 > self.radius ** 2:
            return None
        return angle

    def hide(self):
        self.messagebox.withdraw()

class MainController:
    def __init__(self):
        pass

    def setup_messagebox(self):
        screen_width = self.messagebox.winfo_screenwidth()
        screen_height = self.messagebox.winfo_screenheight()
        x_cordinate = int((screen_width / 2) - self.radius)
        y_cordinate = int((screen_height / 2) - self.radius)
        self.messagebox.geometry(f"{2 * self.radius}x{2 * self.radius}+{x_cordinate}+{y_cordinate}")

    def run(self):
        self.root = tk.Tk()
        self.root.withdraw()

        self.messagebox = tk.Toplevel(self.root)
        self.messagebox.overrideredirect(True)
        self.messagebox.attributes('-topmost', True)
        self.messagebox.attributes('-alpha', 0.5)
        self.messagebox.withdraw()

        self.sector_wheel = SectorWheel(self.messagebox, radius=400)
        self.lock = Lock()
        self.should_run = True
        self.current_categories = None
        self.radius = 400
        self.setup_messagebox()
        main_thread = Thread(target=self.main_loop)
        main_thread.start()
        self.root.mainloop()
        main_thread.join()

    def main_loop(self):
        count = 0
        while self.should_run:
            time.sleep(1)
            new_categories = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                              'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
                              ',', '.', '!', '?']
            count += 1
            selected = None
            with self.lock:
                self.sector_wheel.update_categories(new_categories, layout_type='square')
                print(f"New categories: {new_categories}")
                self.current_categories = new_categories
                time.sleep(3)
                self.sector_wheel.hide()
                selected = self.sector_wheel.selected_sector
                self.current_categories = None
                print(f"Selected sector: {selected}")

    def stop(self):
        self.should_run = False
        self.root.quit()

if __name__ == "__main__":
    controller = MainController()
    Thread(target=controller.run).start()