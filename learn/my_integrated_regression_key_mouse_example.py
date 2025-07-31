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

from my_model_arch.my_l2cs.pipeline import IntegratedRegressionWithKeys9Num
import torch
from pathlib import Path
from multiprocessing import freeze_support
import os
import tkinter as tk
import math
import time
from threading import Thread, Lock
import threading
from multiprocessing import Process, Queue

class SectorWheel:
    def __init__(self, messagebox, radius=100):
        self.messagebox = messagebox
        self.radius = radius
        self.selected_sector = None

        self.canvas = tk.Canvas(self.messagebox, width=2 * self.radius, height=2 * self.radius, bg='white',
                                highlightthickness=0)
        self.canvas.pack()

        self.canvas.bind('<Motion>', self.on_mouse_move)

    def update_categories(self, categories):
        self.categories = categories
        self.num_sectors = len(categories)
        self.draw_sectors()
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

    def on_mouse_move(self, event):
        angle = self.get_angle_from_mouse_position(event)
        if angle is not None:
            sector = int(angle // (360 / self.num_sectors))
        else:
            sector = None
        self.draw_sectors(highlighted_sector=sector)

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
        self.messagebox.withdraw()  # Initially hidden

        self.sector_wheel = SectorWheel(self.messagebox, radius=400)
        self.lock = Lock()
        self.should_run = True
        self.current_categories = None
        self.radius = 400
        self.setup_messagebox()
        main_thread = Thread(target=self.main_loop)
        main_thread.start()
        # Thread(target=self.root.mainloop).start()
        self.root.mainloop()
        main_thread.join()

    def main_loop(self):
        count = 0
        while self.should_run:
            time.sleep(2)  # 每2秒检查一次
            new_categories = self.generate_parameters()
            count += 1
            selected = None
            with self.lock:
                # open
                self.sector_wheel.update_categories(new_categories)
                print(f"New categories: {new_categories}")
                self.current_categories = new_categories
                time.sleep(1)
                # close and get result
                self.sector_wheel.hide()
                selected = self.sector_wheel.selected_sector
                self.current_categories = None
                print(f"Selected sector: {selected}")

    def generate_parameters(self):
        import random
        return ["Category" + str(i) for i in range(random.randint(3, 6))]

    def stop(self):
        self.should_run = False
        self.root.quit()

if __name__ == '__main__':
    freeze_support()
    CWD=Path(__file__).parent

    pipeline = IntegratedRegressionWithKeys9Num(
        regression_model='lassocv',
        weights=CWD / 'models' / 'L2CSNet_gaze360.pkl',
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        func='random',
        render_in_eval=False,
        kalman_filter_std_measurement=2,
        num_points=16,
        images_freq=25,
        each_point_wait_time=1000,
        every_point_has_n_images=25,
    )
    root_dir = 'calibration'
    all_dirs = os.listdir(root_dir)
    useful_dirs = []
    for _dir in all_dirs:
        images = os.listdir(os.path.join(root_dir, _dir, 'images'))
        if len(images) == 400:
            useful_dirs.append(os.path.join(root_dir, _dir, 'train_data.jsonl'))
    pipeline.load_model('model_weights/20240924_092902/model.pkl')
    # pipeline.train_lot_data(
    #     jsonl_path_list=useful_dirs,
    #     model_save_path='models/eye_tracking.pkl'
    # )
    pipeline.demo(0,start_with_calibration=False)
    # threading.Thread(target=pipeline.demo, args=(0, False)).start()
    # pipeline.demo(0,start_with_calibration=True)

    # controller = MainController()
    # # controller.run()
    # Thread(target=controller.run).start()  # 这个也没问题
