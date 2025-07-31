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

from PySide6.QtWidgets import (QApplication, QMainWindow, QTabWidget, 
                              QWidget, QVBoxLayout, QHBoxLayout, QLayout, 
                              QLabel, QSpinBox, QComboBox, QLineEdit,
                              QPushButton, QTreeWidget, QTreeWidgetItem, QDoubleSpinBox,
                              QFileDialog, QMessageBox, QMenu, QGroupBox, QFormLayout, QScrollArea,
                              QDialog, QDialogButtonBox, QFrame, QSizePolicy, QWidgetItem,
                              QTreeWidgetItemIterator, QTableWidget, QHeaderView, QCheckBox, QProgressDialog)
from PySide6.QtCore import Qt, QTimer, QSize, QRect, QPoint, QEvent
from PySide6.QtGui import QAction, QIcon, QPainter, QPen, QColor
import sys
import yaml
import os
from pathlib import Path
import cv2
from PySide6.QtGui import QImage, QPixmap
import keyboard  # 添加到文件开头的导入部分
import time  # 添加到文件开头的导入部分
import warnings
warnings.filterwarnings("ignore")

class SymbolWidget(QWidget):
    def __init__(self, text="", parent=None):
        super().__init__(parent)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        
        # 文本编辑框
        self.edit = QLineEdit(text)
        self.edit.setFixedWidth(40)
        layout.addWidget(self.edit)
        
        # 删除按钮
        self.delete_btn = QPushButton("×")
        self.delete_btn.setFixedSize(20, 20)
        layout.addWidget(self.delete_btn)
        
        self.setLayout(layout)
        self.setFixedHeight(25)

    def sizeHint(self):
        return QSize(70, 25)

    def minimumSizeHint(self):
        return self.sizeHint()

class SymbolListWidget(QWidget):
    """符号列表组件"""
    def __init__(self, symbols=None, parent=None):
        super().__init__(parent)
        self.layout = QHBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(2)
        
        self.symbols_layout = FlowLayout()
        self.symbols_layout.setSpacing(2)
        self.layout.addLayout(self.symbols_layout)
        
        self.setLayout(self.layout)
        
        # 保存原始符号顺序
        self.original_symbols = []
        
        # 先创建所有符号部件但不添加
        self.symbol_widgets = []
        if symbols and isinstance(symbols, list):
            self.original_symbols = list(symbols)
            for symbol in symbols:
                symbol_widget = SymbolWidget(str(symbol))
                symbol_widget.delete_btn.clicked.connect(lambda w=symbol_widget: self.on_symbol_deleted(w))
                self.symbol_widgets.append(symbol_widget)
        
        # 按顺序添加所有符号部件
        for widget in self.symbol_widgets:
            self.symbols_layout.addWidget(widget)
        
        # 最后添加加号按钮
        self.add_btn = QPushButton("+")
        self.add_btn.setFixedSize(20, 20)
        self.add_btn.clicked.connect(lambda: self.add_symbol(""))
        self.symbols_layout.addWidget(self.add_btn)

    def add_symbol(self, text=""):
        """添加新的符号"""
        symbol_widget = SymbolWidget(text)
        # 在最后一个位置（加号按钮之前）添加新符号
        self.symbols_layout.insertWidget(self.symbols_layout.count() - 1, symbol_widget)
        symbol_widget.delete_btn.clicked.connect(lambda: self.on_symbol_deleted(symbol_widget))
        self.update_height()

    def on_symbol_deleted(self, widget):
        """当符号被删除时"""
        widget.deleteLater()
        self.symbols_layout.invalidate()
        self.update_height()

    def get_symbols(self):
        """获取所有符号，保持原始顺序"""
        try:
            print("Starting get_symbols")
            if not hasattr(self, 'symbols_layout'):
                print("No symbols_layout found")
                return []
                
            print(f"Layout count: {self.symbols_layout.count()}")
            current_symbols = []
            
            # 直接获取所有符号
            for i in range(self.symbols_layout.count()):
                try:
                    item = self.symbols_layout.itemAt(i)
                    if not item:
                        print(f"No item at index {i}")
                        continue
                        
                    widget = item.widget()
                    if not widget:
                        print(f"No widget in item at index {i}")
                        continue
                        
                    if not isinstance(widget, SymbolWidget):
                        print(f"Widget at index {i} is not a SymbolWidget")
                        continue
                        
                    text = widget.edit.text().strip()
                    if text:
                        current_symbols.append(text)
                        print(f"Added symbol: {text}")
                        
                except Exception as e:
                    print(f"Error processing item {i}: {str(e)}")
                    continue
            
            print(f"Final symbols list: {current_symbols}")
            return current_symbols
            
        except Exception as e:
            print(f"Error in get_symbols: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 不再调用 update_height

    def update_height(self):
        """更新高度以适应内容"""
        if self.parent() and hasattr(self.parent(), 'tree_item'):
            # 计算实际需要的行数
            available_width = 200  # 固定宽度
            button_width = 46  # 单个按钮宽度
            spacing = 2
            buttons_per_row = available_width // (button_width + spacing)
            
            # 计算总按钮数（包括+按钮）
            total_buttons = self.symbols_layout.count()
            
            # 计算需要的行数（向上取整）
            rows_needed = (total_buttons + buttons_per_row - 1) // buttons_per_row
            
            # 计算总高度（增加行间距）
            button_height = 25  # 单个按钮高度
            row_spacing = 4  # 增加行间距
            total_height = rows_needed * (button_height + row_spacing) + 8  # 增加整体边距
            
            # 通知父级更新树项目高度
            self.parent().tree_item.setSizeHint(1, QSize(0, total_height))

    def sizeHint(self):
        """返回建议的大小"""
        # 根据实际内容计算合适的大小
        available_width = 200
        button_width = 46
        spacing = 2
        buttons_per_row = available_width // (button_width + spacing)
        total_buttons = self.symbols_layout.count()
        rows_needed = (total_buttons + buttons_per_row - 1) // buttons_per_row
        button_height = 25
        total_height = rows_needed * (button_height + spacing) - spacing +button_height
        
        return QSize(available_width, total_height)

    def minimumSizeHint(self):
        """返回最小建议大小"""
        return self.sizeHint()

# 添加 FlowLayout 类来支持自动换行
class FlowLayout(QLayout):
    def __init__(self, parent=None, margin=0, spacing=-1):
        super().__init__(parent)
        self.itemList = []
        self.margin = margin
        self.setSpacing(2)

    def __del__(self):
        # 简化析构函数，避免调用可能已被删除的方法
        while self.itemList:
            self.itemList.pop()

    def addItem(self, item):
        self.itemList.append(item)
        self.invalidate()

    def count(self):
        return len(self.itemList)

    def itemAt(self, index):
        """返回指定索引的项目"""
        try:
            if 0 <= index < len(self.itemList):
                return self.itemList[index]
        except Exception as e:
            print(f"Error in itemAt: {str(e)}")
        return None

    def takeAt(self, index):
        """移除并返回指定索引的项目"""
        try:
            if 0 <= index < len(self.itemList):
                return self.itemList.pop(index)
        except Exception as e:
            print(f"Error in takeAt: {str(e)}")
        return None

    def _is_being_deleted(self):
        """检查对象是否正在被删除"""
        try:
            self.parent()
            return False
        except RuntimeError:
            return True

    def expandingDirections(self):
        return Qt.Orientations()

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        height = self.doLayout(QRect(0, 0, width, 0), True)
        return height

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self.doLayout(rect, False)

    def sizeHint(self):
        width = 0
        height = 0
        
        # 计算所有项目所需的最小空间
        for item in self.itemList:
            size = item.sizeHint()
            width = max(width, size.width())
            height += size.height() + self.spacing()
        
        if height > 0:
            height -= self.spacing()  # 移除最后一个spacing
            
        return QSize(width, height)

    def minimumSize(self):
        width = 0
        height = 0
        
        # 计算所有项目所需的最小空间
        for item in self.itemList:
            size = item.minimumSize()
            width = max(width, size.width())
            height += size.height() + self.spacing()
        
        if height > 0:
            height -= self.spacing()
            
        return QSize(width, height)

    def doLayout(self, rect, testOnly):
        """执行布局"""
        try:
            left = rect.x()
            top = rect.y()
            available_width = rect.width()
            x = left
            y = top
            line_height = 0
            row_items = []
            
            for item in self.itemList:
                if not item:  # 跳过无效项目
                    continue
                    
                width = item.sizeHint().width()
                height = item.sizeHint().height()
                
                current_row_width = sum(i.sizeHint().width() + self.spacing() for i in row_items if i)
                if current_row_width + width > available_width and row_items:
                    x = left
                    y += line_height
                    line_height = 0
                    row_items = []
                
                if not testOnly and item.widget():
                    item.setGeometry(QRect(x, y, width, height))
                
                x += width + self.spacing()
                line_height = max(line_height, height)
                row_items.append(item)
            
            return y + line_height - top
            
        except Exception as e:
            print(f"Error in doLayout: {str(e)}")
            return 0

    def insertWidget(self, index, widget):
        """在指定位置插入部件"""
        self.addChildWidget(widget)
        item = QWidgetItem(widget)
        self.itemList.insert(index, item)
        self.invalidate()

class ConfigWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Eye Tracking Configuration")
        self.resize(1000, 800)
        
        # 初始化变量
        self.pipeline = None
        self.cap = None  # 添加摄像头对象初始化
        self.current_config_path = None  # 添加当前配置文件路径追踪
        
        # 启动检查热键的定时器
        self.hotkey_timer = QTimer(self)
        self.hotkey_timer.timeout.connect(self.check_hotkeys)
        self.hotkey_timer.start(100)  # 每100ms检查一次
        
        # 创建菜单栏
        self.setup_menu_bar()
        
        # 加载配置
        try:
            self.load_config()
        except Exception as e:
            QMessageBox.critical(self, "Configuration Error", 
                               f"Failed to load configuration:\n{str(e)}\n\nPlease check your configuration files.")
            sys.exit(1)  # 退出程序
        
        # 获取所有可用的表情名称和特征
        self.expression_list = list(self.config.get('expression_evaluator_config', {}).get('expressions', {}).keys())
        
        # 从配置中提取所有唯一的表情特征
        self.feature_list = set()
        for expr in self.config.get('expression_evaluator_config', {}).get('expressions', {}).values():
            for condition in expr.get('conditions', []):
                if 'feature' in condition:
                    self.feature_list.add(condition['feature'])
        self.feature_list = sorted(list(self.feature_list))  # 转换为排序列表
        
        # 创建主界面
        self.setup_ui()

    def setup_menu_bar(self):
        """设置菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu('File')
        
        # 打开配置文件
        open_action = QAction('Open Configuration...', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.open_config_file)
        file_menu.addAction(open_action)
        
        # 保存配置文件
        save_action = QAction('Save Configuration', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.save_config)
        file_menu.addAction(save_action)
        
        # 另存为配置文件
        save_as_action = QAction('Save Configuration As...', self)
        save_as_action.setShortcut('Ctrl+Shift+S')
        save_as_action.triggered.connect(self.save_config_as)
        file_menu.addAction(save_as_action)
        
        file_menu.addSeparator()
        
        # 新建配置文件
        new_action = QAction('New Configuration from Template', self)
        new_action.setShortcut('Ctrl+N')
        new_action.triggered.connect(self.new_config)
        file_menu.addAction(new_action)
        
        file_menu.addSeparator()
        
        # 最近打开的文件子菜单
        self.recent_menu = file_menu.addMenu('Recent Files')
        self.update_recent_files_menu()
        
        # 更新窗口标题显示当前配置文件
        self.update_window_title()

    def open_config_file(self):
        """打开配置文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Configuration File",
            str(Path(__file__).parent),
            "YAML files (*.yaml *.yml);;All files (*.*)"
        )
        
        if file_path:
            try:
                self.load_config(file_path)
                self.add_to_recent_files(file_path)
                if hasattr(self, 'config_file_combo'):
                    self.refresh_config_files()  # 刷新配置文件列表
                self.refresh_ui()
                QMessageBox.information(self, "Success", f"Configuration loaded from {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load configuration:\n{str(e)}")

    def save_config_as(self):
        """另存为配置文件"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Configuration As",
            str(Path(__file__).parent / "config.yaml"),
            "YAML files (*.yaml *.yml);;All files (*.*)"
        )
        
        if file_path:
            try:
                self.save_config_to_file(file_path)
                self.current_config_path = file_path
                self.add_to_recent_files(file_path)
                self.update_window_title()
                QMessageBox.information(self, "Success", f"Configuration saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save configuration:\n{str(e)}")

    def new_config(self):
        """新建配置文件"""
        reply = QMessageBox.question(
            self, 
            "New Configuration", 
            "This will create a new configuration based on configs/cpu.yaml template. Unsaved changes will be lost. Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.current_config_path = None
            try:
                self.create_default_config()
                self.refresh_ui()
                self.update_window_title()
            except Exception as e:
                QMessageBox.critical(self, "Error", 
                                   f"Failed to create new configuration:\n{str(e)}\n\nPlease check that configs/cpu.yaml exists and is valid.")
                return

    def add_to_recent_files(self, file_path):
        """添加到最近使用的文件列表"""
        # 简单实现，可以后续改进为持久化存储
        if not hasattr(self, 'recent_files'):
            self.recent_files = []
        
        # 如果文件已在列表中，先移除
        if file_path in self.recent_files:
            self.recent_files.remove(file_path)
        
        # 添加到列表开头
        self.recent_files.insert(0, file_path)
        
        # 限制最多10个最近文件
        self.recent_files = self.recent_files[:10]
        
        self.update_recent_files_menu()

    def update_recent_files_menu(self):
        """更新最近文件菜单"""
        self.recent_menu.clear()
        
        if not hasattr(self, 'recent_files') or not self.recent_files:
            no_recent_action = QAction('No recent files', self)
            no_recent_action.setEnabled(False)
            self.recent_menu.addAction(no_recent_action)
            return
        
        for file_path in self.recent_files:
            if Path(file_path).exists():
                action = QAction(Path(file_path).name, self)
                action.setToolTip(file_path)
                action.triggered.connect(lambda checked, path=file_path: self.load_recent_file(path))
                self.recent_menu.addAction(action)

    def load_recent_file(self, file_path):
        """加载最近使用的文件"""
        try:
            self.load_config(file_path)
            self.refresh_ui()
            QMessageBox.information(self, "Success", f"Configuration loaded from {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load configuration:\n{str(e)}")

    def update_window_title(self):
        """更新窗口标题"""
        base_title = "Eye Tracking Configuration"
        if self.current_config_path:
            config_name = Path(self.current_config_path).name
            self.setWindowTitle(f"{base_title} - {config_name}")
        else:
            self.setWindowTitle(f"{base_title} - New Configuration")
        
        # 同时更新配置文件标签
        if hasattr(self, 'current_config_label'):
            self.update_current_config_label()

    def create_default_config(self):
        """创建默认配置"""
        # 从configs/cpu.yaml加载默认配置
        default_config_path = Path(__file__).parent / 'configs' / 'cpu.yaml'
        
        if not default_config_path.exists():
            raise FileNotFoundError(f"Default configuration file not found: {default_config_path}")
        
        try:
            with open(default_config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
                print(f"Loaded default config from: {default_config_path}")
        except Exception as e:
            raise Exception(f"Failed to load default configuration from {default_config_path}: {str(e)}")

    def refresh_ui(self):
        """刷新UI以反映新加载的配置"""
        # 重新获取表情列表
        self.expression_list = list(self.config.get('expression_evaluator_config', {}).get('expressions', {}).keys())
        
        # 重新获取特征列表
        self.feature_list = set()
        for expr in self.config.get('expression_evaluator_config', {}).get('expressions', {}).values():
            for condition in expr.get('conditions', []):
                if 'feature' in condition:
                    self.feature_list.add(condition['feature'])
        self.feature_list = sorted(list(self.feature_list))
        
        # 重新设置UI（这可能需要重新创建标签页）
        # 为了简化，我们可以提示用户重启应用
        reply = QMessageBox.question(
            self, 
            "Configuration Loaded", 
            "Configuration has been loaded. The application needs to be restarted to fully apply the new configuration. Restart now?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        
        if reply == QMessageBox.Yes:
            # 保存当前配置路径到临时文件
            if self.current_config_path:
                temp_path = Path(__file__).parent / '.last_config_path'
                with open(temp_path, 'w') as f:
                    f.write(self.current_config_path)
            
            # 重启应用
            QApplication.quit()
            os.system(f'python "{__file__}"')

    def setup_ui(self):
        """初始化用户界面"""
        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 创建标签页
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # 设置各个标签页（注意顺序）
        self.setup_camera_tab()  # 新添加的摄像头设置标签页
        self.setup_basic_tab()
        self.setup_expression_tab()
        self.setup_keymap_tab()
        
        # 添加保存按钮
        save_btn = QPushButton("Save Configuration")
        save_btn.clicked.connect(self.save_config)
        main_layout.addWidget(save_btn)

    def setup_camera_tab(self):
        """设置摄像头标签页"""
        camera_tab = QScrollArea()
        camera_tab.setWidgetResizable(True)
        
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        
        # 添加预览标签
        self.preview_label = QLabel()
        self.preview_label.setMinimumSize(640, 480)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setText("No camera preview")
        layout.addWidget(self.preview_label)
        
        # 创建预览定时器
        self.preview_timer = QTimer(self)
        self.preview_timer.timeout.connect(self.update_preview)
        
        # 创建可折叠的摄像头选择组
        self.camera_group = QWidget()
        camera_layout = QVBoxLayout(self.camera_group)
        camera_layout.setContentsMargins(0, 0, 0, 0)
        
        # 标题栏
        header = QPushButton("▼ Camera Selection")
        header.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding: 5px;
                border: none;
                background: #f0f0f0;
            }
            QPushButton:hover {
                background: #e0e0e0;
            }
        """)
        header.clicked.connect(self.toggle_camera_section)
        
        # 内容区域
        self.camera_content = QWidget()
        content_layout = QVBoxLayout(self.camera_content)
        content_layout.setContentsMargins(10, 0, 10, 0)
        
        # 摄像头选择区域
        select_layout = QHBoxLayout()
        self.camera_combo = QComboBox()
        self.camera_combo.currentIndexChanged.connect(self.on_camera_selected)
        select_layout.addWidget(QLabel("Select Camera:"))
        select_layout.addWidget(self.camera_combo)
        content_layout.addLayout(select_layout)
        
        # 确认按钮
        confirm_layout = QHBoxLayout()
        self.camera_confirm_btn = QPushButton("Confirm Selection")
        self.camera_confirm_btn.clicked.connect(self.confirm_camera_selection)
        confirm_layout.addWidget(self.camera_confirm_btn)
        
        # 重新选择按钮
        self.camera_change_btn = QPushButton("Change Camera")
        self.camera_change_btn.clicked.connect(self.change_camera_selection)
        self.camera_change_btn.setEnabled(False)
        confirm_layout.addWidget(self.camera_change_btn)
        
        content_layout.addLayout(confirm_layout)
        
        # 将所有元素添加到布局
        camera_layout.addWidget(header)
        camera_layout.addWidget(self.camera_content)
        layout.addWidget(self.camera_group)
        
        # 校准和评估按钮
        buttons_layout = QHBoxLayout()
        
        self.calibrate_btn = QPushButton("Start Calibration")
        self.calibrate_btn.clicked.connect(self.start_calibration)
        buttons_layout.addWidget(self.calibrate_btn)
        
        self.evaluate_btn = QPushButton("Start Evaluation")
        self.evaluate_btn.clicked.connect(self.start_evaluation)
        buttons_layout.addWidget(self.evaluate_btn)
        
        layout.addLayout(buttons_layout)
        
        camera_tab.setWidget(content_widget)
        
        # 获取可用摄像头列表并初始化按钮状态
        self.list_cameras()
        
        # 初始状态：禁用所有按钮
        self.camera_confirm_btn.setEnabled(False)
        self.calibrate_btn.setEnabled(False)
        self.evaluate_btn.setEnabled(False)
        
        # 根据配置文件中的cam_id设置初始状态
        configured_cam_id = self.config['integrated_config'].get('cam_id')
        if configured_cam_id is not None:
            # 如果配置文件中有cam_id，尝试选择对应的摄像头
            for i in range(self.camera_combo.count()):
                if self.camera_combo.itemData(i) == configured_cam_id:
                    self.camera_combo.setCurrentIndex(i)
                    
                    # 直接确认摄像头选择，不显示预览
                    self.config['integrated_config']['cam_id'] = configured_cam_id
                    self.selected_camera_id = configured_cam_id
                    
                    # 更新界面状态
                    self.camera_confirm_btn.setEnabled(False)
                    self.camera_combo.setEnabled(False)
                    self.camera_change_btn.setEnabled(True)
                    self.calibrate_btn.setEnabled(True)
                    self.evaluate_btn.setEnabled(True)
                    
                    # 显示确认信息
                    self.preview_label.setText(f"Camera {configured_cam_id} confirmed from configuration. Ready for calibration/evaluation.")
                    
                    # 自动折叠摄像头选择区域
                    self.camera_section_expanded = False
                    self.camera_content.setVisible(False)
                    header = self.camera_group.layout().itemAt(0).widget()
                    header.setText("▶ Camera Selection")
                    
                    print(f"Auto-confirmed camera {configured_cam_id} from configuration")
                    break
        
        self.tabs.insertTab(0, camera_tab, "Camera Setup")
        self.tabs.setCurrentIndex(0)

    def list_cameras(self):
        """列出可用摄像头"""
        self.camera_combo.clear()
        self.camera_combo.addItem("None", None)
        
        # 首先尝试使用 DirectShow 后端
        for i in range(10):
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        self.camera_combo.addItem(f"Camera {i}", i)
                    cap.release()
            except Exception as e:
                print(f"Error checking camera {i}: {str(e)}")
                continue
        
        # 如果没有找到摄像头，尝试使用其他后端
        if self.camera_combo.count() <= 1:  # 只有 "None" 选项
            backends = [
                cv2.CAP_MSMF,  # Microsoft Media Foundation
                cv2.CAP_DSHOW,  # DirectShow
                cv2.CAP_ANY  # 自动选择
            ]
            
            for backend in backends:
                for i in range(10):
                    try:
                        cap = cv2.VideoCapture(i, backend)
                        if cap.isOpened():
                            ret, _ = cap.read()
                            if ret:
                                self.camera_combo.addItem(f"Camera {i} ({backend})", i)
                            cap.release()
                    except Exception as e:
                        print(f"Error checking camera {i} with backend {backend}: {str(e)}")
                        continue
        
        # 连接选择变化信号
        self.camera_combo.currentIndexChanged.connect(self.on_camera_selected)

    def on_camera_selected(self, index):
        """当用户选择摄像头时"""
        try:
            camera_id = self.camera_combo.currentData()
            
            # 停止之前的预览
            self.preview_timer.stop()
            
            # 如果已经有打开的摄像头，先关闭
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            
            if camera_id is not None:
                # 打开新选择的摄像头，使用与 pipeline 相同的设置
                self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)  # 使用 DirectShow 后端
                if not self.cap.isOpened():
                    QMessageBox.warning(self, "Warning", f"Failed to open camera {camera_id}")
                    self.cap = None
                    self.camera_confirm_btn.setEnabled(False)
                    return
                
                # 设置摄像头属性，与 pipeline 保持一致
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                
                # 尝试读取一帧来确认摄像头是否正常工作
                ret, frame = self.cap.read()
                if not ret:
                    QMessageBox.warning(self, "Warning", f"Camera {camera_id} is not working properly")
                    self.cap.release()
                    self.cap = None
                    self.camera_confirm_btn.setEnabled(False)
                    return
                
                # 开始预览
                self.preview_timer.start(30)  # 30ms 刷新一次，约 33fps
                
                # 启用确认按钮，但校准和评估按钮保持禁用直到确认
                self.camera_confirm_btn.setEnabled(True)
                self.camera_change_btn.setEnabled(True)  # 启用重新选择按钮
            else:
                # 如果选择None，清除预览并禁用所有按钮
                self.preview_label.clear()
                self.preview_label.setText("No camera preview")
                self.camera_confirm_btn.setEnabled(False)
                self.calibrate_btn.setEnabled(False)
                self.evaluate_btn.setEnabled(False)
                
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error handling camera: {str(e)}")
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.preview_timer.stop()
            self.camera_confirm_btn.setEnabled(False)

    def update_preview(self):
        """更新摄像头预览"""
        if self.cap and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                if ret:
                    # 转换为Qt图像
                    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    
                    # 获取预览标签的大小
                    label_size = self.preview_label.size()
                    
                    # 创建缩放后的图像，保持宽高比
                    scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                        label_size,
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    )
                    
                    self.preview_label.setPixmap(scaled_pixmap)
                    
            except Exception as e:
                print(f"Error updating preview: {str(e)}")
                self.preview_timer.stop()

    def confirm_camera_selection(self):
        """确认摄像头选择并更新配置"""
        camera_id = self.camera_combo.currentData()
        
        # 停止预览
        if hasattr(self, 'preview_timer'):
            self.preview_timer.stop()
        
        # 关闭摄像头预览
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        # 清除预览显示
        self.preview_label.clear()
        self.preview_label.setText("Camera confirmed. Preview stopped.")
        
        # 更新配置中的cam_id
        self.config['integrated_config']['cam_id'] = camera_id
        self.selected_camera_id = camera_id
        
        # 更新按钮状态
        self.camera_confirm_btn.setEnabled(False)
        self.camera_combo.setEnabled(False)  # 禁用摄像头选择，防止误操作
        self.camera_change_btn.setEnabled(True)  # 启用重新选择按钮
        
        if camera_id is not None:
            self.calibrate_btn.setEnabled(True)
            self.evaluate_btn.setEnabled(True)
            QMessageBox.information(self, "Success", f"Camera {camera_id} confirmed and ready for calibration/evaluation!")
        else:
            self.calibrate_btn.setEnabled(False)
            self.evaluate_btn.setEnabled(False)
            QMessageBox.information(self, "Info", "No camera selected.")
        
        # 折叠摄像头选择区域
        if hasattr(self, 'camera_section_expanded'):
            self.camera_section_expanded = False
            self.camera_content.setVisible(False)
            header = self.camera_group.layout().itemAt(0).widget()
            header.setText("▶ Camera Selection")

    def toggle_camera_section(self):
        """切换摄像头选择区域的折叠状态"""
        self.camera_section_expanded = not self.camera_section_expanded
        self.camera_content.setVisible(self.camera_section_expanded)
        
        # 更新按钮文本
        button = self.sender()
        button.setText("▼ Camera Selection" if self.camera_section_expanded else "▶ Camera Selection")

    def start_evaluation(self):
        """开始评估过程"""
        try:
            # 在启动评估前，关闭GUI的摄像头预览以避免资源冲突
            if hasattr(self, 'preview_timer') and self.preview_timer.isActive():
                self.preview_timer.stop()
                print("Stopped camera preview timer")
            
            # 如果存在摄像头对象，释放它
            if self.cap is not None:
                self.cap.release()
                self.cap = None
                print("Released camera resource from GUI")
            
            # 更新预览显示
            if hasattr(self, 'preview_label'):
                self.preview_label.clear()
                self.preview_label.setText("Evaluation running... Camera in use by algorithm.")
            
            if not self.pipeline:
                self.initialize_pipeline()
            
            if self.pipeline:
                # self.pipeline.demo()
                self.pipeline.start_evaluation()
            else:
                QMessageBox.warning(self, "Warning", "Failed to initialize pipeline!")
            
        except Exception as e:
            error_msg = f"Failed to start evaluation:\n{str(e)}\n\nTraceback:\n"
            import traceback
            error_msg += "".join(traceback.format_exc())
            QMessageBox.critical(self, "Error", error_msg)
            print(error_msg)

    def initialize_pipeline(self):
        """初始化 pipeline"""
        try:
            print("\nCreating RealAction instance...")
            from my_model_arch.cpu_fast.pipeline import RealAction
            # 创建新的 pipeline 实例
            self.pipeline = RealAction(
                **self.config['real_action_config'],
                gaze_config=self.config['gaze_config'],
                mouse_control_config=self.config['mouse_control_config'],
                wheel_config=self.config['wheel_config'],
                configuration=self.config['key_config'],
                head_angles_center=self.config['head_angles_center'],
                head_angles_scale=self.config['head_angles_scale'],
                expression_evaluator_config=self.config['expression_evaluator_config'],
                **self.config['integrated_config']
            )

            print("\nPipeline initialized successfully")
            return True
            
        except Exception as e:
            error_msg = f"Error initializing pipeline:\n{str(e)}\n\nTraceback:\n"
            import traceback
            error_msg += "".join(traceback.format_exc())
            QMessageBox.critical(self, "Error", error_msg)
            print(error_msg)
            return False


    def add_priority_rule_row(self, row, rule=None):
        """添加优先级规则行"""
        # When 列（表情选择）
        when_combo = NoWheelComboBox()
        when_combo.addItems(['any'] + self.expression_list)  # 添加 'any' 选项和所有表情
        if rule and 'when' in rule:
            when_combo.setCurrentText(rule['when'])
        self.priority_table.setCellWidget(row, 0, when_combo)
        
        # Disable 列（表情多选）
        disable_widget = QWidget()
        disable_layout = QHBoxLayout(disable_widget)
        disable_layout.setContentsMargins(2, 2, 2, 2)
        disable_edit = QLineEdit()
        if rule and 'disable' in rule:
            disable_edit.setText(','.join(rule['disable']))
        disable_layout.addWidget(disable_edit)
        self.priority_table.setCellWidget(row, 1, disable_widget)
        
        # Except 列（表情多选）
        except_widget = QWidget()
        except_layout = QHBoxLayout(except_widget)
        except_layout.setContentsMargins(2, 2, 2, 2)
        except_edit = QLineEdit()
        if rule and 'except' in rule:
            except_edit.setText(','.join(rule['except']))
        except_layout.addWidget(except_edit)
        self.priority_table.setCellWidget(row, 2, except_widget)

    def setup_keymap_tab(self):
        """设置按键映射标签页"""
        keymap_tab = QWidget()
        layout = QVBoxLayout()
        
        # 按键映射树
        self.keymap_tree = QTreeWidget()
        self.keymap_tree.setHeaderLabels(["Mode/Key", "Wheel Actions"])
        self.keymap_tree.setColumnWidth(0, 150)
        self.keymap_tree.setColumnWidth(1, 600)
        self.keymap_tree.setUniformRowHeights(False)  # 允许不同行高
        self.keymap_tree.setStyleSheet("""
            QTreeWidget {
                border: 1px solid #ccc;
            }
            QTreeWidget::item {
                border-bottom: 1px solid #eee;
                padding: 1px 0;
                margin: 0px;
            }
            QTreeWidget::item:selected {
                background-color: #e0e0e0;
            }
        """)
        
        # 添加调整大小的事件处理
        self.keymap_tree.itemChanged.connect(self.adjust_item_size)
        
        # 从配置加载按键映射
        key_config = self.config.get('key_config', {})
        for mode, mode_config in key_config.items():
            mode_item = QTreeWidgetItem([mode])
            self.keymap_tree.addTopLevelItem(mode_item)
            
            for key, key_config in mode_config.items():
                key_item = QTreeWidgetItem([key])
                mode_item.addChild(key_item)
                actions_widget = KeyItemWidget(mode, key, key_config)
                self.keymap_tree.setItemWidget(key_item, 1, actions_widget)

        layout.addWidget(self.keymap_tree)
        
        # 添加按钮
        buttons_layout = QHBoxLayout()
        
        add_mode_btn = QPushButton("Add Mode")
        add_mode_btn.clicked.connect(self.add_mode)
        buttons_layout.addWidget(add_mode_btn)
        
        add_key_btn = QPushButton("Add Key")
        add_key_btn.clicked.connect(self.add_key)
        buttons_layout.addWidget(add_key_btn)
        
        layout.addLayout(buttons_layout)
        
        keymap_tab.setLayout(layout)
        self.tabs.addTab(keymap_tab, "Key Mapping")

    def edit_induce(self, key_config):
        """编辑 induce 配置"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Edit Induce")
        layout = QFormLayout()
        
        induce_config = key_config.get('induce', {})
        duration = QDoubleSpinBox()
        duration.setRange(0, 10)
        duration.setSingleStep(0.1)
        duration.setValue(induce_config.get('lock_mouse_move', {}).get('duration', 0))
        
        layout.addRow("Lock Duration:", duration)
        
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        
        dialog.setLayout(layout)
        
        if dialog.exec() == QDialog.Accepted:
            key_config['induce'] = {
                'lock_mouse_move': {
                    'duration': duration.value()
                }
            }

    def add_mode(self):
        """添加新的模式"""
        mode_item = QTreeWidgetItem(["new_mode"])
        self.keymap_tree.addTopLevelItem(mode_item)

    def add_key(self):
        """添加新的按键映射"""
        current = self.keymap_tree.currentItem()
        if not current:
            QMessageBox.warning(self, "Warning", "Please select a mode first")
            return
            
        # 如果选中的是按键，获取其父模式
        if current.parent():
            current = current.parent()
            
        key_item = QTreeWidgetItem(["new_key"])
        current.addChild(key_item)
        
        # 创建操作编辑器
        actions_widget = QWidget()
        actions_layout = QHBoxLayout()
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setSpacing(2)
        
        # 创建符号列表
        symbol_list = SymbolListWidget([])
        actions_layout.addWidget(symbol_list)
        
        actions_widget.setLayout(actions_layout)
        self.keymap_tree.setItemWidget(key_item, 1, actions_widget)
        
        # 立即设置项目高度
        height = actions_widget.sizeHint().height()
        key_item.setSizeHint(1, QSize(0, height))

    def get_gaze_config(self):
        """获取凝视配置"""
        return {
            field: widget.value()
            for field, widget in self.gaze_widgets.items()
        }
    
    def get_mouse_control_config(self):
        """获取鼠标控制配置"""
        return {
            field: widget.value()
            for field, widget in self.mouse_widgets.items()
        }
    
    def get_integrated_config(self):
        """获取集成配置"""
        config = {}
        
        # 处理普通字段
        for field, widget in self.integrated_widgets.items():
            if isinstance(widget, tuple):  # 处理列表类型的配置
                if field in ['screen_size', 'gaze_bias']:
                    # 这些字段需要两个值
                    config[field] = [widget[0].value(), widget[1].value()]
                elif field in ['pred_point_color', 'true_point_color']:
                    # 颜色字段需要三个值
                    config[field] = [widget[0].value(), widget[1].value(), widget[2].value()]
            elif isinstance(widget, QLineEdit):
                config[field] = widget.text()
            elif isinstance(widget, QCheckBox):
                config[field] = widget.isChecked()
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                config[field] = widget.value()
            elif isinstance(widget, QComboBox):
                config[field] = widget.currentText()
        
        return config

    def get_head_angles_config(self):
        """获取头部角度配置"""
        center_config = {
            angle: widget.value()
            for angle, widget in self.head_widgets['center'].items()
        }
        
        scale_config = {
            angle: widget.value()
            for angle, widget in self.head_widgets['scale'].items()
        }
        
        return center_config, scale_config
    
    def get_expression_config(self):
        """获取表情配置"""
        expressions = {}
        root = self.expression_tree.invisibleRootItem()
        
        for i in range(root.childCount()):
            expr_item = root.child(i)
            expr_name = expr_item.text(0)
            
            # 获取条件容器
            conditions_widget = self.expression_tree.itemWidget(expr_item, 1)
            combine_widget = self.expression_tree.itemWidget(expr_item, 2)
            
            if conditions_widget and combine_widget:
                conditions = []
                layout = conditions_widget.layout()
                if layout:  # 确保布局存在
                    for j in range(layout.count()):
                        layout_item = layout.itemAt(j)
                        if layout_item and layout_item.widget():  # 确保项目和部件存在
                            condition_widget = layout_item.widget()
                            if isinstance(condition_widget, ExpressionRow):
                                condition = condition_widget.get_condition()
                                conditions.append(condition)
                
                expressions[expr_name] = {
                    'conditions': conditions,
                    'combine': combine_widget.currentText()
                }
        
        # 获取优先级规则
        priority_rules = []
        for i in range(self.priority_table.rowCount()):
            rule = {}
            
            # When
            when_combo = self.priority_table.cellWidget(i, 0)
            if when_combo:
                rule['when'] = when_combo.currentText()
            
            # Disable
            disable_widget = self.priority_table.cellWidget(i, 1)
            if disable_widget and disable_widget.layout():
                disable_edit = disable_widget.layout().itemAt(0).widget()
                if disable_edit:
                    rule['disable'] = [x.strip() for x in disable_edit.text().split(',') if x.strip()]
            
            # Except
            except_widget = self.priority_table.cellWidget(i, 2)
            if except_widget and except_widget.layout():
                except_edit = except_widget.layout().itemAt(0).widget()
                if except_edit:
                    rule['except'] = [x.strip() for x in except_edit.text().split(',') if x.strip()]
            
            if rule:  # 只添加非空规则
                priority_rules.append(rule)
        
        return {'expressions': expressions, 'priority_rules': priority_rules}

    def get_keymap_config(self):
        """获取按键映射配置"""
        try:
            print("Starting get_keymap_config")
            config = {}
            root = self.keymap_tree.invisibleRootItem()
            
            for i in range(root.childCount()):
                try:
                    mode_item = root.child(i)
                    mode_name = mode_item.text(0)
                    print(f"Processing mode: {mode_name}")
                    mode_config = {}
                    
                    for j in range(mode_item.childCount()):
                        try:
                            key_item = mode_item.child(j)
                            key_name = key_item.text(0)
                            print(f"Processing key: {key_name}")
                            actions_widget = self.keymap_tree.itemWidget(key_item, 1)
                            
                            if not actions_widget:
                                print(f"Warning: No actions widget for key {key_name}")
                                continue
                            
                            print(f"Actions widget type: {type(actions_widget)}")
                            print(f"Actions widget attributes: {dir(actions_widget)}")
                            
                            key_data = {}
                            
                            # 获取 wheel actions
                            try:
                                if hasattr(actions_widget, 'symbol_list'):
                                    symbol_list = actions_widget.symbol_list
                                    print(f"Symbol list type: {type(symbol_list)}")
                                    print(f"Symbol list attributes: {dir(symbol_list)}")
                                    if symbol_list:
                                        wheel_actions = symbol_list.get_symbols()
                                        print(f"Wheel actions: {wheel_actions}")
                                        if wheel_actions:
                                            key_data['wheel'] = wheel_actions
                                else:
                                    print(f"No symbol_list attribute")
                            except Exception as e:
                                print(f"Error getting wheel actions: {str(e)}")
                                import traceback
                                traceback.print_exc()
                            
                            # 获取 layout_type
                            try:
                                if hasattr(actions_widget, 'layout_combo'):
                                    layout_combo = actions_widget.layout_combo
                                    if layout_combo:
                                        key_data['layout_type'] = layout_combo.currentText()
                            except Exception as e:
                                print(f"Error getting layout_type: {str(e)}")
                            
                            # 获取 induce 配置
                            try:
                                if hasattr(actions_widget, 'key_config'):
                                    if isinstance(actions_widget.key_config, dict) and 'induce' in actions_widget.key_config:
                                        key_data['induce'] = actions_widget.key_config['induce']
                            except Exception as e:
                                print(f"Error getting induce config: {str(e)}")
                            
                            if key_data:
                                mode_config[key_name] = key_data
                                print(f"Added key_data: {key_data}")
                        
                        except Exception as e:
                            print(f"Error processing key: {str(e)}")
                            import traceback
                            traceback.print_exc()
                    
                    if mode_config:
                        config[mode_name] = mode_config
                        print(f"Added mode_config: {mode_config}")
                
                except Exception as e:
                    print(f"Error processing mode: {str(e)}")
                    import traceback
                    traceback.print_exc()
            
            print(f"Final config: {config}")
            return config
        
        except Exception as e:
            print(f"Error in get_keymap_config: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}

    def save_config(self):
        """保存配置"""
        if self.current_config_path:
            try:
                self.save_config_to_file(self.current_config_path)
                QMessageBox.information(self, "Success", f"Configuration saved to {self.current_config_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save configuration:\n{str(e)}")
        else:
            # 如果没有当前路径，调用另存为
            self.save_config_as()

    def save_config_to_file(self, file_path):
        """保存配置到指定文件"""
        try:
            print("Starting to save configuration...")
            config = {}
            
            # 逐个获取配置，并添加错误检查
            try:
                config['gaze_config'] = self.get_gaze_config()
                print("Gaze config loaded")
            except Exception as e:
                print(f"Error getting gaze config: {str(e)}")
                raise
            
            try:
                config['mouse_control_config'] = self.get_mouse_control_config()
                print("Mouse control config loaded")
            except Exception as e:
                print(f"Error getting mouse control config: {str(e)}")
                raise
            
            try:
                config['wheel_config'] = self.get_wheel_config()
                print("Wheel config loaded")
            except Exception as e:
                print(f"Error getting wheel config: {str(e)}")
                raise
            
            try:
                config['real_action_config'] = self.get_real_action_config()
                print("Real action config loaded")
            except Exception as e:
                print(f"Error getting real action config: {str(e)}")
                raise
            
            try:
                config['integrated_config'] = self.get_integrated_config()
                print("Integrated config loaded")
            except Exception as e:
                print(f"Error getting integrated config: {str(e)}")
                raise
            
            try:
                head_center, head_scale = self.get_head_angles_config()
                config['head_angles_center'] = head_center
                config['head_angles_scale'] = head_scale
                print("Head angles config loaded")
            except Exception as e:
                print(f"Error getting head angles config: {str(e)}")
                raise
            
            try:
                config['expression_evaluator_config'] = self.get_expression_config()
                print("Expression config loaded")
            except Exception as e:
                print(f"Error getting expression config: {str(e)}")
                raise
            
            try:
                config['key_config'] = self.get_keymap_config()
                print("Key config loaded")
            except Exception as e:
                print(f"Error getting key config: {str(e)}")
                raise

            # 保存配置
            print(f"Saving config to {file_path}")
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
            
        except Exception as e:
            import traceback
            error_msg = f"Failed to save configuration:\n{str(e)}\n\n"
            error_msg += "Traceback:\n" + traceback.format_exc()
            print(error_msg)
            raise Exception(error_msg)

    def get_wheel_config(self):
        """获取轮盘配置"""
        return {
            field: (widget.text() if isinstance(widget, QLineEdit) else widget.value())
            for field, widget in self.wheel_widgets.items()
        }

    def get_real_action_config(self):
        """获取真实动作配置"""
        return {
            'scroll_coef': self.real_action_widgets['scroll_coef'].value(),
            'sys_mode': self.real_action_widgets['sys_mode'].currentText(),
            'show_gaze': self.real_action_widgets['show_gaze'].isChecked(),
            'mouse_control': self.real_action_widgets['mouse_control'].isChecked(),
            'key_control': self.real_action_widgets['key_control'].isChecked()
        }

    def adjust_item_size(self, item):
        """调整项目大小以适应内容"""
        if item.childCount() == 0:  # 只处理叶子节点
            widget = self.keymap_tree.itemWidget(item, 1)
            if widget:
                # 获取实际需要的高度
                height = widget.sizeHint().height()
                # 设置项目高度
                item.setSizeHint(1, QSize(0, height + 2))  # +2 为边距

    def add_condition(self, layout):
        """添加新的条件行"""
        row = ExpressionRow()
        row.delete_btn.clicked.connect(lambda _, w=row: w.deleteLater())
        # 在添加按钮之前插入新条件
        layout.insertWidget(layout.count() - 1, row)

    def start_calibration(self):
        """开始校准过程"""
        try:
            # 在启动校准前，关闭GUI的摄像头预览以避免资源冲突
            if hasattr(self, 'preview_timer') and self.preview_timer.isActive():
                self.preview_timer.stop()
                print("Stopped camera preview timer")
            
            # 如果存在摄像头对象，释放它
            if self.cap is not None:
                self.cap.release()
                self.cap = None
                print("Released camera resource from GUI")
            
            # 更新预览显示
            if hasattr(self, 'preview_label'):
                self.preview_label.clear()
                self.preview_label.setText("Calibration running... Camera in use by algorithm.")
            
            if not self.pipeline:
                self.initialize_pipeline()
            
            if self.pipeline:
                try:
                    result = self.pipeline.start_calibration()
                    # 总是调用完成回调，无论校准成功还是失败
                    QTimer.singleShot(100, self.on_calibration_finished)
                    if result is None:
                        print("Calibration failed or was interrupted")
                    else:
                        print("Calibration completed successfully")
                except Exception as e:
                    print(f"Calibration error: {e}")
                    import traceback
                    traceback.print_exc()
                    # 即使出现异常也要调用回调函数
                    QTimer.singleShot(100, self.on_calibration_finished)
                
            else:
                QMessageBox.warning(self, "Warning", "Failed to initialize pipeline!")
            
        except Exception as e:
            error_msg = f"Failed to start calibration:\n{str(e)}\n\nTraceback:\n"
            import traceback
            error_msg += "".join(traceback.format_exc())
            QMessageBox.critical(self, "Error", error_msg)
            print(error_msg)


    def on_calibration_finished(self):
        """校准完成后的回调处理"""
        try:
            # 更新预览显示状态
            if hasattr(self, 'preview_label'):
                self.preview_label.setText("Calibration completed. Click 'Change Camera' to restart preview.")
            
            # 调试信息
            print(f"on_calibration_finished called")
            print(f"pipeline exists: {hasattr(self, 'pipeline') and self.pipeline is not None}")
            if hasattr(self, 'pipeline') and self.pipeline:
                print(f"pipeline calibration_time exists: {hasattr(self.pipeline, 'calibration_time')}")
                if hasattr(self.pipeline, 'calibration_time'):
                    print(f"calibration_time value: {self.pipeline.calibration_time}")
            
            # 更新配置文件中的权重路径
            if hasattr(self, 'pipeline') and self.pipeline and hasattr(self.pipeline, 'calibration_time'):
                # 构建新的权重路径
                new_model_path = f'model_weights/{self.pipeline.calibration_time}/model.pkl'
                print(f"Attempting to update regression_model_path to: {new_model_path}")
                
                # 检查权重文件是否存在
                import os
                if os.path.exists(new_model_path):
                    print(f"Model file exists: {new_model_path}")
                else:
                    print(f"Warning: Model file does not exist yet: {new_model_path}")
                
                # 更新GUI中的权重路径输入框
                if 'regression_model_path' in self.integrated_widgets:
                    self.integrated_widgets['regression_model_path'].setText(new_model_path)
                    print("Updated GUI widget")
                    
                    # 保存配置文件
                    self.save_config()
                    print(f"Configuration saved with new regression_model_path: {new_model_path}")
                else:
                    print("Warning: regression_model_path widget not found")
            else:
                print("Warning: Pipeline or calibration_time not available for updating config")
            
            # 询问用户是否要恢复预览（可选）
            reply = QMessageBox.question(
                self, 
                "Calibration Complete", 
                "Calibration completed successfully!\n\nWould you like to restart camera preview?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            
            if reply == QMessageBox.Yes:
                self.restart_camera_preview()
            
        except Exception as e:
            print(f"Error in calibration finished callback: {e}")
            import traceback
            traceback.print_exc()


    def closeEvent(self, event):
        """关闭窗口时清理资源"""
        try:
            # 停止预览定时器
            if hasattr(self, 'preview_timer'):
                self.preview_timer.stop()
            
            # 如果存在摄像头，关闭它
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            
            # 如果存在 pipeline，关闭它
            if hasattr(self, 'pipeline') and self.pipeline:
                self.pipeline.quit_pipeline()
                self.pipeline = None
                
        except Exception as e:
            print(f"Error closing resources: {str(e)}")
            import traceback
            traceback.print_exc()
        
        super().closeEvent(event)


    def load_config(self, config_path=None):
        """加载配置文件"""
        if config_path is None:
            # 尝试从临时文件加载上次使用的配置路径
            temp_path = Path(__file__).parent / '.last_config_path'
            if temp_path.exists():
                with open(temp_path, 'r') as f:
                    last_config_path = f.read().strip()
                if Path(last_config_path).exists():
                    config_path = last_config_path
                else:
                    temp_path.unlink()  # 删除无效的临时文件
            
            # 如果还是没有路径，使用默认路径 configs/cpu.yaml
            if config_path is None:
                config_path = Path(__file__).parent / 'configs' / 'cpu.yaml'
        
        # 转换为Path对象
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
                print(f"Loaded config from: {config_path}")
                self.current_config_path = str(config_path)
                self.update_window_title()
        except Exception as e:
            raise Exception(f"Failed to load configuration from {config_path}: {str(e)}")



    def setup_basic_tab(self):
        """设置基础配置标签页"""
        basic_tab = QScrollArea()
        basic_tab.setWidgetResizable(True)
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        
        # Configuration File Selector Group (添加在最上面)
        config_file_group = QGroupBox("Configuration File")
        config_file_layout = QHBoxLayout()
        
        # 当前配置文件显示
        self.current_config_label = QLabel()
        self.update_current_config_label()
        config_file_layout.addWidget(QLabel("Current:"))
        config_file_layout.addWidget(self.current_config_label)
        
        # 配置文件选择下拉框
        self.config_file_combo = NoWheelComboBox()
        self.config_file_combo.setMinimumWidth(200)
        self.refresh_config_files()
        config_file_layout.addWidget(QLabel("Switch to:"))
        config_file_layout.addWidget(self.config_file_combo)
        
        # 加载选中配置按钮
        load_config_btn = QPushButton("Load Selected")
        load_config_btn.clicked.connect(self.load_selected_config)
        config_file_layout.addWidget(load_config_btn)
        
        # 刷新配置列表按钮
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_config_files)
        config_file_layout.addWidget(refresh_btn)
        
        config_file_layout.addStretch()  # 添加弹性空间
        config_file_group.setLayout(config_file_layout)
        layout.addWidget(config_file_group)
        
        # Control Options Group
        control_group = QGroupBox("Control Options")
        control_layout = QVBoxLayout()
        
        # Show Gaze Checkbox
        self.show_gaze_cb = QCheckBox("Show Gaze")
        self.show_gaze_cb.setChecked(self.config['real_action_config'].get('show_gaze', False))
        self.show_gaze_cb.stateChanged.connect(self.update_show_gaze)
        control_layout.addWidget(self.show_gaze_cb)
        
        # Mouse Control Checkbox
        self.mouse_control_cb = QCheckBox("Mouse Control")
        self.mouse_control_cb.setChecked(self.config['real_action_config'].get('mouse_control', False))
        self.mouse_control_cb.stateChanged.connect(self.update_mouse_control)
        control_layout.addWidget(self.mouse_control_cb)
        
        # Key Control Checkbox
        self.key_control_cb = QCheckBox("Key Control")
        self.key_control_cb.setChecked(self.config['real_action_config'].get('key_control', False))
        self.key_control_cb.stateChanged.connect(self.update_key_control)
        control_layout.addWidget(self.key_control_cb)
        
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        # Head Angles Configuration (移到最上面)
        head_angles_group = QGroupBox("Head Angles Configuration")
        head_layout = QFormLayout()
        self.head_widgets = {'center': {}, 'scale': {}}

        # Head Angles Center
        for angle in ['yaw', 'pitch', 'roll']:
            widget = QDoubleSpinBox()
            widget.setRange(-90, 90)
            widget.setSingleStep(0.1)
            if angle in self.config.get('head_angles_center', {}):
                widget.setValue(self.config['head_angles_center'][angle])
            self.head_widgets['center'][angle] = widget
            head_layout.addRow(f'Center {angle}:', widget)

        # Head Angles Scale
        for angle in ['yaw', 'pitch', 'roll']:
            widget = QDoubleSpinBox()
            widget.setRange(0, 90)
            widget.setSingleStep(0.1)
            if angle in self.config.get('head_angles_scale', {}):
                widget.setValue(self.config['head_angles_scale'][angle])
            self.head_widgets['scale'][angle] = widget
            head_layout.addRow(f'Scale {angle}:', widget)

        head_angles_group.setLayout(head_layout)
        layout.addWidget(head_angles_group)
        
        # Wheel Config
        wheel_group = QGroupBox("Wheel Configuration")
        wheel_layout = QFormLayout()
        self.wheel_widgets = {}
        
        wheel_fields = {
            'radius': (QSpinBox, {'min': 100, 'max': 2000}),
            'font_size': (QSpinBox, {'min': 8, 'max': 72}),
            'font': (QLineEdit, {})
        }
        
        for field, (widget_class, props) in wheel_fields.items():
            widget = widget_class()
            if isinstance(widget, QSpinBox):
                for prop, value in props.items():
                    if prop == 'min':
                        widget.setMinimum(value)
                    elif prop == 'max':
                        widget.setMaximum(value)
            
            if 'wheel_config' in self.config and field in self.config['wheel_config']:
                if isinstance(widget, QLineEdit):
                    widget.setText(str(self.config['wheel_config'][field]))
                else:
                    widget.setValue(self.config['wheel_config'][field])
            
            self.wheel_widgets[field] = widget
            wheel_layout.addRow(field, widget)
        
        wheel_group.setLayout(wheel_layout)
        layout.addWidget(wheel_group)

        # Real Action Config
        real_action_group = QGroupBox("Real Action Configuration")
        real_action_layout = QFormLayout()
        self.real_action_widgets = {}
        
        # Scroll Coefficient
        scroll_spin = QDoubleSpinBox()
        scroll_spin.setRange(0.1, 10.0)
        scroll_spin.setSingleStep(0.1)
        if 'real_action_config' in self.config and 'scroll_coef' in self.config['real_action_config']:
            scroll_spin.setValue(self.config['real_action_config']['scroll_coef'])
        self.real_action_widgets['scroll_coef'] = scroll_spin
        real_action_layout.addRow('Scroll Coefficient:', scroll_spin)
        
        # System Mode
        sys_mode_combo = NoWheelComboBox()
        sys_mode_combo.addItems(['type', 'game', 'game_cs'])
        if 'real_action_config' in self.config and 'sys_mode' in self.config['real_action_config']:
            sys_mode_combo.setCurrentText(self.config['real_action_config']['sys_mode'])
        self.real_action_widgets['sys_mode'] = sys_mode_combo
        real_action_layout.addRow('System Mode:', sys_mode_combo)
        
        # Show Gaze
        show_gaze_check = QCheckBox()
        if 'real_action_config' in self.config and 'show_gaze' in self.config['real_action_config']:
            show_gaze_check.setChecked(self.config['real_action_config']['show_gaze'])
        self.real_action_widgets['show_gaze'] = show_gaze_check
        real_action_layout.addRow('Show Gaze:', show_gaze_check)
        
        # Mouse Control
        mouse_control_check = QCheckBox()
        if 'real_action_config' in self.config and 'mouse_control' in self.config['real_action_config']:
            mouse_control_check.setChecked(self.config['real_action_config']['mouse_control'])
        self.real_action_widgets['mouse_control'] = mouse_control_check
        real_action_layout.addRow('Mouse Control:', mouse_control_check)
        
        # Key Control
        key_control_check = QCheckBox()
        if 'real_action_config' in self.config and 'key_control' in self.config['real_action_config']:
            key_control_check.setChecked(self.config['real_action_config']['key_control'])
        self.real_action_widgets['key_control'] = key_control_check
        real_action_layout.addRow('Key Control:', key_control_check)
        
        real_action_group.setLayout(real_action_layout)
        layout.addWidget(real_action_group)

        # Gaze Config
        gaze_group = QGroupBox("Gaze Configuration")
        gaze_layout = QFormLayout()
        self.gaze_widgets = {}
        
        gaze_fields = {
            'history_duration': (QDoubleSpinBox, {'min': 0.0, 'max': 1.0, 'step': 0.01}),
            'update_interval': (QDoubleSpinBox, {'min': 0.001, 'max': 0.1, 'step': 0.001}),
            'point_radius': (QSpinBox, {'min': 1, 'max': 100}),
            'point_alpha': (QSpinBox, {'min': 0, 'max': 255}),
            'clear_radius': (QSpinBox, {'min': 1, 'max': 200}),
            'color_r': (QSpinBox, {'min': 0, 'max': 255}),
            'color_g': (QSpinBox, {'min': 0, 'max': 255}),
            'color_b': (QSpinBox, {'min': 0, 'max': 255}),
            'gaussian_sigma_ratio': (QDoubleSpinBox, {'min': 0.1, 'max': 5.0, 'step': 0.1}),
            'window_alpha': (QSpinBox, {'min': 0, 'max': 255})
        }
        
        for field, (widget_class, props) in gaze_fields.items():
            widget = widget_class()
            for prop, value in props.items():
                if prop == 'min':
                    widget.setMinimum(value)
                elif prop == 'max':
                    widget.setMaximum(value)
                elif prop == 'step':
                    widget.setSingleStep(value)
            
            # 从配置文件加载值
            if 'gaze_config' in self.config and field in self.config['gaze_config']:
                widget.setValue(self.config['gaze_config'][field])
            
            self.gaze_widgets[field] = widget
            gaze_layout.addRow(field, widget)
        
        gaze_group.setLayout(gaze_layout)
        layout.addWidget(gaze_group)

        # Mouse Control Config
        mouse_group = QGroupBox("Mouse Control Configuration")
        mouse_layout = QFormLayout()
        self.mouse_widgets = {}
        
        mouse_fields = {
            'dead_zone': (QSpinBox, {'min': 0, 'max': 1000}),
            'max_speed': (QSpinBox, {'min': 1, 'max': 100}),
            'smoothing': (QDoubleSpinBox, {'min': 0.0, 'max': 1.0, 'step': 0.1}),
            'y_speed_coef': (QDoubleSpinBox, {'min': 0.1, 'max': 5.0, 'step': 0.1}),
            'head_coef': (QSpinBox, {'min': 1, 'max': 100}),
            'lock_head_duration': (QSpinBox, {'min': 0, 'max': 10}),
            'wheel_head_coef': (QSpinBox, {'min': 1, 'max': 1000})
        }
        
        for field, (widget_class, props) in mouse_fields.items():
            widget = widget_class()
            for prop, value in props.items():
                if prop == 'min':
                    widget.setMinimum(value)
                elif prop == 'max':
                    widget.setMaximum(value)
                elif prop == 'step':
                    widget.setSingleStep(value)
            
            # 从配置文件加载值
            if 'mouse_control_config' in self.config and field in self.config['mouse_control_config']:
                widget.setValue(self.config['mouse_control_config'][field])
            
            self.mouse_widgets[field] = widget
            mouse_layout.addRow(field, widget)
        
        mouse_group.setLayout(mouse_layout)
        layout.addWidget(mouse_group)
        
        # Integrated Config
        integrated_group = QGroupBox("Integrated Configuration")
        integrated_layout = QFormLayout()
        self.integrated_widgets = {}
        
        # 添加所有integrated_config字段
        integrated_fields = {
            'weights': (QLineEdit, {}),
            'regression_model_type': (NoWheelComboBox, {'items': ['lassocv', 'ridge', 'linear']}),
            'regression_model_path': (QLineEdit, {}),
            'arch': (QLineEdit, {}),
            'device': (NoWheelComboBox, {'items': ['cuda', 'cpu']}),
            'cam_id': (QSpinBox, {'min': 0, 'max': 10}),
            'num_points': (QSpinBox, {'min': 1, 'max': 100}),
            'every_point_has_n_images': (QSpinBox, {'min': 1, 'max': 100}),
            'images_freq': (QSpinBox, {'min': 1, 'max': 60}),
            'each_point_wait_time': (QSpinBox, {'min': 100, 'max': 5000}),
            'edge': (QDoubleSpinBox, {'min': 0.0, 'max': 0.5, 'step': 0.01}),
            'radius': (QSpinBox, {'min': 1, 'max': 100}),
            'func': (NoWheelComboBox, {'items': ['random', 'sequence']}),
            'window_name': (QLineEdit, {}),
            'render_in_eval': (QCheckBox, {}),
            'kalman_filter_std_measurement': (QDoubleSpinBox, {'min': 0.1, 'max': 20.0, 'step': 0.1}),
            'Q_coef': (QDoubleSpinBox, {'min': 0.001, 'max': 1.0, 'step': 0.001}),
            'dt': (QDoubleSpinBox, {'min': 0.001, 'max': 0.1, 'step': 0.001})
        }

        for field, (widget_class, props) in integrated_fields.items():
            widget = widget_class()
            if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                for prop, value in props.items():
                    if prop == 'min':
                        widget.setMinimum(value)
                    elif prop == 'max':
                        widget.setMaximum(value)
                    elif prop == 'step':
                        widget.setSingleStep(value)
            elif isinstance(widget, NoWheelComboBox) and 'items' in props:
                widget.addItems(props['items'])
            
            if 'integrated_config' in self.config and field in self.config['integrated_config']:
                value = self.config['integrated_config'][field]
                if isinstance(widget, QLineEdit):
                    widget.setText(str(value))
                elif isinstance(widget, QCheckBox):
                    widget.setChecked(value)
                else:
                    widget.setValue(value) if isinstance(value, (int, float)) else widget.setCurrentText(str(value))
            
            self.integrated_widgets[field] = widget
            integrated_layout.addRow(field, widget)
        
        # 添加 screen_size 配置
        screen_size_widget = QWidget()
        screen_size_layout = QHBoxLayout(screen_size_widget)
        screen_size_x = QSpinBox()
        screen_size_y = QSpinBox()
        screen_size_x.setRange(0, 9999)
        screen_size_y.setRange(0, 9999)
        screen_size_layout.addWidget(screen_size_x)
        screen_size_layout.addWidget(screen_size_y)
        integrated_layout.addRow('Screen Size:', screen_size_widget)
        self.integrated_widgets['screen_size'] = (screen_size_x, screen_size_y)

        # 添加颜色配置
        for color_field in ['pred_point_color', 'true_point_color']:
            color_widget = QWidget()
            color_layout = QHBoxLayout(color_widget)
            r = QSpinBox()
            g = QSpinBox()
            b = QSpinBox()
            for spin in [r, g, b]:
                spin.setRange(0, 255)
                color_layout.addWidget(spin)
            integrated_layout.addRow(f'{color_field}:', color_widget)
            self.integrated_widgets[color_field] = (r, g, b)

        # 添加 gaze_bias 配置
        gaze_bias_widget = QWidget()
        gaze_bias_layout = QHBoxLayout(gaze_bias_widget)
        gaze_bias_x = QSpinBox()
        gaze_bias_y = QSpinBox()
        gaze_bias_x.setRange(-2000, 2000)
        gaze_bias_y.setRange(-2000, 2000)
        gaze_bias_layout.addWidget(gaze_bias_x)
        gaze_bias_layout.addWidget(gaze_bias_y)
        integrated_layout.addRow('Gaze Bias:', gaze_bias_widget)
        self.integrated_widgets['gaze_bias'] = (gaze_bias_x, gaze_bias_y)

        integrated_group.setLayout(integrated_layout)
        layout.addWidget(integrated_group)

        basic_tab.setWidget(content_widget)
        self.tabs.addTab(basic_tab, "Basic Settings")

    def setup_expression_tab(self):
        """设置表情映射标签页"""
        expression_tab = QWidget()
        layout = QVBoxLayout()
        
        # 表情映射部分
        self.expression_tree = QTreeWidget()
        self.expression_tree.setHeaderLabels(["Expression", "Conditions", "Combine"])
        self.expression_tree.setColumnWidth(0, 100)
        self.expression_tree.setColumnWidth(1, 400)
        self.expression_tree.setUniformRowHeights(False)
        
        # 从配置加载表情映射
        expressions = self.config.get('expression_evaluator_config', {}).get('expressions', {})
        for expr_name, expr_config in expressions.items():
            expr_item = QTreeWidgetItem([expr_name])
            self.expression_tree.addTopLevelItem(expr_item)
            
            # 创建条件容器
            conditions_widget = QWidget()
            conditions_layout = QVBoxLayout()
            conditions_layout.setContentsMargins(0, 0, 0, 0)
            conditions_layout.setSpacing(2)
            
            # 添加现有条件
            for condition in expr_config.get('conditions', []):
                row = ExpressionRow()
                row.expression_combo.setCurrentText(condition['feature'])
                row.operator_combo.setCurrentText(condition['operator'])
                
                # 根据操作符类型设置值
                if condition['operator'] == 'BETWEEN':
                    row.threshold_spin.setValue(float(condition.get('min', 0)))
                elif condition['operator'].startswith('DIFF'):
                    row.threshold_spin.setValue(float(condition.get('threshold', 0)))
                else:
                    row.threshold_spin.setValue(float(condition.get('threshold', 0)))
                
                # 连接删除按钮
                row.delete_btn.clicked.connect(lambda _, w=row: w.deleteLater())
                conditions_layout.addWidget(row)
            
            # 添加新条件按钮
            add_condition_btn = QPushButton("+")
            add_condition_btn.setFixedSize(20, 20)
            add_condition_btn.clicked.connect(lambda _, l=conditions_layout: self.add_condition(l))
            conditions_layout.addWidget(add_condition_btn)
            
            conditions_widget.setLayout(conditions_layout)
            self.expression_tree.setItemWidget(expr_item, 1, conditions_widget)
            
            # 添加组合方式
            combine_combo = NoWheelComboBox()
            combine_combo.addItems(['AND', 'OR'])
            combine_combo.setCurrentText(expr_config.get('combine', 'AND'))
            self.expression_tree.setItemWidget(expr_item, 2, combine_combo)
        
        # Priority Rules 部分
        priority_group = QGroupBox("Priority Rules")
        priority_layout = QVBoxLayout()
        
        # 创建优先级规则表格
        self.priority_table = QTableWidget()
        self.priority_table.setColumnCount(3)
        self.priority_table.setHorizontalHeaderLabels(["When", "Disable", "Except"])
        self.priority_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.priority_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        
        # 从配置加载优先级规则
        priority_rules = self.config.get('expression_evaluator_config', {}).get('priority_rules', [])
        self.priority_table.setRowCount(len(priority_rules))
        for i, rule in enumerate(priority_rules):
            self.add_priority_rule_row(i, rule)
        
        priority_layout.addWidget(self.priority_table)
        
        # 添加规则按钮
        add_rule_btn = QPushButton("Add Priority Rule")
        add_rule_btn.clicked.connect(self.add_priority_rule)
        priority_layout.addWidget(add_rule_btn)
        
        priority_group.setLayout(priority_layout)
        
        layout.addWidget(self.expression_tree)
        layout.addWidget(priority_group)
        expression_tab.setLayout(layout)
        self.tabs.addTab(expression_tab, "Expression Mapping")

    def add_priority_rule(self):
        """添加新的优先级规则"""
        row = self.priority_table.rowCount()
        self.priority_table.insertRow(row)
        self.add_priority_rule_row(row)


    def toggle_control_section(self):
        """切换功能控制区域的折叠状态"""
        self.control_section_expanded = not getattr(self, 'control_section_expanded', True)
        self.control_content.setVisible(self.control_section_expanded)
        
        # 更新按钮文本
        button = self.sender()
        button.setText("▼ Function Controls" if self.control_section_expanded else "▶ Function Controls")


    def update_show_gaze(self, state):
        """更新显示凝视点的状态"""
        show_gaze = bool(state)
        self.config['real_action_config']['show_gaze'] = show_gaze
        if hasattr(self, 'pipeline'):
            self.pipeline.set_show_gaze(show_gaze)

    def update_mouse_control(self, state):
        """更新鼠标控制的状态"""
        mouse_control = bool(state)
        self.config['real_action_config']['mouse_control'] = mouse_control
        if hasattr(self, 'pipeline'):
            self.pipeline.set_mouse_control(mouse_control)

    def update_key_control(self, state):
        """更新键盘控制的状态"""
        key_control = bool(state)
        self.config['real_action_config']['key_control'] = key_control
        if hasattr(self, 'pipeline'):
            self.pipeline.set_key_control(key_control)

    def keyPressEvent(self, event):
        """处理键盘按下事件"""
        if event.key() == Qt.Key_Escape:
            self.esc_pressed = True
        elif event.key() == Qt.Key_Q and self.esc_pressed:
            # ESC + Q 组合键被按下
            if hasattr(self, 'pipeline') and self.pipeline:
                self.pipeline.quit_pipeline()  # 先调用退出方法清理资源
                self.pipeline = None  # 然后设为 None
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        """处理键盘释放事件"""
        if event.key() == Qt.Key_Escape:
            self.esc_pressed = False
        super().keyReleaseEvent(event)

    def check_hotkeys(self):
        """检查热键组合"""
        if keyboard.is_pressed('esc+q'):
            if hasattr(self, 'pipeline') and self.pipeline:
                self.pipeline.quit_pipeline()
                self.pipeline = None

    def update_current_config_label(self):
        """更新当前配置文件标签"""
        if self.current_config_path:
            config_name = Path(self.current_config_path).name
            self.current_config_label.setText(config_name)
        else:
            self.current_config_label.setText("New Configuration")

    def refresh_config_files(self):
        """刷新配置文件列表"""
        if not hasattr(self, 'config_file_combo'):
            return
            
        self.config_file_combo.clear()
        
        # 扫描configs目录下的YAML文件
        configs_dir = Path(__file__).parent / 'configs'
        if configs_dir.exists():
            yaml_files = []
            for pattern in ['*.yaml', '*.yml']:
                yaml_files.extend(configs_dir.glob(pattern))
            
            # 按文件名排序
            yaml_files.sort(key=lambda x: x.name)
            
            for yaml_file in yaml_files:
                # 显示相对于configs目录的路径
                display_name = yaml_file.name
                self.config_file_combo.addItem(display_name, str(yaml_file))
        
        # 扫描根目录下的YAML文件
        root_dir = Path(__file__).parent
        for pattern in ['*.yaml', '*.yml']:
            for yaml_file in root_dir.glob(pattern):
                display_name = f"../{yaml_file.name}"
                self.config_file_combo.addItem(display_name, str(yaml_file))

    def load_selected_config(self):
        """加载选中的配置文件"""
        if not hasattr(self, 'config_file_combo'):
            return
            
        selected_path = self.config_file_combo.currentData()
        if selected_path:
            try:
                # 询问用户是否保存当前配置
                if self.current_config_path:
                    reply = QMessageBox.question(
                        self, 
                        "Load Configuration", 
                        "Do you want to save the current configuration before loading a new one?",
                        QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                        QMessageBox.Yes
                    )
                    
                    if reply == QMessageBox.Cancel:
                        return
                    elif reply == QMessageBox.Yes:
                        self.save_config()
                
                # 加载新配置
                self.load_config(selected_path)
                self.add_to_recent_files(selected_path)
                self.update_current_config_label()
                self.refresh_ui()
                
                QMessageBox.information(self, "Success", f"Configuration loaded from {Path(selected_path).name}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load configuration:\n{str(e)}")

    def change_camera_selection(self):
        """重新选择摄像头"""
        reply = QMessageBox.question(
            self, 
            "Change Camera", 
            "This will stop the current camera and allow you to select a new one. Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # 重新启用摄像头选择
            self.camera_combo.setEnabled(True)
            self.camera_confirm_btn.setEnabled(False)
            self.camera_change_btn.setEnabled(False)
            
            # 禁用校准和评估按钮
            self.calibrate_btn.setEnabled(False)
            self.evaluate_btn.setEnabled(False)
            
            # 清除预览显示
            self.preview_label.clear()
            self.preview_label.setText("Please select a camera")
            
            # 展开摄像头选择区域
            if hasattr(self, 'camera_section_expanded'):
                self.camera_section_expanded = True
                self.camera_content.setVisible(True)
                header = self.camera_group.layout().itemAt(0).widget()
                header.setText("▼ Camera Selection")

    def restart_camera_preview(self):
        """重新启动摄像头预览（在算法结束后可选择调用）"""
        try:
            camera_id = self.config['integrated_config'].get('cam_id')
            if camera_id is not None:
                # 重新打开摄像头
                self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
                if self.cap.isOpened():
                    # 设置摄像头属性
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    
                    # 重新启动预览定时器
                    self.preview_timer.start(30)
                    print("Restarted camera preview")
                else:
                    self.preview_label.setText("Failed to restart camera preview")
            else:
                self.preview_label.setText("No camera configured for preview")
        except Exception as e:
            print(f"Error restarting camera preview: {str(e)}")
            self.preview_label.setText("Error restarting camera preview")


class KeyItemWidget(QWidget):
    def __init__(self, mode, key_name, key_config, parent=None):
        super().__init__(parent)
        self.tree_item = None
        self.key_config = key_config  # 保存配置
        
        layout = QHBoxLayout()
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(2)
        
        # Wheel Actions
        wheel_actions = []
        if isinstance(key_config, dict) and 'wheel' in key_config:
            wheel_actions = key_config['wheel']
        self.symbol_list = SymbolListWidget(wheel_actions)
        layout.addWidget(self.symbol_list)
        
        # Layout Type (如果配置中有 layout_type)
        if isinstance(key_config, dict) and 'layout_type' in key_config:
            self.layout_combo = NoWheelComboBox()
            self.layout_combo.addItems(['circle', 'square'])
            self.layout_combo.setCurrentText(key_config['layout_type'])
            layout.addWidget(self.layout_combo)
        
        # Induce (如果有)
        if isinstance(key_config, dict) and 'induce' in key_config:
            induce_btn = QPushButton("Induce")
            induce_btn.clicked.connect(self.edit_induce)
            layout.addWidget(induce_btn)
        
        self.setLayout(layout)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)

    def get_all_items(self, tree):
        """获取树中所有项"""
        items = []
        iterator = QTreeWidgetItemIterator(tree)
        while iterator.value():
            items.append(iterator.value())
            iterator += 1
        return items

    def sizeHint(self):
        """返回建议的大小"""
        return self.layout().sizeHint()

    def minimumSizeHint(self):
        """返回最小建议大小"""
        return self.layout().minimumSize()

    def edit_induce(self):
        """编辑 induce 配置"""
        if not isinstance(self.key_config, dict) or 'induce' not in self.key_config:
            return
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Edit Induce")
        layout = QFormLayout()
        
        # 创建编辑控件
        widgets = {}
        for action, config in self.key_config['induce'].items():
            group = QGroupBox(action)
            group_layout = QFormLayout()
            
            if isinstance(config, dict):
                for key, value in config.items():
                    if isinstance(value, (int, float)):
                        widget = QDoubleSpinBox()
                        widget.setRange(0, 100)
                        widget.setValue(value)
                    else:
                        widget = QLineEdit(str(value))
                    group_layout.addRow(key, widget)
                    widgets[(action, key)] = widget
            
            group.setLayout(group_layout)
            layout.addRow(group)
        
        # 添加确定和取消按钮
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addRow(button_box)
        
        dialog.setLayout(layout)
        
        # 如果用户点击确定，更新配置
        if dialog.exec() == QDialog.Accepted:
            for (action, key), widget in widgets.items():
                if isinstance(widget, QDoubleSpinBox):
                    self.key_config['induce'][action][key] = widget.value()
                else:
                    self.key_config['induce'][action][key] = widget.text()

class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, event):
        # 完全禁用滚轮事件
        event.ignore()

class NoWheelComboBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, event):
        # 完全禁用滚轮事件
        event.ignore()

class ExpressionRow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 表情特征列表
        feature_list = [
            'mouthSmileLeft', 'mouthSmileRight',
            'mouthLeft', 'mouthRight',
            'mouthRollUpper', 'mouthRollLower',
            'mouthPucker', 'mouthFunnel',
            'mouthUpperUpLeft', 'mouthUpperUpRight',
            'mouthLowerDownLeft', 'mouthLowerDownRight',
            'mouthPressLeft', 'mouthPressRight',
            'jawOpen', 'jawLeft', 'jawRight',
            'browInnerUp',
            'eyeBlinkLeft', 'eyeBlinkRight'
        ]
        
        # Expression ComboBox
        self.expression_combo = NoWheelComboBox()
        self.expression_combo.addItems(feature_list)
        layout.addWidget(self.expression_combo)
        
        # Operator ComboBox
        self.operator_combo = NoWheelComboBox()
        self.operator_combo.addItems(['>', '<', '>=', '<=', '==', 'BETWEEN', 'DIFF>', 'DIFF<'])
        self.operator_combo.currentTextChanged.connect(self.on_operator_changed)
        layout.addWidget(self.operator_combo)
        
        # 创建阈值相关的控件
        self.threshold_widget = QWidget()
        self.threshold_layout = QHBoxLayout()
        self.threshold_layout.setContentsMargins(0, 0, 0, 0)
        
        # 最小值输入框
        self.min_spin = NoWheelDoubleSpinBox()
        self.min_spin.setRange(0, 1)
        self.min_spin.setSingleStep(0.1)
        self.min_spin.setDecimals(2)
        self.min_spin.setVisible(False)
        self.threshold_layout.addWidget(self.min_spin)
        
        # 最大值输入框
        self.max_spin = NoWheelDoubleSpinBox()
        self.max_spin.setRange(0, 1)
        self.max_spin.setSingleStep(0.1)
        self.max_spin.setDecimals(2)
        self.max_spin.setVisible(False)
        self.threshold_layout.addWidget(self.max_spin)
        
        # 阈值输入框
        self.threshold_spin = NoWheelDoubleSpinBox()
        self.threshold_spin.setRange(0, 1)
        self.threshold_spin.setSingleStep(0.1)
        self.threshold_spin.setDecimals(2)
        self.threshold_layout.addWidget(self.threshold_spin)
        
        # 比较对象选择框
        self.compare_to_combo = NoWheelComboBox()
        self.compare_to_combo.addItems(feature_list)
        self.compare_to_combo.setVisible(False)
        self.threshold_layout.addWidget(self.compare_to_combo)
        
        self.threshold_widget.setLayout(self.threshold_layout)
        layout.addWidget(self.threshold_widget)
        
        # Delete Button
        self.delete_btn = QPushButton("Delete")
        layout.addWidget(self.delete_btn)
        
        self.setLayout(layout)
        
        # 初始化显示状态
        self.on_operator_changed(self.operator_combo.currentText())

    def on_operator_changed(self, operator):
        """根据操作符类型更新界面显示"""
        if operator == 'BETWEEN':
            self.min_spin.setVisible(True)
            self.max_spin.setVisible(True)
            self.threshold_spin.setVisible(False)
            self.compare_to_combo.setVisible(False)
        elif operator.startswith('DIFF'):
            self.min_spin.setVisible(False)
            self.max_spin.setVisible(False)
            self.threshold_spin.setVisible(True)
            self.compare_to_combo.setVisible(True)
        else:
            self.min_spin.setVisible(False)
            self.max_spin.setVisible(False)
            self.threshold_spin.setVisible(True)
            self.compare_to_combo.setVisible(False)

    def get_condition(self):
        """获取当前条件配置"""
        condition = {
            'feature': self.expression_combo.currentText(),
            'operator': self.operator_combo.currentText()
        }
        
        operator = condition['operator']
        if operator == 'BETWEEN':
            condition['min'] = self.min_spin.value()
            condition['max'] = self.max_spin.value()
        elif operator.startswith('DIFF'):
            condition['threshold'] = self.threshold_spin.value()
            condition['compare_to'] = self.compare_to_combo.currentText()
        else:
            condition['threshold'] = self.threshold_spin.value()
            
        return condition

    def set_condition(self, condition):
        """设置条件配置"""
        self.expression_combo.setCurrentText(condition['feature'])
        self.operator_combo.setCurrentText(condition['operator'])
        
        operator = condition['operator']
        if operator == 'BETWEEN':
            self.min_spin.setValue(condition.get('min', 0))
            self.max_spin.setValue(condition.get('max', 1))
        elif operator.startswith('DIFF'):
            self.threshold_spin.setValue(condition.get('threshold', 0))
            self.compare_to_combo.setCurrentText(condition.get('compare_to', ''))
        else:
            self.threshold_spin.setValue(condition.get('threshold', 0))

    def get_main_window(self):
        """获取主窗口实例"""
        parent = self.parent()
        while parent:
            if isinstance(parent, ConfigWindow):
                return parent
            parent = parent.parent()
        return None

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 设置样式
    app.setStyle("Fusion")
    
    window = ConfigWindow()
    window.show()
    sys.exit(app.exec())