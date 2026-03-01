# 登录注册&裂隙识别程序
# 功能：包含登录、注册、记住密码、自动登录、找回密码功能
# 包含注册界面中提示词功能
# 跳转到软件主要功能界面
# 新增图片放大缩小功能，局部布局调整
# 新增导出图片功能
# 裂隙数据优化 & 裂隙识别精细处理
# 大图片导入 & 添加提示
# 添加历史记录功能 & 对操作历史中的时间进行修改
# 新增删除操作历史功能&批量删除 & 整体布局调整 & 优化提示词 & 提升响应速度
# 作者：郭宇帆
# 版本：V3.0
# 2026年1月4日17:46:24

import os # 系统交互功能
import tensorflow as tf
# 屏蔽TensorFlow Info级日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# 关闭oneDNN自定义操作提示
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import datetime
import hashlib  # 加密密码
import json  # json格式对象解码为python对象
import math
import pywt
import re  # 字符串匹配
import sqlite3  # 对象与数据库进行交互
import tkinter as tk  # GUI图形库
from tkinter import filedialog
from tkinter import ttk, messagebox
from keras.src.saving import load_model
from keras.src.utils import img_to_array
from keras import layers, Model
from ttkthemes import ThemedTk, ThemedStyle
import cv2  # 进行图像处理
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.cluster import DBSCAN  # 密度聚类算法 #矩阵操作
import pandas as pd  # 数据分析和数据处理
from PIL import Image, ImageTk


# -------------------裂隙识别功能--------------------
class MainApplication:
    # 软件主功能界面

    def __init__(self, master, user_id=0):
        self.user_id = user_id
        self.master = master
        self.style = ThemedStyle(self.master)
        self.style.set_theme("breeze")
        self.master.title("图片裂隙智能分析系统 v3.0")
        self.master.geometry("1250x850")
        self.master.minsize(1000, 700)
        self.master.state('zoomed')

        # 初始化核心变量（原有变量全部保留，确保兼容性）
        self.zoom_level = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.drag_data = {"x": 0, "y": 0, "item": None}
        self.original_image = None
        self.processed_image = None
        self.crack_data = []
        self.model = None
        self.threshold_value = 127  # 传统方法阈值
        self.min_length = 20  # 适配显微镜细裂隙
        self.tooltip = None
        self.tooltip_text = tk.StringVar()
        self.tooltip_visible = False
        self._image_refs = []  # 防止图像被垃圾回收

        # 状态标签（原有代码不变）
        self.status_label = ttk.Label(self.master, text="就绪")
        self.status_label.pack(side='bottom', fill='x', padx=10, pady=5)

        # ========== 关键：创建主容器并调用create_left_workspace，初始化阈值滑块 ==========
        self.main_container = ttk.Frame(self.master)
        self.main_container.pack(fill='both', expand=True)
        # 调用方法创建左侧工作区（含threshold_slider控件）
        self.create_left_workspace(self.main_container)
        # ==============================================================================

        # 加载AI模型和历史记录（原有代码不变）
        self.load_ai_model()
        self.load_history()

        # 画布图像引用初始化（原有代码不变）
        if hasattr(self, 'original_canvas') and hasattr(self, 'processed_canvas'):
            self.original_canvas._image_refs = []
            self.processed_canvas._image_refs = []

    def unet_plus_plus(input_size=(256, 256, 3)):
        """定义完整U-Net++模型结构（消除c3未定义错误）"""
        # 输入层
        inputs = layers.Input(input_size)

        # ==================== 编码器部分（完整定义，包含c3/p3，解决未定义问题） ====================
        # 第1层编码器（浅层：捕捉细粒度裂隙特征）
        c1 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
        c1 = layers.Dropout(0.1)(c1)
        c1 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
        p1 = layers.MaxPooling2D((2, 2))(c1)  # 下采样，尺寸减半

        # 第2层编码器
        c2 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = layers.Dropout(0.1)(c2)
        c2 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
        p2 = layers.MaxPooling2D((2, 2))(c2)  # 下采样，尺寸减半

        # 第3层编码器（定义c3/p3，解决未解析引用错误）
        c3 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = layers.Dropout(0.2)(c3)
        c3 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
        p3 = layers.MaxPooling2D((2, 2))(c3)  # 下采样，尺寸减半

        # 第4层编码器（可选，加深网络，提升深层语义提取能力）
        c4 = layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
        c4 = layers.Dropout(0.2)(c4)
        c4 = layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
        p4 = layers.MaxPooling2D((2, 2))(c4)

        # 瓶颈层（网络最深层）
        c5 = layers.Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
        c5 = layers.Dropout(0.3)(c5)
        c5 = layers.Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

        # ==================== 解码器部分（U-Net++核心：嵌套跳跃连接，基于已定义的c3等变量） ====================
        # 第1层解码（对应c4，上采样+融合c4特征）
        u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = layers.concatenate([u6, c4])  # 融合同尺度编码器特征（c4已定义）
        c6 = layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
        c6 = layers.Dropout(0.2)(c6)
        c6 = layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

        # 第2层解码（对应c3，上采样+融合c3特征，此时c3已定义，无报错）
        u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = layers.concatenate([u7, c3])  # 融合c3特征（已定义，解决未引用问题）
        c7 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = layers.Dropout(0.2)(c7)
        c7 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

        # 第3层解码（对应c2，上采样+融合c2特征）
        u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = layers.concatenate([u8, c2])
        c8 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = layers.Dropout(0.1)(c8)
        c8 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

        # 第4层解码（对应c1，上采样+融合c1特征，恢复原始图像尺寸）
        u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = layers.concatenate([u9, c1])
        c9 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = layers.Dropout(0.1)(c9)
        c9 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

        # 输出层（裂隙二分类：1个通道，sigmoid激活）
        outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

        # 构建完整模型
        model = Model(inputs=[inputs], outputs=[outputs])
        return model

    def load_ai_model(self):
        """加载预训练的U-Net++模型"""
        try:
            # 更改模型路径为U-Net++模型
            model_path = "unet_plus_cos24.h5"  # 新的模型文件名
            if not os.path.exists(model_path):
                model_path = "unet_plus_cos24.h5"  # 备选路径

            if os.path.exists(model_path):
                self.model = load_model(model_path)
                # 编译模型（保持与原代码一致）
                self.model.compile(
                    optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                self.status_label.config(text=f"U-Net++模型加载成功: {os.path.basename(model_path)}")
            else:
                self.status_label.config(text="未找到U-Net++模型文件，请确保模型文件在程序根目录下")
                messagebox.showwarning("模型加载警告", "未找到预训练模型，将使用传统方法进行识别")
        except Exception as e:
            self.status_label.config(text=f"模型加载失败: {str(e)}")
            messagebox.showerror("模型加载错误", f"加载模型时发生错误: {str(e)}\n将使用传统方法进行识别")

    def create_history_panel(self, parent):
        """创建历史记录面板布局"""
        history_frame = ttk.Frame(parent)
        history_frame.grid(row=0, column=0, sticky='nsew')

        # 配置网格权重，使区域可扩展
        history_frame.grid_rowconfigure(1, weight=1)
        history_frame.grid_columnconfigure(0, weight=1)

        # 标题和删除按钮
        header_frame = ttk.Frame(history_frame)
        header_frame.grid(row=0, column=0, sticky='ew', pady=5)

        ttk.Label(header_frame, text="操作历史", font=('Microsoft YaHei', 10, 'bold')).pack(side='left', padx=5)
        ttk.Button(header_frame, text="删除历史记录", command=self.delete_selected_history).pack(side='right', padx=5)

        # 历史记录表格
        self.history_tree = ttk.Treeview(history_frame,
                                         columns=('time', 'action'),
                                         show='headings',
                                         height=25,
                                         selectmode='extended')  # 支持多选

        # 设置列标题和宽度
        self.history_tree.heading('time', text='时间')
        self.history_tree.heading('action', text='操作')
        self.history_tree.column('time', width=120, anchor='center')
        self.history_tree.column('action', width=200, anchor='w')

        self.history_tree.grid(row=1, column=0, sticky='nsew', padx=5)

    def log_history(self, action_type, file_path="", details=""):
        """记录操作历史到数据库"""
        try:
            conn = sqlite3.connect('qq_users.db')
            c = conn.cursor()
            # 获取当前时间（24小时制）
            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            # 插入记录
            c.execute('''INSERT INTO history 
                       (user_id, action_type, action_time, file_path, details)
                       VALUES (?,?,?,?,?)''',
                      (self.user_id, action_type, current_time, file_path, details))
            conn.commit()
            conn.close()
            # 刷新显示
            self.load_history()
        except Exception as e:
            print(f"历史记录保存失败: {str(e)}")
            messagebox.showerror("错误", f"记录操作历史失败: {str(e)}")

    def load_history(self):
        """从数据库加载历史记录并显示"""
        try:
            conn = sqlite3.connect('qq_users.db')
            c = conn.cursor()
            # 查询当前用户的最近100条记录，按时间倒序
            c.execute('''SELECT action_time, action_type, file_path 
                       FROM history 
                       WHERE user_id=? 
                       ORDER BY action_time DESC LIMIT 100''',
                      (self.user_id,))

            # 清空现有记录
            for item in self.history_tree.get_children():
                self.history_tree.delete(item)

            # 添加新记录
            for row in c.fetchall():
                # 显示格式：时间 + 操作 + 文件名
                action_text = f"{row[1]} {os.path.basename(row[2]) if row[2] else ''}"
                self.history_tree.insert('', 'end', values=(row[0], action_text))

            conn.close()
        except Exception as e:
            print(f"历史记录加载失败: {str(e)}")

    def delete_selected_history(self):
        """删除选中的历史记录"""
        selected_items = self.history_tree.selection()
        if not selected_items:
            messagebox.showwarning("提示", "请选择要删除的历史记录")
            return

        try:
            conn = sqlite3.connect('qq_users.db')
            c = conn.cursor()

            # 批量删除数据库记录
            for item in selected_items:
                values = self.history_tree.item(item, 'values')
                if values:
                    time_str = values[0]
                    c.execute('''DELETE FROM history 
                               WHERE user_id=? AND action_time=?''',
                              (self.user_id, time_str))

            conn.commit()
            conn.close()

            # 从界面删除选中项
            for item in selected_items:
                self.history_tree.delete(item)

            messagebox.showinfo("成功", "选中的历史记录已删除")
        except Exception as e:
            messagebox.showerror("错误", f"删除历史记录失败: {str(e)}")

    def clear_all_history(self):
        """清空当前用户所有历史记录"""
        if not messagebox.askyesno("确认", "确定要删除所有历史记录吗？此操作不可恢复"):
            return

        try:
            conn = sqlite3.connect('qq_users.db')
            c = conn.cursor()
            c.execute('DELETE FROM history WHERE user_id=?', (self.user_id,))
            conn.commit()
            conn.close()

            # 清空界面记录
            for item in self.history_tree.get_children():
                self.history_tree.delete(item)

            messagebox.showinfo("成功", "所有历史记录已清空")
        except Exception as e:
            messagebox.showerror("错误", f"清空失败: {str(e)}")

    def create_widgets(self):
        # 确保在所有控件创建前设置主题
        self.style.configure('TButton', font=('Microsoft YaHei', 10))
        self.style.configure('TLabel', font=('Microsoft YaHei', 10))
        # 主容器改为左右结构
        main_pane = ttk.PanedWindow(self.master, orient=tk.HORIZONTAL)
        main_pane.pack(fill='both', expand=True)

        # 初始化提示框
        self.tooltip = None
        self.tooltip_text = tk.StringVar()
        self.tooltip_visible = False

        # 设置grid权重
        main_pane.grid_rowconfigure(0, weight=1)
        main_pane.grid_columnconfigure(0, weight=4)
        main_pane.grid_columnconfigure(1, weight=1)

        # 左侧工作区
        left_frame = ttk.Frame(main_pane)
        main_pane.add(left_frame, weight=4)
        left_frame.grid_rowconfigure(0, weight=1)
        left_frame.grid_columnconfigure(0, weight=1)

        # 右侧历史记录区
        right_frame = ttk.Frame(main_pane)
        main_pane.add(right_frame, weight=1)
        right_frame.grid_rowconfigure(0, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)

        # 创建左侧工作区内容
        self.create_left_workspace(left_frame)

        # 创建右侧历史记录区
        self.create_history_panel(right_frame)

    def create_left_workspace(self, parent):
        # 顶部工具栏（保持原有按钮）
        tool_frame = ttk.Frame(parent)
        tool_frame.pack(pady=10, fill='x')

        # 按钮框架
        button_frame = ttk.Frame(tool_frame)
        button_frame.pack(side='left')

        ttk.Button(button_frame, text="导入图片", command=self.load_image).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(button_frame, text="裂隙识别", command=self.analyze_cracks).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(button_frame, text="导出数据", command=self.export_data).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(button_frame, text="导出图片", command=self.export_image).grid(row=0, column=3, padx=5, pady=5)
        ttk.Button(button_frame, text="统计分析", command=self.show_statistics).grid(row=0, column=4, padx=5, pady=5)

        # 阈值控件
        threshold_frame = ttk.Frame(button_frame)
        threshold_frame.grid(row=0, column=5, padx=5, pady=5)
        ttk.Label(threshold_frame, text="模型阈值:").pack(side='left')
        self.threshold_slider = ttk.Scale(
            threshold_frame, from_=0.0, to=1.0,
            value=0.7, orient='horizontal', length=150
        )
        self.threshold_slider.pack(side='left', padx=5)
        self.threshold_label = ttk.Label(threshold_frame, text="0.5")
        self.threshold_label.pack(side='left')
        self.threshold_slider.bind("<Motion>", lambda e: self.threshold_label.config(
            text=f"{self.threshold_slider.get():.1f}"))
        fusion_frame = ttk.Frame(tool_frame)
        fusion_frame.pack(side='right', padx=10)
        ttk.Label(fusion_frame, text="模型融合权重:").pack(side='left')
        self.fusion_slider = ttk.Scale(  # 必须加self，设为实例属性
            fusion_frame, from_=0.0, to=1.0,
            value=0.7, orient='horizontal', length=150
        )
        self.fusion_slider.pack(side='left', padx=5)
        self.fusion_label = ttk.Label(fusion_frame, text="0.7")  # 加self
        self.fusion_label.pack(side='left')
        self.fusion_slider.bind("<Motion>", lambda e: self.fusion_label.config(
            text=f"{self.fusion_slider.get():.1f}"))

        # 内容布局（左右分栏）
        content_frame = ttk.Frame(parent)
        content_frame.pack(fill='both', expand=True)
        content_frame.grid_columnconfigure(0, weight=3)
        content_frame.grid_columnconfigure(1, weight=1)
        content_frame.grid_rowconfigure(0, weight=1)

        # 左侧图像显示区域
        left_content_frame = ttk.Frame(content_frame)
        left_content_frame.grid(row=0, column=0, sticky='nsew', padx=10, pady=5)

        # 图像显示画布
        img_frame = ttk.Frame(left_content_frame)
        img_frame.pack(fill='both', expand=True, pady=5)

        self.original_canvas = tk.Canvas(img_frame, width=600, height=600, bg='gray')
        self.original_canvas.grid(row=0, column=0, padx=10, pady=5)
        self.original_canvas.bind("<Motion>", self.on_canvas_motion)
        self.original_canvas.bind("<Leave>", self.hide_tooltip)

        self.processed_canvas = tk.Canvas(img_frame, width=600, height=600, bg='gray')
        self.processed_canvas.grid(row=0, column=1, padx=10, pady=5)
        self.processed_canvas.bind("<Motion>", self.on_canvas_motion)
        self.processed_canvas.bind("<Leave>", self.hide_tooltip)

        # 图像操作控件
        self.add_image_controls(self.original_canvas)
        self.add_image_controls(self.processed_canvas)

        # 裂隙结果显示
        result_frame = ttk.Frame(left_content_frame)
        result_frame.pack(fill='x', pady=10)

        ttk.Label(result_frame, text="裂隙识别结果", font=("Arial", 12, "bold")).pack(pady=5)
        self.result_text = tk.Text(result_frame, height=10, width=80)
        self.result_text.pack(pady=5, padx=10, fill='x')

        # 右侧历史记录区（使用Treeview统一显示）
        right_history_frame = ttk.Frame(content_frame)
        right_history_frame.grid(row=0, column=1, sticky='nsew', padx=10, pady=5)

        ttk.Label(right_history_frame, text="历史处理记录", font=("Arial", 12, "bold")).pack(pady=5)

        # 历史记录表格
        self.history_tree = ttk.Treeview(right_history_frame,
                                         columns=('time', 'action'),
                                         show='headings',
                                         height=30,
                                         selectmode='extended')
        self.history_tree.heading('time', text='时间')
        self.history_tree.heading('action', text='操作')
        self.history_tree.column('time', width=120, anchor='center')
        self.history_tree.column('action', width=200, anchor='w')
        self.history_tree.pack(pady=5, padx=5, fill='both', expand=True)

        # 历史记录操作按钮
        btn_frame = ttk.Frame(right_history_frame)
        btn_frame.pack(fill='x', pady=5)
        ttk.Button(btn_frame, text="删除选中记录", command=self.delete_selected_history).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="清空所有记录", command=self.clear_all_history).pack(side='right', padx=5)

    def create_history_panel(self, parent):

        # 历史记录面板
        history_frame = ttk.Frame(parent)
        history_frame.grid(row=0, column=0, sticky='nsew')

        # 拉长页面，设置较大的高度
        history_frame.configure(height=1000)

        # 使 history_frame 的行和列能自适应扩展
        history_frame.grid_rowconfigure(1, weight=1)
        history_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(history_frame, text="操作历史").grid(row=0, column=0, sticky='nw')

        # 修改选择模式为 extended，支持多选
        self.history_tree = ttk.Treeview(history_frame, columns=('time',
                                                                 'action'), show='headings', height=25,
                                         selectmode='extended')
        self.history_tree.heading('time', text='时间')
        self.history_tree.heading('action', text='操作')
        self.history_tree.column('time', width=120)
        self.history_tree.column('action', width=200)
        self.history_tree.grid(row=1, column=0, sticky='nsew')

        # 添加“删除历史记录”按钮
        ttk.Button(history_frame, text="删除历史记录", command=self.delete_selected_history).grid(
            row=0, column=0, pady=5)

    # 删除重复的delete_selected_history方法，保留以下版本
    def delete_selected_history(self):
        """删除选中的历史记录"""
        selected_items = self.history_tree.selection()
        if not selected_items:
            messagebox.showwarning("提示", "请选择要删除的历史记录")
            return

        try:
            conn = sqlite3.connect('qq_users.db')
            c = conn.cursor()

            for item in selected_items:
                values = self.history_tree.item(item, 'values')
                if values:
                    time_str = values[0]
                    c.execute('''DELETE FROM history 
                               WHERE user_id=? AND action_time=?''',
                              (self.user_id, time_str))

            conn.commit()
            conn.close()

            # 从界面删除选中项
            for item in selected_items:
                self.history_tree.delete(item)

            messagebox.showinfo("成功", f"已删除{len(selected_items)}条记录")
        except Exception as e:
            messagebox.showerror("错误", f"删除失败: {str(e)}")

    def add_image_controls(self, canvas):
        """添加图像操作控件"""
        # 创建按钮容器
        control_frame = ttk.Frame(canvas)

        # 创建操作按钮
        ttk.Button(control_frame, text="↔", width=3,
                   command=lambda: self.reset_position(canvas)).pack(side='left', padx=2)
        ttk.Button(control_frame, text="+", width=3,
                   command=lambda: self.zoom_image(canvas, 1.2)).pack(side='left', padx=2)
        ttk.Button(control_frame, text="-", width=3,
                   command=lambda: self.zoom_image(canvas, 0.8)).pack(side='left', padx=2)  # 修复缩小功能

        # 将按钮容器添加到画布右上角
        canvas.create_window(570, 10, window=control_frame, anchor='ne')

        # 绑定鼠标事件
        canvas.bind("<ButtonPress-1>", lambda e: self.start_drag(e, canvas))
        canvas.bind("<B1-Motion>", lambda e: self.on_drag(e, canvas))
        canvas.bind("<MouseWheel>", lambda e: self.on_mousewheel(e, canvas))

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image文件", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        if not file_path:
            return

        try:
            # 使用Pillow打开图像（支持更广泛的格式）
            with Image.open(file_path) as pil_img:
                # 检查图像模式并转换为RGB
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')

                # 转换为numpy数组
                np_image = np.array(pil_img)

                # 转换为OpenCV格式（BGR）
                original_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

                # 处理超大图像
                width, height = pil_img.size
                if width * height > 10000000:
                    scale = math.sqrt(10000000 / (width * height))
                    new_size = (int(width * scale), int(height * scale))
                    original_image = cv2.resize(original_image, new_size)

                self.original_image = original_image

        except Exception as e:
            messagebox.showerror("错误",
                                 f"图像读取失败：{str(e)}\n"
                                 "可能原因：\n"
                                 "1. 文件路径包含特殊字符\n"
                                 "2. 图像文件已损坏\n"
                                 "3. 暂不支持该图像格式\n"
                                 "4. 图像尺寸超过系统内存限制")
            return

        self.show_image(self.original_image, self.original_canvas)
        self.log_history("图片导入", file_path)
        self.status_label.config(text="图像加载完成")
        self.master.update_idletasks()

    def traditional_preprocessing(self, image):
        """传统视觉预处理步骤"""
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 自适应直方图均衡化增强对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_enhanced = clahe.apply(gray)

        # 高斯滤波去噪
        blurred = cv2.GaussianBlur(gray_enhanced, (5, 5), 0)

        # 边缘检测
        edges = cv2.Canny(blurred, 50, 150)

        return edges, gray_enhanced

    def analyze_cracks(self):
        """裂隙分析核心算法：优化后（U-Net+++显微镜专用预处理+滑动窗口）"""
        if self.original_image is None:
            messagebox.showerror("错误", "请先导入图片")
            return

        self.status_label.config(text="正在进行裂隙识别...")
        self.master.update_idletasks()

        # 创建处理用的副本（保持原始图像不变）
        process_copy = self.original_image.copy()
        img_height, img_width = process_copy.shape[:2]

        # 传统视觉预处理（原有代码不变）
        edges, gray_enhanced = self.traditional_preprocessing(process_copy)

        # 初始化处理结果图像
        processed_img = process_copy.copy()
        self.crack_data = []

        try:
            # 如果模型加载成功，使用优化后的混合模型进行识别
            if self.model is not None:
                # 准备模型输入参数
                input_size = (256, 256)  # 与训练时一致
                # 1. 显微镜图像专用预处理（新增）
                img_preprocessed = self.microscope_preprocessing(process_copy)

                self.status_label.config(text="模型正在预测...")
                self.master.update_idletasks()

                # 2. 大图像使用滑动窗口推理，小图像直接推理（新增）
                if img_width > 512 or img_height > 512:
                    prediction = self.sliding_window_inference(img_preprocessed, input_size)
                else:
                    img_resized = cv2.resize(img_preprocessed, input_size)
                    img_normalized = img_to_array(img_resized) / 255.0
                    img_input = np.expand_dims(img_normalized, axis=0)
                    prediction = self.model.predict(img_input, verbose=0)[0]

                # 后处理预测结果（优化版：动态阈值+Otsu+增强形态学）
                prediction = cv2.resize(prediction, (img_width, img_height))

                # 3. 使用滑块动态阈值（替代固定0.5）
                threshold_value = self.threshold_slider.get()
                prediction_binary = (prediction > threshold_value).astype(np.uint8) * 255

                # 4. Otsu算法二次优化（双阈值保障，新增）
                _, prediction_binary_otsu = cv2.threshold(
                    prediction_binary, 0, 255,
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )

                # 5. 增强形态学操作（针对细微裂隙，新增）
                kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
                kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                # 新增：开运算（优先消除微小噪声区域）
                prediction_binary_otsu = cv2.morphologyEx(prediction_binary_otsu, cv2.MORPH_OPEN, kernel_small,
                                                          iterations=1)
                # 原有腐蚀+膨胀（弱化强度，保留有效裂隙）
                prediction_binary_otsu = cv2.erode(prediction_binary_otsu, kernel_small, iterations=1)
                prediction_binary_otsu = cv2.dilate(prediction_binary_otsu, kernel_large, iterations=1)
                # ==========================================

                # 得到 combined_mask 之后（继续下一步修改）
                edges_resized = cv2.resize(edges, (img_width, img_height))
                model_weight = self.fusion_slider.get()
                traditional_weight = 1 - model_weight
                combined_mask = cv2.addWeighted(
                    prediction_binary_otsu.astype(np.float32), model_weight,
                    edges_resized.astype(np.float32), traditional_weight,
                    0
                ).astype(np.uint8)

                # 从掩码中提取裂隙轮廓
                contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                self.status_label.config(text="正在分析裂隙特征...")
                self.master.update_idletasks()

            else:
                # 模型加载失败，使用传统方法作为备选（原有代码不变）
                self.status_label.config(text="使用传统方法进行识别...")
                self.master.update_idletasks()

                # 使用中值滤波减少噪声
                gray = cv2.cvtColor(process_copy, cv2.COLOR_BGR2GRAY)
                gray = cv2.medianBlur(gray, 5)

                # 自适应二值化处理
                binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY_INV, 17, 5)

                # 形态学操作
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                binary = cv2.dilate(binary, kernel, iterations=1)

                # 查找轮廓
                contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


            # 分析每个轮廓（原有代码不变，仅调整最小长度适配细裂隙）
            crack_info_list = []
            for idx, cnt in enumerate(contours):
                # 计算轮廓长度
                length = cv2.arcLength(cnt, False)

                # 忽略小线段
                if length < 30:  # 关键调整：减小最小长度阈值
                    continue

                # 使用Douglas-Peucker算法简化轮廓
                epsilon = 0.001 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)

                # 计算轮廓起点和终点
                start = tuple(approx[0][0])
                end = tuple(approx[-1][0])

                # 计算方向（正北方向）
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                angle = math.degrees(math.atan2(dy, dx))
                if dx >= 0 and dy >= 0:  # 第一象限
                    north_direction = angle
                elif dx < 0 and dy >= 0:  # 第二象限
                    north_direction = 180 + angle
                elif dx < 0 and dy < 0:  # 第三象限
                    north_direction = 180 + angle
                else:  # 第四象限
                    north_direction = 360 + angle

                # 存储裂隙信息
                crack_info_list.append({
                    "原始索引": idx,
                    "编号": idx,
                    "起点X": start[0],
                    "起点Y": start[1],
                    "终点X": end[0],
                    "终点Y": end[1],
                    "长度(像素)": length,
                    "正北方向(度)": north_direction
                })

            # 根据长度对裂隙信息进行排序
            crack_info_list.sort(key=lambda x: x["长度(像素)"], reverse=True)

            # 选取主要裂隙
            self.crack_data = crack_info_list[:200]

            # 重新编号
            for i, item in enumerate(self.crack_data):
                item['编号'] = i

            # 在图像上绘制主要裂隙
            for item in self.crack_data:
                idx = item["编号"]
                start = (item["起点X"], item["起点Y"])
                end = (item["终点X"], item["终点Y"])

                # 从简化后的轮廓中获取点
                approx = cv2.approxPolyDP(contours[item["原始索引"]], 0.001 * cv2.arcLength(
                    contours[item["原始索引"]], True), True)
                pts = approx.reshape((-1, 1, 2)).astype(np.int32)

                # 绘制裂隙线条
                cv2.polylines(processed_img, [pts], False, (0, 0, 255), 1)
                cv2.putText(processed_img, str(idx), start,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # 显示处理结果
            self.processed_image = processed_img
            self.show_image(self.original_image, self.original_canvas)
            self.show_image(processed_img, self.processed_canvas)

            # 更新结果显示
            self.update_result()

            if self.crack_data:
                self.log_history("裂隙识别", "", f"识别到{len(self.crack_data)}条裂隙")
                self.status_label.config(text=f"裂隙识别完成，共发现{len(self.crack_data)}条裂隙")
            else:
                self.status_label.config(text="未识别到裂隙")

        except Exception as e:
            self.status_label.config(text=f"识别过程出错: {str(e)}")
            messagebox.showerror("识别错误", f"裂隙识别过程中发生错误: {str(e)}")

    # 新增显微镜图像预处理函数
    def microscope_preprocessing(self, image):
        """针对显微镜图像的专用预处理：增强细微裂隙对比度+去噪"""
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # ========== 新增1：高斯滤波 ==========
        gray = cv2.GaussianBlur(gray, (3, 3), 0)  # 轻度高斯滤波，减少噪声

        # 自适应直方图均衡化（增强局部对比度，适配显微镜低对比度图像）
        # 优化后：提高clipLimit，细化网格，增强对比度
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(2, 2))  # 提高对比度限制，更细网格
        clahe_img = clahe.apply(gray)

        # 小波去噪（保留边缘的同时去除高频噪声，不破坏细裂隙）
        coeffs = pywt.wavedec2(clahe_img, 'db4', level=3)
        cA3, (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1) = coeffs
        # 阈值处理高频分量，过滤噪声
        threshold = 30
        cH1 = np.where(np.abs(cH1) < threshold, 0, cH1)
        cV1 = np.where(np.abs(cV1) < threshold, 0, cV1)
        cD1 = np.where(np.abs(cD1) < threshold, 0, cD1)
        coeffs_new = cA3, (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1)
        denoised_img = pywt.waverec2(coeffs_new, 'db4').astype(np.uint8)
        # 新增：直方图均衡化二次增强
        denoised_img = cv2.equalizeHist(denoised_img)
        # ========== 新增2：中值滤波 ==========
        denoised_img = cv2.medianBlur(denoised_img, 3)  # 消除孤立噪声点
        # 转回BGR格式，适配模型输入要求
        return cv2.cvtColor(denoised_img, cv2.COLOR_GRAY2BGR)

    def sliding_window_inference(self, image, input_size, stride=128):
        """滑动窗口分块处理大尺寸显微镜图像，避免细节丢失（修复除以0警告）"""
        h, w = image.shape[:2]
        input_h, input_w = input_size
        # 初始化输出掩码和计数矩阵（处理重叠区域）
        output = np.zeros((h, w, 1), dtype=np.float32)
        count = np.zeros((h, w, 1), dtype=np.float32)

        # 滑动窗口遍历图像
        for y in range(0, h - input_h + 1, stride):
            for x in range(0, w - input_w + 1, stride):
                # 提取当前窗口
                window = image[y:y + input_h, x:x + input_w]
                # 窗口预处理+归一化
                window_resized = cv2.resize(window, input_size)
                window_norm = img_to_array(window_resized) / 255.0
                window_input = np.expand_dims(window_norm, axis=0)
                # 模型预测
                pred = self.model.predict(window_input, verbose=0)[0]
                # 将预测结果放回对应位置
                output[y:y + input_h, x:x + input_w] += pred
                count[y:y + input_h, x:x + input_w] += 1

        # 处理底部边界区域（避免遗漏最后一行未覆盖区域）
        if y + input_h < h:
            y = h - input_h
            for x in range(0, w - input_w + 1, stride):
                window = image[y:y + input_h, x:x + input_w]
                window_resized = cv2.resize(window, input_size)
                window_norm = img_to_array(window_resized) / 255.0
                window_input = np.expand_dims(window_norm, axis=0)
                pred = self.model.predict(window_input, verbose=0)[0]
                output[y:y + input_h, x:x + input_w] += pred
                count[y:y + input_h, x:x + input_w] += 1

        # 处理右侧边界区域（避免遗漏最后一列未覆盖区域）
        if x + input_w < w:
            x = w - input_w
            for y in range(0, h - input_h + 1, stride):
                window = image[y:y + input_h, x:x + input_w]
                window_resized = cv2.resize(window, input_size)
                window_norm = img_to_array(window_resized) / 255.0
                window_input = np.expand_dims(window_norm, axis=0)
                pred = self.model.predict(window_input, verbose=0)[0]
                output[y:y + input_h, x:x + input_w] += pred
                count[y:y + input_h, x:x + input_w] += 1

        # 处理右下角最后一个窗口（避免双重遗漏）
        if (y + input_h < h) and (x + input_w < w):
            y = h - input_h
            x = w - input_w
            window = image[y:y + input_h, x:x + input_w]
            window_resized = cv2.resize(window, input_size)
            window_norm = img_to_array(window_resized) / 255.0
            window_input = np.expand_dims(window_norm, axis=0)
            pred = self.model.predict(window_input, verbose=0)[0]
            output[y:y + input_h, x:x + input_w] += pred
            count[y:y + input_h, x:x + input_w] += 1

        # ========== 关键修复：将count中0值替换为1，避免除以0 ==========
        count[count == 0] = 1
        # ============================================================
        # 重叠区域取平均，消除窗口边界效应
        output = output / count
        return output

    def export_image(self):
        # 导出识别图片
        if self.processed_image is None:
            messagebox.showerror("错误", "请先进行裂隙识别")
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG文件", "*.png"), ("JPEG文件", "*.jpg;*.jpeg"), ("BMP文件", "*.bmp"),
                       ("TIFF文件", "*.tif")]
        )
        if file_path:
            try:
                # 将OpenCV的BGR格式图片转换为PIL的RGB格式图片
                pil_image = Image.fromarray(cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB))
                pil_image.save(file_path)
                messagebox.showinfo("成功", "图片导出完成")
            except PermissionError:
                messagebox.showerror("错误", "文件被其他程序占用，请关闭后重试")
            except Exception as e:
                messagebox.showerror("导出失败", f"发生未知错误：{str(e)}")
        if file_path:
            self.log_history("图片导出", file_path)

    def show_statistics(self):
        """统计分析功能（修复Matplotlib错误 + 期刊论文级美观度）"""
        if not self.crack_data:
            messagebox.showwarning("提示", "暂无裂隙数据，无法进行统计分析")
            return

        # ========== 全局样式设置（期刊论文标准，移除无效参数） ==========
        plt.rcParams['font.family'] = 'Times New Roman'  # 期刊常用字体
        plt.rcParams['font.size'] = 8  # 基础字体大小
        plt.rcParams['axes.linewidth'] = 1.0  # 坐标轴线条宽度
        plt.rcParams['xtick.major.width'] = 1.0  # x轴主刻度线宽度
        plt.rcParams['ytick.major.width'] = 1.0  # y轴主刻度线宽度
        plt.rcParams['figure.dpi'] = 300  # 高清分辨率（期刊要求≥300dpi）
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['axes.spines.top'] = False  # 隐藏顶部边框
        plt.rcParams['axes.spines.right'] = False  # 隐藏右侧边框
        # 移除无效的 savefig.bbox_inches 配置（核心修复）

        # 提取统计数据
        lengths = [item["长度(像素)"] for item in self.crack_data]
        directions = [item["正北方向(度)"] for item in self.crack_data]
        start_x = [item["起点X"] for item in self.crack_data]
        start_y = [item["起点Y"] for item in self.crack_data]
        end_x = [item["终点X"] for item in self.crack_data]
        end_y = [item["终点Y"] for item in self.crack_data]

        # 创建2x2子图（长度分析、方向分析、密度热力图、汇总信息）
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(4, 3))
        fig.suptitle('Crack Statistical Analysis', fontsize=10, fontweight='bold', y=0.99)

        # ========== 1. 裂隙长度分析（直方图，期刊样式） ==========
        ax1.hist(lengths, bins=15, color='#2E86AB', edgecolor='black', linewidth=1.0, alpha=0.8)
        # 设置标题和标签
        ax1.set_title('Crack Length Distribution', fontsize=8, fontweight='bold', pad=6)
        ax1.set_xlabel('Length (pixels)', fontsize=8, fontweight='medium')
        ax1.set_ylabel('Number of Cracks', fontsize=8, fontweight='medium')
        # 添加网格线（虚线，更美观）
        ax1.grid(axis='y', linestyle='--', alpha=0.3, linewidth=0.8)
        # 设置坐标轴刻度
        ax1.tick_params(axis='both', labelsize=6)
        # 添加统计标注（均值、最大值）
        mean_len = np.mean(lengths)
        max_len = np.max(lengths)
        ax1.text(0.7, 0.9, f'Mean: {mean_len:.1f}\nMax: {max_len:.1f}',
                 transform=ax1.transAxes, fontsize=6, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

        # ========== 2. 裂隙方向分析（直方图，期刊样式） ==========
        ax2.hist(directions, bins=12, color='#A23B72', edgecolor='black', linewidth=1.0, alpha=0.8)
        ax2.set_title('Crack Direction Distribution', fontsize=8, fontweight='bold', pad=6)
        ax2.set_xlabel('Direction (degrees, North)', fontsize=8, fontweight='medium')
        ax2.set_ylabel('Number of Cracks', fontsize=8, fontweight='medium')
        ax2.set_xlim(0, 360)
        ax2.grid(axis='y', linestyle='--', alpha=0.3, linewidth=0.8)
        ax2.tick_params(axis='both', labelsize=8)
        # 添加方向标注（北、东、南、西）
        direction_annotations = [(0, 'N'), (90, 'E'), (180, 'S'), (270, 'W')]
        for deg, label in direction_annotations:
            ax2.axvline(x=deg, color='red', linestyle=':', linewidth=1.0, alpha=0.6)
            ax2.text(deg, ax2.get_ylim()[1] * 0.9, label, ha='center', fontsize=6, fontweight='bold')

        # ========== 3. 裂隙密度热力图（平滑插值，期刊样式） ==========
        # 合并起点和终点坐标（更全面的密度统计）
        all_x = np.concatenate([start_x, end_x])
        all_y = np.concatenate([start_y, end_y])
        # 计算2D密度直方图
        heatmap, xedges, yedges = np.histogram2d(all_x, all_y, bins=20)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        # 绘制平滑热力图（使用高斯插值）
        im = ax3.imshow(heatmap.T, extent=extent, origin='lower', cmap='viridis', interpolation='gaussian', alpha=0.9)
        # 添加颜色条（带标注）
        cbar = fig.colorbar(im, ax=ax3, shrink=0.8)
        cbar.set_label('Crack Density', fontsize=8, fontweight='medium')
        cbar.ax.tick_params(labelsize=6)
        # 设置标题和标签
        ax3.set_title('Crack Density Heatmap', fontsize=8, fontweight='bold', pad=6)
        ax3.set_xlabel('X Coordinate (pixels)', fontsize=8, fontweight='medium')
        ax3.set_ylabel('Y Coordinate (pixels)', fontsize=8, fontweight='medium')
        ax3.tick_params(axis='both', labelsize=6)
        # 叠加裂隙起点散点（增强可视化）
        ax3.scatter(start_x, start_y, c='red', s=8, alpha=0.6, marker='o', label='Crack Start')
        ax3.legend(loc='upper right', fontsize=6, framealpha=0.8)

        # ========== 4. 汇总信息（文本标注，简洁美观） ==========
        ax4.axis('off')  # 隐藏坐标轴
        total_cracks = len(self.crack_data)
        avg_direction = np.mean(directions)
        stats_text = f"""
        Total Cracks: {total_cracks}
        Average Length: {mean_len:.1f} pixels
        Maximum Length: {max_len:.1f} pixels
        Average Direction: {avg_direction:.1f}° (North)
        Length Std: {np.std(lengths):.1f} pixels
        Direction Std: {np.std(directions):.1f}°
        """
        # 绘制文本框（期刊样式）
        ax4.text(0.1, 0.8, stats_text, transform=ax4.transAxes, fontsize=8,
                 verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='#F1F1F1',
                                                    edgecolor='black', linewidth=1.0))
        ax4.set_title('Statistical Summary', fontsize=8, fontweight='bold', pad=4)
        # 添加历史记录
        self.log_history("统计分析", details="进行了裂隙统计分析")
        # 调整子图间距
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        # 若需要保存图片，在此处添加（指定bbox_inches='tight'，不影响显示）
        # plt.savefig('crack_statistics.png', bbox_inches='tight')
        # 显示图像
        plt.show()

    def show_length_analysis(self):
        """显示长度分析结果"""
        self.stats_canvas.delete("all")
        lengths = [item['长度(像素)'] for item in self.crack_data]

        if not lengths:
            return

        # 计算统计指标
        max_len = max(lengths)
        min_len = min(lengths)
        avg_len = sum(lengths) / len(lengths)

        # 改进后的分箱策略
        bin_count = 6  # 减少分箱数量
        bin_edges = np.linspace(min_len, max_len, bin_count + 1)  # 等间距分箱
        hist, bins = np.histogram(lengths, bins=bin_edges)

        # 设置绘图区域
        width = 700
        height = 500
        margin = 50

        # 绘制坐标轴
        self.stats_canvas.create_line(margin, height - margin, width - margin, height - margin, width=2)  # X轴
        self.stats_canvas.create_line(margin, margin, margin, height - margin, width=2)  # Y轴

        # 绘制柱状图
        bin_width = (width - 2 * margin) / bin_count
        max_height = height - 2 * margin

        for i in range(bin_count):
            x0 = margin + i * bin_width
            x1 = x0 + bin_width
            y0 = height - margin
            y1 = y0 - (hist[i] / max(hist)) * max_height if max(hist) > 0 else y0

            self.stats_canvas.create_rectangle(x0, y0, x1, y1, fill='blue')
            # 显示整数范围并旋转文本
            label = f"{int(bins[i])}-{int(bins[i + 1])}"
            self.stats_canvas.create_text((x0 + x1) / 2, y0 + 20,
                                          text=label,
                                          angle=45,  # 旋转45度
                                          anchor='nw',
                                          font=('Arial', 8))

        # 添加统计信息
        stats_text = f"最大长度: {max_len:.1f} 像素\n最小长度: {min_len:.1f} 像素\n平均长度: {avg_len:.1f} 像素"
        self.stats_canvas.create_text(width - 100, margin + 50, text=stats_text, anchor='ne')

        # 添加标题
        self.stats_canvas.create_text(width / 2, 30, text="裂隙长度分布直方图", font=('Arial', 12, 'bold'))

    def show_direction_analysis(self):
        """显示方向分析结果"""
        self.stats_canvas.delete("all")
        directions = [item['正北方向(度)'] for item in self.crack_data]
        lengths = [item['长度(像素)'] for item in self.crack_data]
        self.draw_rose_diagram(self.stats_canvas, directions, lengths)

    def show_density_heatmap(self):
        """显示密度热力图"""
        self.stats_canvas.delete("all")

        if not self.crack_data:
            return

        # 获取图像尺寸
        img_height, img_width = self.original_image.shape[:2]

        # 创建密度网格
        grid_size = 20  # 网格大小
        grid_width = img_width // grid_size
        grid_height = img_height // grid_size
        density = np.zeros((grid_height, grid_width))

        # 计算每个网格的裂隙密度
        for item in self.crack_data:
            x = int(item['起点X'] / grid_size)
            y = int(item['起点Y'] / grid_size)
            if 0 <= x < grid_width and 0 <= y < grid_height:
                density[y, x] += 1

        # 归一化
        max_density = density.max()
        if max_density > 0:
            density = density / max_density

        # 设置绘图区域
        width = 700
        height = 500
        margin = 50

        # 绘制坐标轴
        self.stats_canvas.create_line(margin, height - margin, width - margin, height - margin, width=2)  # X轴
        self.stats_canvas.create_line(margin, height - margin, margin, margin, width=2)  # Y轴

        # 添加X轴刻度标签
        for i in range(0, img_width + 1, img_width // 5):
            x_pos = margin + (i / img_width) * (width - 2 * margin)
            self.stats_canvas.create_line(x_pos, height - margin, x_pos, height - margin + 5, width=1)
            self.stats_canvas.create_text(x_pos, height - margin + 15, text=str(i), anchor='n')

        # 添加Y轴刻度标签
        for i in range(0, img_height + 1, img_height // 5):
            y_pos = height - margin - (i / img_height) * (height - 2 * margin)
            self.stats_canvas.create_line(margin, y_pos, margin - 5, y_pos, width=1)
            self.stats_canvas.create_text(margin - 10, y_pos, text=str(i), anchor='e')

        # 绘制热力图
        cell_width = (width - 2 * margin) / grid_width
        cell_height = (height - 2 * margin) / grid_height

        for y in range(grid_height):
            for x in range(grid_width):
                x0 = margin + x * cell_width
                y0 = margin + y * cell_height
                x1 = x0 + cell_width
                y1 = y0 + cell_height

                # 根据密度值设置颜色
                val = density[y, x]
                color = self._get_heatmap_color(val)
                self.stats_canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline='')

        # 添加标题
        self.stats_canvas.create_text(width / 2, 30, text="裂隙密度热力图", font=('Arial', 12, 'bold'))

        # 添加范围显示
        self.stats_canvas.create_text(width / 2, height - 0.01,
                                      text=f"X轴范围: 0-{img_width}像素, Y轴范围: 0-{img_height}像素",
                                      font=('Arial', 10))

    def _get_heatmap_color(self, value):
        """根据密度值获取颜色"""
        if value <= 0:
            return '#FFFFFF'
        elif value <= 0.25:
            return '#FFCCCC'
        elif value <= 0.5:
            return '#FF9999'
        elif value <= 0.75:
            return '#FF6666'
        else:
            return '#FF0000'

    def draw_rose_diagram(self, canvas, directions, lengths):
        """绘制方向分布玫瑰图"""
        center_x, center_y = 350, 250
        max_length = max(lengths) if lengths else 1

        # 绘制极坐标网格
        for r in range(100, 201, 100):
            canvas.create_oval(center_x - r, center_y - r, center_x + r, center_y + r, outline='gray')

        # 绘制方向标记
        for angle in range(0, 360, 45):
            rad = math.radians(angle)
            x = center_x + 220 * math.sin(rad)
            y = center_y - 220 * math.cos(rad)
            canvas.create_line(center_x, center_y, x, y, fill='lightgray')
            canvas.create_text(x, y, text=f"{angle}°")

        # 绘制玫瑰图花瓣
        for dir, length in zip(directions, lengths):
            rad = math.radians(dir)
            scaled_length = 200 * (length / max_length)

            # 计算花瓣顶点坐标
            x1 = center_x + scaled_length * math.sin(rad)
            y1 = center_y - scaled_length * math.cos(rad)

            # 绘制花瓣
            canvas.create_line(center_x, center_y, x1, y1, fill='red', width=6)
            # 添加标题
        canvas.create_text(center_x, 10, text="裂隙方向分布玫瑰图", font=('Arial', 12, 'bold'))

    def export_data(self):
        # 导出裂隙数据
        if not self.crack_data:
            messagebox.showerror("错误", "没有可导出的数据")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel文件", "*.xlsx"), ("CSV文件", "*.csv")]
        )

        if file_path:
            try:
                # 重新处理编号
                for i, item in enumerate(self.crack_data):
                    item['编号'] = i
                df = pd.DataFrame(self.crack_data)
                if file_path.endswith('.xlsx'):
                    # 添加engine参数
                    df.to_excel(file_path, index=False, engine='openpyxl')
                else:
                    df.to_csv(file_path, index=False)
                messagebox.showinfo("成功", "数据导出完成")
            except PermissionError:
                messagebox.showerror("错误", "文件被其他程序占用，请关闭后重试")
            except Exception as e:
                messagebox.showerror("导出失败", f"发生未知错误：{str(e)}")
        if file_path:
            self.log_history("数据导出", file_path)

    def show_image(self, image, canvas):
        try:
            canvas.delete("image")
            if image is None:
                return

            # 转换颜色空间
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 获取实际尺寸
            h, w = image_rgb.shape[:2]
            canvas_width = canvas.winfo_width() or 600
            canvas_height = canvas.winfo_height() or 600

            # 计算缩放后的尺寸
            scaled_w = int(w * self.zoom_level)
            scaled_h = int(h * self.zoom_level)

            # 限制偏移量
            self.offset_x = max(0, min(self.offset_x, scaled_w - canvas_width))
            self.offset_y = max(0, min(self.offset_y, scaled_h - canvas_height))

            # 裁剪显示区域
            x1 = int(self.offset_x)
            y1 = int(self.offset_y)
            x2 = min(x1 + canvas_width, scaled_w)
            y2 = min(y1 + canvas_height, scaled_h)

            # 缩放并裁剪图像
            resized = cv2.resize(image_rgb, (scaled_w, scaled_h))
            cropped = resized[y1:y2, x1:x2]

            # 转换为Tkinter格式
            img_pil = Image.fromarray(cropped)
            img_tk = ImageTk.PhotoImage(img_pil)

            # 保持引用
            if not hasattr(canvas, '_image_refs'):
                canvas._image_refs = []
            canvas._image_refs.append(img_tk)

            # 显示图像
            canvas.create_image(0, 0, anchor='nw', image=img_tk, tags="image")
            self.update_scale_text(canvas)  # 更新缩放比例文本

            # 确保图像在底层
            canvas.tag_lower("image")

        except Exception as e:
            print(f"显示错误: {str(e)}")

    def update_scale_text(self, canvas):  # 原图右上角缩放比例

        # 先删除之前的缩放比例文本
        canvas.delete("scale")

        # 创建新的缩放比例文本
        canvas.create_text(500, 50, text=f"{self.zoom_level * 100:.0f}%",
                           fill="red", font=("Arial", 10), tags="scale")

        # 将缩放比例文本提升到顶层
        canvas.tag_raise("scale")

    def show_tooltip(self, canvas, x, y, text):
        """显示工具提示"""
        if self.tooltip_visible:
            self.tooltip.destroy()

        self.tooltip_text.set(text)
        self.tooltip = tk.Toplevel(canvas)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x + canvas.winfo_rootx() + 15}+{y + canvas.winfo_rooty() + 15}")

        label = tk.Label(self.tooltip, textvariable=self.tooltip_text,
                         background="#ffffe0", relief='solid', borderwidth=1,
                         font=('Microsoft YaHei', 10))
        label.pack(ipadx=5, ipady=2)
        self.tooltip_visible = True

    def hide_tooltip(self, event=None):
        """隐藏工具提示"""
        if self.tooltip_visible:
            self.tooltip.destroy()
            self.tooltip_visible = False

    def on_canvas_motion(self, event):
        """处理画布上的鼠标移动事件"""
        if not self.crack_data:
            return

        canvas = event.widget
        x, y = canvas.canvasx(event.x), canvas.canvasy(event.y)

        for item in self.crack_data:
            # 检查鼠标是否在裂隙附近
            if (abs(x - item["起点X"]) < 10 or abs(x - item["终点X"]) < 10 or
                    abs(y - item["起点Y"]) < 10 or abs(y - item["终点Y"]) < 10):
                # 生成提示文本
                text = (f"编号: {item['编号']}\n"
                        f"起点: ({item['起点X']}, {item['起点Y']})\n"
                        f"终点: ({item['终点X']}, {item['终点Y']})\n"
                        f"长度: {item['长度(像素)']:.2f}像素\n"
                        f"方向: {item['正北方向(度)']:.4f}度")
                self.show_tooltip(canvas, event.x, event.y, text)
                return

        # 鼠标不在任何裂隙上时隐藏提示
        self.hide_tooltip()

    def update_result(self):
        # 更新文本显示结果
        self.result_text.delete(1.0, tk.END)
        if self.crack_data:
            headers = ["编号", "起点X", "起点Y", "终点X", "终点Y", "长度(像素)", "正北方向(度)"]
            header_line = "\t".join(headers) + "\n"
            self.result_text.insert(tk.END, header_line)

            for i, item in enumerate(self.crack_data):
                line = f"{i}\t{item['起点X']}\t{item['起点Y']}\t{item['终点X']}\t{item['终点Y']}\t{item['长度(像素)']:.2f}\t{item['正北方向(度)']:.4f}\n"
                self.result_text.insert(tk.END, line)
        else:
            self.result_text.insert(tk.END, "未检测到有效裂隙")

    def zoom_image(self, canvas, factor):
        """缩放图像"""
        # 保持缩放中心点不变
        old_zoom = self.zoom_level
        self.zoom_level *= factor
        self.zoom_level = max(0.1, min(self.zoom_level, 5.0))
        self.update_scale_text(canvas)  # 更新缩放比例文本

        # 调整偏移量保持中心点
        canvas_width = canvas.winfo_width() or 600
        canvas_height = canvas.winfo_height() or 600
        self.offset_x += (canvas_width / 2 * (1 / old_zoom - 1 / self.zoom_level))
        self.offset_y += (canvas_height / 2 * (1 / old_zoom - 1 / self.zoom_level))

        self.show_image(self.original_image if canvas == self.original_canvas else self.processed_image, canvas)

    def start_drag(self, event, canvas):
        """记录拖动起始位置"""
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y
        self.drag_data["canvas"] = canvas

    def on_drag(self, event, canvas):
        """处理拖动事件"""
        if self.original_image is None:
            return

        dx = event.x - self.drag_data["x"]
        dy = event.y - self.drag_data["y"]

        # 计算实际移动距离
        self.offset_x -= dx / self.zoom_level
        self.offset_y -= dy / self.zoom_level

        # 限制最大偏移量
        h, w = self.original_image.shape[:2]
        scaled_w = w * self.zoom_level
        scaled_h = h * self.zoom_level
        canvas_width = canvas.winfo_width() or 600
        canvas_height = canvas.winfo_height() or 600

        self.offset_x = max(0, min(self.offset_x, scaled_w - canvas_width))
        self.offset_y = max(0, min(self.offset_y, scaled_h - canvas_height))

        # 更新显示
        self.show_image(self.original_image if canvas == self.original_canvas else self.processed_image, canvas)

    def on_mousewheel(self, event, canvas):
        """鼠标滚轮缩放"""
        # 滚轮方向判断
        if event.delta > 0:
            self.zoom_image(canvas, 1.2)
        else:
            self.zoom_image(canvas, 0.8)

    def reset_position(self, canvas):
        """重置图像位置和缩放"""
        self.zoom_level = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.show_image(self.original_image if canvas == self.original_canvas else self.processed_image, canvas)


# --------------------主应用程序类，包含完整登录系统功能--------------------

class lxcxLoginApp:
    # 修改原登录系统的登录成功跳转部分
    # 用户配置存储文件
    CONFIG_FILE = "user_config.json"

    def __init__(self):
        # 初始化应用程序
        self.conn = None  # 初始化为 None，延迟数据库连接
        self.c = None
        self.root = tk.Tk()
        self.root.title("图片裂隙自动识别程序V3.0")
        self.root.geometry("400x300")

        # 初始化样式
        self.style = ttk.Style()
        self.style.configure("TButton", padding=6, relief="flat")

        # 用户配置变量
        self.remember_var = tk.BooleanVar()
        self.auto_login_var = tk.BooleanVar()

        self.create_login_widgets()
        self.load_config()  # 加载保存的配置

    def ensure_db_connected(self):
        if self.conn is None:
            self.create_db()

    def create_db(self):
        # 创建或连接数据库（自动更新表结构）
        self.conn = sqlite3.connect('qq_users.db')
        self.c = self.conn.cursor()

        # 先创建users表（如果不存在）
        self.c.execute('''CREATE TABLE IF NOT EXISTS users
                        (id INTEGER PRIMARY KEY AUTOINCREMENT,
                         username TEXT,
                         phone TEXT UNIQUE,
                         password TEXT,
                         hint TEXT,
                         answer_hint TEXT)''')

        # 再创建history表
        self.c.execute('''CREATE TABLE IF NOT EXISTS history
                        (id INTEGER PRIMARY KEY AUTOINCREMENT,
                         user_id INTEGER,
                         action_type TEXT,
                         action_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                         file_path TEXT,
                         details TEXT,
                         FOREIGN KEY(user_id) REFERENCES users(id))''')

        # 检查列是否存在（兼容旧版本）
        self.c.execute("PRAGMA table_info(users)")
        columns = [column[1] for column in self.c.fetchall()]

        # 动态添加缺失列
        for column in ['hint', 'answer_hint']:
            if column not in columns:
                try:
                    self.c.execute(f'ALTER TABLE users ADD COLUMN {column} TEXT')
                except sqlite3.OperationalError as e:
                    print(f"添加字段失败（可能已存在）: {str(e)}")

        self.conn.commit()

    def create_login_widgets(self):
        # 创建登录界面
        # 界面上下左右边距均为40像素
        main_frame = ttk.Frame(self.root)
        main_frame.pack(pady=40, padx=40, fill='both', expand=True)

        # Logo显示
        ttk.Label(main_frame, text="欢迎使用", font=("Arial", 24)).grid(row=0, column=0,
                                                                        columnspan=2, pady=10)

        # 输入区域框架的位置信息
        input_frame = ttk.Frame(main_frame)
        input_frame.grid(row=1, column=0, columnspan=2, pady=10)

        # 账号输入框
        ttk.Label(input_frame, text="账号:").grid(row=0, column=0, sticky='w')
        self.username_entry = ttk.Entry(input_frame)
        self.username_entry.grid(row=0, column=1)

        # 密码输入框
        ttk.Label(input_frame, text="密码:").grid(row=1, column=0, sticky='w')
        self.password_entry = ttk.Entry(input_frame, show="*")
        self.password_entry.grid(row=1, column=1)

        # 功能选项区域
        func_frame = ttk.Frame(main_frame)
        func_frame.grid(row=2, column=0, columnspan=2, pady=10)
        ttk.Checkbutton(func_frame, text="记住密码", variable=self.remember_var).pack(side='left')

        # 按钮位置区域
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=3, column=0, columnspan=2, pady=10)
        ttk.Button(btn_frame, text="登录", command=self.login).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="注册账号", command=self.show_register).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="找回密码", command=self.show_forgot_password).pack(side='left', padx=5)

    # ---------- 配置 ----------
    def save_config(self):
        # 保存用户配置到文件
        config = {
            "remember": self.remember_var.get(),
            "auto_login": self.auto_login_var.get(),
            "username": self.username_entry.get() if self.remember_var.get() else "",
            "password": self.password_entry.get() if self.remember_var.get() else ""
        }
        try:
            with open(self.CONFIG_FILE, 'w') as f:
                json.dump(config, f)
        except Exception as e:
            messagebox.showerror("配置保存失败", str(e))

    def load_config(self):
        # 加载保存的用户配置
        if not os.path.exists(self.CONFIG_FILE):
            return
        try:
            with open(self.CONFIG_FILE, 'r') as f:
                config = json.load(f)
            self.remember_var.set(config["remember"])
            self.auto_login_var.set(config["auto_login"])
            if config["remember"]:
                self.username_entry.insert(0, config["username"])
                self.password_entry.insert(0, config["password"])
        except Exception as e:
            messagebox.showerror("配置加载失败", str(e))

    def show_register(self):
        # 打开注册窗口
        self.ensure_db_connected()  # 确保数据库已连接
        RegisterWindow(self)

    def show_forgot_password(self):
        # 打开找回密码窗口
        self.ensure_db_connected()  # 确保数据库已连接
        ForgotPasswordWindow(self)

    def login(self):
        # 执行登录操作
        self.ensure_db_connected()  # 确保数据库已连接
        username = self.username_entry.get().strip()
        password = self.password_entry.get().strip()

        # 输入验证
        if not username or not password:
            messagebox.showerror("错误", "账号和密码不能为空")
            return

        # 使用MD5加密密码
        hashed_pwd = hashlib.md5(password.encode()).hexdigest()

        try:
            self.c.execute("SELECT * FROM users WHERE (username=? OR phone=?) AND password=?",
                           (username, username, hashed_pwd))
            user = self.c.fetchone()
            if user:
                user_id = user[0]  # 获取用户ID
                self.save_config()
                messagebox.showinfo("登录成功", "欢迎使用图片裂隙智能识别程序，请注意数据保存路径")
                self.root.destroy()
                # 传递用户ID给主程序
                root = tk.Tk()  # 创建标准Tk窗口
                app = MainApplication(root, user_id)  # 传入窗口对象
                root.mainloop()
            else:
                messagebox.showerror("登录失败", "账号或密码错误")
        except Exception as e:
            messagebox.showerror("数据库错误", str(e))


# -------------------注册窗口类-------------------

class RegisterWindow:
    # 新增提示词功能

    def __init__(self, master):
        # 初始化注册窗口
        self.master = master
        self.window = tk.Toplevel()
        self.window.title("注册账号")
        self.window.geometry("350x350")  # 调整窗口大小
        self.create_widgets()

    @staticmethod
    def check_password_strength(password):
        # 密码强度检测函数
        # 返回强度等级和提示信息

        # 长度检测（建议至少8位）
        if len(password) < 8:
            return "弱", "密码强度较弱，密码至少需要8个字符"

        # 字符类型检测（大写字母、小写字母、数字、特殊字符）
        has_upper = re.search(r'[A-Z]', password)
        has_lower = re.search(r'[a-z]', password)
        has_digit = re.search(r'\d', password)
        has_special = re.search(r'[!@#$%^&*(),.?":{}|<>]', password)

        # 计算满足的条件数量
        conditions = sum([bool(has_upper), bool(has_lower), bool(has_digit), bool(has_special)])

        # 强度分级
        if conditions < 2:
            return "弱", "密码强度较弱，需要至少包含字母、数字和特殊字符中的两种"
        elif conditions == 2:
            return "中", "密码强度适中，建议增加字符类型（大写字母、数字、特殊字符）"
        else:
            return "强", "密码强度非常强"

    def create_widgets(self):
        # 创建注册界面（新增提示词输入）
        main_frame = ttk.Frame(self.window)
        main_frame.pack(pady=20, padx=20, fill='both', expand=True)

        # 手机号输入区域
        phone_frame = ttk.Frame(main_frame)
        phone_frame.pack(pady=5, fill='y')
        ttk.Label(phone_frame, text="+86").pack(side='left')
        self.phone_entry = ttk.Entry(phone_frame)
        self.phone_entry.pack(side='left', padx=10)

        # 密码设置区域
        ttk.Label(main_frame, text="密码（6 - 16位）").pack(pady=5)
        self.password_entry = ttk.Entry(main_frame, show="*")
        self.password_entry.pack()

        # 确认密码区域
        ttk.Label(main_frame, text="确认密码").pack(pady=5)
        self.confirm_entry = ttk.Entry(main_frame, show="*")
        self.confirm_entry.pack()

        # 新增提示词输入
        hint_label = ttk.Label(main_frame, text="安全提示词")
        hint_label.pack(pady=5)
        self.hint_entry = ttk.Entry(main_frame)
        self.hint_entry.pack()

        # 新增请回答提示词输入
        answer_hint_label = ttk.Label(main_frame, text="请回答提示词")
        answer_hint_label.pack(pady=5)
        self.answer_hint_entry = ttk.Entry(main_frame)
        self.answer_hint_entry.pack()

        # 注册按钮
        ttk.Button(main_frame, text="立即注册", command=self.register).pack(pady=15)

    def register(self):

        # 执行注册操作（新增提示词验证）
        phone = self.phone_entry.get().strip()
        password = self.password_entry.get().strip()
        confirm = self.confirm_entry.get().strip()
        hint = self.hint_entry.get().strip()
        answer_hint = self.answer_hint_entry.get().strip()

        # 基础验证（新增提示词验证）
        if not all([phone, password, confirm, hint, answer_hint]):
            messagebox.showerror("错误", "请填写所有字段")
            return

        # 长度验证
        if len(password) < 6 or len(password) > 16:
            messagebox.showerror("错误", "密码长度需为6 - 16位")
            return

        # 一致性验证
        if password != confirm:
            messagebox.showerror("错误", "两次密码输入不一致")
            return

        # 手机号格式验证
        if not phone.isdigit() or len(phone) != 11:
            messagebox.showerror("错误", "请输入有效手机号")
            return

        # 密码强度检测
        strength, msg = self.check_password_strength(password)
        if strength in ["弱", "中"]:
            # 允许用户选择是否继续使用
            if not messagebox.askyesno("密码强度不足", f"{msg}\n是否继续使用该密码？"):
                return

        # 密码加密存储
        hashed_pwd = hashlib.md5(password.encode()).hexdigest()

        try:
            # 插入提示词和回答到数据库，确保列名和参数数量一致
            self.master.c.execute("INSERT INTO users (phone, password, hint, answer_hint) VALUES (?, ?, ?, ?)",
                                  (phone, hashed_pwd, hint, answer_hint))
            self.master.conn.commit()
            messagebox.showinfo("成功", "注册成功！")
            self.window.destroy()
        except sqlite3.IntegrityError:
            messagebox.showerror("错误", "该手机号已注册")
        except Exception as e:
            messagebox.showerror("错误", str(e))


# --------------------找回密码窗口类--------------------
class ForgotPasswordWindow:
    # 新增提示词验证

    def __init__(self, master):
        self.master = master
        self.window = tk.Toplevel()
        self.window.title("找回密码")
        self.window.geometry("350x300")  # 调整窗口大小
        self.create_widgets()

    def create_widgets(self):
        # 创建找回密码界面（新增提示词输入）
        main_frame = ttk.Frame(self.window)
        main_frame.pack(pady=10, padx=20, fill='both', expand=True)

        # 手机号输入
        ttk.Label(main_frame, text="手机号（输入后请回车）").pack(pady=2)
        self.phone_entry = ttk.Entry(main_frame)
        self.phone_entry.pack()
        # 绑定回车键事件，触发 show_hint 方法
        self.phone_entry.bind("<Return>", self.show_hint)

        # 显示安全提示词
        self.hint_label = ttk.Label(main_frame, text="安全提示词: ")
        self.hint_label.pack(pady=2)

        # 请回答提示词输入
        answer_hint_label = ttk.Label(main_frame, text="请回答提示词")
        answer_hint_label.pack(pady=2)
        self.answer_hint_entry = ttk.Entry(main_frame)
        self.answer_hint_entry.pack()

        # 新密码输入
        ttk.Label(main_frame, text="新密码").pack(pady=2)
        self.new_password_entry = ttk.Entry(main_frame, show="*")
        self.new_password_entry.pack()

        # 确认密码
        ttk.Label(main_frame, text="确认密码").pack(pady=2)
        self.confirm_entry = ttk.Entry(main_frame, show="*")
        self.confirm_entry.pack()

        # 重置按钮
        ttk.Button(main_frame, text="重置密码", command=self.reset_password).pack(pady=5)

    def show_hint(self, event=None):
        phone = self.phone_entry.get().strip()
        if phone:
            try:
                self.master.c.execute("SELECT hint FROM users WHERE phone=?", (phone,))
                result = self.master.c.fetchone()
                if result:
                    stored_hint = result[0]
                    self.hint_label.config(text=f"安全提示词: {stored_hint}")
                else:
                    self.hint_label.config(text="安全提示词: 该手机号未注册")
            except Exception as e:
                messagebox.showerror("错误", f"查询提示词失败: {str(e)}")

    def reset_password(self):
        # 执行密码重置（新增提示词验证）
        phone = self.phone_entry.get().strip()
        answer_hint = self.answer_hint_entry.get().strip()
        new_pwd = self.new_password_entry.get().strip()
        confirm = self.confirm_entry.get().strip()
        answer_hint = self.answer_hint_entry.get().strip()

        # 输入验证（新增提示词验证）
        if not all([phone, new_pwd, confirm, answer_hint]):
            messagebox.showerror("错误", "请填写所有字段")
            return

        # 验证手机号和获取提示词及回答
        self.master.c.execute("SELECT id, hint, answer_hint FROM users WHERE phone=?", (phone,))
        result = self.master.c.fetchone()
        if not result:
            messagebox.showerror("错误", "该手机号未注册")
            return
        stored_user_id, stored_hint, stored_answer_hint = result

        # 显示安全提示词
        self.hint_label.config(text=f"安全提示词: {stored_hint}")

        if stored_answer_hint != answer_hint:
            messagebox.showerror("错误", "提示词回答不正确")
            return

        # 密码强度检测（复用注册窗口的检测方法）
        strength, msg = RegisterWindow.check_password_strength(new_pwd)
        if strength in ["弱", "中"]:
            if not messagebox.askyesno("密码强度不足", f"{msg}\n是否继续使用该密码？"):
                return

        # 长度验证
        if len(new_pwd) < 6 or len(new_pwd) > 16:
            messagebox.showerror("错误", "密码长度需为6 - 16位")
            return

        # 一致性验证
        if new_pwd != confirm:
            messagebox.showerror("错误", "两次密码输入不一致")
            return

        # 手机号格式验证
        if not phone.isdigit() or len(phone) != 11:
            messagebox.showerror("错误", "请输入有效手机号")
            return

        # 更新密码操作
        hashed_pwd = hashlib.md5(new_pwd.encode()).hexdigest()
        try:
            self.master.c.execute("UPDATE users SET password=? WHERE id=?",
                                  (hashed_pwd, stored_user_id))
            self.master.conn.commit()
            messagebox.showinfo("成功", "密码重置成功！")
            self.window.destroy()
        except Exception as e:
            messagebox.showerror("错误", f"密码重置失败: {str(e)}")


if __name__ == "__main__":
    app = lxcxLoginApp()
    app.root.mainloop()