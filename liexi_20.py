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
# 版本：V2.0
# 2025年2月9日17:46:24

import datetime
import hashlib  # 加密密码
import json  # json格式对象解码为python对象
import math
import os  # 系统交互功能
import re  # 字符串匹配
import sqlite3  # 对象与数据库进行交互
import tkinter as tk  # GUI图形库
from tkinter import filedialog
from tkinter import ttk, messagebox
from ttkthemes import ThemedTk, ThemedStyle
import cv2  # 进行图像处理
import numpy as np
from sklearn.cluster import DBSCAN  # 密度聚类算法 #矩阵操作
import pandas as pd  # 数据分析和数据处理
from PIL import Image, ImageTk


# -------------------裂隙识别功能--------------------
class MainApplication:
    # 软件主功能界面

    def __init__(self, master, user_id):
        self.user_id = user_id  # 新增用户ID
        self.master = master  # 保持使用传入的窗口对象
        self.style = ThemedStyle(self.master)  # 应用主题到现有窗口
        self.style.set_theme("breeze")  # 设置主题
        self.master.title("图片裂隙智能分析系统 v2.0")
        self.master.geometry("1250x850")
        self.master.minsize(1000, 700)  # 设置最小窗口尺寸
        self.master.state('zoomed')  # 启动时最大化窗口（Windows）)

        # 初始化图像变量
        self.zoom_level = 1.0  # 缩放比例
        self.offset_x = 0  # X轴偏移量
        self.offset_y = 0  # Y轴偏移量
        self.drag_data = {"x": 0, "y": 0, "item": None}  # 拖动状态

        # 初始化图像变量
        self.original_image = None
        self.processed_image = None
        self.crack_data = []

        # 创建界面布局
        self.create_widgets()

        # 初始化OpenCV参数
        self.threshold_value = 127  # 二值化阈值
        self.min_length = 100  # 最小裂隙长度（像素）

        # 初始化 _image_refs 属性
        self.original_canvas._image_refs = []
        self.processed_canvas._image_refs = []

        # 初始化提示框变量
        self.tooltip = None
        self.tooltip_text = tk.StringVar()
        self.tooltip_visible = False

        # 添加状态标签
        self.status_label = ttk.Label(self.master, text="就绪")
        self.status_label.pack(side='bottom', fill='x', padx=10, pady=5)

        # 添加历史记录面板
        self.load_history()  # 加载历史记录

    def create_history_panel(self):
        # 历史记录面板
        history_frame = ttk.Frame(self.master, width=200)
        history_frame.pack(side='right', fill='y', padx=10, pady=10)

        ttk.Label(history_frame, text="操作历史").pack()
        self.history_tree = ttk.Treeview(history_frame, columns=('time',
                                                                 'action'), show='headings', height=25)
        self.history_tree.heading('time', text='时间')
        self.history_tree.heading('action', text='操作')
        self.history_tree.column('time', width=120)
        self.history_tree.column('action', width=200)
        self.history_tree.pack()

    def log_history(self, action_type, file_path="", details=""):
        """记录操作历史"""
        try:
            conn = sqlite3.connect('qq_users.db')
            c = conn.cursor()
            # 获取24小时制本地时间
            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            c.execute('''INSERT INTO history 
                       (user_id, action_type, action_time, file_path, details)
                       VALUES (?,?,?,?,?)''',
                      (self.user_id, action_type, current_time, file_path, details))
            conn.commit()
            conn.close()
            self.load_history()  # 刷新历史记录显示
        except Exception as e:
            print("历史记录保存失败:", str(e))

    def load_history(self):
        """加载历史记录"""
        try:
            conn = sqlite3.connect('qq_users.db')
            c = conn.cursor()
            c.execute('''SELECT action_time, action_type FROM history 
                       WHERE user_id=? ORDER BY action_time DESC LIMIT 50''',
                      (self.user_id,))

            # 清空现有显示
            for item in self.history_tree.get_children():
                self.history_tree.delete(item)

            # 添加新记录，直接使用读取到的时间
            for row in c.fetchall():
                formatted_time = row[0]
                self.history_tree.insert('', 'end', values=(formatted_time, row[1]))

            conn.close()
        except Exception as e:
            print("历史记录加载失败:", str(e))

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
        # 顶部工具栏
        tool_frame = ttk.Frame(parent)
        tool_frame.pack(pady=10, fill='x')

        button_frame = ttk.Frame(tool_frame)
        button_frame.pack()

        ttk.Button(button_frame, text="导入图片", command=self.load_image).grid(row=0,
                                                                                column=0, padx=5, pady=5)
        ttk.Button(button_frame, text="裂隙识别", command=self.analyze_cracks).grid(row=0,
                                                                                    column=1, padx=5, pady=5)
        ttk.Button(button_frame, text="导出数据", command=self.export_data).grid(row=0,
                                                                                 column=2, padx=5, pady=5)
        ttk.Button(button_frame, text="导出图片", command=self.export_image).grid(row=0,
                                                                                  column=3, padx=5, pady=5)
        ttk.Button(button_frame, text="统计分析", command=self.show_statistics).grid(row=0,
                                                                                     column=4, padx=5, pady=5)

        # 创建一个新的框架来包含图像显示区域和结果显示区域
        content_frame = ttk.Frame(parent)
        content_frame.pack(fill='both', expand=True)
        content_frame.grid_rowconfigure(0, weight=1)
        content_frame.grid_rowconfigure(1, weight=1)
        content_frame.grid_columnconfigure(0, weight=1)

        # 图像显示区域（使用grid布局）
        img_frame = ttk.Frame(content_frame)
        img_frame.grid(row=0, column=0, sticky='nsew')

        self.original_canvas = tk.Canvas(img_frame, width=600, height=600, bg='gray')
        self.original_canvas.grid(row=0, column=0, padx=10, pady=10)
        self.original_canvas.bind("<Motion>", self.on_canvas_motion)
        self.original_canvas.bind("<Leave>", self.hide_tooltip)

        self.processed_canvas = tk.Canvas(img_frame, width=600, height=600, bg='gray')
        self.processed_canvas.grid(row=0, column=1, padx=10, pady=10)
        self.processed_canvas.bind("<Motion>", self.on_canvas_motion)
        self.processed_canvas.bind("<Leave>", self.hide_tooltip)

        # 为两个画布添加图像操作控件
        self.add_image_controls(self.original_canvas)
        self.add_image_controls(self.processed_canvas)

        # 结果显示区域
        result_frame = ttk.Frame(content_frame)
        result_frame.grid(row=1, column=0, sticky='nsew')

        self.result_text = tk.Text(result_frame, height=20, width=80)
        self.result_text.pack(pady=10, padx=10)

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

    def delete_selected_history(self):
        selected_items = self.history_tree.selection()
        if not selected_items:
            messagebox.showwarning("提示", "请选择要删除的历史记录")
            return
        try:
            conn = sqlite3.connect('qq_users.db')
            c = conn.cursor()
            for item in selected_items:
                values = self.history_tree.item(item, 'values')
                time = values[0]
                c.execute('DELETE FROM history WHERE user_id =? AND action_time =?',
                          (self.user_id, time))
            conn.commit()
            conn.close()
            # 从界面删除选中项
            for item in selected_items:
                self.history_tree.delete(item)
        except Exception as e:
            messagebox.showerror("错误", f"删除历史记录失败: {str(e)}")

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
                   command=lambda: self.zoom_image(canvas, 1.2)).pack(side='left', padx=2)

        # 将按钮容器添加到画布右上角
        canvas.create_window(570, 10, window=control_frame, anchor='ne')

        # 绑定鼠标事件（新增）
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
        self.status_label.config(text="正在加载图像...")
        self.master.update_idletasks()

    def analyze_cracks(self):
        # 裂隙分析核心算法
        if self.original_image is None:
            messagebox.showerror("错误", "请先导入图片")
            return

        # 创建处理用的副本（保持原始图像不变）
        process_copy = self.original_image.copy()

        # 转换为灰度图
        gray = cv2.cvtColor(process_copy, cv2.COLOR_BGR2GRAY)
        # 使用中值滤波减少噪声
        gray = cv2.medianBlur(gray, 5)

        # 自适应二值化处理，代替固定阈值二值化
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 17, 5)

        # 形态学操作：膨胀操作增强裂隙连续性
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.dilate(binary, kernel, iterations=1)

        # 获取所有非零像素点的坐标
        y_coords, x_coords = np.where(binary == 255)
        points = np.column_stack((x_coords, y_coords))

        # 使用K-means聚类替代DBSCAN
        from sklearn.cluster import KMeans

        # 估计聚类数量（基于图像大小和裂隙密度）
        n_clusters = min(30, max(5, int(len(points) / 100)))

        # 执行K-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(points)
        labels = kmeans.labels_

        # 根据聚类结果生成边缘图像
        edges = np.zeros_like(binary)
        for label in np.unique(labels):
            cluster_points = points[labels == label]
            for pt in cluster_points:
                edges[pt[1], pt[0]] = 255

        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # 分析每个轮廓
        # 定义 crack_info_list为空列表
        crack_info_list = []
        processed_img = self.original_image.copy()

        for idx, cnt in enumerate(contours):
            # 计算轮廓长度
            length = cv2.arcLength(cnt, False)

            # 忽略小线段
            if length < self.min_length:
                continue

            # 使用Douglas - Peucker算法简化轮廓
            epsilon = 0.001 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            # 计算轮廓起点和终点
            start = tuple(approx[0][0])
            end = tuple(approx[-1][0])

            # 修正正北方向计算
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

            # 存储裂隙信息，增加原始轮廓索引
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

        # 选取前20条主要裂隙
        self.crack_data = crack_info_list[:200]

        # 重新编号
        for i, item in enumerate(self.crack_data):
            item['编号'] = i

        # 在图像上绘制主要裂隙
        for item in self.crack_data:
            idx = item["编号"]
            start = (item["起点X"], item["起点Y"])
            end = (item["终点X"], item["终点Y"])

            # 从简化后的轮廓中获取点，使用原始索引
            approx = cv2.approxPolyDP(contours[item["原始索引"]], 0.001 * cv2.arcLength(
                contours[item["原始索引"]], True), True)
            pts = approx.reshape((-1, 1, 2)).astype(np.int32)

            # 绘制裂隙线条，调整线条文字粗细
            cv2.polylines(processed_img, [pts], False, (0, 255, 0), 2)
            cv2.putText(processed_img, str(idx), start,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # 显示处理结果
        self.processed_image = processed_img
        self.show_image(self.original_image, self.original_canvas)
        self.show_image(processed_img, self.processed_canvas)

        # 更新结果显示
        self.update_result()

        if self.crack_data:
            self.log_history("裂隙分析", "", f"识别到{len(self.crack_data)}条裂隙")

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
        if not self.crack_data:
            messagebox.showerror("错误", "请先进行裂隙识别")
            return

        # 创建统计窗口
        stats_window = tk.Toplevel(self.master)
        stats_window.title("裂隙统计分析")
        stats_window.geometry("900x700")

        # 主布局
        main_frame = ttk.Frame(stats_window)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # 左侧选项面板
        option_frame = ttk.Frame(main_frame, width=200)
        option_frame.pack(side='left', fill='y', padx=5, pady=5)

        # 右侧展示区
        display_frame = ttk.Frame(main_frame)
        display_frame.pack(side='right', fill='both', expand=True)

        # 创建画布
        self.stats_canvas = tk.Canvas(display_frame, bg='white')
        self.stats_canvas.pack(fill='both', expand=True, padx=10, pady=10)

        # 添加分析功能按钮
        ttk.Button(option_frame, text="裂隙长度分析",
                   command=lambda: self.show_length_analysis()).pack(fill='x', pady=5)
        ttk.Button(option_frame, text="方向分布分析",
                   command=lambda: self.show_direction_analysis()).pack(fill='x', pady=5)
        ttk.Button(option_frame, text="密度热力图",
                   command=lambda: self.show_density_heatmap()).pack(fill='x', pady=5)

        # 默认显示方向分布分析
        self.show_direction_analysis()

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

            # 确保图像在底层（显示调节）
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
        """缩放图像（增强版）"""
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

        self.show_image(self.original_image, canvas)

    def start_drag(self, event, canvas):
        """记录拖动起始位置"""
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y
        self.drag_data["canvas"] = canvas

    def on_drag(self, event, canvas):
        """处理拖动事件（增强版）"""
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

        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y

        # 根据当前画布显示对应图像
        if canvas == self.original_canvas:
            self.show_image(self.original_image, canvas)
        else:
            self.show_image(self.processed_image, canvas)

    def on_mousewheel(self, event, canvas):
        """鼠标滚轮缩放"""
        if event.delta > 0:
            self.zoom_image(canvas, 1.1)
        else:
            self.zoom_image(canvas, 0.9)

    def reset_position(self, canvas):
        """重置视图"""
        self.zoom_level = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.update_scale_text(canvas)  # 更新缩放比例文本
        if canvas == self.original_canvas:
            self.show_image(self.original_image, canvas)
        else:
            self.show_image(self.processed_image, canvas)


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
        self.root.title("图片裂隙自动识别程序V2.0")
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