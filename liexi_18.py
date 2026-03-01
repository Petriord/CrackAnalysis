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
# 版本：V1.0
# 2025年2月9日17:46:24

import datetime
import hashlib #加密密码
import json #json格式对象解码为python对象
import math
import os #系统交互功能
import re #字符串匹配
import sqlite3 #对象与数据库进行交互
import tkinter as tk #GUI图形库
from tkinter import filedialog
from tkinter import ttk, messagebox
import cv2 #进行图像处理
import numpy as np #矩阵操作
import pandas as pd #数据分析和数据处理
from PIL import Image, ImageTk

#-------------------裂隙识别功能--------------------
class MainApplication:
    # 软件主功能界面

    def __init__(self, master, user_id):
        self.user_id = user_id  # 新增用户ID
        self.master = master
        self.master.title("图片裂隙智能分析系统 v3.0")
        self.master.geometry("1250x850")
        self.master.minsize(1000, 700)  # 设置最小窗口尺寸
        self.master.state('zoomed')  # 启动时最大化窗口（Windows）

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
        # 主容器改为左右结构
        main_pane = ttk.PanedWindow(self.master, orient=tk.HORIZONTAL)
        main_pane.pack(fill='both', expand=True)

        # 左侧工作区（使用grid布局）
        left_frame = ttk.Frame(main_pane)
        main_pane.add(left_frame, weight=4)  # 占4/5宽度
        left_frame.grid_rowconfigure(0, weight=1)
        left_frame.grid_columnconfigure(0, weight=1)

        # 右侧历史记录区
        right_frame = ttk.Frame(main_pane)
        main_pane.add(right_frame, weight=1)  # 占1/5宽度
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

        self.processed_canvas = tk.Canvas(img_frame, width=600, height=600, bg='gray')
        self.processed_canvas.grid(row=0, column=1, padx=10, pady=10)

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
            row=0, column=0,pady=5)

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

        # Canny算子边缘检测
        edges = cv2.Canny(binary, 30, 100)

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
        self.crack_data = crack_info_list[:20]

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

    def update_result(self):
        # 更新文本显示结果
        self.result_text.delete(1.0, tk.END)
        if self.crack_data:
            headers = ["编号", "起点X", "起点Y", "终点X", "终点Y", "长度(像素)", "正北方向(度)"]
            header_line = "\t".join(headers) + "\n"
            self.result_text.insert(tk.END, header_line)

            for i, item in enumerate(self.crack_data):
                line = f"{i}\t{item['起点X']}\t{item['起点Y']}\t" \
                       f"{item['终点X']}\t{item['终点Y']}\t{item['长度(像素)']:.2f}\t{
                       item['正北方向(度)']:.4f}\n"
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
        self.root.title("图片裂隙自动识别程序")
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
                root = tk.Tk()
                app = MainApplication(root, user_id)
                root.mainloop()
            else:
                messagebox.showerror("登录失败", "账号或密码错误")
        except Exception as e:
            messagebox.showerror("数据库错误", str(e))

#-------------------注册窗口类-------------------

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

#--------------------找回密码窗口类--------------------
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