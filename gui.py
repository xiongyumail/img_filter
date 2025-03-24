import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import argparse
import threading
from main import main

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Detector Config")
        self.last_error = None  # 用于存储错误信息
        self.setup_ui()
        # 绑定事件处理
        self.root.bind("<<Success>>", self.on_success)
        self.root.bind("<<Error>>", self.on_error)

    def setup_ui(self):
        self.create_input_section()
        self.create_model_section()
        self.create_processing_section()
        self.create_output_section()
        self.create_actions()

    def create_input_section(self):
        frame = ttk.LabelFrame(self.root, text="输入配置")
        frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")

        # 输入方式选择
        self.input_method = tk.StringVar(value="folder")
        ttk.Radiobutton(frame, text="JSON文件", variable=self.input_method, value="json", command=self.toggle_input).grid(row=0, column=0)
        ttk.Radiobutton(frame, text="文件夹", variable=self.input_method, value="folder", command=self.toggle_input).grid(row=0, column=1)

        # JSON文件输入
        self.json_frame = ttk.Frame(frame)
        self.json_path = tk.StringVar()
        ttk.Entry(self.json_frame, textvariable=self.json_path, width=40).grid(row=0, column=0, padx=5)
        ttk.Button(self.json_frame, text="浏览", command=lambda: self.json_path.set(filedialog.askopenfilename())).grid(row=0, column=1)

        # 文件夹输入
        self.folder_frame = ttk.Frame(frame)
        self.folders = tk.Listbox(self.folder_frame, height=4)
        self.folders.grid(row=0, column=0, columnspan=2)
        ttk.Button(self.folder_frame, text="添加", command=self.add_folder).grid(row=1, column=0)
        ttk.Button(self.folder_frame, text="删除", command=lambda: self.folders.delete(tk.ANCHOR)).grid(row=1, column=1)

        self.toggle_input()

    def toggle_input(self):
        if self.input_method.get() == "json":
            self.json_frame.grid(row=1, column=0, pady=5)
            self.folder_frame.grid_forget()
        else:
            self.json_frame.grid_forget()
            self.folder_frame.grid(row=1, column=0, pady=5)

    def add_folder(self):
        if folder := filedialog.askdirectory():
            self.folders.insert(tk.END, folder)

    def create_model_section(self):
        frame = ttk.LabelFrame(self.root, text="模型配置")
        frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        ttk.Label(frame, text="YOLO模型路径:").grid(row=0, column=0)
        self.yolo_path = tk.StringVar(value="models/yoloface_8n.onnx")
        ttk.Entry(frame, textvariable=self.yolo_path, width=40).grid(row=0, column=1)
        ttk.Button(frame, text="浏览", command=lambda: self.yolo_path.set(filedialog.askopenfilename())).grid(row=0, column=2)

        ttk.Label(frame, text="FAN模型路径:").grid(row=1, column=0)
        self.fan_path = tk.StringVar(value="models/2dfan4.onnx")
        ttk.Entry(frame, textvariable=self.fan_path, width=40).grid(row=1, column=1)
        ttk.Button(frame, text="浏览", command=lambda: self.fan_path.set(filedialog.askopenfilename())).grid(row=1, column=2)

        # ONNX Provider配置
        ttk.Label(frame, text="ONNX Provider:").grid(row=2, column=0)
        self.provider = ttk.Combobox(frame, 
                                values=[
                                    "CPUExecutionProvider",
                                    "CUDAExecutionProvider",
                                    "OpenVINOExecutionProvider"
                                ], 
                                state="readonly")
        self.provider.set("OpenVINOExecutionProvider")
        self.provider.grid(row=2, column=1)
        self.provider.bind("<<ComboboxSelected>>", self.update_device_options)

        # 设备类型配置
        ttk.Label(frame, text="目标设备:").grid(row=3, column=0)
        self.device = ttk.Combobox(frame, state="readonly")
        self.device.grid(row=3, column=1)
        self.update_device_options()  # 初始化设备选项

    def update_device_options(self, event=None):
        """动态更新设备类型选项"""
        provider = self.provider.get()
        if provider == "CPUExecutionProvider":
            options = ["CPU"]
        elif provider == "CUDAExecutionProvider":
            options = ["CUDA", "CPU"]
        elif provider == "OpenVINOExecutionProvider":
            options = ["GPU", "CPU"]  # OpenVINO支持的设备类型
        
        current_val = self.device.get()
        self.device["values"] = options
        if current_val not in options:
            self.device.set(options[0])  # 自动重置为第一个有效选项

    def create_processing_section(self):
        frame = ttk.LabelFrame(self.root, text="处理参数")
        frame.grid(row=2, column=0, padx=10, pady=5, sticky="ew")

        ttk.Label(frame, text="YOLO尺寸:").grid(row=0, column=0)
        self.yolo_size = tk.StringVar(value="640x640")
        ttk.Entry(frame, textvariable=self.yolo_size, width=15).grid(row=0, column=1)

        ttk.Label(frame, text="2DFAN尺寸:").grid(row=1, column=0)
        self.fan_size = tk.StringVar(value="256x256")
        ttk.Entry(frame, textvariable=self.fan_size, width=15).grid(row=1, column=1)

        ttk.Label(frame, text="检测阈值:").grid(row=2, column=0)
        self.det_thresh = tk.DoubleVar(value=0.8)
        ttk.Spinbox(frame, from_=0, to=1, increment=0.1, textvariable=self.det_thresh, width=5).grid(row=2, column=1)

        ttk.Label(frame, text="关键点阈值:").grid(row=3, column=0)
        self.landmark_thresh = tk.DoubleVar(value=0.95)
        ttk.Spinbox(frame, from_=0, to=1, increment=0.05, textvariable=self.landmark_thresh, width=5).grid(row=3, column=1)

    def create_output_section(self):
        frame = ttk.LabelFrame(self.root, text="输出选项")
        frame.grid(row=3, column=0, padx=10, pady=5, sticky="ew")

        # 输出路径配置
        ttk.Label(frame, text="输出路径:").grid(row=0, column=0)
        self.output_path = tk.StringVar(value="face.json")
        self.output_entry = ttk.Entry(frame, textvariable=self.output_path, width=40)
        self.output_entry.grid(row=0, column=1)
        ttk.Button(frame, text="浏览", command=lambda: self.output_path.set(filedialog.asksaveasfilename())).grid(row=0, column=2)

        # 完整数据选项
        self.full_data = tk.BooleanVar()
        ttk.Checkbutton(frame, text="输出完整数据", variable=self.full_data).grid(row=1, column=0)

        # 删除选项
        self.delete_imgs = tk.BooleanVar()
        ttk.Checkbutton(frame, text="删除低分图片", variable=self.delete_imgs).grid(row=1, column=1)

        # 复制选项（关键修改部分）
        self.copy_imgs = tk.BooleanVar()
        ttk.Checkbutton(frame, text="复制图片到:", variable=self.copy_imgs).grid(row=2, column=0)
        
        self.copy_path = tk.StringVar(value="./copied_images")
        self.copy_path_entry = ttk.Entry(frame, textvariable=self.copy_path, state=tk.DISABLED, width=30)
        self.copy_path_entry.grid(row=2, column=1)
        
        self.copy_browse_btn = ttk.Button(
            frame, 
            text="浏览", 
            command=lambda: self.copy_path.set(filedialog.askdirectory()),
            state=tk.DISABLED
        )
        self.copy_browse_btn.grid(row=2, column=2)

        # 绑定状态变更（修改跟踪方法）
        self.copy_imgs.trace_add("write", self.toggle_copy)

    def toggle_copy(self, *args):
        """动态切换复制路径控件的可用状态"""
        if self.copy_imgs.get():
            new_state = tk.NORMAL
        else:
            new_state = tk.DISABLED
        
        # 同时更新输入框和浏览按钮
        self.copy_path_entry.config(state=new_state)
        self.copy_browse_btn.config(state=new_state)

    def create_actions(self):
        frame = ttk.Frame(self.root)
        frame.grid(row=4, column=0, pady=10)

        ttk.Button(frame, text="运行", command=self.run).grid(row=0, column=0, padx=5)
        ttk.Button(frame, text="退出", command=self.root.quit).grid(row=0, column=1, padx=5)

    def run(self):
        # 输入验证
        if self.input_method.get() == "json":
            if not self.json_path.get().strip():
                messagebox.showwarning("错误", "请选择JSON文件。")
                return
            if not os.path.exists(self.json_path.get()):
                messagebox.showwarning("错误", "指定的JSON文件不存在。")
                return
        else:
            if not self.folders.get(0, tk.END):
                messagebox.showwarning("错误", "请至少添加一个文件夹。")
                return

        # 验证模型路径是否存在
        if not os.path.exists(self.yolo_path.get()):
            messagebox.showwarning("错误", "YOLO模型路径不存在。")
            return
        if not os.path.exists(self.fan_path.get()):
            messagebox.showwarning("错误", "FAN模型路径不存在。")
            return
        args_dict = {
            "input_json": self.json_path.get() if self.input_method.get() == "json" else None,
            "folder_path": list(self.folders.get(0, tk.END)),
            "size_yoloface": self.yolo_size.get(),
            "size_2dfan4": self.fan_size.get(),
            "face_detector_score": self.det_thresh.get(),
            "output_json": self.output_path.get(),
            "output_full_data": self.full_data.get(),
            "onnx_provider": self.provider.get(),
            "device_type": self.device.get(),
            "onnx_model_path_yoloface": self.yolo_path.get(),
            "onnx_model_path_2dfan4": self.fan_path.get(),
            "delete": self.delete_imgs.get(),
            "copy": self.copy_path.get() if self.copy_imgs.get() else None,
            "landmark_score": self.landmark_thresh.get(),
            "display": False  # 可根据需要添加显示选项
        }

        try:
            args = argparse.Namespace(**args_dict)
            threading.Thread(target=self.execute, args=(args,), daemon=True).start()
        except Exception as e:
            messagebox.showerror("错误", str(e))

    def execute(self, args):
        try:
            main(args)
            self.root.event_generate("<<Success>>")
        except Exception as e:
            self.last_error = str(e)
            self.root.event_generate("<<Error>>")  # 注意这里不再传递参数

    def on_success(self, event):
        messagebox.showinfo("成功", "处理完成！")

    def on_error(self, event):
        if self.last_error:
            messagebox.showerror("错误", self.last_error)
            self.last_error = None  # 清除错误信息