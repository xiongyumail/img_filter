import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import argparse
import threading
from main import main
from version import __version__

class App:
    def __init__(self, root):
        self.root = root
        self.root.title(f"IMG FLITER - 版本: {__version__}")
        self.last_error = None
        self.setup_ui()
        self.root.bind("<<Success>>", self.on_success)
        self.root.bind("<<Error>>", self.on_error)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.running = False  # 跟踪任务运行状态

    def setup_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.create_input_section()
        self.create_model_section()
        self.create_device_section() 
        self.create_processing_section()
        self.create_output_section()
        self.create_actions()
        self.create_status_bar()

    def create_status_bar(self):
        self.status_var = tk.StringVar(value="准备就绪")
        status_bar = ttk.Label(
            self.root, 
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        status_bar.grid(row=6, column=0, sticky="ew", padx=10, pady=5)

    def create_path_selector(self, parent, label, var, default_path, row, is_file=True):
        """优化后的路径选择组件"""
        container = ttk.Frame(parent)
        container.grid(row=row, column=0, columnspan=2, sticky="nsew", padx=5, pady=3)
        container.columnconfigure(1, weight=1)

        # 标签（右对齐）
        ttk.Label(container, text=label).grid(
            row=0, column=0, sticky="e", padx=(5,10), pady=2)
        
        # 输入框（自适应宽度）
        entry = ttk.Entry(container, textvariable=var)
        entry.grid(row=0, column=1, sticky="ew", padx=(0,5), pady=2)
        
        # 浏览按钮（固定宽度）
        browse_text = "浏览文件" if is_file else "浏览目录"
        browse_command = lambda: var.set(
            filedialog.askopenfilename(initialdir=os.path.dirname(var.get())) if is_file 
            else filedialog.askdirectory(initialdir=var.get())
        )
        ttk.Button(container, text=browse_text, command=browse_command, width=8).grid(
            row=0, column=2, padx=(0,5), pady=2)
        
        var.set(os.path.abspath(default_path))
        return entry

    def create_input_section(self):
        """优化后的输入配置区块"""
        container = ttk.LabelFrame(self.root, text="输入配置")
        container.grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")
        
        # 配置列比例（操作列固定宽度）
        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, minsize=120)  # 操作按钮列
        
        # 输入方式切换
        self.input_method = tk.StringVar(value="folder")
        switch_frame = ttk.Frame(container)
        switch_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=3)
        
        ttk.Radiobutton(switch_frame, text="JSON文件", variable=self.input_method, 
                        value="json", command=self.toggle_input).pack(side=tk.LEFT, padx=(10,5))
        ttk.Radiobutton(switch_frame, text="文件夹", variable=self.input_method, 
                        value="folder", command=self.toggle_input).pack(side=tk.LEFT, padx=5)

        # JSON输入区块
        self.json_frame = ttk.Frame(container)
        self.json_frame.columnconfigure(1, weight=1)
        self.json_path = tk.StringVar()
        self.create_path_selector(self.json_frame, "JSON路径:", self.json_path, 
                                "input.json", 0, is_file=True)
        
        # 文件夹输入区块
        self.folder_frame = ttk.Frame(container)
        list_btn_frame = ttk.Frame(self.folder_frame)
        
        # 文件夹列表（自适应高度）
        self.folders = tk.Listbox(list_btn_frame, height=4, selectmode=tk.EXTENDED)
        self.folders.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,5))
        
        # 操作按钮列
        btn_frame = ttk.Frame(list_btn_frame)
        ttk.Button(btn_frame, text="添加", command=self.add_folder, width=6).pack(pady=2)
        ttk.Button(btn_frame, text="删除", command=self.remove_folders, width=6).pack(pady=2)
        btn_frame.pack(side=tk.LEFT)
        list_btn_frame.pack(fill=tk.BOTH, expand=True)

        self.toggle_input()

    def toggle_input(self):
        """优化后的切换逻辑"""
        for widget in [self.json_frame, self.folder_frame]:
            widget.grid_forget()
        
        if self.input_method.get() == "json":
            self.json_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=5)
            self.root.grid_rowconfigure(1, weight=1)
        else:
            self.folder_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=5)
            self.root.grid_rowconfigure(1, weight=1)

    def add_folder(self):
        """添加多个文件夹"""
        folders = filedialog.askdirectory(mustexist=True, title="选择输入文件夹")
        if folders:
            if not isinstance(folders, (list, tuple)):  # 单目录选择返回字符串
                folders = [folders]
            for folder in folders:
                if folder not in self.folders.get(0, tk.END):
                    self.folders.insert(tk.END, folder)

    def remove_folders(self):
        """删除选中的文件夹"""
        for i in reversed(self.folders.curselection()):
            self.folders.delete(i)

    def create_model_section(self):
        """模型路径配置区块"""
        frame = ttk.LabelFrame(self.root, text="模型配置")
        frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        frame.columnconfigure(1, weight=1)

        # YOLO模型路径
        self.yolo_path = tk.StringVar()
        self.create_path_selector(frame, "YOLO模型路径:", self.yolo_path, 
                                "models/yoloface_8n.onnx", 0)
        
        # FAN模型路径
        self.fan_path = tk.StringVar()
        self.create_path_selector(frame, "2DFAN模型路径:", self.fan_path, 
                                "models/2dfan4.onnx", 1)

    def create_device_section(self):
        """设备配置区块（主窗口级左右分栏版本）"""
        # 主容器配置
        container = ttk.Frame(self.root)
        container.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")
        
        # 配置主容器列比例（左侧40% | 间隔 | 右侧60%）
        container.columnconfigure(0, weight=4)
        container.columnconfigure(1, minsize=10)  # 间隔列
        container.columnconfigure(2, weight=5)

        # 左侧加速引擎区块
        left_frame = ttk.LabelFrame(container, text="加速引擎配置")
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0,5))
        left_frame.columnconfigure(1, weight=1)
        
        ttk.Label(left_frame, text="加速引擎:").grid(
            row=0, column=0, sticky="e", padx=(10,5), pady=3)
        
        self.provider = ttk.Combobox(
            left_frame,
            values=["CPUExecutionProvider", "CUDAExecutionProvider", "OpenVINOExecutionProvider"],
            state="readonly"
        )
        self.provider.set("OpenVINOExecutionProvider")
        self.provider.grid(row=0, column=1, sticky="ew", padx=5)
        self.provider.bind("<<ComboboxSelected>>", self.update_device_options)

        # 右侧运算设备区块
        right_frame = ttk.LabelFrame(container, text="运算设备配置") 
        right_frame.grid(row=0, column=2, sticky="nsew", padx=(5,0))
        right_frame.columnconfigure(1, weight=1)
        
        ttk.Label(right_frame, text="运算设备:").grid(
            row=0, column=0, sticky="e", padx=(10,5), pady=3)
        
        self.device = ttk.Combobox(
            right_frame,
            state="readonly"
        )
        self.device.grid(row=0, column=1, sticky="ew", padx=5)
        self.update_device_options()

    def update_device_options(self, event=None):
        """更新设备选项的逻辑保持不变"""
        provider = self.provider.get()
        options = []
        if provider == "CPUExecutionProvider":
            options = ["CPU"]
        elif provider == "CUDAExecutionProvider":
            options = ["CUDA", "CPU"]
        elif provider == "OpenVINOExecutionProvider":
            options = ["GPU", "CPU"]
        
        current = self.device.get()
        self.device["values"] = options
        if current not in options:
            self.device.set(options[0])

    def create_processing_section(self):
        """处理参数区块（采用与设备配置相同的左右分栏布局）"""
        # 主容器配置
        container = ttk.Frame(self.root)
        container.grid(row=3, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")
        
        # 配置主容器列比例（左侧40% | 间隔 | 右侧60%）
        container.columnconfigure(0, weight=4)
        container.columnconfigure(1, minsize=10)  # 间隔列
        container.columnconfigure(2, weight=6)

        # 左侧尺寸参数区块
        left_frame = ttk.LabelFrame(container, text="尺寸设置")
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0,5))
        left_frame.columnconfigure(1, weight=1)  # 输入框列自适应
        
        # YOLO尺寸控件
        ttk.Label(left_frame, text="YOLO尺寸:").grid(
            row=0, column=0, sticky="e", padx=(10,5), pady=3)
        self.yolo_size = tk.StringVar(value="640x640")
        self.create_validated_entry(left_frame, self.yolo_size, 0, "640x640").grid(
            row=0, column=1, sticky="ew", padx=(0,5), pady=3)
        
        # 2DFAN尺寸控件
        ttk.Label(left_frame, text="2DFAN尺寸:").grid(
            row=1, column=0, sticky="e", padx=(10,5), pady=3)
        self.fan_size = tk.StringVar(value="256x256")
        self.create_validated_entry(left_frame, self.fan_size, 1, "256x256").grid(
            row=1, column=1, sticky="ew", padx=(0,5), pady=3)

        # 右侧阈值参数区块
        right_frame = ttk.LabelFrame(container, text="阈值设置")
        right_frame.grid(row=0, column=2, sticky="nsew", padx=(5,0))
        right_frame.columnconfigure(1, weight=1)  # 输入框列自适应
        
        # 检测阈值控件
        ttk.Label(right_frame, text="检测阈值:").grid(
            row=0, column=0, sticky="e", padx=(10,5), pady=3)
        self.det_thresh = tk.DoubleVar(value=0.8)
        ttk.Spinbox(
            right_frame, 
            from_=0, 
            to=1, 
            increment=0.1,
            textvariable=self.det_thresh,
            width=8
        ).grid(row=0, column=1, sticky="ew", padx=(0,5), pady=3)
        
        # 关键点阈值控件
        ttk.Label(right_frame, text="关键点阈值:").grid(
            row=1, column=0, sticky="e", padx=(10,5), pady=3)
        self.landmark_thresh = tk.DoubleVar(value=0.95)
        ttk.Spinbox(
            right_frame,
            from_=0, 
            to=1, 
            increment=0.05,
            textvariable=self.landmark_thresh,
            width=8
        ).grid(row=1, column=1, sticky="ew", padx=(0,5), pady=3)

    def create_validated_entry(self, parent, var, row, default):
        """创建带格式验证的输入框"""
        validate_cmd = (self.root.register(self.validate_resolution), "%P")
        entry = ttk.Entry(parent, textvariable=var, width=15,
                         validate="key", validatecommand=validate_cmd)
        entry.grid(row=row, column=1, sticky="w", padx=5)
        var.set(default)
        return entry

    def validate_resolution(self, value):
        """验证分辨率格式"""
        if value == "":
            return True
        try:
            w, h = map(int, value.split("x"))
            return w > 0 and h > 0
        except:
            return False

    def create_output_section(self):
        frame = ttk.LabelFrame(self.root, text="输出选项")
        frame.grid(row=4, column=0, padx=10, pady=5, sticky="ew")
        frame.columnconfigure(1, weight=1)

        # 输出路径
        self.output_path = tk.StringVar(value=os.path.abspath("face.json"))
        self.create_path_selector(frame, "输出路径:", self.output_path, 
                                 "face.json", 0, is_file=True)
        
        # 选项按钮
        options_frame = ttk.Frame(frame)
        options_frame.grid(row=1, column=0, columnspan=3, pady=5, sticky="ew")
        
        self.full_data = tk.BooleanVar()
        ttk.Checkbutton(options_frame, text="输出完整数据", variable=self.full_data).grid(row=0, column=0, padx=10)
        
        self.delete_imgs = tk.BooleanVar()
        ttk.Checkbutton(options_frame, text="删除低分图片", variable=self.delete_imgs).grid(row=0, column=1, padx=10)
        
        # 复制选项
        self.copy_imgs = tk.BooleanVar()
        ttk.Checkbutton(options_frame, text="复制图片到:", variable=self.copy_imgs).grid(row=0, column=2, padx=10)
        
        self.copy_path = tk.StringVar(value=os.path.abspath("./copied_images"))
        self.copy_entry = ttk.Entry(options_frame, textvariable=self.copy_path, 
                                   state=tk.DISABLED, width=30)
        self.copy_entry.grid(row=0, column=3, padx=5)
        
        self.copy_browse_btn = ttk.Button(
            options_frame,
            text="浏览",
            command=lambda: self.copy_path.set(filedialog.askdirectory()),
            state=tk.DISABLED
        )
        self.copy_browse_btn.grid(row=0, column=4)
        self.copy_imgs.trace_add("write", self.toggle_copy)

    def toggle_copy(self, *args):
        """切换复制路径控件的可用状态"""
        state = tk.NORMAL if self.copy_imgs.get() else tk.DISABLED
        self.copy_entry.config(state=state)
        self.copy_browse_btn.config(state=state)

    def create_actions(self):
        frame = ttk.Frame(self.root)
        frame.grid(row=5, column=0, pady=10)
        
        self.run_button = ttk.Button(frame, text="运行", command=self.run)
        self.run_button.grid(row=0, column=0, padx=5)
        
        ttk.Button(frame, text="退出", command=self.on_close).grid(row=0, column=1, padx=5)

    def validate_inputs(self):
        """执行输入验证"""
        # 检查输入源
        if self.input_method.get() == "json":
            json_path = self.json_path.get()
            if not json_path:
                return "请选择JSON输入文件"
            if not os.path.exists(json_path):
                return f"JSON文件不存在: {json_path}"
        else:
            if not self.folders.size():
                return "请至少添加一个输入文件夹"
        
        # 检查模型路径
        for name, path in [("YOLO", self.yolo_path.get()), ("FAN", self.fan_path.get())]:
            if not os.path.exists(path):
                return f"{name}模型路径不存在: {path}"
        
        # 检查分辨率格式
        for res, name in [(self.yolo_size.get(), "YOLO尺寸"), 
                         (self.fan_size.get(), "FAN尺寸")]:
            if not self.validate_resolution(res):
                return f"{name}格式无效，应为 宽x高"
        
        # 检查输出路径可写
        try:
            with open(self.output_path.get(), 'w') as f:
                pass
            os.remove(self.output_path.get())
        except Exception as e:
            return f"输出路径不可写: {str(e)}"
        
        return None

    def run(self):
        if self.running:
            messagebox.showwarning("警告", "当前已有任务正在运行")
            return
        
        # 输入验证
        if error := self.validate_inputs():
            messagebox.showerror("输入错误", error)
            return
        
        # 禁用控件
        self.running = True
        self.run_button.config(state=tk.DISABLED)
        self.status_var.set("正在处理...")
        
        # 准备参数
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
            "display": False
        }
        
        try:
            args = argparse.Namespace(**args_dict)
            threading.Thread(target=self.execute, args=(args,), daemon=True).start()
        except Exception as e:
            self.handle_error(f"参数错误: {str(e)}")

    def execute(self, args):
        try:
            main(args)
            self.root.event_generate("<<Success>>")
        except Exception as e:
            self.last_error = str(e)
            self.root.event_generate("<<Error>>")

    def on_success(self, event):
        self.running = False
        self.run_button.config(state=tk.NORMAL)
        self.status_var.set("处理完成")
        messagebox.showinfo("成功", "处理完成！")

    def on_error(self, event):
        self.running = False
        self.run_button.config(state=tk.NORMAL)
        self.status_var.set("处理出错")
        if self.last_error:
            messagebox.showerror("错误", self.last_error)
            self.last_error = None

    def on_close(self):
        if self.running:
            if messagebox.askokcancel("退出", "当前有任务正在运行，确定要退出吗？"):
                self.root.destroy()
        else:
            self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()