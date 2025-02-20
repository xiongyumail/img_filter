# IMG_FACEDETECTOR 
`img_facedetector` 是一款基于Python开发的图像人脸检测工具，它借助ONNX运行时和YOLOFace模型，能精准检测图像中的人脸。该工具支持处理单张图像或整个文件夹内的图像，根据检测结果，可选择删除未检测到人脸的图像，或者将检测到人脸的图像路径输出到JSON文件中。经过实际测试，搭配Intel Arc B580显卡，结合`onnxruntime-openvino`，可实现极为高效的推理性能，大大提升检测效率。

## 功能亮点
- **高效人脸检测**：运用YOLOFace模型与ONNX运行时，快速且精准地检测图像中的人脸，满足各类场景下的检测需求。
- **中文路径支持**：在读取与处理图像过程中，能够完美识别并处理包含中文字符的文件路径，消除因路径问题导致的使用障碍。
- **多线程处理**：通过`concurrent.futures.ThreadPoolExecutor`实现多线程处理机制，显著提升图像检测速度，尤其适用于处理大量图像的情况。
- **灵活配置选项**：用户可通过命令行参数灵活配置人脸检测器的分辨率、检测分数阈值、是否删除非人脸图像以及输出JSON文件的路径等，适应不同的检测任务和需求。
- **Intel GPU支持**：搭配`onnxruntime-openvino`，充分释放Intel显卡的强大计算能力，加速人脸检测进程，大幅提高检测效率。

## 依赖安装
### 通用依赖
使用本工具前，请确保已安装以下依赖库：
- `opencv-python`：用于图像处理和图像读取，为图像操作提供丰富的功能和接口。
- `numpy`：用于数值计算和数组操作，是科学计算的基础库。
- `onnxruntime`：用于运行ONNX模型，实现模型的推理和预测。
- `argparse`：用于命令行参数解析，方便用户根据需求配置工具参数。
- `json`：用于处理JSON数据，实现数据的存储和读取。
- `concurrent.futures`：用于多线程处理，提升图像检测的效率。

可使用`pip`进行安装：
```bash
pip install opencv-python numpy onnxruntime
```

### Intel GPU及onnxruntime-openvino依赖
若要利用Intel显卡进行加速，需安装`onnxruntime-openvino`：
```bash
pip install onnxruntime-openvino
```

同时，务必确保系统已安装Intel显卡驱动程序，以保障显卡正常工作。可前往[Intel官方网站](https://www.intel.com/content/www/us/en/download-center/home.html)下载并安装最新的显卡驱动。

## 使用步骤
1. **下载ONNX模型**：获取`yoloface_8n.onnx`模型文件，并放置在正确路径下。可从模型官方来源获取。
2. **运行命令**：通过命令行运行工具，支持以下参数：
    - `folder_path`：必填参数，指定包含图像的文件夹路径。
    - `--face_detector_size`：可选参数，指定人脸检测器的分辨率，格式为`widthxheight`，默认为`640x640`。
    - `--face_detector_score`：可选参数，指定人脸检测分数阈值，默认为`0.7`。
    - `--delete_non_face`：可选参数，指定是否删除未检测到人脸的图像，默认为不删除。
    - `--output_json`：可选参数，指定输出人脸图片路径的JSON文件路径，默认为`face.json`。
    - `--onnx_provider`：可选参数，指定ONNX运行时的提供者，若使用Intel显卡，建议设置为`OpenVINOExecutionProvider`，默认为`OpenVINOExecutionProvider`。
    - `--device_type`：可选参数，指定提供者的设备类型，若使用Intel显卡，设置为`GPU`，默认为`GPU`。
    - `--onnx_model_path`：可选参数，指定ONNX模型的路径，默认为`yoloface_8n.onnx`。

示例命令（使用Intel显卡）：
```bash
python img_facedetector.py /path/to/images --face_detector_size 640x640 --face_detector_score 0.7 --delete_non_face --output_json face.json --onnx_provider OpenVINOExecutionProvider --device_type GPU --onnx_model_path yoloface_8n.onnx
```
简化命令(默认参数):
```bash
python img_facedetector.py /path/to/images 
```

## 代码架构
1. **核心函数**：
    - `init_onnx_session`：初始化ONNX推理会话，设置全局变量`onnx_session`和`input_name`，为后续模型推理做准备。
    - `unpack_resolution`：将分辨率字符串拆分为宽度和高度，方便参数处理和配置。
    - `resize_frame_resolution`：调整帧的分辨率，使其符合模型输入要求。
    - `prepare_detect_frame`：准备用于检测的帧，包括归一化、转置和扩展维度等操作，确保数据格式正确。
    - `forward_with_yoloface`：使用YOLOFace模型进行前向推理，得出初步检测结果。
    - `detect_with_yoloface`：使用YOLOFace模型进行人脸检测，返回边界框、分数和5点人脸关键点，提供详细的检测信息。
    - `process_single_image`：处理单张图片，判断是否检测到人脸，输出检测结果。
    - `process_images_in_folder`：处理文件夹中的图像，根据选项删除未检测到人脸的图像或输出人脸图片路径JSON文件，实现批量处理功能。
2. **命令行参数解析**：利用`argparse`模块解析命令行参数，根据用户输入的参数灵活配置工具行为，满足不同用户的多样化需求。

## 注意要点
1. **模型兼容性**：确保使用的ONNX模型与代码兼容，模型的输入输出格式与代码中的处理逻辑保持一致，避免因格式不匹配导致错误。
2. **设备配置**：根据硬件设备和实际需求，合理配置`--onnx_provider`和`--device_type`参数，以获取最佳性能。若使用Intel显卡，务必将`--onnx_provider`设置为`OpenVINOExecutionProvider`，`--device_type`设置为`GPU`。
3. **文件路径**：处理文件路径时，确保路径的正确性和可读性，尤其是包含特殊字符或中文的路径，避免因路径错误导致文件读取失败。
4. **检测阈值**：根据实际应用场景，灵活调整`--face_detector_score`参数，平衡检测的准确性和召回率，满足不同场景下的检测精度要求。
5. **驱动更新**：定期更新Intel显卡驱动程序，确保显卡性能的稳定性和兼容性，充分发挥显卡的最佳性能。

希望这款工具能助力您快速高效地进行图像人脸检测！若它对您有所帮助，欢迎为我们的项目点个Star，您的支持是我们前进的动力！ 