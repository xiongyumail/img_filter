# IMG_FACEDETECTOR 

`img_facedetector`是一款基于Python开发的功能强大的图像人脸检测工具，借助ONNX运行时和YOLOFace模型，能以极高的精度检测图像中的人脸。它不仅能处理单张图像，还能批量处理整个文件夹内的图像。根据检测结果，用户可以选择删除未检测到人脸的图像，也可以将检测到人脸的图像路径输出到JSON文件中。在实际测试中，搭配Intel Arc B580显卡并结合`onnxruntime-openvino`，能实现极为高效的推理性能，大幅提升检测效率。

## 功能亮点
- **高效人脸检测**：依托YOLOFace模型与ONNX运行时，能迅速且精准地检测图像中的人脸，满足安防监控、图像分析、人脸识别等各类场景的检测需求。
- **中文路径支持**：在读取和处理图像时，可完美识别并处理包含中文字符的文件路径，解决了因路径问题导致的使用困扰。
- **多线程处理**：通过`concurrent.futures.ThreadPoolExecutor`实现多线程处理机制，显著加快图像检测速度，尤其适用于处理大量图像的任务。
- **灵活配置选项**：用户可通过命令行参数灵活配置人脸检测器的分辨率、检测分数阈值、是否删除非人脸图像以及输出JSON文件的路径等，以适应不同的检测任务和需求。
- **Intel GPU支持**：搭配`onnxruntime-openvino`，可充分发挥Intel显卡的强大计算能力，加速人脸检测进程，极大地提高检测效率。
- **图像筛选与管理**：新增的删除和复制功能，可根据人脸检测分数和人脸关键点分数，自动删除不符合条件的图像，或复制符合条件的图像到指定目录，方便用户对图像数据集进行筛选和管理 。通过`--landmark_score`参数，用户可以灵活控制图像的筛选标准；通过`--delete`选项和`--copy`选项，以及`--target_path`参数，用户可以控制图像的删除复制操作和目标路径。

## 依赖安装
### 通用依赖
使用本工具前，请确保已安装以下依赖库：
- `opencv-python`：用于图像处理和图像读取，提供丰富的功能和接口。
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
1. **下载ONNX模型**：获取`yoloface_8n.onnx`和`2dfan4.onnx`模型文件，并放置在正确路径下。可从模型官方来源获取。
2. **运行`main.py`**：通过命令行运行工具，支持以下参数：
    - `folder_path`：必填参数，指定包含图像的文件夹路径。
    - `--face_detector_size`：可选参数，指定人脸检测器的分辨率，格式为`widthxheight`，默认为`640x640`。
    - `--face_detector_score`：可选参数，指定人脸检测分数阈值，默认为`0.7`。
    - `--output_json`：可选参数，指定输出人脸图片路径的JSON文件路径，默认为`face.json`。
    - `--onnx_provider`：可选参数，指定ONNX运行时的提供者，若使用Intel显卡，建议设置为`OpenVINOExecutionProvider`，默认为`OpenVINOExecutionProvider`。
    - `--device_type`：可选参数，指定提供者的设备类型，若使用Intel显卡，设置为`GPU`，默认为`GPU`。
    - `--onnx_model_path_yoloface`：可选参数，指定YOLOFace ONNX模型的路径，默认为`yoloface_8n.onnx`。
    - `--onnx_model_path_2dfan4`：可选参数，指定2DFAN4 ONNX模型的路径，默认为`2dfan4.onnx`。
    - `--delete`：可选参数，指定是否根据人脸检测分数和人脸关键点分数删除不符合条件的图像，默认为不删除。
    - `--copy`：可选参数，指定是否根据人脸关键点分数复制符合条件的图像到指定目录。
    - `--landmark_score`：可选参数，指定人脸关键点分数阈值，默认为`0.9`，当`--delete`或`--copy`选项开启时生效。
    - `--target_path`：可选参数，指定复制图像的目标路径，默认为`./copied_images`，当`--copy`选项开启时生效。

示例命令（使用Intel显卡，删除不符合条件图像，复制符合条件图像）：
```bash
python main.py /path/to/images --face_detector_size 640x640 --face_detector_score 0.7 --delete --landmark_score 0.9 --output_json face.json --onnx_provider OpenVINOExecutionProvider --device_type GPU --onnx_model_path_yoloface yoloface_8n.onnx --onnx_model_path_2dfan4 2dfan4.onnx --copy --target_path ./new_copied_images
```
简化命令(默认参数):
```bash
python main.py /path/to/images 
```

3. **单独运行`file.py`（可选）**：如果您已经有了包含检测结果的JSON文件，也可以单独运行`file.py`来进行图像的删除或复制操作。支持以下参数：
    - `json_file_path`：可选参数，指定包含人脸检测结果的JSON文件路径，默认为`face.json`。
    - `--landmark_score`：可选参数，指定人脸关键点分数阈值，默认为`0.9`。
    - `--target_path`：可选参数，指定复制图像的目标路径，默认为`./copied_images`，当复制操作开启时生效。
    - `--delete`：可选参数，指定是否根据人脸检测分数和人脸关键点分数删除不符合条件的图像，默认为不删除。
    - `--copy`：可选参数，指定是否根据人脸关键点分数复制符合条件的图像到指定目录。

示例命令（删除原始目录不符合条件图像，复制符合条件图像到新目录）：
```bash
python file.py --landmark_score 0.8 --delete --copy --target_path ./new_copied_images
```
此命令会使用默认的`face.json`文件，根据`0.8`的关键点分数阈值来删除不符合条件的图像，并将符合条件的图像复制到`./new_copied_images`目录。

## 代码架构
### main.py
1. **核心函数**：
    - `init_onnx_session`：初始化ONNX推理会话，设置全局变量`onnx_session`和`input_name`，为后续模型推理做准备。
    - `unpack_resolution`：将分辨率字符串拆分为宽度和高度，方便参数处理和配置。
    - `resize_frame_resolution`：调整帧的分辨率，使其符合模型输入要求。
    - `prepare_detect_frame`：准备用于检测的帧，包括归一化、转置和扩展维度等操作，确保数据格式正确。
    - `forward_with_yoloface`：使用YOLOFace模型进行前向推理，得出初步检测结果。
    - `detect_with_yoloface`：使用YOLOFace模型进行人脸检测，返回边界框、分数和5点人脸关键点，提供详细的检测信息。
    - `detect_with_2dfan4`：使用2DFAN4模型获取人脸68个关键点及其分数，为图像筛选提供更全面的数据。
    - `process_single_image`：处理单张图片，判断是否检测到人脸，并获取人脸关键点分数，输出检测结果。
    - `process_images_in_folder`：处理文件夹中的图像，根据选项删除未检测到人脸的图像或输出人脸图片路径JSON文件，实现批量处理功能。
2. **命令行参数解析**：利用`argparse`模块解析命令行参数，根据用户输入的参数灵活配置工具行为，满足不同用户的多样化需求。

### file.py
1. **核心函数**：
    - `process_images_based_on_scores`：读取JSON文件中的人脸检测结果，根据`face_scores`是否为空或者`face_landmark_scores_68`的均值是否小于指定的`landmark_score`来判断是否删除对应的图片；根据`face_landmark_scores_68`的均值是否大于指定的`landmark_score`来判断是否复制对应的图片到指定目录。

## 注意要点
1. **模型兼容性**：确保使用的ONNX模型与代码兼容，模型的输入输出格式与代码中的处理逻辑保持一致，避免因格式不匹配导致错误。
2. **设备配置**：根据硬件设备和实际需求，合理配置`--onnx_provider`和`--device_type`参数，以获取最佳性能。若使用Intel显卡，务必将`--onnx_provider`设置为`OpenVINOExecutionProvider`，`--device_type`设置为`GPU`。
3. **文件路径**：处理文件路径时，确保路径的正确性和可读性，尤其是包含特殊字符或中文的路径，避免因路径错误导致文件读取失败。
4. **检测阈值**：根据实际应用场景，灵活调整`--face_detector_score`和`--landmark_score`参数，平衡检测的准确性和召回率，满足不同场景下的检测精度要求。
5. **驱动更新**：定期更新Intel显卡驱动程序，确保显卡性能的稳定性和兼容性，充分发挥显卡的最佳性能。

希望这款工具能助力您快速高效地进行图像人脸检测！若它对您有所帮助，欢迎为我们的项目点个Star，您的支持是我们前进的动力！ 