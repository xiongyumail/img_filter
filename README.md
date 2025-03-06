# IMG_FILTER

`IMG_FILTER` 是一款基于 Python 开发的功能强大的图像人脸检测工具，借助 ONNX 运行时和 YOLOFace 模型，能以极高的精度检测图像中的人脸。它不仅能处理单张图像，还能批量处理多个文件夹内的图像。根据检测结果，用户可以选择删除未检测到人脸的图像，也可以将检测到人脸的图像路径输出到 JSON 文件中。在实际测试中，搭配 Intel Arc B580 显卡并结合 `onnxruntime-openvino`，能实现极为高效的推理性能，大幅提升检测效率。

## 功能亮点

- **高效人脸检测**：依托 YOLOFace 模型与 ONNX 运行时，能迅速且精准地检测图像中的人脸，满足安防监控、图像分析、人脸识别等各类场景的检测需求。
- **中文路径支持**：在读取和处理图像时，可完美识别并处理包含中文字符的文件路径，解决了因路径问题导致的使用困扰。
- **多线程处理**：通过 `concurrent.futures.ThreadPoolExecutor` 实现多线程处理机制，显著加快图像检测速度，尤其适用于处理大量图像的任务。
- **灵活配置选项**：用户可通过命令行参数灵活配置人脸检测器的分辨率、检测分数阈值、是否删除非人脸图像以及输出 JSON 文件的路径等，以适应不同的检测任务和需求。
- **Intel GPU 支持**：搭配 `onnxruntime-openvino`，可充分发挥 Intel 显卡的强大计算能力，加速人脸检测进程，极大地提高检测效率。
- **图像筛选与管理**：新增的删除和复制功能，可根据人脸检测分数和人脸关键点分数，自动删除不符合条件的图像，或复制符合条件的图像到指定目录，方便用户对图像数据集进行筛选和管理。通过 `--landmark_score` 参数，用户可以灵活控制图像的筛选标准；通过 `--delete` 选项和 `--copy` 选项，用户可以控制图像的删除复制操作，`--copy` 选项可选择指定目标路径。
- **图像信息展示**：新增基于 Flask 的图像信息展示功能，可在网页上查看图像的分类、文件名、人脸质量均分、特征点精度均分等信息，并支持分页浏览和分类筛选，方便用户直观地查看和管理检测结果。

## 依赖安装

### 通用依赖

使用本工具前，请确保已安装以下依赖库：

- `opencv-python`：用于图像处理和图像读取，提供丰富的功能和接口。
- `numpy`：用于数值计算和数组操作，是科学计算的基础库。
- `onnxruntime`：用于运行 ONNX 模型，实现模型的推理和预测。
- `argparse`：用于命令行参数解析，方便用户根据需求配置工具参数。
- `json`：用于处理 JSON 数据，实现数据的存储和读取。
- `concurrent.futures`：用于多线程处理，提升图像检测的效率。
- `flask`：用于搭建 Web 应用，展示图像信息。
- `urllib.parse`：用于 URL 编码和解码，处理网页链接。
- `webbrowser`：用于自动打开浏览器访问 Web 应用。

可使用 `pip` 进行安装：

```bash
pip install opencv-python numpy onnxruntime flask
```

### Intel GPU 及 onnxruntime-openvino 依赖

若要利用 Intel 显卡进行加速，需安装 `onnxruntime-openvino`：

```bash
pip install onnxruntime-openvino
```

同时，务必确保系统已安装 Intel 显卡驱动程序，以保障显卡正常工作。可前往 [Intel 官方网站](https://www.intel.com/content/www/us/en/download-center/home.html) 下载并安装最新的显卡驱动。

## 使用步骤

1. **下载 ONNX 模型**：获取 `yoloface_8n.onnx` 和 `2dfan4.onnx` 模型文件，并放置在项目目录下的 `models` 文件夹中。
2. **运行 `main.py`**：通过命令行运行工具，支持以下参数：
    - `--input_json`：可选参数，指定包含文件夹路径的 JSON 文件路径。
    - `folder_path`：可选参数，可指定一个或多个包含图像的文件夹路径。若不指定 `--input_json`，则此参数为必填。（注意使用双引号）
    - `--size_yoloface`：可选参数，指定 YOLOFace 模型的输入尺寸，格式为 `widthxheight`，默认为 `640x640`。
    - `--size_2dfan4`：可选参数，指定 2DFAN4 模型的输入尺寸，格式为 `widthxheight`，默认为 `256x256`。
    - `--face_detector_score`：可选参数，指定人脸检测分数阈值，默认为 `0.7`。
    - `--output_json`：可选参数，指定输出人脸图片路径的 JSON 文件路径，默认为 `face.json`。
    - `--onnx_provider`：可选参数，指定 ONNX 运行时的提供者，若使用 Intel 显卡，建议设置为 `OpenVINOExecutionProvider`，默认为 `OpenVINOExecutionProvider`。
    - `--device_type`：可选参数，指定提供者的设备类型，若使用 Intel 显卡，设置为 `GPU`，默认为 `GPU`。
    - `--onnx_model_path_yoloface`：可选参数，指定 YOLOFace ONNX 模型的路径，默认为 `models/yoloface_8n.onnx`。
    - `--onnx_model_path_2dfan4`：可选参数，指定 2DFAN4 ONNX 模型的路径，默认为 `models/2dfan4.onnx`。
    - `--delete`：可选参数，指定是否根据人脸检测分数和人脸关键点分数删除不符合条件的图像，默认为不删除。
    - `--copy`：可选参数，指定是否根据人脸关键点分数复制符合条件的图像到指定目录。可选择指定目标路径，默认目标路径为 `./copied_images`。（注意使用双引号）
    - `--landmark_score`：可选参数，指定人脸关键点分数阈值，默认为 `0.9`，当 `--delete` 或 `--copy` 选项开启时生效。
    - `--display`：可选参数，指定是否使用 Flask 应用展示处理后的图像信息。开启后会自动启动 Flask 应用，并在浏览器中打开展示页面。
    - `--output_full_data`：可选参数，指定是否输出完整数据包括边界框和关键点，默认是 `False`。

示例命令（使用 Intel 显卡，删除不符合条件图像，复制符合条件图像到默认目录，处理多个文件夹，展示图像信息）：

```bash
python main.py "/path/to/images1" "/path/to/images2" --size_yoloface 640x640 --size_2dfan4 256x256 --face_detector_score 0.7 --delete --landmark_score 0.9 --output_json face.json --onnx_provider OpenVINOExecutionProvider --device_type GPU --onnx_model_path_yoloface models/yoloface_8n.onnx --onnx_model_path_2dfan4 models/2dfan4.onnx --copy --display
```

示例命令（使用 Intel 显卡，删除不符合条件图像，复制符合条件图像到指定目录，通过 JSON 文件指定文件夹路径，展示图像信息）：

```bash
python main.py --input_json folders.json --size_yoloface 640x640 --size_2dfan4 256x256 --face_detector_score 0.7 --delete --landmark_score 0.9 --output_json face.json --onnx_provider OpenVINOExecutionProvider --device_type GPU --onnx_model_path_yoloface models/yoloface_8n.onnx --onnx_model_path_2dfan4 models/2dfan4.onnx --copy "./new_copied_images" --display
```

简化命令（默认参数，处理单个文件夹，展示图像信息）：

```bash
python main.py "/path/to/images" --display
```

3. **单独运行 `file.py`（可选）**：如果您已经有了包含检测结果的 JSON 文件，也可以单独运行 `file.py` 来进行图像的删除或复制操作。支持以下参数：
    - `output_json`：可选参数，指定包含人脸检测结果的 JSON 文件路径，默认为 `face.json`。
    - `--landmark_score`：可选参数，指定人脸关键点分数阈值，默认为 `0.9`。
    - `--delete`：可选参数，指定是否根据人脸检测分数和人脸关键点分数删除不符合条件的图像，默认为不删除。
    - `--copy`：可选参数，指定是否根据人脸关键点分数复制符合条件的图像到指定目录。可选择指定目标路径，默认目标路径为 `./copied_images`。

示例命令（删除原始目录不符合条件图像，复制符合条件图像到默认目录）：

```bash
python file.py --landmark_score 0.8 --delete --copy
```

示例命令（删除原始目录不符合条件图像，复制符合条件图像到指定目录）：

```bash
python file.py --landmark_score 0.8 --delete --copy "./new_copied_images"
```

4. **使用图像信息展示功能（运行 `display.py`）**：当在 `main.py` 中使用 `--display` 参数或单独运行 `img_display/display.py` 时：
    - 可通过命令行参数配置展示页面的相关设置，如 `--per_page` 指定每页显示的图像数量，`--input_json` 指定包含检测结果的 JSON 文件路径，`--host` 指定 Flask 应用运行的主机地址，`--port` 指定端口号，`--debug` 指定是否以调试模式运行。
    - 启动后，会自动在浏览器中打开展示页面。页面包含分类目录，可点击分类查看该分类下的图像；也可通过“查看所有图片”按钮查看全部图像。图像列表支持分页浏览，点击图像可查看详细信息，包括完整路径、人脸质量均分、特征点精度均分等。

## 代码架构变更说明

1. **`face_detector.py`**：
    - **导入优化**：移除了不必要的 `json` 导入，简化了代码结构。
    - **方法功能调整**：`process_images_in_folder` 方法更名为 `process_images`，并移除了 `output_json` 和 `output_full_data` 参数，改为直接返回检测结果，使得方法更加专注于图像处理逻辑。
    - **返回值优化**：`process_images` 方法现在直接返回检测结果，而不是将结果保存到 JSON 文件中，提高了代码的灵活性。

2. **`file.py`**：
    - **新增 `save_output_json` 函数**：新增了 `save_output_json` 函数，用于将检测结果保存到 JSON 文件中，支持跨磁盘/网络路径的智能路径压缩存储，提高了 JSON 文件的结构化和可读性。
    - **`process_images_based_on_scores` 函数优化**：优化了 `process_images_based_on_scores` 函数的逻辑，使其能够处理新的 JSON 文件结构，支持更灵活的图像删除和复制操作。
    - **命令行参数调整**：`json_file_path` 参数更名为 `output_json`，使其与 `main.py` 中的参数命名保持一致，提高了一致性。

3. **`main.py`**：
    - **函数调用调整**：`detector.process_images_in_folder` 方法调用改为 `detector.process_images`，并新增了 `save_output_json` 函数的调用，将检测结果保存到 JSON 文件中。
    - **参数传递优化**：`process_images` 方法的参数传递更加简洁，移除了不必要的 `output_json` 和 `output_full_data` 参数，使得代码更加清晰。

## 注意要点

1. **模型兼容性**：确保使用的 ONNX 模型与代码兼容，模型的输入输出格式与代码中的处理逻辑保持一致，避免因格式不匹配导致错误。
2. **设备配置**：根据硬件设备和实际需求，合理配置 `--onnx_provider` 和 `--device_type` 参数，以获取最佳性能。若使用 Intel 显卡，务必将 `--onnx_provider` 设置为 `OpenVINOExecutionProvider`，`--device_type` 设置为 `GPU`。
3. **文件路径**：处理文件路径时，确保路径的正确性和可读性，尤其是包含特殊字符或中文的路径，避免因路径错误导致文件读取失败。
4. **检测阈值**：根据实际应用场景，灵活调整 `--face_detector_score` 和 `--landmark_score` 参数，平衡检测的准确性和召回率，满足不同场景下的检测精度要求。
5. **驱动更新**：定期更新 Intel 显卡驱动程序，确保显卡性能的稳定性和兼容性，充分发挥显卡的最佳性能。

希望这款工具能助力您快速高效地进行图像人脸检测！若它对您有所帮助，欢迎为我们的项目点个 Star，您的支持是我们前进的动力！