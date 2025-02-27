# IMG_FILTER 
`IMG_FILTER`是一款基于Python开发的功能强大的图像人脸检测工具，借助ONNX运行时和YOLOFace模型，能以极高的精度检测图像中的人脸。它不仅能处理单张图像，还能批量处理多个文件夹内的图像。根据检测结果，用户可以选择删除未检测到人脸的图像，也可以将检测到人脸的图像路径输出到JSON文件中。在实际测试中，搭配Intel Arc B580显卡并结合`onnxruntime-openvino`，能实现极为高效的推理性能，大幅提升检测效率。

## 功能亮点
- **高效人脸检测**：依托YOLOFace模型与ONNX运行时，能迅速且精准地检测图像中的人脸，满足安防监控、图像分析、人脸识别等各类场景的检测需求。
- **中文路径支持**：在读取和处理图像时，可完美识别并处理包含中文字符的文件路径，解决了因路径问题导致的使用困扰。
- **多线程处理**：通过`concurrent.futures.ThreadPoolExecutor`实现多线程处理机制，显著加快图像检测速度，尤其适用于处理大量图像的任务。
- **灵活配置选项**：用户可通过命令行参数灵活配置人脸检测器的分辨率、检测分数阈值、是否删除非人脸图像以及输出JSON文件的路径等，以适应不同的检测任务和需求。
- **Intel GPU支持**：搭配`onnxruntime-openvino`，可充分发挥Intel显卡的强大计算能力，加速人脸检测进程，极大地提高检测效率。
- **图像筛选与管理**：新增的删除和复制功能，可根据人脸检测分数和人脸关键点分数，自动删除不符合条件的图像，或复制符合条件的图像到指定目录，方便用户对图像数据集进行筛选和管理 。通过`--landmark_score`参数，用户可以灵活控制图像的筛选标准；通过`--delete`选项和`--copy`选项，用户可以控制图像的删除复制操作，`--copy`选项可选择指定目标路径。
- **图像信息展示**：新增基于Flask的图像信息展示功能，可在网页上查看图像的分类、文件名、人脸质量均分、特征点精度均分等信息，并支持分页浏览和分类筛选，方便用户直观地查看和管理检测结果。

## 依赖安装
### 通用依赖
使用本工具前，请确保已安装以下依赖库：
- `opencv-python`：用于图像处理和图像读取，提供丰富的功能和接口。
- `numpy`：用于数值计算和数组操作，是科学计算的基础库。
- `onnxruntime`：用于运行ONNX模型，实现模型的推理和预测。
- `argparse`：用于命令行参数解析，方便用户根据需求配置工具参数。
- `json`：用于处理JSON数据，实现数据的存储和读取。
- `concurrent.futures`：用于多线程处理，提升图像检测的效率。
- `flask`：用于搭建Web应用，展示图像信息。
- `urllib.parse`：用于URL编码和解码，处理网页链接。
- `webbrowser`：用于自动打开浏览器访问Web应用。
可使用`pip`进行安装：
```bash
pip install opencv-python numpy onnxruntime flask
```

### Intel GPU及onnxruntime-openvino依赖
若要利用Intel显卡进行加速，需安装`onnxruntime-openvino`：
```bash
pip install onnxruntime-openvino
```
同时，务必确保系统已安装Intel显卡驱动程序，以保障显卡正常工作。可前往[Intel官方网站](https://www.intel.com/content/www/us/en/download-center/home.html)下载并安装最新的显卡驱动。

## 使用步骤
1. ​**下载ONNX模型**：获取`yoloface_8n.onnx`和`2dfan4.onnx`模型文件，并放置在项目目录下的`models`文件夹中。
2. **运行`main.py`**：通过命令行运行工具，支持以下参数：
    - `--input_json`：可选参数，指定包含文件夹路径的JSON文件路径。
    - `folder_path`：可选参数，可指定一个或多个包含图像的文件夹路径。若不指定`--input_json`，则此参数为必填。（注意使用双引号）
    - `--size_yoloface`：可选参数，指定YOLOFace模型的输入尺寸，格式为`widthxheight`，默认为`640x640`。
    - `--size_2dfan4`：可选参数，指定2DFAN4模型的输入尺寸，格式为`widthxheight`，默认为`256x256`。
    - `--face_detector_score`：可选参数，指定人脸检测分数阈值，默认为`0.7`。
    - `--output_json`：可选参数，指定输出人脸图片路径的JSON文件路径，默认为`face.json`。
    - `--onnx_provider`：可选参数，指定ONNX运行时的提供者，若使用Intel显卡，建议设置为`OpenVINOExecutionProvider`，默认为`OpenVINOExecutionProvider`。
    - `--device_type`：可选参数，指定提供者的设备类型，若使用Intel显卡，设置为`GPU`，默认为`GPU`。
    - `--onnx_model_path_yoloface`：可选参数，指定YOLOFace ONNX模型的路径，默认为`models/yoloface_8n.onnx`。
    - `--onnx_model_path_2dfan4`：可选参数，指定2DFAN4 ONNX模型的路径，默认为`models/2dfan4.onnx`。
    - `--delete`：可选参数，指定是否根据人脸检测分数和人脸关键点分数删除不符合条件的图像，默认为不删除。
    - `--copy`：可选参数，指定是否根据人脸关键点分数复制符合条件的图像到指定目录。可选择指定目标路径，默认目标路径为`./copied_images`。（注意使用双引号）
    - `--landmark_score`：可选参数，指定人脸关键点分数阈值，默认为`0.9`，当`--delete`或`--copy`选项开启时生效。
    - `--display`：可选参数，指定是否使用Flask应用展示处理后的图像信息。开启后会自动启动Flask应用，并在浏览器中打开展示页面。
    - `--output_full_data`：可选参数，指定是否输出完整数据包括边界框和关键点，默认是`False`。
示例命令（使用Intel显卡，删除不符合条件图像，复制符合条件图像到默认目录，处理多个文件夹，展示图像信息）：
```bash
python main.py "/path/to/images1" "/path/to/images2" --size_yoloface 640x640 --size_2dfan4 256x256 --face_detector_score 0.7 --delete --landmark_score 0.9 --output_json face.json --onnx_provider OpenVINOExecutionProvider --device_type GPU --onnx_model_path_yoloface models/yoloface_8n.onnx --onnx_model_path_2dfan4 models/2dfan4.onnx --copy --display
```
示例命令（使用Intel显卡，删除不符合条件图像，复制符合条件图像到指定目录，通过JSON文件指定文件夹路径，展示图像信息）：
```bash
python main.py --input_json folders.json --size_yoloface 640x640 --size_2dfan4 256x256 --face_detector_score 0.7 --delete --landmark_score 0.9 --output_json face.json --onnx_provider OpenVINOExecutionProvider --device_type GPU --onnx_model_path_yoloface models/yoloface_8n.onnx --onnx_model_path_2dfan4 models/2dfan4.onnx --copy "./new_copied_images" --display
```
简化命令(默认参数，处理单个文件夹，展示图像信息)：
```bash
python main.py "/path/to/images" --display
```
3. **单独运行`file.py`（可选）**：如果您已经有了包含检测结果的JSON文件，也可以单独运行`file.py`来进行图像的删除或复制操作。支持以下参数：
    - `json_file_path`：可选参数，指定包含人脸检测结果的JSON文件路径，默认为`face.json`。
    - `--landmark_score`：可选参数，指定人脸关键点分数阈值，默认为`0.9`。
    - `--delete`：可选参数，指定是否根据人脸检测分数和人脸关键点分数删除不符合条件的图像，默认为不删除。
    - `--copy`：可选参数，指定是否根据人脸关键点分数复制符合条件的图像到指定目录。可选择指定目标路径，默认目标路径为`./copied_images`。
示例命令（删除原始目录不符合条件图像，复制符合条件图像到默认目录）：
```bash
python file.py --landmark_score 0.8 --delete --copy
```
示例命令（删除原始目录不符合条件图像，复制符合条件图像到指定目录）：
```bash
python file.py --landmark_score 0.8 --delete --copy "./new_copied_images"
```
4. **使用图像信息展示功能（运行`display.py`）**：当在`main.py`中使用`--display`参数或单独运行`display.py`时：
    - 可通过命令行参数配置展示页面的相关设置，如`--per_page`指定每页显示的图像数量，`--input_json`指定包含检测结果的JSON文件路径，`--host`指定Flask应用运行的主机地址，`--port`指定端口号，`--debug`指定是否以调试模式运行。
    - 启动后，会自动在浏览器中打开展示页面。页面包含分类目录，可点击分类查看该分类下的图像；也可通过“查看所有图片”按钮查看全部图像。图像列表支持分页浏览，点击图像可查看详细信息，包括完整路径、人脸质量均分、特征点精度均分等。

## 代码架构变更说明
1. **`face_detector.py`**：
    - **导入优化**：移除了不必要的`onnxruntime as ort`导入，同时也移除了`create_static_model_set`的导入，因为相关逻辑已调整。
    - **类型安全增强**：添加了`VisionFrame`、`BoundingBox`、`Score`等类型别名，提高代码的可读性和类型安全性，使得代码在处理相关数据时更加清晰和健壮。
    - **初始化逻辑调整**：`FaceDetector`类的`__init__`方法中，移除了`self.onnx_session`和`self.input_name`的直接初始化，改为使用`InferencePool`来管理模型会话，使得代码结构更加模块化和可维护。
    - **方法功能细化与注释完善**：多个方法添加了详细的文档字符串，包括`get_inference_pool`、`unpack_resolution`、`resize_frame_resolution`等，解释了方法的功能、参数和返回值，方便开发者理解和使用。同时，部分方法的参数名进行了修改，如`face_detector_size`改为`size_yoloface`，使参数含义更加明确。
    - **新增参数**：`process_images_in_folder`方法新增`output_full_data`参数，用于控制是否输出完整数据，增加了灵活性。
2. **`inference_utils.py`**：
    - **独立初始化函数**：新增`init_onnx_session`函数，将初始化ONNX会话的逻辑封装起来，便于复用和维护，同时添加了详细的文档字符串说明其功能、参数和返回值。
    - **`InferencePool`类重构**：`InferencePool`类的`__init__`方法进行了重构，添加了详细的文档字符串，解释各个参数的含义。调用`init_onnx_session`方法来初始化`yoloface`和`2dfan4`模型的会话，并在`pool`字典中存储`yoloface`会话和输入名称的元组，使得模型管理更加清晰。`get`方法也添加了文档字符串，解释其功能和返回值。
    - **函数注释补充**：`conditional_thread_semaphore`和`transform_points`函数添加了文档字符串，解释函数的功能、参数和返回值，提高代码的可读性。
3. **`main.py`**：
    - **参数解析更新**：`--face_detector_size`参数改为`--size_yoloface`，并新增`--size_2dfan4`参数，分别用于指定YOLOFace和2DFAN4模型的输入尺寸，使参数设置更加明确和灵活。同时新增`--output_full_data`参数，用于控制是否输出完整数据。
    - **函数调用适配**：`detector.process_images_in_folder`方法的调用参数相应修改为`args.size_yoloface`和`args.size_2dfan4`，以适配代码中参数名的变更，并传入`args.output_full_data`参数。
    - **新增功能支持**：新增`--display`参数，用于控制是否启动Flask应用展示图像信息，并在代码中添加相应逻辑，通过`subprocess`启动`display.py`脚本。
4. **`display.py`**：
    - **新增文件**：新增`display.py`文件，用于基于Flask搭建Web应用展示图像信息。实现了图像数据加载、分类处理、分页展示、图像详情查看等功能，并通过命令行参数配置应用的运行参数。
    - **模板文件**：新增`templates`目录下的`404.html`、`categories.html`、`index.html`模板文件，用于定义Web页面的布局和样式，实现页面的展示效果和交互功能。
    - **新增功能**：新增点赞功能，可对图片进行点赞或取消点赞操作，状态会实时更新；新增了关闭服务器的功能，可以通过`/shutdown`接口关闭Flask应用。
5. **新增文件**：
    - **静态资源**：新增`static/css/categories.css`和`static/css/style.css`文件，分别用于定义分类页面和图像展示页面的样式，增强了页面的美观性和用户体验。
    - **JavaScript脚本**：新增`static/js/main.js`文件，用于实现点赞功能的交互逻辑，包括点赞状态的初始化、图片加载时的处理以及点赞操作的发送和反馈。

## 注意要点
1. **模型兼容性**：确保使用的ONNX模型与代码兼容，模型的输入输出格式与代码中的处理逻辑保持一致，避免因格式不匹配导致错误。
2. **设备配置**：根据硬件设备和实际需求，合理配置`--onnx_provider`和`--device_type`参数，以获取最佳性能。若使用Intel显卡，务必将`--onnx_provider`设置为`OpenVINOExecutionProvider`，`--device_type`设置为`GPU`。
3. **文件路径**：处理文件路径时，确保路径的正确性和可读性，尤其是包含特殊字符或中文的路径，避免因路径错误导致文件读取失败。
4. **检测阈值**：根据实际应用场景，灵活调整`--face_detector_score`和`--landmark_score`参数，平衡检测的准确性和召回率，满足不同场景下的检测精度要求。
5. **驱动更新**：定期更新Intel显卡驱动程序，确保显卡性能的稳定性和兼容性，充分发挥显卡的最佳性能。
6. **Flask应用配置**：在使用Flask展示图像信息时，注意配置`--per_page`、`--input_json`等参数，确保展示的数据和页面布局符合需求。若遇到问题，可通过调试模式（`--debug`参数）排查错误。
7. **点赞功能**：使用点赞功能时，确保网络连接正常，以便点赞状态能够及时更新和保存。
8. **关闭服务器**：通过`/shutdown`接口关闭服务器时，确保已经保存好所有需要的数据，避免数据丢失。

希望这款工具能助力您快速高效地进行图像人脸检测！若它对您有所帮助，欢迎为我们的项目点个Star，您的支持是我们前进的动力！