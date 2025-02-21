import os
import cv2
import numpy as np
import onnxruntime as ort
import argparse
import json
from typing import List, Tuple
from collections import namedtuple
import concurrent.futures

# 定义必要的类和函数
VisionFrame = np.ndarray
BoundingBox = np.ndarray
Score = float
FaceLandmark5 = np.ndarray
StateManager = namedtuple('StateManager', ['get_item'])

# 初始化 ONNX 推理会话，避免每次检测都重新创建
onnx_session = None
input_name = None

def init_onnx_session(onnx_model_path, provider, device_type):
    global onnx_session, input_name
    provider_options = [{'device_type': device_type}] if device_type else []
    onnx_session = ort.InferenceSession(onnx_model_path, providers=[provider], provider_options=provider_options)
    input_name = onnx_session.get_inputs()[0].name

def unpack_resolution(resolution_str: str) -> Tuple[int, int]:
    """
    将分辨率字符串拆分为宽度和高度
    :param resolution_str: 分辨率字符串，格式为 'widthxheight'
    :return: 宽度和高度的元组
    """
    width, height = map(int, resolution_str.split('x'))
    return width, height

def resize_frame_resolution(frame: VisionFrame, size: Tuple[int, int]) -> VisionFrame:
    """
    调整帧的分辨率
    :param frame: 输入的帧
    :param size: 目标大小，格式为 (width, height)
    :return: 调整分辨率后的帧
    """
    return cv2.resize(frame, size)

def prepare_detect_frame(frame: VisionFrame, size: Tuple[int, int]) -> np.ndarray:
    """
    准备用于检测的帧
    :param frame: 输入的帧
    :param size: 目标大小，格式为 (width, height)
    :return: 准备好的帧
    """
    # 这里简单归一化，实际可能需要根据模型要求调整
    frame = frame.astype(np.float32) / 255.0
    frame = np.transpose(frame, (2, 0, 1))
    frame = np.expand_dims(frame, axis=0)
    return frame

def forward_with_yoloface(frame: np.ndarray) -> np.ndarray:
    """
    使用 YOLOFace 模型进行前向推理
    :param frame: 输入的帧
    :return: 模型输出
    """
    global onnx_session, input_name
    output = onnx_session.run(None, {input_name: frame})[0]
    return output

def detect_with_yoloface(vision_frame: VisionFrame, face_detector_size: str, face_detector_score: float) -> Tuple[List[BoundingBox], List[Score], List[FaceLandmark5]]:
    """
    使用 YOLOFace 模型进行人脸检测
    :param vision_frame: 输入的帧
    :param face_detector_size: 人脸检测器的分辨率字符串
    :param face_detector_score: 人脸检测分数阈值
    :return: 边界框列表、分数列表和 5 点人脸关键点列表
    """
    bounding_boxes = []
    face_scores = []
    face_landmarks_5 = []
    face_detector_width, face_detector_height = unpack_resolution(face_detector_size)
    temp_vision_frame = resize_frame_resolution(vision_frame, (face_detector_width, face_detector_height))
    ratio_height = vision_frame.shape[0] / temp_vision_frame.shape[0]
    ratio_width = vision_frame.shape[1] / temp_vision_frame.shape[1]
    detect_vision_frame = prepare_detect_frame(temp_vision_frame, (face_detector_width, face_detector_height))
    detection = forward_with_yoloface(detect_vision_frame)
    detection = np.squeeze(detection).T
    bounding_box_raw, score_raw, face_landmark_5_raw = np.split(detection, [4, 5], axis=1)
    keep_indices = np.where(score_raw > face_detector_score)[0]

    if keep_indices.size > 0:
        bounding_box_raw = bounding_box_raw[keep_indices]
        face_landmark_5_raw = face_landmark_5_raw[keep_indices]
        score_raw = score_raw[keep_indices]

        # 计算边界框
        bounding_boxes = [
            np.array([
                (box[0] - box[2] / 2) * ratio_width,
                (box[1] - box[3] / 2) * ratio_height,
                (box[0] + box[2] / 2) * ratio_width,
                (box[1] + box[3] / 2) * ratio_height
            ]) for box in bounding_box_raw
        ]

        face_scores = score_raw.ravel().tolist()

        # 调整关键点坐标
        face_landmark_5_raw[:, 0::3] *= ratio_width
        face_landmark_5_raw[:, 1::3] *= ratio_height

        face_landmarks_5 = [
            np.array(landmark.reshape(-1, 3)[:, :2]) for landmark in face_landmark_5_raw
        ]

    return bounding_boxes, face_scores, face_landmarks_5

def process_single_image(file_path: str, face_detector_size: str, face_detector_score: float) -> bool:
    """
    处理单张图片，判断是否检测到人脸
    :param file_path: 图片文件路径
    :param face_detector_size: 人脸检测器的分辨率字符串
    :param face_detector_score: 人脸检测分数阈值
    :return: 是否检测到人脸
    """
    try:
        # 使用 numpy 读取图片以支持中文路径
        image = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is not None:
            bounding_boxes, _, _ = detect_with_yoloface(image, face_detector_size, face_detector_score)
            return bool(bounding_boxes)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    return False

def process_images_in_folder(folder_path: str, face_detector_size: str = '640x640', face_detector_score: float = 0.5, delete_non_face: bool = True, output_json: str = None):
    """
    处理文件夹中的图像，根据选项删除未检测到人脸的图像或输出人脸图片路径 JSON 文件
    :param folder_path: 文件夹路径
    :param face_detector_size: 人脸检测器的分辨率字符串
    :param face_detector_score: 人脸检测分数阈值
    :param delete_non_face: 是否删除非人脸图片
    :param output_json: 输出人脸图片路径 JSON 文件的路径
    """
    image_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))

    face_image_paths = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda file: (file, process_single_image(file, face_detector_size, face_detector_score)), image_files))

    for file_path, has_face in results:
        if has_face:
            face_image_paths.append(file_path)

    if output_json:
        try:
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump(face_image_paths, f, indent=4, ensure_ascii=False)
            print(f"Face image paths saved to {output_json}")
        except Exception as e:
            print(f"Error saving JSON file {output_json}: {e}")

    if delete_non_face and output_json:
        import subprocess
        try:
            # 调用删除脚本
            subprocess.run(['python', 'delete_non_face.py', folder_path, '--face_image_json', output_json], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running delete script: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process images in a folder and either delete non-face images or output face image paths to a JSON file.')
    parser.add_argument('folder_path', type=str, help='Path to the folder containing images.')
    parser.add_argument('--face_detector_size', type=str, default='640x640', help='Resolution for the face detector, in the format "widthxheight".')
    parser.add_argument('--face_detector_score', type=float, default=0.7, help='Score threshold for face detection.')
    parser.add_argument('--delete_non_face', action='store_true', help='Delete non-face images.')
    parser.add_argument('--output_json', type=str, default='face.json', help='Output JSON file path for face image paths.')
    parser.add_argument('--onnx_provider', type=str, default="OpenVINOExecutionProvider", help='ONNX runtime provider.')
    parser.add_argument('--device_type', type=str, default="GPU", help='Device type for the provider.')
    parser.add_argument('--onnx_model_path', type=str, default='yoloface_8n.onnx', help='Path to the ONNX model.')

    args = parser.parse_args()
    init_onnx_session(args.onnx_model_path, args.onnx_provider, args.device_type)
    process_images_in_folder(args.folder_path, args.face_detector_size, args.face_detector_score, args.delete_non_face, args.output_json)