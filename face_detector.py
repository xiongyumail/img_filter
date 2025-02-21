import os
import cv2
import numpy as np
import json
import concurrent.futures
import threading
from typing import List, Tuple
from inference_utils import InferencePool, conditional_thread_semaphore, transform_points
from image_utils import conditional_optimize_contrast, warp_face_by_translation, create_rotated_matrix_and_size

# 定义必要的类型别名
VisionFrame = np.ndarray
BoundingBox = np.ndarray
Score = float
FaceLandmark5 = np.ndarray
FaceLandmark68 = np.ndarray
Resolution = Tuple[int, int]
Detection = np.ndarray
Angle = float

class FaceDetector:
    def __init__(self, onnx_model_path_yoloface, onnx_model_path_2dfan4, provider: str, device_type: str):
        """
        初始化人脸检测器。
        :param onnx_model_path_yoloface: YOLOFace 模型的 ONNX 路径
        :param onnx_model_path_2dfan4: 2DFAN4 模型的 ONNX 路径
        :param provider: 推理提供者，如 'CPUExecutionProvider'
        :param device_type: 设备类型，如 'CPU'
        """
        self.semaphore = threading.Semaphore(1)
        self.inference_pool = InferencePool(onnx_model_path_yoloface, onnx_model_path_2dfan4, provider, device_type)

    def get_inference_pool(self):
        """
        获取推理池中的 YOLOFace 会话和输入名称。
        :return: 包含 YOLOFace 会话和输入名称的字典
        """
        yoloface_session, yoloface_input_name = self.inference_pool.get('yoloface')
        return {'yoloface': yoloface_session, 'yoloface_input_name': yoloface_input_name}

    def unpack_resolution(self, resolution: str) -> Resolution:
        """
        将分辨率字符串解析为宽度和高度的元组。
        :param resolution: 分辨率字符串，格式为 'widthxheight'
        :return: 宽度和高度的元组
        """
        try:
            width, height = map(int, resolution.split('x'))
            return width, height
        except ValueError:
            raise ValueError(f"Invalid resolution format: {resolution}. Expected 'widthxheight'.")

    def resize_frame_resolution(self, vision_frame: VisionFrame, max_resolution: Resolution) -> VisionFrame:
        """
        调整图像帧的分辨率，使其不超过最大分辨率。
        :param vision_frame: 输入的图像帧
        :param max_resolution: 最大分辨率
        :return: 调整后的图像帧
        """
        height, width = vision_frame.shape[:2]
        max_width, max_height = max_resolution

        if height > max_height or width > max_width:
            scale = min(max_height / height, max_width / width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            return cv2.resize(vision_frame, (new_width, new_height))
        return vision_frame

    def prepare_detect_frame(self, temp_vision_frame: VisionFrame, size_yoloface: str) -> VisionFrame:
        """
        准备用于人脸检测的图像帧。
        :param temp_vision_frame: 临时图像帧
        :param size_yoloface: YOLOFace 模型的输入尺寸，格式为 'widthxheight'
        :return: 准备好的图像帧
        """
        face_detector_width, face_detector_height = self.unpack_resolution(size_yoloface)
        detect_vision_frame = np.zeros((face_detector_height, face_detector_width, 3))
        detect_vision_frame[:temp_vision_frame.shape[0], :temp_vision_frame.shape[1], :] = temp_vision_frame
        detect_vision_frame = (detect_vision_frame - 127.5) / 128.0
        detect_vision_frame = np.expand_dims(detect_vision_frame.transpose(2, 0, 1), axis=0).astype(np.float32)
        return detect_vision_frame

    def forward_with_yoloface(self, detect_vision_frame: VisionFrame) -> Detection:
        """
        使用 YOLOFace 模型进行前向推理。
        :param detect_vision_frame: 准备好的用于检测的图像帧
        :return: 检测结果
        """
        face_detector = self.get_inference_pool().get('yoloface')
        input_name = self.get_inference_pool().get('yoloface_input_name')

        with self.semaphore:
            try:
                detection = face_detector.run(None, {input_name: detect_vision_frame})
            except Exception as e:
                raise RuntimeError(f"Error during forward pass: {e}")

        return detection

    def detect_with_yoloface(self, vision_frame: VisionFrame, size_yoloface: str, face_detector_score: float) -> Tuple[List[BoundingBox], List[Score], List[FaceLandmark5]]:
        """
        使用 YOLOFace 模型进行人脸检测。
        :param vision_frame: 输入的图像帧
        :param size_yoloface: YOLOFace 模型的输入尺寸，格式为 'widthxheight'
        :param face_detector_score: 人脸检测的分数阈值
        :return: 检测到的边界框、人脸分数和 5 点人脸关键点
        """
        bounding_boxes = []
        face_scores = []
        face_landmarks_5 = []
        face_detector_width, face_detector_height = self.unpack_resolution(size_yoloface)
        temp_vision_frame = self.resize_frame_resolution(vision_frame, (face_detector_width, face_detector_height))
        ratio_height = vision_frame.shape[0] / temp_vision_frame.shape[0]
        ratio_width = vision_frame.shape[1] / temp_vision_frame.shape[1]
        detect_vision_frame = self.prepare_detect_frame(temp_vision_frame, size_yoloface)
        detection = self.forward_with_yoloface(detect_vision_frame)
        detection = np.squeeze(detection).T
        bounding_box_raw, score_raw, face_landmark_5_raw = np.split(detection, [4, 5], axis=1)
        keep_indices = np.where(score_raw > face_detector_score)[0]

        if keep_indices.size > 0:
            bounding_box_raw = bounding_box_raw[keep_indices]
            face_landmark_5_raw = face_landmark_5_raw[keep_indices]
            score_raw = score_raw[keep_indices]

            bounding_boxes = [
                np.array([
                    (box[0] - box[2] / 2) * ratio_width,
                    (box[1] - box[3] / 2) * ratio_height,
                    (box[0] + box[2] / 2) * ratio_width,
                    (box[1] + box[3] / 2) * ratio_height
                ]) for box in bounding_box_raw
            ]

            face_scores = score_raw.ravel().tolist()

            face_landmark_5_raw[:, 0::3] *= ratio_width
            face_landmark_5_raw[:, 1::3] *= ratio_height

            face_landmarks_5 = [
                np.array(landmark.reshape(-1, 3)[:, :2]) for landmark in face_landmark_5_raw
            ]

        return bounding_boxes, face_scores, face_landmarks_5

    def detect_with_2dfan4(self, temp_vision_frame: VisionFrame, size_2dfan4: str, bounding_box: BoundingBox, face_angle: Angle) -> Tuple[FaceLandmark68, Score]:
        """
        使用 2DFAN4 模型进行 68 点人脸关键点检测。
        :param temp_vision_frame: 临时图像帧
        :param bounding_box: 人脸的边界框
        :param face_angle: 人脸的角度
        :param size_2dfan4: 2DFAN4 模型的输入尺寸，格式为 'widthxheight'
        :return: 68 点人脸关键点和关键点分数
        """
        model_size = self.unpack_resolution(size_2dfan4)
        scale = 195 / np.subtract(bounding_box[2:], bounding_box[:2]).max().clip(1, None)
        translation = (np.array(model_size) - np.add(bounding_box[2:], bounding_box[:2]) * scale) * 0.5
        rotated_matrix, rotated_size = create_rotated_matrix_and_size(face_angle, model_size)
        crop_vision_frame, affine_matrix = warp_face_by_translation(temp_vision_frame, translation, scale, model_size)
        crop_vision_frame = cv2.warpAffine(crop_vision_frame, rotated_matrix, rotated_size)
        crop_vision_frame = conditional_optimize_contrast(crop_vision_frame)
        crop_vision_frame = crop_vision_frame.transpose(2, 0, 1).astype(np.float32) / 255.0
        face_landmarker = self.inference_pool.get('2dfan4')
        try:
            with conditional_thread_semaphore():
                prediction = face_landmarker.run(None, {'input': [crop_vision_frame]})
        except Exception as e:
            print(f"Error in forward_with_2dfan4: {e}")
            return np.array([]), 0

        face_landmark_68, face_heatmap = prediction
        face_landmark_68 = face_landmark_68[:, :, :2][0] / 64 * 256
        face_landmark_68 = transform_points(face_landmark_68, cv2.invertAffineTransform(rotated_matrix))
        face_landmark_68 = transform_points(face_landmark_68, cv2.invertAffineTransform(affine_matrix))
        face_landmark_score_68 = np.amax(face_heatmap, axis=(2, 3))
        face_landmark_score_68 = np.mean(face_landmark_score_68)
        face_landmark_score_68 = np.interp(face_landmark_score_68, [0, 0.9], [0, 1])
        return face_landmark_68, face_landmark_score_68

    def process_single_image(self, file_path: str, size_yoloface: str, size_2dfan4: str, face_detector_score: float):
        """
        处理单张图像，进行人脸检测和关键点检测。
        :param file_path: 图像文件的路径
        :param size_yoloface: YOLOFace 模型的输入尺寸，格式为 'widthxheight'
        :param size_2dfan4: 2DFAN4 模型的输入尺寸，格式为 'widthxheight'
        :param face_detector_score: 人脸检测的分数阈值
        :return: 检测到的边界框、人脸分数、5 点人脸关键点、68 点人脸关键点和关键点分数
        """
        try:
            image = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is not None:
                bounding_boxes, face_scores, face_landmarks_5 = self.detect_with_yoloface(image, size_yoloface, face_detector_score)
                face_landmarks_68 = []
                face_landmark_scores_68 = []
                for bounding_box in bounding_boxes:
                    face_landmark_68, face_landmark_score_68 = self.detect_with_2dfan4(image, size_2dfan4, bounding_box, 0)
                    face_landmarks_68.append(face_landmark_68)
                    face_landmark_scores_68.append(face_landmark_score_68)
                return bounding_boxes, face_scores, face_landmarks_5, face_landmarks_68, face_landmark_scores_68
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
        return [], [], [], [], []

    def process_images_in_folder(self, folder_paths: List[str], size_yoloface: str = '640x640', size_2dfan4: str = '256x256', face_detector_score: float = 0.5, output_json: str = 'face.json'):
        """
        处理多个文件夹中的图像，进行人脸检测和关键点检测，并将结果保存到 JSON 文件中。
        :param folder_paths: 文件夹路径列表
        :param size_yoloface: YOLOFace 模型的输入尺寸，格式为 'widthxheight'
        :param size_2dfan4: 2DFAN4 模型的输入尺寸，格式为 'widthxheight'
        :param face_detector_score: 人脸检测的分数阈值
        :param output_json: 保存检测结果的 JSON 文件路径
        """
        image_files = []
        # 遍历多个文件夹路径
        for folder_path in folder_paths:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_files.append(os.path.join(root, file))

        image_result_dict = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda file: (file, self.process_single_image(file, size_yoloface, size_2dfan4, face_detector_score)), image_files))

        for file_path, (bounding_boxes, face_scores, face_landmarks_5, face_landmarks_68, face_landmark_scores_68) in results:
            image_result_dict[file_path] = {
                'bounding_boxes': [box.tolist() for box in bounding_boxes],
                'face_scores': face_scores,
                'face_landmarks_5': [landmark.tolist() for landmark in face_landmarks_5],
                'face_landmarks_68': [landmark.tolist() for landmark in face_landmarks_68],
                'face_landmark_scores_68': face_landmark_scores_68
            }

        if output_json:
            try:
                with open(output_json, 'w', encoding='utf-8') as f:
                    json.dump(image_result_dict, f, indent=4, ensure_ascii=False)
                print(f"Detection results saved to {output_json}")
            except Exception as e:
                print(f"Error saving JSON file {output_json}: {e}")
