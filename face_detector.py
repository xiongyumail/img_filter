import os
import cv2
import numpy as np
import onnxruntime as ort
import json
import concurrent.futures
import threading
from typing import List, Tuple
from inference_utils import create_static_model_set, InferencePool, conditional_thread_semaphore, transform_points
from image_utils import conditional_optimize_contrast, warp_face_by_translation, create_rotated_matrix_and_size

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
        self.onnx_session = None
        self.input_name = None
        self.semaphore = threading.Semaphore(1)
        self.init_onnx_session(onnx_model_path_yoloface, provider, device_type)
        self.inference_pool = InferencePool(onnx_model_path_2dfan4)

    def init_onnx_session(self, onnx_model_path: str, provider: str, device_type: str):
        provider_options = [{'device_type': device_type}] if device_type else []
        self.onnx_session = ort.InferenceSession(onnx_model_path, providers=[provider], provider_options=provider_options)
        self.input_name = self.onnx_session.get_inputs()[0].name

    def get_inference_pool(self):
        return {'yoloface': self.onnx_session}

    def unpack_resolution(self, resolution: str) -> Resolution:
        try:
            width, height = map(int, resolution.split('x'))
            return width, height
        except ValueError:
            raise ValueError(f"Invalid resolution format: {resolution}. Expected 'widthxheight'.")

    def resize_frame_resolution(self, vision_frame: VisionFrame, max_resolution: Resolution) -> VisionFrame:
        height, width = vision_frame.shape[:2]
        max_width, max_height = max_resolution

        if height > max_height or width > max_width:
            scale = min(max_height / height, max_width / width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            return cv2.resize(vision_frame, (new_width, new_height))
        return vision_frame

    def prepare_detect_frame(self, temp_vision_frame: VisionFrame, face_detector_size: str) -> VisionFrame:
        face_detector_width, face_detector_height = self.unpack_resolution(face_detector_size)
        detect_vision_frame = np.zeros((face_detector_height, face_detector_width, 3))
        detect_vision_frame[:temp_vision_frame.shape[0], :temp_vision_frame.shape[1], :] = temp_vision_frame
        detect_vision_frame = (detect_vision_frame - 127.5) / 128.0
        detect_vision_frame = np.expand_dims(detect_vision_frame.transpose(2, 0, 1), axis=0).astype(np.float32)
        return detect_vision_frame

    def forward_with_yoloface(self, detect_vision_frame: VisionFrame) -> Detection:
        face_detector = self.get_inference_pool().get('yoloface')

        class ThreadSemaphore:
            def __init__(self, semaphore):
                self.semaphore = semaphore

            def __enter__(self):
                self.semaphore.acquire()

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.semaphore.release()

        with ThreadSemaphore(self.semaphore):
            try:
                detection = face_detector.run(None, {self.input_name: detect_vision_frame})
            except Exception as e:
                raise RuntimeError(f"Error during forward pass: {e}")

        return detection

    def detect_with_yoloface(self, vision_frame: VisionFrame, face_detector_size: str, face_detector_score: float) -> Tuple[List[BoundingBox], List[Score], List[FaceLandmark5]]:
        bounding_boxes = []
        face_scores = []
        face_landmarks_5 = []
        face_detector_width, face_detector_height = self.unpack_resolution(face_detector_size)
        temp_vision_frame = self.resize_frame_resolution(vision_frame, (face_detector_width, face_detector_height))
        ratio_height = vision_frame.shape[0] / temp_vision_frame.shape[0]
        ratio_width = vision_frame.shape[1] / temp_vision_frame.shape[1]
        detect_vision_frame = self.prepare_detect_frame(temp_vision_frame, face_detector_size)
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

    def detect_with_2dfan4(self, temp_vision_frame: VisionFrame, bounding_box: BoundingBox, face_angle: Angle) -> Tuple[FaceLandmark68, Score]:
        model_size = create_static_model_set('full').get('2dfan4').get('size')
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

    def process_single_image(self, file_path: str, face_detector_size: str, face_detector_score: float):
        try:
            image = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is not None:
                bounding_boxes, face_scores, face_landmarks_5 = self.detect_with_yoloface(image, face_detector_size, face_detector_score)
                face_landmarks_68 = []
                face_landmark_scores_68 = []
                for bounding_box in bounding_boxes:
                    face_landmark_68, face_landmark_score_68 = self.detect_with_2dfan4(image, bounding_box, 0)
                    face_landmarks_68.append(face_landmark_68)
                    face_landmark_scores_68.append(face_landmark_score_68)
                return bounding_boxes, face_scores, face_landmarks_5, face_landmarks_68, face_landmark_scores_68
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
        return [], [], [], [], []

    def process_images_in_folder(self, folder_path: str, face_detector_size: str = '640x640', face_detector_score: float = 0.5, output_json: str = 'face.json'):
        image_files = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(root, file))

        image_result_dict = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda file: (file, self.process_single_image(file, face_detector_size, face_detector_score)), image_files))

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