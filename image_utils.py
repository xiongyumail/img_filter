import cv2
import numpy as np
from typing import Tuple

VisionFrame = np.ndarray
Translation = np.ndarray
Size = Tuple[int, int]
Matrix = np.ndarray
Angle = float

def conditional_optimize_contrast(crop_vision_frame: VisionFrame) -> VisionFrame:
    crop_vision_frame = cv2.cvtColor(crop_vision_frame, cv2.COLOR_RGB2Lab)
    if np.mean(crop_vision_frame[:, :, 0]) < 30:
        crop_vision_frame[:, :, 0] = cv2.createCLAHE(clipLimit=2).apply(crop_vision_frame[:, :, 0])
    crop_vision_frame = cv2.cvtColor(crop_vision_frame, cv2.COLOR_Lab2RGB)
    return crop_vision_frame

def warp_face_by_translation(temp_vision_frame: VisionFrame, translation: Translation, scale: float, crop_size: Size) -> Tuple[VisionFrame, Matrix]:
    affine_matrix = np.array([[scale, 0, translation[0]], [0, scale, translation[1]]])
    crop_vision_frame = cv2.warpAffine(temp_vision_frame, affine_matrix, crop_size)
    return crop_vision_frame, affine_matrix

def create_rotated_matrix_and_size(angle: Angle, size: Size) -> Tuple[Matrix, Size]:
    rotated_matrix = cv2.getRotationMatrix2D((size[0] / 2, size[1] / 2), angle, 1)
    rotated_size = np.dot(np.abs(rotated_matrix[:, :2]), size)
    rotated_matrix[:, -1] += (rotated_size - size) * 0.5
    rotated_size = int(rotated_size[0]), int(rotated_size[1])
    return rotated_matrix, rotated_size