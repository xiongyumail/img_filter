import onnxruntime as ort
import threading
from collections import namedtuple
from typing import List, Tuple
import numpy as np

# 定义必要的类型别名
VisionFrame = np.ndarray
BoundingBox = np.ndarray
Score = float
FaceLandmark5 = np.ndarray
FaceLandmark68 = np.ndarray
StateManager = namedtuple('StateManager', ['get_item'])
Resolution = Tuple[int, int]
Detection = np.ndarray
Angle = float
Translation = np.ndarray
Size = Tuple[int, int]
Matrix = np.ndarray
Prediction = np.ndarray

# 假设的模型设置
def create_static_model_set(model_type):
    if model_type == 'full':
        return {'2dfan4': {'size': (256, 256)}}
    return {}

# 获取推理池
class InferencePool:
    def __init__(self, onnx_model_path_2dfan4):
        self.pool = {
            '2dfan4': ort.InferenceSession(onnx_model_path_2dfan4)
        }

    def get(self, key):
        return self.pool.get(key)

# 条件线程信号量
def conditional_thread_semaphore():
    semaphore = threading.Semaphore(1)

    class ThreadSemaphore:
        def __enter__(self):
            semaphore.acquire()

        def __exit__(self, exc_type, exc_val, exc_tb):
            semaphore.release()

    return ThreadSemaphore()

# 点的变换函数
def transform_points(points, matrix):
    ones = np.ones((points.shape[0], 1))
    points_homogeneous = np.hstack((points, ones))
    transformed_points = np.dot(matrix, points_homogeneous.T).T[:, :2]
    return transformed_points