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

# 初始化 ONNX 会话
def init_onnx_session(onnx_model_path: str, provider: str, device_type: str):
    """
    初始化 ONNX 会话。
    :param onnx_model_path: ONNX 模型的路径
    :param provider: 推理提供者，如 'CPUExecutionProvider'
    :param device_type: 设备类型，如 'CPU'
    :return: ONNX 会话和输入名称
    """
    provider_options = [{'device_type': device_type}] if device_type else []
    onnx_session = ort.InferenceSession(onnx_model_path, providers=[provider], provider_options=provider_options)
    input_name = onnx_session.get_inputs()[0].name
    return onnx_session, input_name

# 获取推理池
class InferencePool:
    def __init__(self, onnx_model_path_yoloface, onnx_model_path_2dfan4, provider: str, device_type: str):
        """
        初始化推理池。
        :param onnx_model_path_yoloface: YOLOFace 模型的 ONNX 路径
        :param onnx_model_path_2dfan4: 2DFAN4 模型的 ONNX 路径
        :param provider: 推理提供者，如 'CPUExecutionProvider'
        :param device_type: 设备类型，如 'CPU'
        """
        yoloface_session, yoloface_input_name = init_onnx_session(onnx_model_path_yoloface, provider, device_type)
        dfan4_session, _ = init_onnx_session(onnx_model_path_2dfan4, provider, device_type)
        self.pool = {
            'yoloface': (yoloface_session, yoloface_input_name),
            '2dfan4': dfan4_session
        }

    def get(self, key):
        """
        从推理池中获取会话。
        :param key: 模型名称，如 'yoloface' 或 '2dfan4'
        :return: 对应的会话或会话和输入名称
        """
        return self.pool.get(key)

# 条件线程信号量
def conditional_thread_semaphore():
    """
    创建条件线程信号量。
    :return: 线程信号量上下文管理器
    """
    semaphore = threading.Semaphore(1)

    class ThreadSemaphore:
        def __enter__(self):
            semaphore.acquire()

        def __exit__(self, exc_type, exc_val, exc_tb):
            semaphore.release()

    return ThreadSemaphore()

# 点的变换函数
def transform_points(points, matrix):
    """
    对输入的点进行仿射变换。
    :param points: 输入的点，形状为 (N, 2)
    :param matrix: 仿射变换矩阵，形状为 (2, 3)
    :return: 变换后的点，形状为 (N, 2)
    """
    ones = np.ones((points.shape[0], 1))
    points_homogeneous = np.hstack((points, ones))
    transformed_points = np.dot(matrix, points_homogeneous.T).T[:, :2]
    return transformed_points