import argparse
from face_detector import FaceDetector
from file import process_images_based_on_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process images in a folder and save detection results to face.json.')
    parser.add_argument('folder_path', type=str, help='Path to the folder containing images.')
    parser.add_argument('--face_detector_size', type=str, default='640x640', help='Resolution for the face detector, in the format "widthxheight".')
    parser.add_argument('--face_detector_score', type=float, default=0.7, help='Score threshold for face detection.')
    parser.add_argument('--output_json', type=str, default='face.json', help='Output JSON file path for detection results.')
    parser.add_argument('--onnx_provider', type=str, default="OpenVINOExecutionProvider", help='ONNX runtime provider.')
    parser.add_argument('--device_type', type=str, default="GPU", help='Device type for the provider.')
    parser.add_argument('--onnx_model_path_yoloface', type=str, default='yoloface_8n.onnx', help='Path to the YOLOFace ONNX model.')
    parser.add_argument('--onnx_model_path_2dfan4', type=str, default='2dfan4.onnx', help='Path to the 2DFAN4 ONNX model.')
    parser.add_argument('--delete', action='store_true', help='Delete images based on face scores and landmark scores.')
    parser.add_argument('--copy', action='store_true', help='Copy images based on face landmark scores.')
    parser.add_argument('--landmark_score', type=float, default=0.9, help='Threshold for face landmark scores.')
    parser.add_argument('--target_path', type=str, default='./copied_images', help='Target path to copy the images. Default is./copied_images.')

    args = parser.parse_args()
    detector = FaceDetector(args.onnx_model_path_yoloface, args.onnx_model_path_2dfan4, args.onnx_provider, args.device_type)
    detector.process_images_in_folder(args.folder_path, args.face_detector_size, args.face_detector_score, args.output_json)

    if args.delete or args.copy:
        process_images_based_on_scores(
            args.output_json,
            args.landmark_score,
            args.target_path,
            args.delete,
            args.copy
        )