import argparse
import json
import os
import subprocess
from face_detector import FaceDetector
from file import process_images_based_on_scores, save_output_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process images in one or more folders and save detection results to face.json.')
    parser.add_argument('--input_json', type=str, help='Path to the JSON file containing folder paths.')
    parser.add_argument('folder_path', type=str, nargs='*', help='Path to one or more folders containing images.')
    parser.add_argument('--size_yoloface', type=str, default='640x640', help='Resolution for the YOLOFace detector, in the format "widthxheight".')
    parser.add_argument('--size_2dfan4', type=str, default='256x256', help='Resolution for the 2DFAN4 detector, in the format "widthxheight".')
    parser.add_argument('--face_detector_score', type=float, default=0.7, help='Score threshold for face detection.')
    parser.add_argument('--output_json', type=str, default='face.json', help='Output JSON file path for detection results.')
    parser.add_argument('--output_full_data', action='store_true', help='Output full data including bounding boxes and landmarks. Default is False.')
    parser.add_argument('--onnx_provider', type=str, default="OpenVINOExecutionProvider", help='ONNX runtime provider.')
    parser.add_argument('--device_type', type=str, default="GPU", help='Device type for the provider.')
    parser.add_argument('--onnx_model_path_yoloface', type=str, default='models/yoloface_8n.onnx', help='Path to the YOLOFace ONNX model.')
    parser.add_argument('--onnx_model_path_2dfan4', type=str, default='models/2dfan4.onnx', help='Path to the 2DFAN4 ONNX model.')
    parser.add_argument('--delete', action='store_true', help='Delete images based on face scores and landmark scores.')
    parser.add_argument('--copy', nargs='?', const='./copied_images', type=str,
                        help='Copy images based on face landmark scores. Optionally specify the target path.')
    parser.add_argument('--landmark_score', type=float, default=0.9, help='Threshold for face landmark scores.')
    parser.add_argument('--display', action='store_true', help='Display processed image information using Flask app.')

    args = parser.parse_args()

    folder_paths = []
    if args.input_json:
        try:
            with open(args.input_json, 'r', encoding='utf-8') as f:
                folder_paths = json.load(f)
        except FileNotFoundError:
            print(f"Error: The file {args.input_json} was not found.")
        except json.JSONDecodeError:
            print(f"Error: Failed to decode the JSON file {args.input_json}.")
    elif args.folder_path:
        folder_paths = args.folder_path
    else:
        print("Error: No folder paths were provided. Please specify either --input_json or folder_path.")
        exit(1)

    detector = FaceDetector(args.onnx_model_path_yoloface, args.onnx_model_path_2dfan4, args.onnx_provider, args.device_type)
    score_results = detector.process_images(
        folder_paths, 
        size_yoloface=args.size_yoloface,
        size_2dfan4=args.size_2dfan4,
        face_detector_score=args.face_detector_score
    )

    save_output_json(
        results=score_results,
        output_json=args.output_json,
        output_full_data=args.output_full_data
    )

    if args.delete or args.copy:
        target_path = args.copy if args.copy else './copied_images'
        process_images_based_on_scores(
            args.output_json,
            args.landmark_score,
            target_path,
            args.delete,
            bool(args.copy)
        )

    if args.display:  # 检查 --display 选项是否被设置
        try:
            # 更新为调用子模块中的 display.py
            subprocess.run(["python", "img_display/display.py", "--input_json", os.path.abspath(args.output_json)], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: Failed to start img_display/display.py. Error: {e}")