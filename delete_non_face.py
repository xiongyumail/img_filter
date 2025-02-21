import os
import json

def delete_non_face_images(folder_path, face_image_json):
    # 读取包含人脸图片路径的 JSON 文件
    try:
        with open(face_image_json, 'r', encoding='utf-8') as f:
            face_image_paths = set(json.load(f))
    except Exception as e:
        print(f"Error loading JSON file {face_image_json}: {e}")
        return

    # 遍历文件夹中的所有图片文件
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                # 如果图片路径不在人脸图片列表中，则删除该图片
                if file_path not in face_image_paths:
                    try:
                        # 删除文件时支持中文路径
                        os.remove(file_path)
                        print(f"Deleted: {file_path}")
                    except Exception as e:
                        print(f"Error deleting {file_path}: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Delete non-face images based on a JSON file containing face image paths.')
    parser.add_argument('folder_path', type=str, help='Path to the folder containing images.')
    parser.add_argument('--face_image_json', type=str, default='face.json', help='Path to the JSON file containing face image paths.')

    args = parser.parse_args()
    delete_non_face_images(args.folder_path, args.face_image_json)