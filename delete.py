import os
import json
import argparse


def delete_images_based_on_scores(json_file_path, landmark_score=0.9):
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            image_result_dict = json.load(f)

        for file_path, results in image_result_dict.items():
            face_scores = results.get('face_scores', [])
            face_landmark_scores_68 = results.get('face_landmark_scores_68', [])

            # 判断是否需要删除图片
            if not face_scores or (face_landmark_scores_68 and sum(face_landmark_scores_68) / len(face_landmark_scores_68) < landmark_score):
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Deleted {file_path}")
    except Exception as e:
        print(f"Error deleting images: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Delete images based on face detection scores in a JSON file.')
    parser.add_argument('json_file_path', type=str, nargs='?', default='face.json',
                        help='Path to the JSON file containing face detection results. Default is face.json.')
    parser.add_argument('--landmark_score', type=float, default=0.9,
                        help='Threshold for face landmark scores.')

    args = parser.parse_args()
    delete_images_based_on_scores(args.json_file_path, args.landmark_score)