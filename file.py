import os
import json
import shutil
import argparse


def process_images_based_on_scores(json_file_path, landmark_score=0.9, target_path='./copied_images', delete=False, copy=False, verbose=False):
    try:
        target_path = os.path.abspath(target_path)
        with open(json_file_path, 'r', encoding='utf-8') as f:
            image_result_dict = json.load(f)

        files_to_delete = []
        files_to_copy = []

        for file_path, results in image_result_dict.items():
            face_scores = results.get('face_scores', [])
            face_landmark_scores_68 = results.get('face_landmark_scores_68', [])

            if face_landmark_scores_68:
                avg_landmark_score = sum(face_landmark_scores_68) / len(face_landmark_scores_68)
            else:
                avg_landmark_score = 0

            if delete and (not face_scores or avg_landmark_score < landmark_score):
                if os.path.exists(file_path):
                    files_to_delete.append(file_path)
            if copy and face_landmark_scores_68 and avg_landmark_score > landmark_score:
                parent_dir_name = os.path.basename(os.path.dirname(file_path))
                target_sub_dir = os.path.join(target_path, parent_dir_name)
                file_name = os.path.basename(file_path)
                target_file_path = os.path.join(target_sub_dir, file_name)
                files_to_copy.append((file_path, target_file_path))

        # 批量删除文件
        for file in files_to_delete:
            os.remove(file)
            if verbose:
                print(f"Deleted {file}")

        # 批量创建目录和拷贝文件
        for source, target in files_to_copy:
            target_dir = os.path.dirname(target)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            shutil.copyfile(source, target)
            if verbose:
                print(f"Copied {source} to {target}")

    except Exception as e:
        if verbose:
            print(f"Error processing images: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process images based on face landmark scores in a JSON file.')
    parser.add_argument('json_file_path', type=str, nargs='?', default='face.json',
                        help='Path to the JSON file containing face detection results. Default is face.json.')
    parser.add_argument('--landmark_score', type=float, default=0.9,
                        help='Threshold for face landmark scores.')
    parser.add_argument('--delete', action='store_true', help='Delete images based on face scores and landmark scores.')
    parser.add_argument('--copy', nargs='?', const='./copied_images', type=str,
                        help='Copy images based on face landmark scores. Optionally specify the target path.')
    parser.add_argument('--verbose', action='store_true', help='Print detailed information during processing.')

    args = parser.parse_args()

    if not args.delete and args.copy is None:
        print("Please specify at least one of --delete or --copy options.")
    else:
        target_path = args.copy if args.copy else './copied_images'
        process_images_based_on_scores(
            args.json_file_path,
            args.landmark_score,
            target_path,
            args.delete,
            bool(args.copy),
            args.verbose
        )