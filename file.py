import os
import json
import shutil
import argparse
from collections import defaultdict
from datetime import datetime  # 添加导入

def save_output_json(results, output_json: str = 'face.json', output_full_data: bool = False):
    """
    支持跨磁盘/网络路径的智能路径压缩存储
    """
    drive_groups = defaultdict(list)
    for file_path, data in results:
        abs_path = os.path.abspath(file_path)
        drive = os.path.splitdrive(abs_path)[0]
        drive_groups[drive].append((abs_path, data))

    img_data = {}

    for drive, files in drive_groups.items():
        abs_paths = [fp for fp, _ in files]
        if not abs_paths:
            continue

        # 使用commonpath获取公共路径并确保以分隔符结尾
        common_prefix = os.path.commonpath(abs_paths)
        common_prefix = os.path.join(common_prefix, '')  # 确保目录格式

        path_tree = {}
        for abs_path, (bounding_boxes, face_scores, landmarks_5, landmarks_68, scores_68) in files:
            rel_path = os.path.relpath(abs_path, common_prefix).replace('\\', '/')
            parts = rel_path.split('/')
            
            current_level = path_tree
            for part in parts[:-1]:
                current_level = current_level.setdefault(part, {})
            
            file_data = {
                'face_scores': face_scores,
                'face_landmark_scores_68': scores_68
            } if not output_full_data else {
                'bounding_boxes': [box.tolist() for box in bounding_boxes],
                'face_scores': face_scores,
                'face_landmarks_5': [lm.tolist() for lm in landmarks_5],
                'face_landmarks_68': [lm.tolist() for lm in landmarks_68],
                'face_landmark_scores_68': scores_68
            }
            current_level[parts[-1]] = file_data

        normalized_prefix = common_prefix.replace('\\', '/')
        if not normalized_prefix.endswith('/'):
            normalized_prefix += '/'
        img_data[normalized_prefix] = path_tree

    # 创建包含新结构的数据
    current_time = datetime.now().astimezone().isoformat()
    output_data = {
        "version": "3",
        "date_created": current_time,
        "date_updated": current_time,
        "img": img_data
    }

    try:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        print(f"results saved to {output_json}")
    except Exception as e:
        print(f"Error saving JSON: {str(e)}")

def process_images_based_on_scores(output_json, landmark_score=0.9, target_path='./copied_images', 
                                  delete=False, copy=False, verbose=False):
    try:
        with open(output_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
            img_data = data.get('img', {})  # 获取img字段内容
        
        files_to_process = []

        def walk_tree(node, norm_base, current_rel_path=""):
            for key, value in node.items():
                new_rel_path = os.path.join(current_rel_path, key)
                if isinstance(value, dict):
                    if any(k in value for k in ['face_scores', 'face_landmark_scores_68']):
                        files_to_process.append((norm_base, new_rel_path, value))
                    else:
                        walk_tree(value, norm_base, new_rel_path)

        for base_path, subtree in img_data.items():  # 遍历img_data而不是data
            norm_base = os.path.normpath(base_path)
            walk_tree(subtree, norm_base)

        filtered_files = []
        for norm_base, rel_path, results in files_to_process:
            scores = results.get('face_landmark_scores_68', [])
            avg_score = sum(scores) / len(scores) if scores else 0
            if (delete and avg_score < landmark_score) or (copy and avg_score >= landmark_score):
                filtered_files.append((norm_base, rel_path, results, avg_score))

        if delete:
            for norm_base, rel_path, _, score in filtered_files:
                src_path = os.path.join(norm_base, rel_path)
                if os.path.exists(src_path):
                    os.remove(src_path)
                    if verbose:
                        print(f"Deleted [{score:.2f}]: {src_path}")

        if copy:
            for norm_base, rel_path, _, score in filtered_files:
                src_path = os.path.join(norm_base, rel_path)
                dst_path = os.path.join(target_path, rel_path)
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy2(src_path, dst_path)
                if verbose:
                    print(f"Copied [{score:.2f}]: {src_path} → {dst_path}")

    except Exception as e:
        if verbose:
            print(f"Processing error: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process images based on optimized JSON structure.')
    parser.add_argument('output_json', type=str, nargs='?', default='face.json',
                      help='Path to the JSON file containing face detection results. Default is face.json.')
    parser.add_argument('--landmark_score', type=float, default=0.9,
                      help='Threshold for face landmark scores.')
    parser.add_argument('--delete', action='store_true', 
                      help='Delete images based on face scores and landmark scores.')
    parser.add_argument('--copy', nargs='?', const='./copied_images', type=str,
                      help='Copy images based on face landmark scores. Optionally specify the target path.')
    parser.add_argument('--verbose', action='store_true', 
                      help='Print detailed information during processing.')

    args = parser.parse_args()

    if not args.delete and args.copy is None:
        print("Please specify at least one of --delete or --copy options.")
    else:
        target_path = args.copy if args.copy else './copied_images'
        process_images_based_on_scores(
            args.output_json,
            args.landmark_score,
            target_path,
            args.delete,
            bool(args.copy),
            args.verbose
        )