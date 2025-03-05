import os
import json
import shutil
import argparse
from collections import defaultdict

def save_output_json(results, output_json: str = 'face.json', output_full_data: bool = False):
    """
    支持跨磁盘/网络路径的智能路径压缩存储
    """
    # 按磁盘/挂载点分组
    drive_groups = defaultdict(list)
    for file_path, data in results:
        abs_path = os.path.abspath(file_path)
        drive = os.path.splitdrive(abs_path)[0]  # 获取磁盘标识（如 'C:' 或 '\\server'）
        drive_groups[drive].append((abs_path, data))

    final_output = {}

    # 处理每个磁盘组
    for drive, files in drive_groups.items():
        # 获取该磁盘下的所有绝对路径
        abs_paths = [fp for fp, _ in files]
        
        # 计算该磁盘内的公共前缀
        common_prefix = os.path.commonprefix(abs_paths)
        if not common_prefix.endswith(os.sep):
            common_prefix = os.path.dirname(common_prefix) + os.sep
        
        # 构建路径树
        path_tree = {}
        for abs_path, (bounding_boxes, face_scores, landmarks_5, landmarks_68, scores_68) in files:
            # 计算该磁盘内的相对路径
            rel_path = os.path.relpath(abs_path, common_prefix)
            
            # 转换为跨平台路径格式
            rel_path = rel_path.replace('\\', '/')  # 统一使用正斜杠
            parts = rel_path.split('/')
            
            # 构建嵌套字典结构
            current_level = path_tree
            for part in parts[:-1]:
                current_level = current_level.setdefault(part, {})
            
            # 存储文件数据
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

        # 转换公共前缀为网络路径格式
        normalized_prefix = common_prefix.replace('\\', '/')
        if not normalized_prefix.endswith('/'):
            normalized_prefix += '/'
        final_output[normalized_prefix] = path_tree

    # 保存结果
    try:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=4, ensure_ascii=False)
        print(f"Cross-drive optimized results saved to {output_json}")
    except Exception as e:
        print(f"Error saving JSON: {str(e)}")

def process_images_based_on_scores(output_json, landmark_score=0.9, target_path='./copied_images', 
                                  delete=False, copy=False, verbose=False):
    try:
        with open(output_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        files_to_process = []
        
        # 增强版路径展开器
        def walk_tree(node, base_path, current_rel_path=""):
            for key, value in node.items():
                # 构建当前层级相对路径
                new_rel_path = os.path.join(current_rel_path, key)
                
                if isinstance(value, dict):
                    if any(k in value for k in ['face_scores', 'face_landmark_scores_68']):
                        # 文件节点：base_path + 完整相对路径
                        full_path = os.path.join(base_path, new_rel_path)
                        files_to_process.append( (full_path, value) )
                    else:
                        # 目录节点：继续递归
                        walk_tree(value, base_path, new_rel_path)

        # 遍历所有根路径（保持不变）
        for base_path, subtree in data.items():
            norm_base = os.path.normpath(base_path)
            walk_tree(subtree, norm_base)

        # 新增评分过滤逻辑
        filtered_files = []
        for src_path, results in files_to_process:
            scores = results.get('face_landmark_scores_68', [])
            avg_score = sum(scores)/len(scores) if scores else 0
            
            # 核心修复点：应用landmark_score阈值
            if (delete and avg_score < landmark_score) or (copy and avg_score >= landmark_score):
                filtered_files.append( (src_path, results, avg_score) )

        # 处理删除操作
        if delete:
            for path, _, score in filtered_files:
                if os.path.exists(path):
                    os.remove(path)
                    if verbose:
                        print(f"Deleted [{score:.2f}]: {path}")

        # 处理复制操作
        if copy:
            for src_path, _, score in filtered_files:
                rel_path = os.path.relpath(src_path, norm_base)
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
    # 保持主函数不变，参数处理逻辑兼容原有用法
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