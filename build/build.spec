import sys
import os
from pathlib import Path

block_cipher = None

# 自动获取当前Python环境的路径（适用于conda和virtualenv）
conda_env = sys.prefix

# 检查目录是否存在
def check_path_exists(path):
    return os.path.exists(path)

a = Analysis(
    ['../main.py'],
    pathex=['.'],
    binaries=[
        # 核心OpenVINO库
        (f'{conda_env}/Library/bin/openvino.dll', '.'),
        (f'{conda_env}/Library/bin/openvino_c.dll', '.'),
        (f'{conda_env}/Library/bin/openvino_ir_frontend.dll', '.'),
        (f'{conda_env}/Library/bin/openvino_onnx_frontend.dll', '.'),
        (f'{conda_env}/Library/bin/openvino_paddle_frontend.dll', '.'),
        (f'{conda_env}/Library/bin/openvino_pytorch_frontend.dll', '.'),
        (f'{conda_env}/Library/bin/openvino_tensorflow_frontend.dll', '.'),
        (f'{conda_env}/Library/bin/openvino_tensorflow_lite_frontend.dll', '.'),
        
        # OpenVINO插件
        (f'{conda_env}/Library/bin/openvino-2025.0.0/openvino_auto_batch_plugin.dll', '.'),
        (f'{conda_env}/Library/bin/openvino-2025.0.0/openvino_auto_plugin.dll', '.'),
        (f'{conda_env}/Library/bin/openvino-2025.0.0/openvino_hetero_plugin.dll', '.'),
        (f'{conda_env}/Library/bin/openvino-2025.0.0/openvino_intel_cpu_plugin.dll', '.'),
        (f'{conda_env}/Library/bin/openvino-2025.0.0/openvino_intel_gpu_plugin.dll', '.'),
    ],
    datas=[('version.txt', '.')],
    hiddenimports=[
        'openvino',
        'openvino.runtime',
        'openvino.tools',
        'onnxruntime_openvino',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# 添加share目录（如果存在）
openvino_share_path = f'{conda_env}/Library/share/openvino'
if check_path_exists(openvino_share_path):
    a.datas.append((openvino_share_path, 'openvino'))

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    name='img_filter',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)