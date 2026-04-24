"""
config.py
项目配置模块
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


DATASETS_DIR = PROJECT_ROOT / "datasets"

XOR_DIR = DATASETS_DIR / "xor"
SPIRAL_DIR = DATASETS_DIR / "spiral"
SINE_DIR = DATASETS_DIR / "sine"

# xor的文件路径
XOR_FILE = XOR_DIR / "xor.npz"

# spiral的文件路径
SPIRAL_TRAIN_FILE = SPIRAL_DIR / "train.npz"
SPIRAL_TEST_FILE = SPIRAL_DIR / "test.npz"
SPIRAL_VALID_FILE = SPIRAL_DIR / "valid.npz"

# sine的文件路径
SINE_TRAIN_FILE = SINE_DIR / "train.npz"
SINE_TEST_FILE = SINE_DIR / "test.npz"
SINE_VALID_FILE = SINE_DIR / "valid.npz"
