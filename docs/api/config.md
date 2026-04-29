# Config — 配置 API

`config.py` — 项目配置模块，使用 `pathlib` 管理所有路径。

## 路径常量

所有路径均通过 `pathlib.Path` 定义，根目录为 `config.py` 所在目录。

```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# 数据集根目录
DATASETS_DIR = PROJECT_ROOT / "datasets"

# XOR 数据集
XOR_DIR = DATASETS_DIR / "xor"
XOR_FILE = XOR_DIR / "xor.npz"

# Spiral 数据集
SPIRAL_DIR = DATASETS_DIR / "spiral"
SPIRAL_TRAIN_FILE = SPIRAL_DIR / "train.npz"
SPIRAL_VALID_FILE = SPIRAL_DIR / "valid.npz"
SPIRAL_TEST_FILE = SPIRAL_DIR / "test.npz"

# Sine 数据集
SINE_DIR = DATASETS_DIR / "sine"
SINE_TRAIN_FILE = SINE_DIR / "train.npz"
SINE_VALID_FILE = SINE_DIR / "valid.npz"
SINE_TEST_FILE = SINE_DIR / "test.npz"
```

## 路径说明

| 路径常量 | 值（相对 PROJECT_ROOT） | 说明 |
|---|---|---|
| `DATASETS_DIR` | `datasets/` | 数据集根目录 |
| `XOR_FILE` | `datasets/xor/xor.npz` | XOR 数据集（4 样本，无划分） |
| `SPIRAL_TRAIN_FILE` | `datasets/spiral/train.npz` | Spiral 训练集（2100 样本） |
| `SPIRAL_VALID_FILE` | `datasets/spiral/valid.npz` | Spiral 验证集（450 样本） |
| `SPIRAL_TEST_FILE` | `datasets/spiral/test.npz` | Spiral 测试集（450 样本） |
| `SINE_TRAIN_FILE` | `datasets/sine/train.npz` | Sine 训练集（420 样本） |
| `SINE_VALID_FILE` | `datasets/sine/valid.npz` | Sine 验证集（90 样本） |
| `SINE_TEST_FILE` | `datasets/sine/test.npz` | Sine 测试集（90 样本） |

## 使用方式

```python
import config

# 使用 pathlib 路径
dataDir = config.DATASETS_DIR
xorFile = config.XOR_FILE

# 路径判断
if xorFile.exists():
    data = np.load(xorFile)
```
