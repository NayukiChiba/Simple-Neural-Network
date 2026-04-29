# 快速开始

## 环境要求

| 项 | 说明 |
|---|---|
| Python | `>= 3.11` |
| 运行依赖 | `numpy >= 2.4.4` |
| 开发依赖 | `pytest`（测试）、`ruff`（代码格式化与检查） |
| Node.js | `>= 18`（仅构建文档时需要） |

## 安装

### 1. 克隆仓库

```bash
git clone https://github.com/NayukiChiba/Simple-Neural-Network.git
cd Simple-Neural-Network
```

### 2. 创建虚拟环境（推荐）

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 4. 安装开发依赖（可选）

```bash
pip install pytest ruff
```

## 第一个任务：生成数据集

```bash
python main.py
```

运行后，`datasets/` 目录下将生成以下文件：

```text
datasets/
├── xor/
│   └── xor.npz              # XOR 二分类（4 个样本）
├── spiral/
│   ├── train.npz             # 训练集（2100 样本）
│   ├── valid.npz             # 验证集（450 样本）
│   └── test.npz              # 测试集（450 样本）
└── sine/
    ├── train.npz             # 训练集（420 样本）
    ├── valid.npz             # 验证集（90 样本）
    └── test.npz              # 测试集（90 样本）
```

## 运行测试

```bash
# 运行全部 11 个测试文件
pytest tests -q

# 按模块运行
pytest tests/testLinearLayer.py -q
pytest tests/testActivationLayer.py -q
pytest tests/testTrainer.py -q
```

## 项目结构

```text
Simple-Neural-Network/
├── main.py                      # 数据生成入口
├── config.py                    # 路径与超参数配置
├── requirements.txt             # Python 运行依赖
├── package.json                 # Node 依赖（VitePress 文档）
│
├── src/nn/                      # 核心源码
│   ├── data/                    #   数据生成与加载
│   ├── layers/                  #   网络层（Base + Linear + Activation）
│   ├── losses/                  #   损失函数（MSE + CrossEntropy）
│   ├── models/                  #   顺序模型容器
│   ├── optimizers/              #   SGD 优化器
│   ├── training/                #   训练器 + 评估指标
│   └── persistence/             #   模型检查点读写
│
├── tests/                       # 单元测试（11 个文件）
├── docs/                        # VitePress 文档
└── datasets/                    # 生成的数据集
```

## 下一步

- 查看 [快速示例](/guide/quick-example) 了解端到端训练流程
- 从 [数学基础 — 感知机与线性变换](/math/perceptron) 开始深入理解原理
- 查阅 [API 参考](/api/layers) 了解每个模块的完整接口
