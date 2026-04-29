# Simple-Neural-Network

一个基于 `NumPy` 从零实现的简单神经网络项目。

这个项目的目标不是追求复杂功能，而是把神经网络从
数据准备、模型搭建、前向传播、反向传播、损失计算、
参数更新、训练评估到模型保存这一整条主流程清晰地实现出来。

## 项目状态

已经完成，已具备完整的基础训练链路：

- 支持生成并加载示例数据集
- 支持手写网络层、激活函数、损失函数与优化器
- 支持顺序模型组织
- 支持分类与回归训练
- 支持 `mini-batch` 训练
- 支持分类准确率与回归均方误差统计
- 支持模型参数保存与加载
- 支持完整测试

## 已实现能力

### 数据集

- `xor`
  - 二分类任务
  - 用于验证网络学习非线性关系的能力
- `spiral`
  - 三分类二维螺旋任务
  - 用于验证多分类训练流程
- `sine`
  - 一维回归任务
  - 用于验证回归训练流程

### 网络层

- `BaseLayer`
- `LinearLayer`
- `ReLULayer`
- `SigmoidLayer`
- `TanhLayer`

### 损失函数

- `CrossEntropyLoss`
- `MSELoss`

### 模型

- `SequentialModel`

### 优化器

- `SGDOptimizer`

### 训练流程

- `Trainer`
- `calculateAccuracy`
- `calculateMeanSquaredError`

### 持久化

- `CheckpointIO`

## 项目结构

```text
Simple-Neural-Network/
├── config.py
├── main.py
├── requirements.txt
├── pyproject.toml
├── datasets/
│   ├── xor/
│   ├── spiral/
│   └── sine/
├── src/
│   └── nn/
│       ├── data/
│       │   ├── dataGenerator.py
│       │   └── dataLoader.py
│       ├── layers/
│       │   ├── baseLayer.py
│       │   ├── linearLayer.py
│       │   └── activationLayer.py
│       ├── losses/
│       │   ├── crossEntropyLoss.py
│       │   └── mseLoss.py
│       ├── models/
│       │   └── sequentialModel.py
│       ├── optimizers/
│       │   └── sgdOptimizer.py
│       ├── training/
│       │   ├── trainer.py
│       │   └── metrics.py
│       └── persistence/
│           └── checkpointIO.py
└── tests/
    ├── testDataGenerator.py
    ├── testDatasetLoader.py
    ├── testLinearLayer.py
    ├── testActivationLayer.py
    ├── testCrossEntropyLoss.py
    ├── testMSELoss.py
    ├── testSequentialModel.py
    ├── testSgdOptimizer.py
    ├── testMetrics.py
    ├── testTrainer.py
    └── testCheckpointIO.py
```

## 环境要求

- Python `>= 3.11`
- `numpy >= 2.4.4`

开发依赖：

- `pytest`
- `ruff`

## 安装依赖

```bash
pip install -r requirements.txt
pre-commit install
```

如果你使用虚拟环境，先激活再安装依赖即可。

## 运行方式

项目统一通过根目录 `main.py` 作为入口。

### 运行 XOR 分类任务

```bash
python main.py --task xor
```

### 运行 Spiral 分类任务

```bash
python main.py --task spiral
```

### 运行 Sine 回归任务

```bash
python main.py --task sine
```

运行时会自动：

1. 检查数据集是否存在
2. 如果不存在则自动生成
3. 加载对应任务的数据
4. 根据 `config.py` 中的任务配置构建模型
5. 执行训练并打印训练日志
6. 输出测试集结果

## 配置说明

项目的路径配置和任务超参数统一放在根目录 [config.py](/d:/Nayey/Code/NayukiChiba/Simple-Neural-Network/config.py) 中。

当前超参数按任务类型分类存储：

- `CLASSIFICATION_TASK_CONFIGS`
- `REGRESSION_TASK_CONFIGS`

配置内容包括：

- `epochCount`
- `batchSize`
- `learningRate`
- `hiddenDims`
- `activation`

## 当前默认训练效果

按当前默认配置，本地验证结果大致如下：

- `xor`
  - 测试集 `accuracy` 可达到 `1.0`
- `spiral`
  - 测试集 `accuracy` 可达到约 `0.9978`
- `sine`
  - 测试集 `mse` 可稳定下降

## 代码示例

### 构建模型

```python
from src.nn.layers import LinearLayer, ReLULayer
from src.nn.models.sequentialModel import SequentialModel

model = SequentialModel(
    [
        LinearLayer(inputDim=2, outputDim=64, randomSeed=42),
        ReLULayer(),
        LinearLayer(inputDim=64, outputDim=64, randomSeed=43),
        ReLULayer(),
        LinearLayer(inputDim=64, outputDim=3, randomSeed=44),
    ]
)
```

### 训练模型

```python
from src.nn.losses import CrossEntropyLoss
from src.nn.optimizers import SGDOptimizer
from src.nn.training import Trainer

trainer = Trainer(
    model=model,
    lossFunction=CrossEntropyLoss(),
    optimizer=SGDOptimizer(learning_rate=0.03),
    taskType="classification",
    batchSize=64,
    shuffle=True,
    randomSeed=42,
)
```

### 保存与加载模型

```python
from src.nn.persistence import CheckpointIO

checkpointIO = CheckpointIO()
checkpointIO.saveCheckpoint(model, "checkpoints/model.npz")
checkpointIO.loadCheckpoint(model, "checkpoints/model.npz")
```

## 测试

运行全部测试：

```bash
pytest tests -q
```

按模块运行示例：

```bash
pytest tests/testLinearLayer.py -q
pytest tests/testActivationLayer.py -q
pytest tests/testCrossEntropyLoss.py -q
pytest tests/testMSELoss.py -q
pytest tests/testSequentialModel.py -q
pytest tests/testSgdOptimizer.py -q
pytest tests/testMetrics.py -q
pytest tests/testTrainer.py -q
```

## 设计原则

- 显式优于隐式
- 简单优于复杂
- 可读性优先
- 模块职责清晰
- 先保证主流程正确，再考虑扩展功能

## 后续可扩展方向

虽然第一版已经完成，但后续仍然可以继续扩展：

- 更多优化器，例如 `Momentum`、`Adam`
- 更多层，例如 `Dropout`、`BatchNorm`
- 学习率调度器
- 可视化训练曲线
- 更丰富的真实数据集支持

## 总结

这个项目已经完成了一个“从零实现简单神经网络”的完整最小闭环：

- 有数据
- 有模型
- 有训练
- 有评估
- 有测试
- 有保存加载

适合作为后续继续扩展深度学习基础组件的起点，也适合作为手写神经网络的学习项目。
