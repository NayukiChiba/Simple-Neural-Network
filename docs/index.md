---
layout: home

hero:
  name: Simple Neural Network
  text: 基于 NumPy 从零实现神经网络
  tagline: 从数学原理到代码实战 — 覆盖线性层、激活函数、损失函数、反向传播、梯度下降的完整训练管线
  actions:
    - theme: brand
      text: 快速开始
      link: /guide/getting-started
    - theme: alt
      text: GitHub
      link: https://github.com/NayukiChiba/Simple-Neural-Network

features:
  - icon: 📐
    title: 数学优先
    details: 每个模块从数学定义开始，逐步推导到代码实现。包含完整的 LaTeX 公式——前向传播、链式法则求导、梯度下降更新公式。
  - icon: 🔍
    title: 代码透明
    details: 仅依赖 NumPy 完成所有矩阵运算。每个 `forward()` 和 `backward()` 都可逐行追踪，无黑盒。
  - icon: 🔗
    title: 全流程覆盖
    details: 数据生成 → 模型构建 → 前向传播 → 损失计算 → 反向传播 → 参数更新 → 评估 → 持久化，完整闭环。

  - icon: 🧪
    title: 充分测试
    details: 10 个单元测试文件覆盖所有模块。XOR 准确率 1.0，Spiral 分类准确率 ~0.998。
  - icon: 📦
    title: 零外部深度学习依赖
    details: 不需要 PyTorch、TensorFlow。仅需 NumPy >= 2.4.4，Python >= 3.11。
  - icon: 📖
    title: 适合学习
    details: 代码注释清晰，模块职责单一。适合从零理解神经网络核心机制。
---

## 项目总览

**Simple Neural Network** 是一个纯 NumPy 实现的神经网络练习项目，旨在以最清晰的方式展示神经网络的核心计算流程。

### 已实现模块

| 模块 | 实现 |
|---|---|
| 网络层 | `LinearLayer`（全连接，Xavier 初始化）、`ReLULayer`、`SigmoidLayer`、`TanhLayer` |
| 损失函数 | `MSELoss`（回归）、`CrossEntropyLoss`（分类，内置 Softmax） |
| 模型 | `SequentialModel`（顺序容器，继承 BaseLayer） |
| 优化器 | `SGDOptimizer`（随机梯度下降） |
| 训练器 | `Trainer`（mini-batch、shuffle、eval、predict、epoch 日志） |
| 评估指标 | 分类准确率、回归均方误差 |
| 数据 | `DataGenerator`（XOR/Spiral/Sine）、`DataLoader` |
| 持久化 | `CheckpointIO`（模型参数 .npz 读写） |

### 快速命令

```bash
# 生成数据集
python main.py

# 运行全部测试
pytest tests -q

# 启动文档开发服务器
npm run docs:dev

# 构建文档
npm run docs:build
```
