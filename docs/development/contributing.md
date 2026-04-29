# 贡献指南

## 贡献流程

### 1. Fork 和分支

```bash
git checkout -b feat/your-feature-name
```

分支命名规范：
- `feat/` — 新功能
- `fix/` — 修复 bug
- `docs/` — 文档变更
- `refactor/` — 重构
- `test/` — 测试

### 2. 开发

遵循项目的[代码规范](/development/code-style)。确保：
- 新代码有类型标注
- 公共方法有中文文档字符串
- 使用 `pathlib` 处理路径
- 使用 `np.random.default_rng()` 而非全局随机状态

### 3. 测试

为所有新功能编写测试。运行全部测试确保未引入回归：

```bash
pytest tests -q
```

### 4. 代码格式化

```bash
ruff format .
ruff check . --fix
```

### 5. 提交

提交信息使用中文，遵循 [Conventional Commits](https://www.conventionalcommits.org/) 格式：

```
feat(layers): 添加 Dropout 层

实现训练时随机置零的 Dropout 层，评估时缩放激活值。
```

### 6. 发起 Pull Request

推送到你的 fork 后发起 PR。CI 将自动运行格式检查和 lint。

---

## 如何扩展

### 添加新的网络层

1. 在 `src/nn/layers/` 下创建新文件
2. 继承 `BaseLayer`，实现 `forward()` 和 `backward()`
3. 如果有参数，覆盖 `getParameters()` 和 `getGradients()`
4. 在 `src/nn/layers/__init__.py` 中导出
5. 在 `tests/` 下添加测试

### 添加新的优化器

1. 在 `src/nn/optimizers/` 下创建新文件
2. 实现 `step(layers)` 和 `zeroGrad(layers)` 方法
3. 可选：维护内部状态（如 Momentum 的速度变量）

### 添加新的损失函数

1. 在 `src/nn/losses/` 下创建新文件
2. 实现 `forward(predictions, targets)` 和 `backward()` 方法
3. 缓存所需的值（predictions, targets, probabilities 等）

### 添加新的数据集

1. 在 `config.py` 中添加新目录和文件路径
2. 在 `DataGenerator` 中添加 `generateXxxDataset()` 方法
3. 在 `DataLoader` 中添加 `loadXxxDataset()` 方法
4. 在 `loadDataset()` 中添加新的调度分支

---

## 未来路线图

| 优先级 | 项目 |
|---|---|
| 高 | Momentum 优化器 |
| 高 | Adam 优化器 |
| 中 | Dropout 层 |
| 中 | Batch Normalization 层 |
| 中 | 学习率调度器 |
| 低 | 训练可视化（loss/accuracy 曲线） |
| 低 | 更丰富的数据集 |
| 低 | Jupyter Notebook 教程 |
