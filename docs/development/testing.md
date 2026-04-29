# 测试

## 测试框架

本项目使用 [pytest](https://docs.pytest.org/) 进行单元测试。`tests/` 目录下共有 **11 个测试文件**，覆盖所有核心模块。

## 测试结构

| 测试文件 | 测试模块 | 测试项数 |
|---|---|---|
| `testBaseLayer.py` | `BaseLayer` 接口 | 状态管理、模式切换 |
| `testLinearLayer.py` | `LinearLayer` | 初始化、前向、反向、zeroGrad、异常 |
| `testActivationLayer.py` | `ReLU/Sigmoid/Tanh` | 前向、反向、无参数、异常 |
| `testMSELoss.py` | `MSELoss` | 前向、反向、缓存、异常 |
| `testCrossEntropyLoss.py` | `CrossEntropyLoss` | 前向、反向、数值稳定、异常 |
| `testSequentialModel.py` | `SequentialModel` | 前向、反向、添加层、模式传播 |
| `testSgdOptimizer.py` | `SGDOptimizer` | step、zeroGrad、多层、异常 |
| `testTrainer.py` | `Trainer` | createBatches、trainStep、evaluate、fit、predict |
| `testMetrics.py` | 评估指标 | 准确率、MSE、标签转换 |
| `testCheckpointIO.py` | `CheckpointIO` | 保存、加载、校验、异常 |
| `testDataGenerator.py` | `DataGenerator` | XOR/Spiral/Sine 生成、划分 |
| `testDatasetLoader.py` | `DataLoader` | 加载、验证、异常 |

## 运行命令

```bash
# 运行全部测试
pytest tests -q

# 按模块运行
pytest tests/testLinearLayer.py -q
pytest tests/testTrainer.py -q

# 运行单个测试函数
pytest tests/testLinearLayer.py::test_forward_with_bias -q

# 显示详细输出
pytest tests -v

# 运行并显示 print 输出
pytest tests -v -s
```

## 测试模式

### 1. 数学正确性验证

使用 `np.testing.assert_allclose` 对比代码输出与手动计算的期望值：

```python
def test_forward_with_bias():
    layer = LinearLayer(inputDim=2, outputDim=3, randomSeed=42)
    # 手动设定权重以便精确验证
    layer.weights = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    layer.bias = np.array([[0.01, 0.02, 0.03]])
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    output = layer.forward(x)
    # 手工计算: X @ W + b
    expected = np.array([
        [1*0.1+2*0.4+0.01, 1*0.2+2*0.5+0.02, 1*0.3+2*0.6+0.03],
        [3*0.1+4*0.4+0.01, 3*0.2+4*0.5+0.02, 3*0.3+4*0.6+0.03],
    ])
    np.testing.assert_allclose(output, expected)
```

### 2. 异常测试

验证非法输入时正确抛出异常：

```python
def test_invalid_input_dim():
    with pytest.raises(ValueError):
        LinearLayer(inputDim=0, outputDim=5)
```

### 3. 状态测试

验证缓存、模式标记等内部状态：

```python
def test_sigmoid_caching():
    layer = SigmoidLayer()
    x = np.array([0.5, -0.3])
    layer.forward(x)
    assert layer.inputCache is not None
    assert layer.outputCache is not None
```

### 4. 隔离测试

使用 `tmp_path` 和 `monkeypatch` 确保文件 I/O 测试不污染真实文件系统：

```python
def test_generate_xor(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "XOR_FILE", tmp_path / "xor.npz")
    generator = DataGenerator(seed=42)
    generator.generateXorDataset()
    assert (tmp_path / "xor.npz").exists()
```

## 测试哲学

- **每一行代码都应该有对应的测试**：包括正向路径和异常路径
- **数学验证优先**：用独立的手工计算验证代码结果
- **确定性**：所有测试通过固定随机种子保证可复现
- **隔离性**：每个测试不依赖其他测试的执行结果
