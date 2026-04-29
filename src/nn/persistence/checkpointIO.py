"""
模型检查点读写模块

1. 保存模型参数到 .npz 文件
2. 从 .npz 文件加载模型参数
3. 支持顺序模型的参数持久化
"""

from pathlib import Path

import numpy as np

from src.nn.models.sequentialModel import SequentialModel


class CheckpointIO:
    """
    模型检查点读写器
    """

    def saveCheckpoint(self, model: SequentialModel, filePath: str | Path) -> None:
        """
        保存模型参数

        Args:
            model (SequentialModel): 要保存的顺序模型
            filePath (str | Path): 检查点文件路径
        """
        checkpointPath = Path(filePath)

        # 确保使用 .npz 扩展名
        if checkpointPath.suffix != ".npz":
            raise ValueError("检查点文件必须使用 .npz 扩展名")

        # 确保目录存在
        checkpointPath.parent.mkdir(parents=True, exist_ok=True)

        parameterDict: dict[str, np.ndarray] = {}
        parameterCount = 0

        for layerIndex, layer in enumerate(model.layers):
            parameters = layer.getParameters()

            for parameterIndex, parameter in enumerate(parameters):
                parameterName = f"layer_{layerIndex}_param_{parameterIndex}"
                parameterDict[parameterName] = parameter.copy()
                parameterCount += 1

        parameterDict["parameter_count"] = np.array([parameterCount], dtype=np.int64)

        np.savez(checkpointPath, **parameterDict)

    def loadCheckpoint(self, model: SequentialModel, filePath: str | Path) -> None:
        """
        加载模型参数

        Args:
            model (SequentialModel): 要写入参数的顺序模型
            filePath (str | Path): 检查点文件路径
        """
        checkpointPath = Path(filePath)

        if not checkpointPath.exists():
            raise FileNotFoundError(f"检查点文件不存在: {checkpointPath}")

        if checkpointPath.suffix != ".npz":
            raise ValueError("检查点文件必须使用 .npz 扩展名")

        # 计算模型中参数的总数量
        expectedParameterCount = 0
        for layer in model.layers:
            expectedParameterCount += len(layer.getParameters())

        with np.load(checkpointPath, allow_pickle=False) as checkpointData:
            if "parameter_count" not in checkpointData.files:
                raise ValueError("检查点文件缺少 parameter_count 字段")

            savedParameterCount = int(checkpointData["parameter_count"][0])

            if savedParameterCount != expectedParameterCount:
                raise ValueError(
                    f"参数数量不匹配: 保存的参数数量为 {savedParameterCount}, "
                    f"当前模型期望参数数量为 {expectedParameterCount}"
                )

            # 实际加载的参数数量，用于验证与声明数量的一致性
            actualLoadedCount = 0

            for layerIndex, layer in enumerate(model.layers):
                parameters = layer.getParameters()

                for parameterIndex, parameter in enumerate(parameters):
                    parameterName = f"layer_{layerIndex}_param_{parameterIndex}"

                    if parameterName not in checkpointData.files:
                        raise ValueError(f"检查点文件缺少参数: {parameterName}")

                    savedParameter = checkpointData[parameterName]

                    if savedParameter.shape != parameter.shape:
                        raise ValueError(
                            f"参数形状不匹配: {parameterName} 的保存形状为 "
                            f"{savedParameter.shape}, 当前模型形状为 {parameter.shape}"
                        )

                    parameter[...] = savedParameter
                    actualLoadedCount += 1

            if actualLoadedCount != savedParameterCount:
                raise ValueError(
                    f"实际加载参数数量不匹配: 实际加载 {actualLoadedCount}, "
                    f"检查点声明 {savedParameterCount}"
                )
