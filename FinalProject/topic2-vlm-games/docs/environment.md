# 环境说明

评测面向 Linux、单张 NVIDIA GPU，调用 ms-swift 的 vLLM 推理后端；不调用外部 API 或裁判模型。Sokoban 生成器使用 Python，Maze 生成器使用 JDK 17 或更高版本。

## Python 依赖

`requirements-eval.txt` 记录课程开发环境已使用的评测依赖版本，`requirements-generation.txt` 记录 Python 生成器依赖。这些依赖仅约束所提供工具，不要求学生使用相同训练框架。

已有课程环境时直接使用，无需重复安装。自行准备环境时，可在独立 Python 3.12 环境安装：

```bash
python -m pip install -r requirements-eval.txt
python -m pip install -r requirements-generation.txt
```

GPU 驱动、CUDA 和 PyTorch/vLLM 的兼容性取决于运行平台，依赖文件不能替代平台环境配置。以下检查不加载模型权重：

```bash
python evaluation/infer.py doctor --model /path/to/model
python evaluate.py --model /path/to/model --dry-run
```

`doctor` 检查模型架构、依赖和框架注册情况；`--dry-run` 校验测试文件并展示正式命令。它们均不代表 GPU 推理已成功，可用 `--limit 2` 做少量样本试运行，结果会标记为非完整测试。

## 生成器

仅运行 Sokoban 时只需安装 `requirements-generation.txt`。Maze 需要：

```bash
java -version
javac -version
```

Maze 的 `generate.sh` 使用附带的 Gson JAR 编译并运行，无需联网构建。也保留 Gradle 工程，但首次使用 Gradle 可能下载工具及依赖。
