# Assignment 1

本次作业围绕评测指标、决策树和 k 近邻分类展开，包含问答、代码实现及实验结果分析。

## 作业组成

| 部分 | 内容 | 占本次作业分数 |
| --- | --- | --- |
| [Part 1](classification/README.md) | 分类评测指标 | 20% |
| [Part 2](decision_tree/README.md) | 决策树 | 40% |
| [Part 3](k_nearest_neighbors/README.md) | k 近邻（kNN） | 40% |

**请务必仔细阅读并理解每个 Part 的 README，逐项确认作业要求，避免遗漏。** 本文说明总体安排和统一提交格式，各部分的具体任务以对应 README 为准。

## 报告要求

- 按 Part 1、Part 2、Part 3 的顺序组织报告。
- 报告须包含每个 Part 要求的代码以外的全部内容，包括相应的问答、计算过程、测试日志或截图、实验图像及结果分析。
- 保证文字、图像和截图清晰可读。
- 报告使用 PDF 格式，命名为 `report.pdf`。

## 提交要求

提交报告和作业要求完成的代码文件。将各 Part 要求的图像、日志或截图放入报告中，无需单独提交；代码只需提交以下四个文件，无需提交数据集、测试脚本、可视化脚本或其他框架文件。

请在原作业目录中完成实现与测试，提交时再将下列文件复制到提交目录。

提交目录命名为 `学号-姓名`，报告和代码文件直接放在该目录下，结构如下：

```text
学号-姓名/
├── report.pdf
├── accuracy_error.py
├── evaluation_metrics.py
├── criterion.py
└── knn_student.py
```

其中，`accuracy_error.py` 和 `evaluation_metrics.py` 对应 Part 1，`criterion.py` 对应 Part 2，`knn_student.py` 对应 Part 3。

将 `学号-姓名` 目录打包后提交至 eLearning，具体提交要求见 eLearning 上的说明。
