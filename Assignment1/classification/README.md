# Assignment 1 - Part 1: 分类评测指标

本实验围绕分类任务的评测指标展开，包含问答和代码实现两部分。你需要回答指标定义与计算题，补全指标函数，并运行给定脚本进行基础自测。

分数占比 - 20%

仓库主要文件：`classification/`

- 基础指标实现：`accuracy_error.py`
- 二分类指标实现：`evaluation_metrics.py`
- 测试脚本：`test.py`

本部分不依赖额外数据集，`test.py` 内置了基础自测样例。

## 实验目标

- 理解并实现基础评测指标
    - accuracy_score（准确率）
    - mean_squared_error（均方误差）
- 理解并实现二分类评测指标
    - precision、recall、f1（基于 TP/FP/FN/TN）
- 使用测试脚本验证实现是否正确

## 实验内容

### 一、问答

#### 1. 指标定义

用自己的话简要说明以下指标的含义及使用场景，填写下表。

| 指标 | 含义 | 使用场景 |
| --- | --- | --- |
| Accuracy | | |
| MSE | | |
| Precision | | |
| Recall | | |
| F1 | | |

#### 2. 分类指标计算

给定以下真实标签和预测标签，其中 `1` 为正类，`0` 为负类：

```python
Y_true = [1, 0, 1, 1, 0, 1, 0, 1, 1]
Y_predict = [1, 1, 0, 0, 0, 1, 0, 0, 0]
```

请计算 Accuracy、混淆矩阵中的 TP、FP、FN、TN，以及 Precision、Recall 和 F1。

**请写出各项的计算过程，不能只给出最终答案。**

### 二、代码实现

完成代码中的 `TODO` 部分：

- 在 `accuracy_error.py` 中实现：
    - `accuracy_score(y_true, y_pred)`
    - `mean_squared_error(y_true, y_pred)`

- 在 `evaluation_metrics.py` 中实现二分类指标：
    - `precision_score(y_true, y_pred)`
    - `recall_score(y_true, y_pred)`
    - `f1_score(y_true, y_pred)`

已提供辅助函数 `_binary_counts` 与常量 `EPS`。完成实现后，删除或注释占位的 `raise NotImplementedError(...)`。

## 评测与运行

在 `test.py` 中填写姓名和学号，然后在本目录运行：

```bash
python test.py
```

脚本将打印期望值与实际值，并标记 `PASS` / `FAIL`。在所提供的样例下：

- Accuracy 期望为 `0.75`。
- MSE 期望约为 `0.6667`。
- Precision、Recall 和 F1 期望各为 `0.50`。

本脚本仅用于基础自测，通过此处的小样例不代表实现一定完全正确。

## 提交内容

- 提交问答部分的回答，包括填写后的指标定义表格和分类指标计算过程。
- 提交完成后的 `accuracy_error.py` 与 `evaluation_metrics.py`。
- 附上一次 `test.py` 运行的截图或日志（显示 PASS 结果）。
