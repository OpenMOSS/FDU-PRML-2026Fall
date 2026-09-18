# Assignment 1 - Part 2: 决策树

本实验围绕决策树分类展开，要求实现四种分裂准则，并在 UCI Wine 三分类数据集上训练和评估决策树。

分数占比 - 40%

仓库主要文件：`decision_tree/`

- 分裂准则实现：`criterion.py`
- 决策树主体：`decision_tree.py`
- 测试与可视化：`test_decision_tree.py`、`viz_tree.py`
- 数据集：`dataset/wine_train.csv`、`dataset/wine_test.csv`

## 实验目标

- 理解并实现四种分裂度量：
  - Information Gain（信息增益）
  - Information Gain Ratio（信息增益率）
  - Gini Index（Gini 指数）
  - Classification Error（分类误差率）
- 理解多分类标签在决策树分裂中的处理方式。
- 观察不同分裂准则对树深度、叶节点数量和测试准确率的影响。

## 数据说明

本实验使用 UCI Wine 数据集。数据包含 178 条样本、13 个数值特征和 3 个类别。原始类别标签已经转换为整数：`0`、`1`、`2`。

数据已经按照固定随机种子进行分层划分：

- `wine_train.csv`：142 条样本
- `wine_test.csv`：36 条样本

两个 CSV 文件的最后一列是 `label`，其余列是数值特征。测试脚本会自动读取这两个文件，不需要联网下载数据。

## 实验内容

在 `criterion.py` 中完成四个函数的 `TODO` 部分：

- `__info_gain(y, l_y, r_y)`：父节点熵减去子节点加权熵。
- `__info_gain_ratio(y, l_y, r_y)`：信息增益除以分裂信息。
- `__gini_index(y, l_y, r_y)`：分裂前 Gini 减去分裂后加权 Gini。
- `__error_rate(y, l_y, r_y)`：分裂前分类误差减去分裂后加权分类误差。

完成实现后，删除或注释占位的 `raise NotImplementedError(...)`。

## 运行测试

在本目录运行：

```bash
python test_decision_tree.py
```

脚本会分别使用四种 criterion 训练决策树，并检查测试集准确率。每种准则的最低准确率要求为 `0.85`，同时会在 `output/` 中保存对应的决策树图像：

脚本还会进行基本结构检查：特征重要性的维度和归一化结果，以及树的深度和叶节点数量。测试不会要求某一种固定的树结构。

```text
output/wine_info_gain.png
output/wine_info_gain_ratio.png
output/wine_gini.png
output/wine_error_rate.png
```

## 提交内容

- 完成后的 `criterion.py`
- 运行测试脚本的日志或截图
- 简要比较四种准则生成的树结构和测试结果
