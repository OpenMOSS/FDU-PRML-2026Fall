# Assignment 1 - Part 3: k-Nearest Neighbors

本实验围绕 kNN 分类 展开，要求实现距离计算、kNN 预测与验证集选参，并在 2D 多类别数据上可视化决策边界与“val-accuracy vs k” 曲线。

分数占比 - 40%

仓库主要文件: `k_nearest_neighbors/`
- 数据生成与读取：`data_generate.py`
- 学生实现入口：`knn_student.py`
- 可视化：`viz_knn.py`
- 测试脚本：`test_knn.py`
- 数据集与输出集：`input_knn/`、`output/`

## 实验目标
- 理解 kNN 的基本流程（距离度量 → 近邻检索 → 投票决策）
- 实现 L2 距离的两种方式（two_loops / no_loops）并保证数值一致
- 实现 多数表决的 kNN 预测, 在验证集上选 k（网格搜索）
- 在 2D 多类别数据上可视化 val-accuracy vs k 与 决策边界。

## 实验内容
1) 数据与文件说明
- 运行 `data_generate.py` 生成并保存数据到 `input_knn/`，或直接使用已经生成的`input_knn/`数据，读取请调用 `load_prepared_dataset`（会做维度与样本数的最小校验）

2) 你需要在代码中完成 `TODO` 部分
- 在 `knn_student.py` 中补全三个函数的计算逻辑：
    - `pairwise_dist(X_test, X_train, metric, mode)`：
        实现 L2 的 two_loops 与 no_loops
        实现 cosine
        no_loops 展开式的平方距离先截断负值再开方
    - `knn_predict(X_test, X_train, y_train, k, metric, mode)`：
        基于距离的多数表决输出类别
        平票返回最小标签
    - `select_k_by_validation(X_train, y_train, X_val, y_val, ks, metric, mode)`：
        在 ks 上网格搜索并返回 (best_k, accs)
        平票返回最小的 k

- 注意：完成后删除/注释占位的 raise NotImplementedError(...)

选做：调整数据生成参数（类数、噪声、样本量）观察曲线与边界变化

## 评测与运行
- 先运行 `test_knn.py` 执行基础单测：L2 两实现一致性、k=1 与 k=3 的预测、平票处理、选 k 的接口与返回值范围等。
- 通过后运行 `knn_student.py` 生成图像（l2 与 cosine 各一组）：
    - `output/knn_k_curve_{metric}.png`
    - `output/knn_boundary_grid_{metric}.png`（并列展示多种 K 的决策边界，含误分类点标记）

## 提交内容
- 提交你完成后的 `knn_student.py`
- `knn_k_curve_{metric}.png` 与 `knn_boundary_grid_{metric}.png` (l2和cosine各一份)
- 根据图像简短分析：最优 k 与测试集表现、不同 k 的边界差异（过拟合/欠拟合）
