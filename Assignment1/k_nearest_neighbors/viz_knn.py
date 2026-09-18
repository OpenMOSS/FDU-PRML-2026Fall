from __future__ import annotations
import math, os, numpy as np, matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_k_curve(ks, accs, out_path: str):
    ks = np.asarray(list(ks))
    accs = np.asarray(list(accs))
    order = np.argsort(ks)
    ks, accs = ks[order], accs[order]

    fig, ax = plt.subplots()
    ax.plot(ks, accs, marker="o")
    ax.set_xlabel("k")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("kNN: Validation Accuracy vs k")

    ax.set_xticks(ks)
    ax.set_xlim(ks.min() - 0.5, ks.max() + 0.5)

    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_decision_boundary_multi(predict_fn_for_k,
                                 X_train, y_train, X_test, y_test,
                                 ks=(1, 3, 7),
                                 out_path="./output/knn_boundary_grid.png",
                                 grid_n=200,          # ← 网格密度
                                 batch_size=4096):    # ← 分批预测
    """
    多类别版本：并排显示多种 K 的决策边界（仅 2D）。
    predict_fn_for_k(k) 返回一个 f(X) -> 预测标签(0..C-1) 的函数。
    """
    assert X_train.shape[1] == 2

    # 统一坐标范围
    margin = 0.5
    x_min = min(X_train[:,0].min(), X_test[:,0].min()) - margin
    x_max = max(X_train[:,0].max(), X_test[:,0].max()) + margin
    y_min = min(X_train[:,1].min(), X_test[:,1].min()) - margin
    y_max = max(X_train[:,1].max(), X_test[:,1].max()) + margin
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_n),
                         np.linspace(y_min, y_max, grid_n))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # 类别数与配色（tab10 前 n 色）
    n_classes = int(max(y_train.max(), y_test.max())) + 1
    cmap = ListedColormap(plt.cm.tab10.colors[:n_classes])
    levels = np.arange(-0.5, n_classes + 0.5, 1.0)  # 离散等间隔

    ks = list(ks)
    ncols = min(len(ks), 4)
    nrows = math.ceil(len(ks) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 4.2*nrows),
                             sharex=True, sharey=True, squeeze=False,
                             constrained_layout=True)
    axes_flat = axes.ravel()
    for ax in axes_flat[len(ks):]:
        ax.axis("off")

    for ax, k in zip(axes_flat, ks):
        f = predict_fn_for_k(k)

        # 分批预测网格
        Z_flat = np.empty(grid.shape[0], dtype=int)
        for s in range(0, grid.shape[0], batch_size):
            e = min(s + batch_size, grid.shape[0])
            Z_flat[s:e] = f(grid[s:e])
        Z = Z_flat.reshape(xx.shape)

        # 背景色块（多类别）
        cs = ax.contourf(xx, yy, Z, levels=levels, cmap=cmap, alpha=0.75, antialiased=False)

        # 训练/测试散点
        ax.scatter(X_train[:,0], X_train[:,1], s=18, c=y_train, cmap=cmap, vmin=0, vmax=n_classes-1, edgecolors="k", linewidths=0.2, label="Train")
        ax.scatter(X_test[:,0],  X_test[:,1],  s=36, c=y_test,  cmap=cmap, vmin=0, vmax=n_classes-1, marker="^", edgecolors="k", linewidths=0.2, label="Test")

        # 误分类测试点
        y_pred_test = f(X_test)
        mis = (y_pred_test != y_test)
        if np.any(mis):
            ax.scatter(X_test[mis,0], X_test[mis,1], marker="x", s=45, linewidths=1.4, c="k", label="Miscls.")

        ax.set_title(f"K = {k}")
        ax.set_xlabel("x1"); ax.set_ylabel("x2")
        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")

    # 统一图例 + 颜色条便于阅读
    handles, labels = axes_flat[0].get_legend_handles_labels()
    axes_flat[0].legend(handles, labels, loc="upper right")
    fig.colorbar(cs, ax=axes, shrink=0.8, ticks=np.arange(n_classes), label="class id")

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Figure] saved -> {out_path}")
