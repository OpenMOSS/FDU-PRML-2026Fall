import os
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# ===================== 可调参数 =====================
DATA_DIR     = "./input_knn"    # 输入数据目录
RANDOM_STATE = 42               # 随机种子
N_SAMPLES    = 500              # 样本总数
N_CLASSES    = 4                # 类别数 （在boundary图上会有几个色块）
CLUSTER_STD  = 4.0              # 类内标准差（数据难度）
TEST_SIZE    = 0.25             # 测试集比例    
VAL_SIZE     = 0.25             # 验证集比例
# ===================================================

REQUIRED_FILES = [
    "X_train.npy", "y_train.npy",
    "X_val.npy",   "y_val.npy",
    "X_test.npy",  "y_test.npy",
]

def _summary(X_tr, y_tr, X_val, y_val, X_te, y_te):
    print(f"[DATA] train={len(X_tr)}, val={len(X_val)}, test={len(X_te)}, "
          f"D={X_tr.shape[1]}, classes={len(np.unique(y_tr))}")

def generate_and_save(
    data_dir: str = DATA_DIR,
    n_samples: int = N_SAMPLES,
    n_classes: int = N_CLASSES,
    cluster_std: float = CLUSTER_STD,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_STATE,
    force: bool = True,
) -> None:
    """
    生成数据并保存到 data_dir；默认 force=True（覆盖同名文件）。
    若不想每次都覆盖，可传 force=False（存在就跳过）。
    """
    os.makedirs(data_dir, exist_ok=True)
    paths = [os.path.join(data_dir, f) for f in REQUIRED_FILES]

    if (not force) and all(os.path.exists(p) for p in paths):
        print(f"[Skip] Found existing dataset in {data_dir}")
        # 打印概要，便于日志比对
        X_tr = np.load(os.path.join(data_dir, "X_train.npy"))
        y_tr = np.load(os.path.join(data_dir, "y_train.npy"))
        X_val = np.load(os.path.join(data_dir, "X_val.npy"))
        y_val = np.load(os.path.join(data_dir, "y_val.npy"))
        X_te = np.load(os.path.join(data_dir, "X_test.npy"))
        y_te = np.load(os.path.join(data_dir, "y_test.npy"))
        _summary(X_tr, y_tr, X_val, y_val, X_te, y_te)
        return

    # 1) 生成原始数据
    X, y = make_blobs(
        n_samples=n_samples,
        centers=n_classes,
        n_features=2,
        cluster_std=cluster_std,
        random_state=random_state,
    )
    X = X.astype(np.float64)
    y = y.astype(np.int64)

    # 2) train / (val+test)
    X_tr, X_rest, y_tr, y_rest = train_test_split(
        X, y,
        test_size=(test_size + val_size),
        random_state=random_state,
        stratify=y,
    )

    # 3) 在剩余部分里按比例切 val / test
    rel_test = test_size / (test_size + val_size)
    X_val, X_te, y_val, y_te = train_test_split(
        X_rest, y_rest,
        test_size=rel_test,
        random_state=random_state,
        stratify=y_rest,
    )

    # 4) 保存
    np.save(os.path.join(data_dir, "X_train.npy"), X_tr)
    np.save(os.path.join(data_dir, "y_train.npy"), y_tr)
    np.save(os.path.join(data_dir, "X_val.npy"),   X_val)
    np.save(os.path.join(data_dir, "y_val.npy"),   y_val)
    np.save(os.path.join(data_dir, "X_test.npy"),  X_te)
    np.save(os.path.join(data_dir, "y_test.npy"),  y_te)

    print(f"[OK] Saved dataset to {data_dir} (seed={random_state})")
    _summary(X_tr, y_tr, X_val, y_val, X_te, y_te)

def load_prepared_dataset(data_dir: str = DATA_DIR):
    """
    仅负责读取准备好的数据
    test_knn.py / knn_student.py 直接调用该函数。
    """
    X_tr = np.load(os.path.join(data_dir, "X_train.npy"))
    y_tr = np.load(os.path.join(data_dir, "y_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    y_val = np.load(os.path.join(data_dir, "y_val.npy"))
    X_te = np.load(os.path.join(data_dir, "X_test.npy"))
    y_te = np.load(os.path.join(data_dir, "y_test.npy"))
    return X_tr, y_tr, X_val, y_val, X_te, y_te

if __name__ == "__main__":
    # 只有“直接运行 data_generate.py”时才会生成数据
    generate_and_save(force=True)
