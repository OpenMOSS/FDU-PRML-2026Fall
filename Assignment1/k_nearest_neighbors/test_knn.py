"""Unit tests for kNN functions in knn_student.py

Run with:  pytest -q  (if pytest installed) or simply python test_knn.py
"""
import numpy as np
import math
from knn_student import pairwise_dist, knn_predict, select_k_by_validation
from data_generate import load_prepared_dataset
from datetime import datetime

# === 学生信息 ===
STUDENT_NAME = "张三"
STUDENT_ID   = "2026123456"
# ================

def _almost_equal(a, b, tol=1e-9):
    return np.max(np.abs(a - b)) <= tol

def _acc(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float((y_true == y_pred).mean())

def _print_header():
    name = (STUDENT_NAME or "").strip() or "未填写"
    sid  = (STUDENT_ID   or "").strip() or "未填写"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print("实验：Part 3 - kNN 分类")
    print(f"姓名：{name}    学号：{sid}    时间：{ts}")
    print("=" * 70)

def _print_data_summary(Xtr, ytr, Xval, yval, Xte, yte):
    n_classes = int(np.max([ytr.max(), yval.max(), yte.max()])) + 1
    print("[DATA] Summary")
    print(f" - train = {len(Xtr)}, val = {len(Xval)}, test = {len(Xte)}, D = {Xtr.shape[1]}, classes = {n_classes}")
    # 每类样本数
    for split_name, yy in [("train", ytr), ("val", yval), ("test", yte)]:
        uniq, cnt = np.unique(yy, return_counts=True)
        line = ", ".join([f"class {int(u)}: {int(c)}" for u, c in zip(uniq, cnt)])
        print(f"   · {split_name} per-class: {line}")
    print("-" * 70)

# ----------------- unit tests (for pytest) -----------------
def test_pairwise_l2_equivalence():
    Xtr = np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
    Xte = np.array([[0.5,0.5],[2.,2.]])
    d_two = pairwise_dist(Xte, Xtr, metric="l2", mode="two_loops")
    d_no  = pairwise_dist(Xte, Xtr, metric="l2", mode="no_loops")
    assert d_two.shape == (2,4)
    assert _almost_equal(d_two, d_no), "L2 two_loops and no_loops mismatch"

def test_pairwise_cosine_basic():
    Xtr = np.array([[1.,0.],[0.,1.]])
    Xte = np.array([[1.,1.]])
    d_cos = pairwise_dist(Xte, Xtr, metric="cosine", mode="two_loops")
    expected0 = 1 - 1/math.sqrt(2)
    expected = np.array([[expected0, expected0]])
    assert _almost_equal(d_cos, expected, tol=1e-9), f"Cosine distance mismatch: {d_cos} vs {expected}"

def test_knn_predict_basic():
    X_train = np.array([[0,0],[0,1],[1,0],[1,1],   # class 0
                        [3,3],[3,4],[4,3],[4,4]])  # class 1
    y_train = np.array([0,0,0,0, 1,1,1,1])
    X_test  = np.array([[0.2,0.2],[3.6,3.4],[2.0,2.0]])
    y_pred_k1 = knn_predict(X_test, X_train, y_train, k=1, metric="l2", mode="no_loops")
    assert np.array_equal(y_pred_k1, np.array([0,1,0])), f"k=1 predictions wrong: {y_pred_k1}"
    y_pred_k3 = knn_predict(X_test, X_train, y_train, k=3, metric="l2", mode="no_loops")
    assert np.array_equal(y_pred_k3, np.array([0,1,0])), f"k=3 predictions wrong: {y_pred_k3}"

def test_knn_predict_tie():
    X_train = np.array([[0,0],[2,0],[0,2],[2,2]])
    y_train = np.array([0,0,1,1])
    X_test  = np.array([[1,1]])
    y_pred = knn_predict(X_test, X_train, y_train, k=4, metric="l2", mode="two_loops")
    assert y_pred[0] == 0, f"Tie should pick smallest label. Got {y_pred[0]}"

def test_select_k_by_validation():
    X_train = np.array([[0,0],[0,1],[1,0],[1,1]])
    y_train = np.array([0,0,0,0])
    X_val = np.array([[0.2,0.3],[0.9,0.8]])
    y_val = np.array([0,0])
    ks = [1,3]
    best_k, accs = select_k_by_validation(X_train, y_train, X_val, y_val, ks, metric="l2", mode="no_loops")
    assert best_k in ks, "Returned k not in search list"
    assert len(accs) == len(ks), "Acc list length mismatch"
    assert all(abs(a-1.0) < 1e-12 for a in accs), "All accuracies should be 1.0 here"
    assert best_k == min(ks), f"Tie should pick smallest k. Got {best_k}"

# ----------------- verbose report (when run as script) -----------------
def _verbose_report():
    _print_header()

    # 1) 数据集载入与汇总
    Xtr, ytr, Xval, yval, Xte, yte = load_prepared_dataset()
    _print_data_summary(Xtr, ytr, Xval, yval, Xte, yte)

    # 2) 距离函数数值检查
    print("[DIST] L2 two_loops vs no_loops")
    Xtr_small = np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
    Xte_small = np.array([[0.5,0.5],[2.,2.]])
    d_two = pairwise_dist(Xte_small, Xtr_small, metric="l2", mode="two_loops")
    d_no  = pairwise_dist(Xte_small, Xtr_small, metric="l2", mode="no_loops")
    max_err = float(np.max(np.abs(d_two - d_no)))
    print(f" - shape: {d_two.shape}, max|diff| = {max_err:.3e}  -> {'PASS' if max_err < 1e-9 else 'FAIL'}")

    print("[DIST] Cosine basic case")
    Xtr_c = np.array([[1.,0.],[0.,1.]])
    Xte_c = np.array([[1.,1.]])
    d_cos = pairwise_dist(Xte_c, Xtr_c, metric="cosine", mode="two_loops")
    expected0 = 1 - 1/math.sqrt(2)
    print(f" - expected = [{expected0:.6f}, {expected0:.6f}], got = [{d_cos[0,0]:.6f}, {d_cos[0,1]:.6f}]",
          " -> PASS" if _almost_equal(d_cos, np.array([[expected0, expected0]])) else " -> FAIL")
    print("-" * 70)

    # 3) 预测与 Tie 规则检查
    print("[PRED] sanity checks (k=1 / k=3 / tie rule)")
    X_train = np.array([[0,0],[0,1],[1,0],[1,1],
                        [3,3],[3,4],[4,3],[4,4]])
    y_train = np.array([0,0,0,0, 1,1,1,1])
    X_test  = np.array([[0.2,0.2],[3.6,3.4],[2.0,2.0]])
    y1 = knn_predict(X_test, X_train, y_train, k=1, metric="l2", mode="no_loops")
    y3 = knn_predict(X_test, X_train, y_train, k=3, metric="l2", mode="no_loops")
    print(f" - k=1 pred = {y1.tolist()}  (expect [0,1,0])  -> {'PASS' if np.array_equal(y1,[0,1,0]) else 'FAIL'}")
    print(f" - k=3 pred = {y3.tolist()}  (expect [0,1,0])  -> {'PASS' if np.array_equal(y3,[0,1,0]) else 'FAIL'}")
    # tie
    X_train_t = np.array([[0,0],[2,0],[0,2],[2,2]])
    y_train_t = np.array([0,0,1,1])
    y_tie = knn_predict(np.array([[1,1]]), X_train_t, y_train_t, k=4, metric="l2", mode="two_loops")
    print(f" - tie case (k=4) -> pred = {int(y_tie[0])} (expect 0) -> {'PASS' if y_tie[0]==0 else 'FAIL'}")
    print("-" * 70)

    # 4) 选参（验证集）与曲线打印
    print("[MODEL SELECTION] validation curve")
    ks = [1,3,5,7,9,11,13]
    best_k, accs = select_k_by_validation(Xtr, ytr, Xval, yval, ks, metric="l2", mode="no_loops")
    accs_line = ", ".join([f"k={k}:{a:.4f}" for k, a in zip(ks, accs)])
    print(f" - ks & val_accs: {accs_line}")
    print(f" - best_k = {best_k} (val_acc = {accs[ks.index(best_k)]:.4f})")
    print("-" * 70)

    # 5) 端到端：用 best_k 在 test 上评估
    print("[E2E] train+val -> test (metric=l2, mode=no_loops)")
    X_tv = np.vstack([Xtr, Xval])
    y_tv = np.concatenate([ytr, yval])
    y_pred_test = knn_predict(Xte, X_tv, y_tv, k=best_k, metric="l2", mode="no_loops")
    test_acc = _acc(yte, y_pred_test)
    print(f" - test_acc(best_k={best_k}) = {test_acc:.4f}")
    print("=" * 70)
    print("All kNN tests passed.\n")

# ----------------- main -----------------
if __name__ == "__main__":
    # 仍运行一次断言，若失败能第一时间发现
    test_pairwise_l2_equivalence()
    test_pairwise_cosine_basic()
    test_knn_predict_basic()
    test_knn_predict_tie()
    test_select_k_by_validation()

    # 打印详细报告
    _verbose_report()