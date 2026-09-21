# 本脚本仅用于基础自测，通过此处的小样例不代表实现一定完全正确。

from datetime import datetime

import numpy as np

import accuracy_error as acc_err
import evaluation_metrics as metrics

# ============== 学生信息（请填写） ==============
STUDENT_NAME = "张三"  # 例如：张三
STUDENT_ID = "2026123456"  # 例如：2026123456
# ==============================================


def test_basic():
    print("[BASIC] accuracy / MSE")
    # Accuracy test case
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    expected_accuracy = 0.75

    try:
        accuracy = acc_err.accuracy_score(y_true, y_pred)
        ok_accuracy = abs(accuracy - expected_accuracy) < 1e-8
        print(
            f" - accuracy expected {expected_accuracy:.2f} "
            f"-> got {accuracy:.2f} : "
            f"{'PASS' if ok_accuracy else 'FAIL'}"
        )
    except NotImplementedError:
        print(" - accuracy NOT IMPLEMENTED")

    # MSE test case
    y_true = np.array([0.0, 1.0, 2.0])
    y_pred = np.array([0.0, 2.0, 1.0])
    expected_mse = 2 / 3

    try:
        mse = acc_err.mean_squared_error(y_true, y_pred)
        ok_mse = abs(mse - expected_mse) < 1e-6
        print(
            f" - mse      expected {expected_mse:.4f} -> got {mse:.4f} : "
            f"{'PASS' if ok_mse else 'FAIL'}"
        )
    except NotImplementedError:
        print(" - mse NOT IMPLEMENTED")


def test_eval():
    print("[PRF] precision / recall / f1 (binary)")
    # Binary classification test case
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0])
    expected_precision = 0.5
    expected_recall = 0.5
    expected_f1 = 0.5

    try:
        precision = metrics.precision_score(y_true, y_pred)
        recall = metrics.recall_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred)
        ok_precision = abs(precision - expected_precision) < 1e-8
        ok_recall = abs(recall - expected_recall) < 1e-8
        ok_f1 = abs(f1 - expected_f1) < 1e-8
        print(
            f" - precision expected {expected_precision:.2f} "
            f"-> got {precision:.2f} : "
            f"{'PASS' if ok_precision else 'FAIL'}"
        )
        print(
            f" - recall    expected {expected_recall:.2f} "
            f"-> got {recall:.2f} : "
            f"{'PASS' if ok_recall else 'FAIL'}"
        )
        print(
            f" - f1        expected {expected_f1:.2f} -> got {f1:.2f} : "
            f"{'PASS' if ok_f1 else 'FAIL'}"
        )
    except NotImplementedError:
        print(" - PRF NOT IMPLEMENTED")


def _print_header():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 60)
    print("实验：Part 1 - 分类评测指标")
    print(f"姓名：{STUDENT_NAME}    学号：{STUDENT_ID}")
    print(f"时间：{timestamp}")
    print("=" * 60)


if __name__ == "__main__":
    _print_header()
    test_basic()
    test_eval()
