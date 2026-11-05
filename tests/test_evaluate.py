"""Kiểm thử cho going_modular.evaluate (Phase 6).

Chạy nhanh (CPU, Agg, không cần mạng/GPU/dataset thật):
    MPLBACKEND=Agg pytest tests/test_evaluate.py -v

Ánh xạ mã test theo spec: V1–V24.
"""

from __future__ import annotations

import json
import math
import os
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from going_modular import evaluate
from going_modular.evaluate import (
    build_cost_record,
    classification_metrics,
    compute_confusion_matrix,
    denormalize,
    evaluate_predictions,
    file_size_mb,
    finalize_figure,
    format_comparison_table,
    get_mean_std,
    measure_inference_time,
    misclassified_indices,
    plot_confusion_matrix,
    plot_history,
    plot_per_class_accuracy,
    plot_sample_predictions,
    predict_dataloader,
    predict_probs,
    run_inference,
    top_confusions,
    wilson_interval,
)
from going_modular.model_builder import count_params


def make_toy_loader(n_samples=10, n_features=4, n_classes=3, batch_size=4, seed=0, labels=None):
    """Tạo DataLoader giả với dữ liệu ngẫu nhiên cố định."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n_samples, n_features, generator=g)
    if labels is None:
        y = torch.randint(0, n_classes, (n_samples,), generator=g)
    else:
        y = torch.tensor(labels, dtype=torch.long)
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch_size)


def make_toy_model(n_features=4, n_classes=3):
    return nn.Linear(n_features, n_classes)


# ---------------------------------------------------------------------------
# Nhóm A: Suy luận một lượt (V1–V4)
# ---------------------------------------------------------------------------

def test_v1_run_inference_shapes_and_loss_none():
    """V1: logits (10,3), y_true (10,), loss is None khi không truyền loss_fn."""
    model = make_toy_model()
    loader = make_toy_loader(n_samples=10, batch_size=4)
    result = run_inference(model, loader, torch.device("cpu"))

    assert result.logits.shape == (10, 3)
    assert result.y_true.shape == (10,)
    assert result.loss is None
    assert result.y_pred.shape == (10,)


def test_v2_loss_matches_full_batch():
    """V2: Loss từ run_inference khớp loss tính một lần trên toàn bộ tập (sai số 1e-6)."""
    model = make_toy_model()
    loss_fn = nn.CrossEntropyLoss()
    loader = make_toy_loader(n_samples=10, batch_size=4)  # batch 4,4,2

    result = run_inference(model, loader, torch.device("cpu"), loss_fn=loss_fn)
    expected = loss_fn(result.logits, result.y_true).item()
    assert abs(result.loss - expected) < 1e-6


def test_v3_label_order_preserved():
    """V3: shuffle=False, nhãn tăng dần → y_true giữ đúng thứ tự dữ liệu."""
    labels = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
    loader = make_toy_loader(n_samples=10, batch_size=3, labels=labels)
    result = run_inference(make_toy_model(), loader, torch.device("cpu"))

    assert result.y_true.tolist() == labels


def test_v4_empty_dataloader_and_eval_mode():
    """V4: Dataloader rỗng → ValueError; model ở eval sau khi gọi."""
    model = make_toy_model()
    empty_ds = TensorDataset(torch.randn(0, 4), torch.randint(0, 3, (0,)))
    empty_loader = DataLoader(empty_ds, batch_size=4)

    with pytest.raises(ValueError):
        run_inference(model, empty_loader, torch.device("cpu"))

    assert model.training is False


# ---------------------------------------------------------------------------
# Nhóm B: Ma trận nhầm lẫn (V5–V8)
# ---------------------------------------------------------------------------

def test_v5_confusion_matrix_hand_calculated():
    """V5: y_true=[0,0,1,2,2], y_pred=[0,1,1,2,0], C=3 → [[1,1,0],[0,1,0],[1,0,1]]."""
    y_true = torch.tensor([0, 0, 1, 2, 2])
    y_pred = torch.tensor([0, 1, 1, 2, 0])
    cm = compute_confusion_matrix(y_true=y_true, y_pred=y_pred, num_classes=3)

    expected = torch.tensor([[1, 1, 0], [0, 1, 0], [1, 0, 1]])
    assert torch.equal(cm, expected)


def test_v6_perfect_predictions_diagonal():
    """V6: Dự đoán hoàn hảo → ma trận chéo; tổng mỗi hàng = số mẫu thật của lớp."""
    y_true = torch.tensor([0, 0, 1, 1, 2, 2])
    y_pred = torch.tensor([0, 0, 1, 1, 2, 2])
    cm = compute_confusion_matrix(y_true=y_true, y_pred=y_pred, num_classes=3)

    expected = torch.tensor([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    assert torch.equal(cm, expected)
    assert cm.sum(dim=1).tolist() == [2, 2, 2]


def test_v7_matrix_direction_true_is_row():
    """V7: Tất cả thật=2, dự đoán=1 → cm[2,1]==3, cm[1,2]==0 (hàng là thật)."""
    y_true = torch.tensor([2, 2, 2])
    y_pred = torch.tensor([1, 1, 1])
    cm = compute_confusion_matrix(y_true=y_true, y_pred=y_pred, num_classes=3)

    assert cm[2, 1].item() == 3
    assert cm[1, 2].item() == 0


def test_v8_invalid_inputs_and_positional_rejected():
    """V8: Nhãn ngoài miền, tensor sai shape/độ dài, num_classes=0 → ValueError;
    gọi positional → TypeError."""
    y_true = torch.tensor([0, 1])
    y_pred = torch.tensor([0, 1])

    with pytest.raises(ValueError):
        compute_confusion_matrix(y_true=y_true, y_pred=y_pred, num_classes=0)
    with pytest.raises(ValueError):
        compute_confusion_matrix(y_true=y_true, y_pred=torch.tensor([0, 3]), num_classes=3)
    with pytest.raises(ValueError):
        compute_confusion_matrix(y_true=torch.tensor([[0, 1]]), y_pred=y_pred, num_classes=3)
    with pytest.raises(ValueError):
        compute_confusion_matrix(y_true=y_true, y_pred=torch.tensor([0]), num_classes=3)
    with pytest.raises(TypeError):
        compute_confusion_matrix(y_true, y_pred, 3)


# ---------------------------------------------------------------------------
# Nhóm C: Độ đo và Wilson CI (V9–V12)
# ---------------------------------------------------------------------------

def test_v9_classification_metrics_hand_calculated():
    """V9: Ma trận V5 → accuracy 0.6; precision [0.5,0.5,1.0]; recall [0.5,1.0,0.5]; F1 [0.5,0.6667,0.6667]."""
    cm = torch.tensor([[1, 1, 0], [0, 1, 0], [1, 0, 1]])
    m = classification_metrics(cm)

    assert abs(m["accuracy"] - 0.6) < 1e-6
    assert torch.allclose(m["precision"], torch.tensor([0.5, 0.5, 1.0]), atol=1e-6)
    assert torch.allclose(m["recall"], torch.tensor([0.5, 1.0, 0.5]), atol=1e-6)
    assert torch.allclose(m["f1"], torch.tensor([0.5, 2 / 3, 2 / 3]), atol=1e-6)


def test_v10_no_nan_when_class_never_predicted():
    """V10: Lớp 2 không bao giờ được dự đoán → precision[2]=0; không có NaN."""
    y_true = torch.tensor([0, 1, 2])
    y_pred = torch.tensor([0, 1, 0])
    cm = compute_confusion_matrix(y_true=y_true, y_pred=y_pred, num_classes=3)
    m = classification_metrics(cm)

    assert m["precision"][2].item() == 0.0
    assert not torch.isnan(m["precision"]).any()
    assert not torch.isnan(m["recall"]).any()
    assert not torch.isnan(m["f1"]).any()
    assert not math.isnan(m["macro_precision"])
    assert not math.isnan(m["macro_recall"])
    assert not math.isnan(m["macro_f1"])


def test_v11_macro_averages():
    """V11: Ma trận V5 → macro precision ≈ 0.6667, macro recall ≈ 0.6667, macro F1 ≈ 0.6111."""
    cm = torch.tensor([[1, 1, 0], [0, 1, 0], [1, 0, 1]])
    m = classification_metrics(cm)

    assert abs(m["macro_precision"] - 2 / 3) < 1e-6
    assert abs(m["macro_recall"] - 2 / 3) < 1e-6
    assert abs(m["macro_f1"] - (0.5 + 2 / 3 + 2 / 3) / 3) < 1e-6


def test_v12_wilson_interval():
    """V12: wilson_interval(50,100) ≈ [0.4038, 0.5962]; cận trong [0,1]; (5,0) và (11,10) → ValueError."""
    low, high = wilson_interval(50, 100)
    assert abs(low - 0.4038) < 1e-4
    assert abs(high - 0.5962) < 1e-4
    assert 0.0 <= low <= high <= 1.0

    low, high = wilson_interval(0, 10)
    assert low == 0.0
    assert high <= 1.0

    low, high = wilson_interval(10, 10)
    assert high == 1.0

    with pytest.raises(ValueError):
        wilson_interval(5, 0)
    with pytest.raises(ValueError):
        wilson_interval(11, 10)


# ---------------------------------------------------------------------------
# Nhóm D: Phân tích lỗi (V13–V14)
# ---------------------------------------------------------------------------

def test_v13_top_confusions():
    """V13: Ma trận chéo → danh sách rỗng; có nhầm → số đếm >0, giảm dần."""
    cm_diag = torch.tensor([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    names = ["cat", "dog", "bird"]
    assert top_confusions(cm_diag, names, k=5) == []

    cm = torch.tensor([[1, 2, 0], [0, 1, 3], [0, 0, 1]])
    conf = top_confusions(cm, names, k=5)
    assert all(c > 0 for _, _, c in conf)
    assert conf == sorted(conf, key=lambda x: -x[2])


def test_v14_misclassified_indices():
    """V14: Không logits → thứ tự dữ liệu; có logits → theo độ tự tin giảm dần; max_items cắt đúng."""
    y_true = torch.tensor([0, 1, 2, 0, 1])
    y_pred = torch.tensor([0, 2, 2, 1, 1])

    wrong = misclassified_indices(y_true, y_pred)
    assert wrong.tolist() == [1, 3]  # index 1 (1→2), index 3 (0→1)

    logits = torch.tensor([
        [10, 0, 0],   # idx0: đúng (0→0)
        [0, 0, 10],   # idx1: sai, conf cao (1→2)
        [0, 0, 10],   # idx2: đúng (2→2)
        [10, 0, 0],   # idx3: sai, conf cao (0→1)
        [0, 10, 0],   # idx4: đúng (1→1)
    ], dtype=torch.float32)
    wrong_conf = misclassified_indices(y_true, y_pred, logits=logits)
    assert wrong_conf.tolist() == [1, 3]  # cả hai conf=10, thứ tự ưu tiên index thấp

    wrong_limited = misclassified_indices(y_true, y_pred, max_items=1)
    assert len(wrong_limited) == 1


# ---------------------------------------------------------------------------
# Nhóm E: Giải chuẩn hóa (V15–V16)
# ---------------------------------------------------------------------------

def test_v15_denormalize_roundtrip_and_clamp():
    """V15: Chuẩn hóa rồi denormalize → khôi phục giá trị (tol 1e-6); clamp [0,1]; kênh sai → ValueError."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    img = torch.rand(3, 32, 32)
    norm = (img - torch.tensor(mean).view(3, 1, 1)) / torch.tensor(std).view(3, 1, 1)
    back = denormalize(norm, mean, std)
    assert torch.allclose(back, img, atol=1e-6)

    # Clamp: giá trị ngoài [0,1] bị cắt
    extreme = torch.tensor([[[10.0]], [[-10.0]], [[10.0]]])
    clamped = denormalize(extreme, [0, 0, 0], [1, 1, 1])
    assert torch.all(clamped >= 0.0) and torch.all(clamped <= 1.0)

    # Kênh mismatch
    with pytest.raises(ValueError):
        denormalize(torch.rand(3, 4, 4), [0.5], [0.5])


def test_v16_get_mean_std_from_transform_and_weights():
    """V16: Transform có Normalize → trả stats; thiếu stats nhưng có weights → fallback;
    thiếu stats và không xác định weights → ValueError."""
    from torchvision import transforms

    # Transform có Normalize
    t = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2])
    mean, std = get_mean_std(t)
    assert mean == [0.5, 0.5, 0.5]
    assert std == [0.2, 0.2, 0.2]

    # Transform không có stats, weights có preprocessing
    class FakeWeights:
        class preprocessing:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

    t2 = transforms.Compose([transforms.ToTensor()])
    mean, std = get_mean_std(t2, transform_weights=FakeWeights())
    assert mean == [0.485, 0.456, 0.406]

    # Không stats, không weights hợp lệ → ValueError
    with pytest.raises(ValueError):
        get_mean_std(t2)


# ---------------------------------------------------------------------------
# Nhóm F: Chi phí (V17–V19)
# ---------------------------------------------------------------------------

def test_v17_measure_inference_time_structure():
    """V17: Có đủ khóa; thời gian >0; ms_per_image * batch_size ≈ ms_per_batch;
    peak_gpu_mb is None trên CPU; tham số sai → ValueError."""
    model = make_toy_model()
    images = torch.randn(8, 4)
    timing = measure_inference_time(model, images, torch.device("cpu"), warmup=1, repeats=3)

    assert set(timing.keys()) == {
        "batch_size", "ms_per_batch", "ms_per_image", "repeats", "device", "peak_gpu_mb"
    }
    assert timing["batch_size"] == 8
    assert timing["ms_per_batch"] > 0
    assert timing["ms_per_image"] > 0
    assert abs(timing["ms_per_image"] * 8 - timing["ms_per_batch"]) < 1e-6
    assert timing["peak_gpu_mb"] is None
    assert timing["device"] == "cpu"

    with pytest.raises(ValueError):
        measure_inference_time(model, images, warmup=-1, repeats=3)
    with pytest.raises(ValueError):
        measure_inference_time(model, images, warmup=1, repeats=0)


def test_v18_file_size_mb():
    """V18: 1,048,576 byte → 1.0 MiB; tệp thiếu → FileNotFoundError."""
    tmp = "tmp_size_test.bin"
    with open(tmp, "wb") as f:
        f.write(b"\x00" * 1_048_576)
    try:
        assert abs(file_size_mb(tmp) - 1.0) < 1e-6
    finally:
        os.remove(tmp)

    with pytest.raises(FileNotFoundError):
        file_size_mb("nonexistent_file.bin")


def test_v19_build_cost_record():
    """V19: history có epoch_time → train_time_phase_s = tổng; thiếu timing → None;
    source_history cộng vào total; thiếu model_path → None."""
    num_params = count_params(make_toy_model())

    # Có history, có timing
    history = {"epoch_time": [1.0, 2.0, 3.0]}
    timing = {"ms_per_image": 5.0, "ms_per_batch": 160.0, "batch_size": 32, "device": "cpu", "peak_gpu_mb": None}
    record = build_cost_record(num_params, model_path=None, timing=timing, history=history)
    assert record["train_time_phase_s"] == 6.0
    assert record["train_time_total_s"] == 6.0
    assert record["ms_per_image"] == 5.0

    # Thiếu timing
    record2 = build_cost_record(num_params, model_path=None, timing=None, history=history)
    assert record2["ms_per_image"] is None
    assert record2["train_time_phase_s"] == 6.0

    # source_history cộng vào total
    source_history = {"epoch_time": [10.0, 10.0]}
    record3 = build_cost_record(num_params, model_path=None, timing=None,
                                history=history, source_history=source_history)
    assert record3["train_time_phase_s"] == 6.0
    assert record3["train_time_total_s"] == 26.0

    # Không history
    record4 = build_cost_record(num_params, model_path=None, timing=None, history=None)
    assert record4["train_time_phase_s"] is None
    assert record4["train_time_total_s"] is None


# ---------------------------------------------------------------------------
# Nhóm G: Hình vẽ và tổng hợp (V20–V24)
# ---------------------------------------------------------------------------

def test_v20_plots_return_figure_and_save(tmp_path):
    """V20: Các hàm plot trả về Figure; lưu file không rỗng; số figure pyplot không tăng."""
    import matplotlib.pyplot as plt

    initial_count = len(plt.get_fignums())

    cm = torch.tensor([[5, 1], [0, 4]])
    names = ["cat", "dog"]
    fig = plot_confusion_matrix(cm, names, save_path=str(tmp_path / "cm.png"))
    assert isinstance(fig, plt.Figure)
    assert os.path.getsize(tmp_path / "cm.png") > 0

    fig = plot_per_class_accuracy(torch.tensor([0.8, 0.9]), names, save_path=str(tmp_path / "acc.png"))
    assert isinstance(fig, plt.Figure)
    assert os.path.getsize(tmp_path / "acc.png") > 0

    images = [torch.rand(3, 8, 8) for _ in range(4)]
    fig = plot_sample_predictions(
        images, [0, 1, 0, 1], [0, 1, 1, 0], names, save_path=str(tmp_path / "samples.png")
    )
    assert isinstance(fig, plt.Figure)
    assert os.path.getsize(tmp_path / "samples.png") > 0

    histories = {"run1": {"train_loss": [0.5, 0.4], "val_loss": [0.6, 0.5],
                          "train_acc": [0.8, 0.85], "val_acc": [0.75, 0.8]}}
    fig = plot_history(histories, save_path=str(tmp_path / "history.png"))
    assert isinstance(fig, plt.Figure)
    assert os.path.getsize(tmp_path / "history.png") > 0

    assert len(plt.get_fignums()) == initial_count  # Không rò rỉ figure


def test_v21_plot_history_start_epochs_offset(tmp_path):
    """V21: start_epochs={"ft":5} → đường ft bắt đầu từ epoch 6."""
    import matplotlib.pyplot as plt

    histories = {
        "base": {"train_loss": [0.5, 0.4], "val_loss": [0.6, 0.5],
                 "train_acc": [0.8, 0.85], "val_acc": [0.75, 0.8]},
        "ft": {"train_loss": [0.35, 0.3], "val_loss": [0.45, 0.4],
               "train_acc": [0.9, 0.92], "val_acc": [0.85, 0.88]},
    }
    fig = plot_history(histories, start_epochs={"ft": 5}, save_path=str(tmp_path / "h.png"))

    # Kiểm tra dữ liệu trục x của đường ft (line thứ 3 và 4 trong axes[0])
    lines = fig.axes[0].lines
    ft_val_line = lines[2]  # base val, base train, ft val, ft train
    xdata = ft_val_line.get_xdata()
    assert list(xdata) == [6, 7]


def test_v22_plot_sample_predictions_empty_and_mismatch():
    """V22: Danh sách rỗng → ValueError; độ dài nhãn lệch ảnh → ValueError."""
    names = ["cat", "dog"]

    with pytest.raises(ValueError):
        plot_sample_predictions([], [], [], names)

    with pytest.raises(ValueError):
        plot_sample_predictions([torch.rand(3, 8, 8)], [0], [0, 1], names)


def test_v23_evaluate_predictions_json_serializable():
    """V23: json.dumps(record) không lỗi; có accuracy_ci95, confusion_matrix là danh sách lồng nhau;
    predictions có y_true, y_pred, confidence."""
    model = make_toy_model()
    loader = make_toy_loader(n_samples=6, batch_size=3)
    result = run_inference(model, loader, torch.device("cpu"))
    names = ["a", "b", "c"]

    record = evaluate_predictions(result, names)
    json_str = json.dumps(record)  # Không lỗi → toàn bộ JSON-safe
    assert "accuracy_ci95" in record
    assert isinstance(record["confusion_matrix"], list)
    assert isinstance(record["confusion_matrix"][0], list)
    assert "predictions" in record
    assert set(record["predictions"].keys()) == {"y_true", "y_pred", "confidence"}
    assert len(record["predictions"]["y_true"]) == 6


def test_v24_format_comparison_table_missing_values():
    """V24: Có header, separator, mỗi run một dòng; giá trị thiếu → dấu gạch dài."""
    runs = {
        "run_full": {
            "metrics": {
                "accuracy": 0.88,
                "accuracy_ci95": [0.85, 0.91],
                "macro_f1": 0.87,
                "confusion_matrix": [[1, 0], [0, 1]],
            },
            "cost": {
                "num_params_total": 1000,
                "num_params_trainable": 500,
                "model_file_mb": 2.5,
                "ms_per_image": 3.0,
                "train_time_phase_s": 100.0,
                "train_time_total_s": 100.0,
            },
            "best_val_acc": 0.9,
        },
        "run_missing": {
            "metrics": {
                "accuracy": 0.8,
                "accuracy_ci95": [0.75, 0.85],
                "macro_f1": 0.79,
                "confusion_matrix": [[1, 0], [0, 1]],
            },
        },
    }

    table = format_comparison_table(runs)
    lines = table.splitlines()
    assert len(lines) >= 3  # header + separator + ít nhất 1 dòng dữ liệu
    assert "| Run |" in lines[0]
    assert "| :--- |" in lines[1]
    assert "run_full" in lines[2]
    assert "run_missing" in lines[3]
    # Giá trị thiếu hiển thị bằng dấu gạch dài
    assert "—" in lines[3]
