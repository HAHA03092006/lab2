# Phase 6 — Checklist `evaluate.py` và tests

> **Nguồn chính:** `phase6_implementation_requirements.md`, Phase 6 trong `docs/plans/implementation_plan.md`  
> **Đặc tả module:** `docs/define&designt/05_evaluation_results.md`  
> **Đặc tả test:** `docs/define&designt/tests/05_test_evaluate.md`  
> **Điều kiện đầu vào:** Phase 2 (`count_params`), Phase 4 (metrics persistence), Phase 5 (`history["epoch_time"]`) đã đạt.  
> **Phạm vi triển khai:** `going_modular/evaluate.py`, `tests/test_evaluate.py` và exports evaluate trong `going_modular/__init__.py`.  
> **Mục tiêu:** Helper đánh giá độc lập (6A) và cổng tích hợp (6B); test V1–V24 chạy CPU với dữ liệu/model đồ chơi, Matplotlib backend `Agg`; không tải dataset/weights, không truy cập mạng, không yêu cầu GPU.

## 1. Xác nhận phạm vi và trạng thái ban đầu

- [x] Đọc lại `phase6_implementation_requirements.md`, `05_evaluation_results.md`, `05_test_evaluate.md`; dùng test IDs V1–V24 làm căn cứ kiểm thử.
- [x] Kiểm tra trạng thái Git và tích hợp thay đổi hiện có trong các file thuộc Phase 6; không ghi đè thay đổi ngoài phạm vi.
- [x] Ghi nhận API hiện tại và call sites của `evaluate.py` (notebook, engine, utils) trước khi thay đổi chữ ký hoặc schema.
- [x] Giữ scope ở evaluate và unit tests; không chuyển sang sửa model, training, data_setup, Notebook orchestration hoặc artifacts.
- [x] Xác nhận test nhanh không tải CIFAR-10/ImageNet, không cần GPU, không truy cập mạng; Matplotlib dùng backend `Agg`.

## 2. Phase 6A — Helper đánh giá độc lập

### 2.1 Suy luận một lượt (V1–V4)

- [x] `run_inference(model, dataloader, device, loss_fn=None)` trả về `InferenceResult(logits, y_true, loss)`; logits shape `[N, C]`, y_true shape `[N]`.
- [x] Xử lý batch cuối không đầy; giữ thứ tự dự đoán/nhãn (`shuffle=False`).
- [x] Loss tính trên toàn tập (không lấy trung bình không trọng số theo batch); `loss is None` khi không truyền `loss_fn`.
- [x] Dataloader rỗng báo `ValueError`; model vào `eval()` và không tạo gradient (`torch.inference_mode`).
- [x] `predict_dataloader`, `predict_probs` giữ nguyên cho backward compatibility.

### 2.2 Ma trận nhầm lẫn (V5–V8)

- [x] `compute_confusion_matrix(*, y_true, y_pred, num_classes)` tham số keyword-only; hàng là nhãn thật, cột là dự đoán.
- [x] Ví dụ tính tay: `y_true=[0,0,1,2,2]`, `y_pred=[0,1,1,2,0]`, `C=3` → `[[1,1,0],[0,1,0],[1,0,1]]`.
- [x] Dự đoán hoàn hảo → ma trận chéo; tổng mỗi hàng bằng số mẫu thật của lớp đó.
- [x] Báo `ValueError` cho `num_classes < 1`, tensor không phải 1-D, khác độ dài, nhãn ngoài `[0, C-1]`, `num_classes=0`.
- [x] Gọi với tham số vị trí (không keyword) báo `TypeError`.

### 2.3 Độ đo và Wilson CI (V9–V12)

- [x] `classification_metrics(cm)`: accuracy, precision, recall, F1 theo lớp và macro-average; chia cho 0 → kết quả 0, không bao giờ `NaN`.
- [x] Ví dụ tính tay: accuracy `0.6`; precision `[0.5, 0.5, 1.0]`; recall `[0.5, 1.0, 0.5]`; F1 `[0.5, 0.6667, 0.6667]`.
- [x] Macro tính trên mọi lớp, kể cả lớp có precision 0.
- [x] `wilson_interval(correct, total, z=1.96)`: `wilson_interval(50, 100)` xấp xỉ `[0.4038, 0.5962]`; mọi cận trong `[0, 1]`; `total <= 0` hoặc `correct` ngoài `[0, total]` báo `ValueError`.

### 2.4 Phân tích lỗi (V13–V14)

- [x] `top_confusions(cm, class_names, k=5)`: chỉ liệt kê ô khác 0, thứ tự giảm dần; ma trận chéo trả về danh sách rỗng.
- [x] `misclassified_indices(y_true, y_pred, logits=None, max_items=None)`: không có logits → đúng thứ tự dữ liệu; có logits → theo độ tự tin giảm dần; `max_items` cắt đúng.

### 2.5 Giải chuẩn hóa (V15–V16)

- [x] `denormalize(images, mean, std)`: tensor `[C,H,W]` hoặc `[N,C,H,W]`; công thức `clamp(X_norm * std + mean, 0, 1)`; số kênh mismatch báo `ValueError`.
- [x] `get_mean_std(transform, transform_weights=None)`: ưu tiên đọc mean/std từ transform; nếu thiếu thì lấy từ preprocessing preset của `transform_weights` (mock trong test, không tải model weights); nếu không có stats và không xác định được preset → báo `ValueError` (không đoán ImageNet).

### 2.6 Chi phí (V17–V19)

- [x] `measure_inference_time(model, images, device, warmup=3, repeats=10)`: trả về `batch_size`, `ms_per_batch`, `ms_per_image`, `device`, `peak_gpu_mb` (`None` trên CPU); `warmup < 0` hoặc `repeats < 1` báo `ValueError`.
- [x] `file_size_mb(path)`: dùng MiB; thiếu tệp báo `FileNotFoundError`.
- [x] `build_cost_record(num_params, model_path=None, timing=None, history=None, source_history=None)`: trả về `train_time_phase_s` (tổng `epoch_time` của phase hiện tại) và `train_time_total_s` (cộng thêm `source_history` nếu có); thiếu `timing` → `None`, không lỗi; thiếu `model_path` → `FileNotFoundError`.

## 3. Phase 6B — Cổng tích hợp

- [x] `build_cost_record` tích hợp `model_builder.count_params` (num_params), `engine` history (epoch_time) và `source_history`.
- [x] Thời gian train phân biệt `train_time_phase_s` (phase hiện tại) và `train_time_total_s` (cộng phase nguồn), đúng field names theo thiết kế mới nhất.
- [x] `evaluate_predictions(result, class_names)` trả cả block metrics lẫn `predictions` (`y_true`, `y_pred`, `confidence` tối đa, JSON-safe, không lưu logits).
- [x] Xác minh tương thích với `utils.save_test_metrics`/`load_test_metrics`; persistence chỉ ở library/helper boundary nếu phù hợp, không triển khai orchestration reuse trong Notebook (thuộc Phase 7).
- [x] Cập nhật evaluate exports trong `going_modular/__init__.py` cho API công khai đã chọn; test các imports cần hỗ trợ, không nhất thiết export mọi helper nội bộ.

## 4. Tạo `tests/test_evaluate.py`

- [x] Viết test V1–V24 theo test spec chi tiết; mỗi test kiểm tra hành vi quan sát được, dùng toy data và `tmp_path`.
- [x] Dùng Matplotlib backend `Agg`; ghi tệp ảnh vào `tmp_path`; kiểm tra figure không rò rỉ (`plt.close`).
- [x] Kiểm tra JSON serialization của bản ghi kết quả (`json.dumps` không lỗi).
- [x] Kiểm tra tensor không rơi vào bản ghi JSON-ready; không có `NaN` trong precision/recall/F1/macro.
- [x] Đánh dấu test cần thiết (nếu có) bằng `@pytest.mark.slow`; test nhanh không tải dataset/weights, không truy cập mạng.
- [x] Marker `slow` đã được đăng ký trong `tests/conftest.py` (từ Phase 1/2).

## 5. Tương thích và ranh giới kiến trúc

- [x] Không sửa `compute_confusion_matrix` call site trong Notebook ở Phase 6 (Phase 7 là nơi sửa positional call sang keyword-only).
- [x] Không sửa Notebook hoặc hoàn thiện RUN_MODE/reuse orchestration (thuộc Phase 7).
- [x] Không thêm scikit-learn hoặc McNemar testing (không thuộc hợp đồng hiện hành).
- [x] Không thay đổi contracts thuộc `model_builder`, `utils` hoặc `engine` trừ khi có lỗi tích hợp được chứng minh.

## 6. Kiểm thử và cổng nghiệm thu

- [x] Chạy toàn bộ test nhanh: `python -m pytest tests/test_evaluate.py -v`.
- [x] Chạy full suite: `python -m pytest -q` để xác nhận không regression.
- [x] Chạy collection: `python -m pytest --collect-only tests/test_evaluate.py`.
- [x] Chạy smoke test package: `python -m pytest tests/test_smoke.py -v`.
- [x] Xác nhận V1–V24 đều được thu thập và đạt; kiểm tra Agg/headless, JSON serialization, tích hợp helper persistence.
- [x] Xác nhận test không truy cập mạng/tải dữ liệu thật, không cần GPU.
- [x] Chạy `git diff --check` và rà soát diff, không bao gồm thay đổi Notebook.

## 7. Cổng hoàn tất Phase 6

- [x] Helper đánh giá (run_inference, confusion matrix, metrics, Wilson CI, error analysis, denormalize, cost, plots, comparison table) đáp ứng hợp đồng `05_evaluation_results.md`.
- [x] Cổng tích hợp (build_cost_record với source_history, train_time_phase_s/total_s, evaluate_predictions với predictions block) đáp ứng thiết kế.
- [x] Test V1–V24 đạt; test chạy CPU, Agg, không mạng/GPU/dataset thật.
- [x] Package exports evaluate được cập nhật; smoke test và collection đạt; `git diff --check` sạch.
- [x] Không có thay đổi ngoài phạm vi Phase 6.

**Kết quả Phase 6:** ☐ Chưa đạt  ☒ Đạt  
**Ngày rà soát:** 2026-11-05  
**Ghi chú / quyết định còn mở:**  

---

Sau khi đạt cổng này, tiếp tục **Phase 7 — tích hợp Notebook (RUN_MODE, reuse, call sites)** theo `docs/plans/implementation_plan.md`.
