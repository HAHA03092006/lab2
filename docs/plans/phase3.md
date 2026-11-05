# Phase 3 — Checklist `data_setup.py` và tests

> **Nguồn chính:** Phase 3 trong `docs/plans/implementation_plan.md`  
> **Đặc tả module:** `docs/define&designt/01_data.md`  
> **Đặc tả test:** `docs/define&designt/tests/01_test_data_setup.md`  
> **Điều kiện đầu vào:** Phase 1 đạt; các API transform model cần thiết từ Phase 2 sẵn sàng cho Notebook gọi qua Dependency Injection.  
> **Phạm vi triển khai:** `going_modular/data_setup.py`, `tests/test_data_setup.py` và exports data liên quan trong `going_modular/__init__.py`.

## 1. Xác nhận phạm vi và trạng thái ban đầu

- [x] Đọc lại hợp đồng module và test spec; dùng danh sách test chi tiết D1–D15, S1–S3 làm nguồn test IDs.
- [x] Kiểm tra trạng thái Git và tích hợp thay đổi hiện có trong các file sẽ sửa; không ghi đè thay đổi ngoài phạm vi.
- [x] Ghi nhận API hiện tại và call sites của `create_train_val_test_dataloaders` trước khi đổi chữ ký.
- [x] Giữ scope ở data setup và unit tests; không chuyển sang sửa model, training/evaluation, Notebook orchestration hoặc artifacts.
- [x] Xác nhận test nhanh không tải CIFAR-10 thật; các kiểm thử cần download phải được cô lập trong nhóm `slow`.

## 2. Split phân tầng và kiểm tra đầu vào

- [x] Bảo đảm `stratified_split_indices` tạo tập Train/Validation không giao nhau và đầy đủ chỉ số mẫu.
- [x] Bảo đảm số mẫu từng lớp được phân bổ đúng theo chiến lược và quy tắc phân phối phần dư của đặc tả.
- [x] Bảo đảm kết quả split lặp lại khi cùng seed và không làm thay đổi global RNG ngoài ý muốn.
- [x] Trả đúng kiểu dữ liệu công khai đã đặc tả cho danh sách indices.
- [x] Báo `ValueError` rõ ràng với `val_size` không hợp lệ, labels không hợp lệ hoặc trường hợp không thể chia theo đặc tả.
- [x] Không mở rộng cam kết thành hỗ trợ mọi dữ liệu mất cân bằng/rare-class ngoài phạm vi CIFAR-10 cân bằng nếu đặc tả không yêu cầu.

## 3. Transform và `_TransformSubset`

- [x] Giữ `get_pretrained_transform` độc lập với `model_builder`, hỗ trợ đúng model/preset được đặc tả.
- [x] Kiểm tra kích thước, kiểu tensor và normalization của transform theo test spec; không tự thêm augmentation vào eval transform.
- [x] Bảo đảm `augment=True` chỉ thêm augmentation train đã đặc tả; `augment=False` giữ preprocessing deterministic.
- [x] Báo lỗi rõ ràng cho tên model không hỗ trợ.
- [x] Bảo đảm `_TransformSubset` ánh xạ chỉ số đúng về underlying dataset và áp transform khi truy cập mẫu.
- [x] Bảo đảm Train/Validation có thể chia sẻ cùng raw underlying dataset mà không ghi đè transform lẫn nhau; Test dùng preprocessing eval.

## 4. Factory ba DataLoader

- [x] Giữ factory nhận `train_transform` và `eval_transform` qua Dependency Injection; `data_setup` không import `model_builder`.
- [x] Khi `train_transform` không được truyền, fallback Train phải gọi transform với `augment=True`.
- [x] Khi `eval_transform` không được truyền, fallback Validation/Test phải dùng preprocessing eval không augmentation.
- [x] Bổ sung tham số `generator` cho Train DataLoader và truyền generator đó vào loader để cách ly thứ tự shuffle.
- [x] Thêm tham số mới mà không đổi vị trí/ý nghĩa positional của các tham số hiện có; ưu tiên thêm ở cuối chữ ký hoặc cách tương thích tương đương.
- [x] Chỉ Train loader shuffle; Validation và Test không shuffle.
- [x] Bảo đảm `pin_memory` theo khả năng CUDA của runtime và `num_workers` theo cấu hình truyền vào.
- [x] Bảo đảm các kích thước dataset/split và batch behavior đúng hợp đồng.
- [x] Kiểm tra chế độ test nhanh có thể thay/mock dataset để không cần download CIFAR-10 thật.
- [x] Cập nhật exports data trong `going_modular/__init__.py` nếu API công khai thay đổi; không sửa các exports thuộc module khác.

## 5. Tạo `tests/test_data_setup.py`

- [x] Viết test theo đặc tả chi tiết, mỗi test kiểm tra hành vi quan sát được:
  - [x] **D1–D5:** split disjoint/complete, cân bằng theo lớp, kiểu indices, tái lập theo seed và phân phối phần dư.
  - [x] **D6–D7:** từ chối `val_size` và labels không hợp lệ bằng exception đã đặc tả.
  - [x] **D8–D11:** transform output, normalization, cờ augmentation và tên model không hỗ trợ.
  - [x] **D12–D13:** độ dài/index mapping của subset và chia sẻ underlying dataset.
  - [x] **D14–D15:** eval transform deterministic và augmentation Train có thể tạo biến thiên.
- [x] Bổ sung kiểm thử factory cho fallback Train augmentation, eval fallback, generator/shuffle isolation, loader settings và positional compatibility; nếu test spec ánh xạ một hành vi vào ID cụ thể thì giữ nguyên ID đó.
- [x] Test nhanh dùng dữ liệu nhỏ/fake hoặc monkeypatch; không tải CIFAR-10 thật.
- [x] Đánh dấu kiểm thử cần CIFAR-10 thật bằng `@pytest.mark.slow` và dùng đúng IDs:
  - [x] **S1** `test_s1_dataset_sizes`
  - [x] **S2** `test_s2_batch_shape_and_dtype`
  - [x] **S3** `test_s3_shuffle_flags`
- [x] Nếu không tải được CIFAR-10, S1–S3 phải `pytest.skip` với nguyên nhân môi trường rõ ràng; không xem skip là pass của hành vi dataset thật.
- [x] Xác nhận marker `slow` đã được đăng ký trong `tests/conftest.py`; không khai báo trùng nếu đã có.

## 6. Tương thích và ranh giới kiến trúc

- [x] Không thay đổi tên API hoặc ý nghĩa tham số đang được Notebook sử dụng ngoài phần cần thiết và có kiểm tra tương thích.
- [x] Giữ transform dependency injection để Notebook lấy preprocessing từ model layer; fallback trong data module vẫn hoạt động độc lập.
- [x] Không thêm lưu `config.json`, checkpoint, TensorBoard hoặc logic resume vào data module.
- [x] Không chỉnh sửa notebook để tích hợp toàn pipeline; tích hợp đó thuộc Phase 7.
- [x] Không refactor `model_builder.py`, `utils.py`, `engine.py` hoặc `evaluate.py`.

## 7. Kiểm thử và cổng nghiệm thu

- [x] Chạy toàn bộ test nhanh:
  - [x] `python -m pytest tests/test_data_setup.py -v -m "not slow"`
- [x] Chạy test slow riêng khi môi trường cho phép tải CIFAR-10:
  - [x] `python -m pytest tests/test_data_setup.py -v -m slow`
- [x] Chạy collection:
  - [x] `python -m pytest --collect-only tests/test_data_setup.py`
- [x] Chạy smoke test package sau khi cập nhật exports:
  - [x] `python -m pytest tests/test_smoke.py -v`
- [x] Xác nhận D1–D15 đều được thu thập và đạt; S1–S3 đều được thu thập và kết quả pass/skip có lý do được ghi nhận.
- [x] Xác nhận test nhanh không truy cập mạng/tải dữ liệu thật.
- [x] Xác nhận `generator` điều khiển được thứ tự shuffle Train mà không làm Validation/Test shuffle.
- [x] Chạy `git diff --check` và rà soát diff, đặc biệt chữ ký factory và thay đổi positional compatibility.

## 8. Ngoài phạm vi Phase 3

- Refactor kiến trúc model hoặc thay đổi transform preset do `model_builder` sở hữu.
- Triển khai logic seed toàn hệ thống, lưu cấu hình/artifacts, checkpoint/resume hoặc huấn luyện.
- Chạy CIFAR-10 experiments hoặc đánh giá mô hình.
- Sửa toàn bộ Notebook hoặc thay đổi behavior ngoài tương thích cần thiết.
- Xóa artifacts/checkpoints hoặc API cũ.

## 9. Cổng hoàn tất Phase 3

- [x] Split, transform, subset và ba DataLoader đáp ứng hợp đồng `01_data.md`.
- [x] Fallback Train augmentation và generator riêng cho Train đã được triển khai, kiểm thử.
- [x] Test nhanh D1–D15 đạt; test S1–S3 được đánh dấu slow, có kết quả pass hoặc skip với lý do rõ ràng.
- [x] Dependency Injection, tính độc lập khỏi `model_builder` và tương thích call sites được xác nhận.
- [x] Smoke test package và collection test đạt; `git diff --check` không có lỗi.
- [x] Không có thay đổi ngoài phạm vi Phase 3.

**Kết quả Phase 3:** ☐ Chưa đạt  ☒ Đạt  
**Ngày rà soát:** 2026-11-05  
**Ghi chú / quyết định còn mở:**  

---

Sau khi đạt cổng này, tiếp tục **Phase 4 — `utils.py` và `test_utils.py`** theo `docs/plans/implementation_plan.md`.