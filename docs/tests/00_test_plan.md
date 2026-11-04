# Kế hoạch Kiểm thử — Tổng hợp (đặc tả)

> Nguồn: tách từ `docs/define&designt/01_data.md` đến `05_evaluation_results.md` (mục Kế hoạch kiểm thử của từng module). Loại trừ `00_lab2_description.md` và `06_implementation_plan.md`.

## 1. Mục đích

Bộ đặc tả test này định nghĩa những gì cần kiểm thử, cách kiểm thử và tiêu chí chấp nhận cho từng module trong `going_modular/`. Mục tiêu:

* Kiểm thử **độc lập** từng module trước khi tích hợp.
* Không đánh giá độ chính xác mô hình (việc của `05_evaluation_results.md`); chỉ kiểm tra **tính đúng đắn của cấu trúc, trạng thái và hoạt động**.
* Test nhanh, không cần mạng (trừ nhóm `slow` dùng CIFAR-10 / trọng số ImageNet thật).

## 2. Cấu trúc các file đặc tả test

| File | Nguồn | Nội dung |
| :--- | :--- | :--- |
| `01_test_data_setup.md` | `01_data.md` mục 2.11 | Test `data_setup.py` (chia phân tầng, transform, DataLoader) |
| `02_test_model_builder.md` | `02_model.md` mục 2.11 | Test `model_builder.py` (cấu trúc, đóng băng, BatchNorm, cấu hình) |
| `03_test_engine.md` | `03_engine.md` mục 2.9 | Test `engine.py` (train_step, evaluate_step, vòng lặp, TensorBoard) |
| `04_test_utils.md` | `04_utils.md` mục 2.8 | Test `utils.py` (seed, device, atomic write, artifacts) |
| `05_test_evaluate.md` | `05_evaluation_results.md` mục 2.11 | Test `evaluate.py` (inference, ma trận, độ đo, vẽ) |

## 3. Nguyên tắc viết test (áp dụng chung)

1. **Nhanh, không cần mạng:** dùng nhãn giả / dataset giả / `weights=None`; nhóm `slow` chỉ chạy khi có mạng.
2. **Kiểm tra hành vi quan sát được:** shape, kiểu, `requires_grad`, giá trị tensor, nội dung tệp — không kiểm tra cách cài đặt bên trong.
3. **Mỗi test một ý chính:** tên test cho biết ngay lỗi nằm ở đâu.
4. **Test lỗi dùng `pytest.raises`:** đầu vào sai phải báo `ValueError` / `TypeError` rõ ràng.
5. **Tham số hóa theo kiến trúc:** các test chung chạy lặp cho cả `resnet18` và `vgg16` bằng `@pytest.mark.parametrize`.
6. **Marker `slow`:** đăng ký trong `tests/conftest.py`, dùng cho test cần tải CIFAR-10 hoặc trọng số ImageNet.

```python
# tests/conftest.py
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: tests that download CIFAR-10 or ImageNet weights")
```

## 4. Cách chạy

```bash
# Tất cả test (không slow)
pytest tests/ -v -m "not slow"

# Theo module
pytest tests/test_data_setup.py -v -m "not slow"
pytest tests/test_model_builder.py -v -m "not slow"
pytest tests/test_engine.py -v
pytest tests/test_utils.py -v
pytest tests/test_evaluate.py -v -m "not slow"

# Chỉ test chậm (cần mạng)
pytest tests/ -v -m "slow"
```

## 5. Bản tóm tắt mã test

| Module | Mã test | Số lượng | Đánh dấu |
| :--- | :--- | :--- | :--- |
| data_setup | D1–D15, S1–S3 | 18 | S = slow |
| model_builder | T1–T14c, S1–S2 | ~27 | S = slow |
| engine | E1–E12 | 12 | — |
| utils | U1–U11 | 11 | — |
| evaluate | V1–V24 | 24 | — |

**Tổng:** ~92 case test (trong đó 2 case chậm cho data_setup, 2 case chậm cho model_builder).

---
*Được tạo từ tách quy trình kiểm thử trong `docs/define&designt/`. Không phải mã pytest thực thi — là đặc tả để triển khai vào `tests/`.*
