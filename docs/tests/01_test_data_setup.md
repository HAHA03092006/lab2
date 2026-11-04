# Đặc tả Kiểm thử — `data_setup.py`

> Nguồn: `docs/define&designt/01_data.md` mục 2.11 (Kế hoạch kiểm thử `tests/test_data_setup.py`).

## 1. Mục tiêu

Kiểm chứng `data_setup.py` **độc lập** trước khi tích hợp. Test không đánh giá độ chính xác mô hình; chỉ kiểm tra **tính đúng đắn của phép chia, biến đổi và đóng gói dữ liệu**.

## 2. Nguyên tắc

1. **Nhanh, không cần mạng:** test thuật toán dùng nhãn giả (`labels = [i % 10 for i in range(50000)]`) và dataset giả (`torchvision.datasets.FakeData`), không tải CIFAR-10.
2. **Kiểm tra hành vi quan sát được:** tập chỉ mục, shape, kiểu dữ liệu, giá trị tensor — không kiểm tra cách cài đặt bên trong.
3. **Mỗi test một ý chính.**
4. **Test lỗi dùng `pytest.raises`:** đầu vào sai phải báo `ValueError` rõ ràng.
5. **Test cần dữ liệu thật đánh dấu `slow`:** chỉ chạy khi tải được CIFAR-10; nếu không thì tự bỏ qua.

## 3. Sáu nhóm kiểm thử

| Nhóm | Khía cạnh | Câu hỏi cần trả lời | Test |
| :--- | :--- | :--- | :--- |
| A | Chia phân tầng đúng | Hai tập có tách biệt, đủ số lượng, đúng tỷ lệ mỗi lớp không? | D1, D2, D3 |
| B | Tái lập và phần dư | Cùng seed có cùng kết quả? Phần dư được phân bổ đúng? | D4, D5 |
| C | Kiểm tra đầu vào | Đầu vào sai có bị chặn bằng lỗi rõ ràng không? | D6, D7 |
| D | Bộ tiền xử lý | Đầu ra đúng shape, kiểu, chuẩn hóa; augmentation đúng chỗ? | D8, D9, D10, D11 |
| E | Wrapper và cách ly biến đổi | Ánh xạ chỉ mục đúng? Không nhân bản dữ liệu? Train có ngẫu nhiên còn Validation thì không? | D12, D13, D14, D15 |
| F | DataLoader thật (chậm) | Ba loader có đúng kích thước, shape, kiểu, cờ shuffle? | S1, S2, S3 |

## 4. Mô tả chi tiết từng test

### Nhóm A: Chia phân tầng đúng

| Mã | Hàm test (dự kiến) | Cách kiểm thử (ví dụ) | Kỳ vọng | Lỗi bắt được |
| :--- | :--- | :--- | :--- | :--- |
| D1 | `test_d1_disjoint_and_complete` | Nhãn giả 50,000 phần tử; gọi `stratified_split_indices(labels, 5000)`. Chuyển hai danh sách thành `set`. | Giao rỗng; hợp bằng `set(range(50000))`; kích thước 45,000 và 5,000 | Một chỉ mục nằm ở cả hai tập (rò rỉ); mất chỉ mục. |
| D2 | `test_d2_balanced_per_class` | Đếm số mẫu từng lớp trong `val_indices` và `train_indices` bằng `Counter` trên nhãn tương ứng. | Mỗi lớp 0–9 có đúng 500 mẫu Validation và 4,500 mẫu Train | Chia ngẫu nhiên thuần túy làm lệch tỷ lệ lớp. |
| D3 | `test_d3_returns_python_lists_of_ints` | Kiểm tra kiểu trả về. | Hai giá trị là `list` các `int` | Trả về tensor làm `_TransformSubset` hoặc `Subset` lỗi kiểu chỉ mục. |

### Nhóm B: Tái lập và phần dư

| Mã | Hàm test (dự kiến) | Cách kiểm thử (ví dụ) | Kỳ vọng | Lỗi bắt được |
| :--- | :--- | :--- | :--- | :--- |
| D4 | `test_d4_reproducible_by_seed` | Gọi hai lần cùng `seed=42`; gọi thêm một lần `seed=7`. Đặt `torch.manual_seed` toàn cục khác nhau giữa hai lần gọi đầu. | Hai lần `seed=42` cho kết quả giống hệt dù seed toàn cục khác; `seed=7` cho kết quả khác | Dùng bộ sinh toàn cục thay vì `Generator` riêng. |
| D5 | `test_d5_remainder_distribution` | Nhãn có số mẫu `[4, 3, 3]`, `val_size=3` (ví dụ mục 2.5). | Mỗi lớp có đúng 1 mẫu Validation; tổng bằng 3; mỗi lớp còn ≥ 1 mẫu Train | Phân bổ phần dư sai (tổng lệch `val_size`, sót lớp). |

### Nhóm C: Kiểm tra đầu vào

| Mã | Hàm test (dự kiến) | Cách kiểm thử (ví dụ) | Kỳ vọng | Lỗi bắt được |
| :--- | :--- | :--- | :--- | :--- |
| D6 | `test_d6_invalid_val_size` | `val_size=0`, `val_size=-5`, `val_size=N`. | Cả ba đều `ValueError` | Giá trị biên lọt qua và tạo tập rỗng. |
| D7 | `test_d7_invalid_labels` | Một lớp chỉ có 1 mẫu; một `val_size` lớn đến mức một lớp hết mẫu Train. | `ValueError` | Một lớp không còn mẫu Train, gây lỗi khó đoán ở bước huấn luyện. |

### Nhóm D: Bộ tiền xử lý

| Mã | Hàm test (dự kiến) | Cách kiểm thử (ví dụ) | Kỳ vọng | Lỗi bắt được |
| :--- | :--- | :--- | :--- | :--- |
| D8 | `test_d8_transform_output` | Tạo ảnh PIL 32×32, áp `get_pretrained_transform()` cho cả `"vgg16"` và `"resnet18"`. | Tensor shape `(3, 224, 224)`, kiểu `float32` | Quên phóng to hoặc cắt; sai kiểu dữ liệu. |
| D9 | `test_d9_normalization_values` | Ảnh trắng toàn 255 qua transform (không augment). | Mỗi kênh ≈ `(1 - mean_c) / std_c`, sai số nhỏ | Bỏ chuẩn hóa hoặc dùng sai mean/std. |
| D10 | `test_d10_augment_flag` | Kiểm tra `transforms` bên trong `Compose`. | `augment=True`: phần tử đầu là `RandomHorizontalFlip`; `augment=False`: không có | Augmentation lọt vào luồng Validation/Test. |
| D11 | `test_d11_unknown_model_name` | `model_name="foo"`. | `ValueError` | Báo `KeyError` khó hiểu hoặc âm thầm dùng mặc định. |

### Nhóm E: Wrapper và cách ly biến đổi

| Mã | Hàm test (dự kiến) | Cách kiểm thử (ví dụ) | Kỳ vọng | Lỗi bắt được |
| :--- | :--- | :--- | :--- | :--- |
| D12 | `test_d12_len_and_index_mapping` | Dataset giả có 20 mẫu; `indices=[3, 7, 11]`; `transform` ghi lại ảnh gốc. | `len == 3`; phần tử thứ `i` trả về mẫu gốc `indices[i]` cùng nhãn đúng | Ánh xạ chỉ mục sai (dùng `index` thay vì `indices[index]`). |
| D13 | `test_d13_shares_underlying_dataset` | Dựng hai wrapper trên cùng một dataset gốc. | `train.dataset is val.dataset` (cùng một đối tượng) | Nhân bản dataset, tốn gấp đôi RAM. |
| D14 | `test_d14_eval_is_deterministic` | Lấy cùng một mẫu Validation hai lần qua `eval_transform`. | `torch.equal` đúng | Biến đổi ngẫu nhiên lọt vào luồng Validation. |
| D15 | `test_d15_train_augmentation_varies` | Lấy cùng một mẫu Train 20 lần qua `train_transform` có `augment=True`. | Có ít nhất 2 kết quả khác nhau | Augmentation không hoạt động trên tập Train. |

### Nhóm F: DataLoader thật (đánh dấu `slow`)

| Mã | Hàm test (dự kiến) | Cách kiểm thử (ví dụ) | Kỳ vọng | Lỗi bắt được |
| :--- | :--- | :--- | :--- | :--- |
| S1 | `test_s1_dataset_sizes` | Gọi `create_train_val_test_dataloaders(batch_size=32)`; đo `len(loader.dataset)`. | Train 45,000; Validation 5,000; Test 10,000; `class_names` có 10 phần tử | Chia sai kích thước; trả về sai tên lớp. |
| S2 | `test_s2_batch_shape_and_dtype` | Lấy một batch từ mỗi loader. | `images.shape == (32, 3, 224, 224)`, `float32`; `labels.shape == (32,)` | Sai shape hoặc kiểu dữ liệu đầu vào mô hình. |
| S3 | `test_s3_shuffle_flags` | Kiểm tra `loader.sampler`. | Train dùng `RandomSampler`; Validation và Test dùng `SequentialSampler` | Validation bị xáo trộn làm kết quả không so sánh được giữa các lần chạy. |

Nếu không tải được CIFAR-10 (mất mạng, bị chặn), các test nhóm F bắt lỗi tải và **bỏ qua** thay vì báo lỗi.

## 5. Hai ví dụ đọc test chi tiết

**Ví dụ 1: D2 (chia cân bằng theo lớp)**

```python
labels = [i % 10 for i in range(50000)]           # nhãn giả, mỗi lớp 5,000 mẫu
train_idx, val_idx = stratified_split_indices(labels, val_size=5000, seed=42)

val_counts = Counter(labels[i] for i in val_idx)
train_counts = Counter(labels[i] for i in train_idx)

assert all(val_counts[c] == 500 for c in range(10))
assert all(train_counts[c] == 4500 for c in range(10))
```

Test chỉ dùng nhãn giả nên chạy tức thì, không cần tải CIFAR-10. Nếu thuật toán dùng `random_split`, số mẫu mỗi lớp sẽ dao động quanh 500 và test thất bại ngay ở dòng `assert` đầu tiên.

**Ví dụ 2: D14 và D15 (cách ly biến đổi giữa Train và Validation)**

```python
raw = FakeData(size=100, image_size=(3, 32, 32))   # dataset gốc, transform=None
train = _TransformSubset(raw, indices=[0, 1, 2], transform=get_pretrained_transform(augment=True))
val   = _TransformSubset(raw, indices=[0, 1, 2], transform=get_pretrained_transform(augment=False))

# D14: Validation luôn cho cùng một kết quả
assert torch.equal(val[0][0], val[0][0])

# D15: Train có ngẫu nhiên (lật ngang), nhiều lần lấy phải thấy ít nhất 2 kết quả khác nhau
outputs = [train[0][0] for _ in range(20)]
assert any(not torch.equal(outputs[0], o) for o in outputs[1:])
```

Hai wrapper cùng trỏ vào `raw` nhưng khác `transform`. Cặp test này chứng minh hai luồng biến đổi tách biệt mà không cần hai bản dữ liệu.

## 6. Cách chạy và ánh xạ tiêu chí chấp nhận

```bash
pytest tests/test_data_setup.py -v                 # tất cả
pytest tests/test_data_setup.py -v -m "not slow"   # bỏ qua test cần tải CIFAR-10
```

Marker `slow` được đăng ký trong `tests/conftest.py`.

| Tiêu chí (mục 1.5) | Test chứng minh |
| :--- | :--- |
| 1. Giao rỗng, hợp đủ 50,000 | D1 |
| 2. Validation 500/lớp, Train 4,500/lớp | D2, S1 |
| 3. Batch `(B, 3, 224, 224)` kiểu `float32` | D8, S2 |
| 4. Tái lập theo `seed` | D4 |
| 5. Validation và Test không biến đổi ngẫu nhiên | D10, D14 |
| 6. Toàn bộ test đạt | Mọi test ở trên |

**Giới hạn:** Không kiểm tra tốc độ nạp dữ liệu, hiệu năng đa tiến trình (`num_workers > 0`) hay `pin_memory` trên GPU thật; không kiểm tra chất lượng thị giác của ảnh sau khi phóng to; nhóm F cần tải CIFAR-10 (khoảng 170 MB) ở lần chạy đầu.
