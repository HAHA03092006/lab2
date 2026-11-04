# Đặc tả Kiểm thử — `evaluate.py`

> Nguồn: `docs/define&designt/05_evaluation_results.md` mục 2.11 (Kế hoạch kiểm thử `tests/test_evaluate.py`).

## 1. Mục tiêu

Test kiểm chứng **tính đúng đắn của các phép tính và hàm vẽ**, không đánh giá chất lượng mô hình.

## 2. Nguyên tắc

1. **Mô hình và dữ liệu đồ chơi:** mạng `Linear` nhỏ, tập tổng hợp vài chục mẫu; ma trận nhầm lẫn kiểm tra bằng ví dụ **tính tay** ở mục 2.4 của `05_evaluation_results.md`.
2. **Chạy không màn hình:** đặt backend `Agg`; ghi tệp ảnh vào `tmp_path`.
3. **Chạy trên CPU và không cần mạng.** Hàm đo thời gian chỉ kiểm tra cấu trúc kết quả và tính hợp lệ, không kiểm tra giá trị tuyệt đối.
4. **Kiểm tra hành vi quan sát được** (giá trị, kiểu, shape, nội dung tệp), mỗi test một ý chính.

## 3. Bảy nhóm kiểm thử

| Nhóm | Khía cạnh | Câu hỏi cần trả lời | Test |
| :--- | :--- | :--- | :--- |
| A | Suy luận một lượt | Kết quả đúng shape, đúng thứ tự, loss khớp tính trên toàn tập? | V1, V2, V3, V4 |
| B | Ma trận nhầm lẫn | Dựng đúng, đúng hướng, đầu vào sai bị chặn? | V5, V6, V7, V8 |
| C | Độ đo và khoảng tin cậy | Công thức đúng? Không có `NaN`? | V9, V10, V11, V12 |
| D | Phân tích lỗi | Chỉ liệt kê lỗi thật, đúng thứ tự? | V13, V14 |
| E | Giải chuẩn hóa | Đảo ngược đúng, lấy mean/std đúng? | V15, V16 |
| F | Chi phí | Cấu trúc kết quả đúng, đầu vào sai bị chặn? | V17, V18, V19 |
| G | Hình vẽ và tổng hợp | Trả về `Figure`, tạo tệp, JSON hợp lệ, bảng đúng? | V20, V21, V22, V23, V24 |

## 4. Mô tả chi tiết từng test

### Nhóm A: Suy luận một lượt

| Mã | Cách kiểm thử (ví dụ) | Kỳ vọng | Lỗi bắt được |
| :--- | :--- | :--- | :--- |
| V1 | `run_inference` trên `Linear(4, 3)` và 10 mẫu, `batch_size=4`. | `logits` shape `(10, 3)`; `y_true` shape `(10,)`; `loss is None` khi không truyền `loss_fn` | Sai shape; nối batch sai. |
| V2 | Truyền `loss_fn=CrossEntropyLoss()`; so với loss tính một lần trên cả 10 mẫu. | Hai giá trị bằng nhau (sai số `1e-6`) | Lấy trung bình cộng loss theo batch thay vì trên toàn tập. |
| V3 | `shuffle=False`; nhãn tăng dần `0, 1, 2, …` | `y_true` giữ đúng thứ tự dữ liệu | Thứ tự bị xáo, làm hỏng việc ánh xạ ảnh sai về chỉ mục. |
| V4 | Dataloader rỗng; kiểm tra mô hình sau khi gọi. | `ValueError`; `model.training == False` | Lỗi mơ hồ khi rỗng; quên `eval()`. |

### Nhóm B: Ma trận nhầm lẫn

| Mã | Cách kiểm thử (ví dụ) | Kỳ vọng | Lỗi bắt được |
| :--- | :--- | :--- | :--- |
| V5 | `y_true=[0,0,1,2,2]`, `y_pred=[0,1,1,2,0]`, `num_classes=3`. | `[[1,1,0],[0,1,0],[1,0,1]]` | Công thức chỉ số sai. |
| V6 | Dự đoán hoàn hảo. | Ma trận chéo; tổng mỗi hàng bằng số mẫu thật của lớp | Ô ngoài đường chéo khác `0`. |
| V7 | Ví dụ lệch: mọi mẫu thật là lớp 2 nhưng dự đoán là lớp 1. | Số đếm nằm ở **hàng 2, cột 1** (hàng là nhãn thật) | Ma trận bị chuyển vị. |
| V8 | Nhãn bằng `num_classes`; nhãn âm; hai tensor khác độ dài; tensor 2-D; `num_classes=0`. | Tất cả đều `ValueError`; gọi với tham số vị trí (không keyword) báo `TypeError` | Lỗi `reshape` khó hiểu; hoán đổi tham số không bị phát hiện. |

### Nhóm C: Độ đo và khoảng tin cậy

| Mã | Cách kiểm thử (ví dụ) | Kỳ vọng | Lỗi bắt được |
| :--- | :--- | :--- | :--- |
| V9 | Ma trận ở V5. | accuracy `= 0.6`; precision `[0.5, 0.5, 1.0]`; recall `[0.5, 1.0, 0.5]`; F1 `[0.5, 0.6667, 0.6667]` | Nhầm precision và recall. |
| V10 | `y_true=[0,1,2]`, `y_pred=[0,1,0]` (lớp 2 không bao giờ được dự đoán). | precision lớp 2 bằng `0`; không có giá trị `NaN` nào trong kết quả | `NaN` lan vào macro-average. |
| V11 | Ma trận ở V5. | macro precision `≈ 0.6667`; macro recall `≈ 0.6667`; macro F1 `≈ 0.6111` | Macro tính nhầm (ví dụ theo mẫu thay vì theo lớp). |
| V12 | `wilson_interval(50, 100)`; `(0, 10)`; `(10, 10)`; `(5, 0)`; `(11, 10)`. | Khoảng xấp xỉ `[0.4038, 0.5962]`; mọi cận trong `[0, 1]`; hai trường hợp cuối `ValueError` | Công thức sai; cận vượt `[0, 1]` ở biên. |

### Nhóm D: Phân tích lỗi

| Mã | Cách kiểm thử (ví dụ) | Kỳ vọng | Lỗi bắt được |
| :--- | :--- | :--- | :--- |
| V13 | `top_confusions` trên ma trận chéo; trên ma trận có ít hơn `k` kiểu nhầm. | Ma trận chéo trả về danh sách rỗng; mọi mục có số đếm `> 0` và giảm dần | Liệt kê dòng vô nghĩa như `("cat", "cat", 0)`. |
| V14 | `misclassified_indices` với và không có `logits`; `max_items`. | Không có `logits`: đúng thứ tự dữ liệu; có `logits`: theo độ tự tin giảm dần; `max_items` cắt đúng | Sai thứ tự; bỏ sót hoặc thêm chỉ mục. |

### Nhóm E: Giải chuẩn hóa

| Mã | Cách kiểm thử (ví dụ) | Kỳ vọng | Lỗi bắt được |
| :--- | :--- | :--- | :--- |
| V15 | Tensor ngẫu nhiên trong `[0, 1]` qua chuẩn hóa rồi `denormalize`; tensor ngoài miền; số kênh sai. | Khôi phục giá trị ban đầu (sai số `1e-6`); kết quả luôn trong `[0, 1]`; số kênh sai gây `ValueError` | Công thức đảo ngược sai; quên `clamp`. |
| V16 | `get_mean_std` trên `Compose([RandomHorizontalFlip(), Normalize(mean, std)])`; trên transform không có mean/std. | Trả về đúng `mean`, `std`; trường hợp sau trả về mean/std mặc định ImageNet | Không gãy khi transform thiếu thuộc tính. |

### Nhóm F: Chi phí

| Mã | Cách kiểm thử (ví dụ) | Kỳ vọng | Lỗi bắt được |
| :--- | :--- | :--- | :--- |
| V17 | `measure_inference_time` với `warmup=1`, `repeats=3`; sau đó `warmup=-1`, `repeats=0`. | Có đủ khóa `batch_size`, `ms_per_batch`, `ms_per_image`, `device`; thời gian `> 0`; `ms_per_image * batch_size` xấp xỉ `ms_per_batch`; `peak_gpu_mb is None` trên CPU; hai tham số sai `ValueError` | Thiếu khóa; sai đơn vị; chia sai theo batch. |
| V18 | `file_size_mb` trên tệp có kích thước biết trước (ví dụ 1,048,576 byte); tệp không tồn tại. | `1.0` MiB; `FileNotFoundError` | Nhầm MB và MiB. |
| V19 | `build_cost_record` với `history` có và không có `epoch_time`; không truyền `timing`. | `train_time_s` bằng tổng `epoch_time` hoặc `None`; thiếu `timing` cho giá trị `None`, không lỗi | Lỗi `KeyError` khi dữ liệu thiếu. |

### Nhóm G: Hình vẽ và tổng hợp

| Mã | Cách kiểm thử (ví dụ) | Kỳ vọng | Lỗi bắt được |
| :--- | :--- | :--- | :--- |
| V20 | Gọi `plot_confusion_matrix` (cả `normalize` hai giá trị), `plot_per_class_accuracy`, `plot_sample_predictions`, `plot_history` với `save_path` trong `tmp_path`. | Mỗi hàm trả về `Figure`; tệp ảnh tồn tại và có kích thước `> 0`; số figure đang mở của pyplot không tăng | Hàm không trả về `Figure`; không đóng figure (rò bộ nhớ). |
| V21 | `plot_history` với `start_epochs={"ft": 5}`; kiểm tra dữ liệu trục hoành của đường vẽ. | Đường của `ft` bắt đầu ở epoch 6 | Đường cong Giai đoạn 2 chồng lên Giai đoạn 1. |
| V22 | `plot_sample_predictions` với danh sách rỗng; với độ dài nhãn lệch ảnh. | `ValueError` cả hai trường hợp | Lỗi hình vẽ khó hiểu khi đầu vào rỗng. |
| V23 | `evaluate_predictions` trên kết quả đồ chơi; `json.dumps(record)`. | Không lỗi; có đủ khóa của mục 2.2; `accuracy_ci95` chứa `accuracy`; `confusion_matrix` là danh sách lồng nhau | Còn tensor trong bản ghi làm JSON thất bại. |
| V24 | `format_comparison_table` với một run đủ dữ liệu và một run thiếu `cost`/`best_val_acc`. | Có dòng tiêu đề, dòng phân cách, mỗi run một dòng; giá trị thiếu hiển thị bằng dấu gạch dài; không lỗi | Lỗi định dạng khi `None`; bảng lệch cột. |

## 5. Hai ví dụ đọc test chi tiết

**Ví dụ 1: V10 (không có `NaN` khi một lớp không bao giờ được dự đoán)**

```python
y_true = torch.tensor([0, 1, 2])
y_pred = torch.tensor([0, 1, 0])                 # lớp 2 không bao giờ được dự đoán
cm = compute_confusion_matrix(y_true=y_true, y_pred=y_pred, num_classes=3)
m = classification_metrics(cm)

assert m["precision"][2].item() == 0.0           # 0/0 được quy ước là 0
assert not torch.isnan(m["precision"]).any()
assert not math.isnan(m["macro_f1"])
```

**Ví dụ 2: V7 (đúng hướng ma trận, hàng là nhãn thật)**

```python
y_true = torch.tensor([2, 2, 2])                 # nhãn thật: lớp 2
y_pred = torch.tensor([1, 1, 1])                 # dự đoán: lớp 1
cm = compute_confusion_matrix(y_true=y_true, y_pred=y_pred, num_classes=3)

assert cm[2, 1].item() == 3                      # hàng 2 (thật), cột 1 (dự đoán)
assert cm[1, 2].item() == 0                      # ô đối xứng phải rỗng

with pytest.raises(TypeError):
    compute_confusion_matrix(y_pred, y_true, 3)  # không cho phép tham số vị trí
```

## 6. Cách chạy và ánh xạ tiêu chí chấp nhận

```bash
MPLBACKEND=Agg pytest tests/test_evaluate.py -v
```

| Tiêu chí (mục 1.5) | Test chứng minh |
| :--- | :--- |
| 1. Ma trận đúng trên ví dụ tính tay | V5, V6 |
| 2. Không có `NaN` | V10 |
| 3. Loss khớp loss trên toàn tập | V2 |
| 4. Không hoán đổi âm thầm `y_true`/`y_pred` | V7, V8 |
| 5. Khoảng tin cậy khớp giá trị đã biết | V12 |
| 6. Hàm vẽ trả về `Figure` và tạo tệp | V20 |
| 7. Bản ghi serialize được bằng JSON | V23 |
| 8. Toàn bộ test đạt | Mọi test ở trên |
