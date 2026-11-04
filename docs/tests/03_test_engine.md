# Đặc tả Kiểm thử — `engine.py`

> Nguồn: `docs/define&designt/03_engine.md` mục 2.9 (Kế hoạch kiểm thử `tests/test_engine.py`).

## 1. Mục tiêu

Test kiểm chứng tính đúng đắn của **cơ chế huấn luyện**, không đánh giá độ chính xác mô hình hay utilities (xem `04_utils.md`).

## 2. Nguyên tắc

1. **Mô hình và dữ liệu đồ chơi:** dùng một mạng `Linear` nhỏ và tập dữ liệu tổng hợp vài chục mẫu, nên mỗi test chạy trong chớp mắt, không tải gì.
2. **Cô lập bằng giả lập (mock):** để kiểm tra logic chọn mô hình tốt nhất và `start_epoch`, thay `train_step`/`evaluate_step` bằng bản có kết quả được định sẵn (`monkeypatch`) và `writer` bằng đối tượng giả.
3. **Kiểm tra hành vi quan sát được:** so sánh giá trị tham số, nội dung tệp, số lần gọi, không kiểm tra cách cài đặt.
4. **Mỗi test một ý chính** và dùng thư mục tạm `tmp_path` cho mọi tệp.

## 3. Bốn nhóm kiểm thử

| Nhóm | Khía cạnh | Câu hỏi cần trả lời | Test |
| :--- | :--- | :--- | :--- |
| A | Đúng đắn của một bước | Chỉ tham số trainable thay đổi? Loss tính đúng? Đánh giá không đổi trọng số? | E1, E2, E3, E4 |
| B | BatchNorm trong huấn luyện | Chế độ train tùy chỉnh được gọi? BN đã freeze có đứng yên? | E5, E6 |
| C | Vòng lặp nhiều epoch | `history` đúng? Chọn và khôi phục mô hình tốt nhất đúng? | E7, E8, E9, E10 |
| D | TensorBoard và nối tiếp giai đoạn | Log đúng tag và đúng bước? | E11, E12 |

## 4. Mô tả chi tiết từng test

### Nhóm A: Đúng đắn của một bước

| Mã | Cách kiểm thử (ví dụ) | Kỳ vọng | Lỗi bắt được |
| :--- | :--- | :--- | :--- |
| E1 | Gọi `train_step` trên mô hình đồ chơi. | Trả về hai `float`; accuracy thuộc `[0, 1]` | Sai kiểu trả về; accuracy tính nhầm thành phần trăm hoặc tổng. |
| E2 | Mô hình có hai tầng, đóng băng tầng đầu; sao chép trọng số trước, chạy `train_step`, so sánh sau. | Trọng số tầng đóng băng giữ nguyên; trọng số tầng trainable đã đổi | Optimizer cập nhật nhầm tham số đóng băng; `loss.backward()` hoặc `step()` bị bỏ sót. |
| E3 | Dùng optimizer `lr=0` (trọng số không đổi), tập 10 mẫu, `batch_size=4` (các batch 4, 4, 2). So `train_step` với loss tính một lần trên cả 10 mẫu. | Hai giá trị bằng nhau (sai số `1e-6`) | Lấy trung bình cộng loss theo batch (lệch do batch cuối nhỏ hơn). |
| E4 | Gọi `evaluate_step` với mô hình luôn dự đoán lớp 0 và nhãn biết trước; sao chép trọng số trước và sau. | Accuracy bằng đúng tỷ lệ nhãn 0; trọng số không đổi; `model.training == False` sau khi gọi | Quên `eval()`; tính gradient khi đánh giá; accuracy tính sai. |

### Nhóm B: BatchNorm trong huấn luyện

| Mã | Cách kiểm thử (ví dụ) | Kỳ vọng | Lỗi bắt được |
| :--- | :--- | :--- | :--- |
| E5 | Truyền `train_mode_fn` là hàm giả đếm số lần gọi; chạy `train_step`. | Hàm được gọi đúng 1 lần mỗi `train_step`; khi không truyền thì mô hình ở chế độ train | `train_mode_fn` bị bỏ qua. |
| E6 | Tích hợp: ResNet-18 (`weights=None`), Giai đoạn 1, đầu vào nhỏ `(4, 3, 64, 64)`; truyền `set_train_mode` làm `train_mode_fn`; chạy `train_step`. | `bn1.running_mean` không đổi | Engine gọi `model.train()` trực tiếp, ghi đè thống kê BatchNorm đã đóng băng. |

### Nhóm C: Vòng lặp nhiều epoch

| Mã | Cách kiểm thử (ví dụ) | Kỳ vọng | Lỗi bắt được |
| :--- | :--- | :--- | :--- |
| E7 | Chạy `train` với `epochs=3`. | `history` có đủ 6 khóa; mỗi danh sách dài 3; `lr` mỗi phần tử là danh sách | Thiếu khóa; ghi sai độ dài. |
| E8 | Thay `evaluate_step` bằng bản trả `val_acc` lần lượt `[0.5, 0.9, 0.7]`; thay `train_step` bằng bản cộng 1 vào một trọng số mỗi epoch. | `best_model.pth` chứa trạng thái của epoch 2 (không phải epoch 3) | Lưu mô hình ở epoch cuối thay vì epoch tốt nhất; lưu khi giá trị giảm. |
| E9 | Như E8, đặt `restore_best=True` rồi `False`. | `True`: trọng số mô hình bằng trạng thái epoch 2; `False`: bằng trạng thái epoch 3 | Bước đánh giá cuối dùng nhầm mô hình ở epoch cuối. |
| E10 | `epochs=0`, `start_epoch=-1`, `monitor="foo"`. | Cả ba đều `ValueError` | Tham số sai lọt qua và gây lỗi khó đoán. |

### Nhóm D: TensorBoard và nối tiếp giai đoạn

| Mã | Cách kiểm thử (ví dụ) | Kỳ vọng | Lỗi bắt được |
| :--- | :--- | :--- | :--- |
| E11 | Dùng `writer` giả ghi lại lời gọi `add_scalar`; chạy `train` với `epochs=2`, `start_epoch=5`. | Các bước ghi là 6 và 7; có đủ tag `Loss/train`, `Loss/val`, `Accuracy/train`, `Accuracy/val`, `LR/group_0` | Đường cong Giai đoạn 2 chồng lên Giai đoạn 1; thiếu hoặc sai tên tag. |
| E12 | Chạy `train` với `writer=None`. | Chạy bình thường, không lỗi | Giả định luôn có writer. |

## 5. Hai ví dụ đọc test chi tiết

**Ví dụ 1: E3 (loss trung bình có trọng số)**

```python
model = nn.Linear(4, 3)
X, y = torch.randn(10, 4), torch.randint(0, 3, (10,))
loader = DataLoader(TensorDataset(X, y), batch_size=4)       # các batch: 4, 4, 2
optimizer = torch.optim.SGD(model.parameters(), lr=0.0)     # trọng số không đổi
loss_fn = nn.CrossEntropyLoss()

loss, _ = train_step(model, loader, loss_fn, optimizer, torch.device("cpu"))
expected = loss_fn(model(X), y).item()                      # loss trên cả 10 mẫu

assert abs(loss - expected) < 1e-6
```

Đặt `lr=0` để trọng số không đổi giữa các batch, nên loss gộp phải bằng loss trên toàn tập. Nếu code lấy trung bình cộng ba giá trị loss của ba batch, batch cuối chỉ có 2 mẫu sẽ được tính nặng bằng batch 4 mẫu và test thất bại.

**Ví dụ 2: E8 (chọn đúng epoch tốt nhất)**

```python
scripted = iter([0.5, 0.9, 0.7])                            # val_acc định sẵn cho 3 epoch
monkeypatch.setattr(engine, "evaluate_step", lambda *a, **k: (0.1, next(scripted)))
monkeypatch.setattr(engine, "train_step", fake_step_adds_one)  # mỗi epoch cộng 1 vào w

engine.train(model, train_dl, val_dl, loss_fn, optimizer, epochs=3,
             device=cpu, save_path=tmp_path / "best.pth")

saved = torch.load(tmp_path / "best.pth", weights_only=True)
assert saved["weight"].item() == initial + 2                # trạng thái cuối epoch 2
```

Bằng cách định sẵn `val_acc` và làm cho trọng số thay đổi theo từng epoch, test xác định chính xác epoch nào đã được lưu mà không cần huấn luyện thật.

## 6. Cách chạy và ánh xạ tiêu chí chấp nhận

```bash
pytest tests/test_engine.py -v
```

| Tiêu chí (mục 1.5) | Test chứng minh |
| :--- | :--- |
| 1. Chỉ tham số trainable thay đổi | E2 |
| 2. Loss không lệch do batch cuối | E3 |
| 3. BN đã freeze không đổi `running_mean` | E6 |
| 4. `history` đúng độ dài; checkpoint đúng epoch | E7, E8, E9 |
| 5. Toàn bộ test engine đạt | Mọi test ở trên |

**Giới hạn:** Không kiểm tra tốc độ, bộ nhớ GPU hay hành vi riêng của GPU/MPS (chỉ chạy trên CPU). Không kiểm tra giao diện TensorBoard thật; chỉ kiểm tra các lời gọi ghi log và sự tồn tại của tệp event. Tái lập chỉ được kiểm chứng trên cùng môi trường.
