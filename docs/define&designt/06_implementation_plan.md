# Kế hoạch Triển khai Lab 2 — Pre-trained Models trên CIFAR-10

> **Căn cứ:** Đặc tả tổng thể `00` và các đặc tả module `01`–`05`.
> **Notebook điều phối:** `notebooks/pretrained_models.ipynb`.
> **Trạng thái:** Phạm vi và thiết kế đã chốt; kế hoạch sẵn sàng để bắt đầu triển khai.

## 1. Trạng thái và nguyên tắc triển khai

Các tài liệu `00`–`05` mô tả thiết kế mục tiêu, không mặc nhiên xác nhận code hiện tại đã đáp ứng thiết kế. Các module và notebook cũ cần được đối chiếu, refactor và kiểm thử theo từng cổng bên dưới. Không thay đổi ngoài phạm vi Lab 2 trong quá trình refactor.

Phạm vi thực nghiệm đã chốt gồm sáu runs:

1. `v0_vgg16` — VGG16 Feature Extraction, 5 epochs.
2. `v1_resnet18` — ResNet-18 Feature Extraction, 5 epochs.
3. `v1_resnet18_finetune` — tiếp tục từ checkpoint tốt nhất của `v1_resnet18`, mở `layer4` và head, 3 epochs.
4. `v2_resnet18_lr1e-2` — ResNet-18 Feature Extraction, head LR `1e-2`.
5. `v2_resnet18_lr1e-4` — ResNet-18 Feature Extraction, head LR `1e-4`.
6. `v2_resnet18_bs64` — ResNet-18 Feature Extraction, batch size `64`.

Ba runs `v2` mỗi run chỉ thay một yếu tố so với `v1_resnet18`; không bổ sung run đối chứng 8 epoch. Đây là khảo sát có kiểm soát LR và batch size, không phải grid search toàn diện.

## 2. Các quyết định thiết kế được dùng làm chuẩn

- `docs/define&designt/00_lab2_description.md` là nguồn duy nhất cho cấu hình chung, schema `config.json`, sáu runs, artifacts và các quy ước toàn hệ thống.
- Notebook chuẩn là `notebooks/pretrained_models.ipynb`.
- Notebook lấy transforms từ `model_builder.get_transforms(cfg, weights=data.transform_weights)` và truyền vào `data_setup`; `data_setup.get_pretrained_transform(...)` là fallback độc lập, không import `model_builder`.
- Tạo DataLoader riêng cho mỗi run sau khi gọi `set_seed(42)`; Train DataLoader nhận `torch.Generator` riêng để thứ tự shuffle không bị ảnh hưởng bởi thao tác ngẫu nhiên khác.
- `history` có sáu keys: `train_loss`, `val_loss`, `train_acc`, `val_acc`, `lr`, `epoch_time`.
- Mỗi run lưu `best_model.pth`, `last_checkpoint.pth`, `history.json`, `config.json`; sau đánh giá bổ sung `test_metrics.json`. Không commit checkpoint `.pth`.
- Fine-tuning có optimizer mới; LR head là `1e-3`, LR backbone mở khóa là `1e-4`; trục epoch TensorBoard tiếp tục từ epoch 5.
- `deterministic=True` tăng tính tái lập trong cùng môi trường phần cứng/phần mềm, không bảo đảm kết quả giống hệt giữa các môi trường khác nhau.
- `train.epochs` là số epoch mục tiêu của phase; `start_epoch` chỉ là offset log/báo cáo. Resume mặc định tắt, chỉ tiếp tục ở ranh giới epoch với phase config khớp; RNG không được khôi phục.
- Fine-tune và `reuse` dựng kiến trúc với `model.weights=null`, rồi nạp checkpoint; `data.transform_weights` là cấu hình preprocessing riêng.
- `reuse` yêu cầu đủ `best_model.pth`, `history.json`, `config.json`; nếu chỉ thiếu `test_metrics.json`, chạy một lần inference Test để sinh tệp. Không tự chuyển sang train.
- `test_metrics.json` giữ `y_true`, `y_pred` và confidence tối đa theo mẫu, không lưu logits. `train_time_phase_s` và `train_time_total_s` phân biệt thời gian phase với thời gian cộng phase nguồn.
- `pin_memory=torch.cuda.is_available()` theo khả năng CUDA của runtime, không phụ thuộc thiết bị được yêu cầu cho model. Stratified split được cam kết cho cấu hình CIFAR-10 cân bằng của lab; mean/std lấy từ preprocessing preset, chỉ fallback ImageNet khi config xác nhận.
- Tập Test chỉ dùng để đánh giá cuối; chọn checkpoint bằng Validation.

## 3. Các giai đoạn triển khai

### Phase 0 — Hoàn tất Define & Design

**Trạng thái: Hoàn tất.**

- Đồng bộ `00`–`05` theo các quyết định ở mục 2 và nhật ký `Spec_Review_Report.md`.
- Ghi rõ ba runs cốt lõi và ba runs khảo sát LR/batch size; không đưa run đối chứng epoch vào phạm vi.
- Ghi rõ mọi quyết định về epoch/resume, khởi tạo fine-tune, reuse, metrics, thiết bị dữ liệu và preprocessing trong `Spec_Review_Report.md`.

**Cổng:** tên notebook, schema, artifacts, transform flow, phạm vi sáu runs và các chính sách đã chốt thống nhất xuyên tài liệu. Cổng này chỉ xác nhận tài liệu thiết kế; không xác nhận code đã đáp ứng.

### Phase 1 — Hạ tầng kiểm thử và cấu hình dự án

- Đối chiếu `.gitignore`, `requirements.txt`, README và cấu trúc package với `00`.
- Hoàn thiện `tests/conftest.py` (marker `slow`, backend Matplotlib không tương tác, đường dẫn import).
- Xác nhận các test được thu thập mà không lỗi import.

**Cổng:** `pytest --collect-only` thành công; dữ liệu, logs và checkpoints không bị thêm nhầm vào Git.

### Phase 2 — `model_builder.py` và `test_model_builder.py`

- Hiện thực `ModelConfig`, registry VGG16/ResNet-18, thay head, freeze/unfreeze, BatchNorm mode, transforms và parameter groups.
- Bổ sung `count_params` và `describe_model` cho yêu cầu khảo sát kiến trúc Step 3.
- Kiểm tra cấu hình `weights=None` dùng được để dựng lại kiến trúc khi nạp checkpoint, đồng thời transform vẫn lấy riêng từ `data.transform_weights`.

**Cổng:** toàn bộ test nhanh của `02` đạt; xác nhận số tham số kỳ vọng; test cần tải ImageNet weights được đánh dấu `slow`.

### Phase 3 — `data_setup.py` và `test_data_setup.py`

- Hiện thực stratified split, `_TransformSubset`, transform fallback và ba DataLoader.
- Nhận transform chuẩn qua tham số do Notebook cung cấp; giữ module dữ liệu độc lập với `model_builder`.
- Kiểm tra loader riêng cho từng run, shuffle chỉ ở Train, Validation/Test không augmentation.

**Cổng:** test chia dữ liệu, transform và wrapper đạt; test CIFAR-10 thật chỉ chạy khi có dữ liệu/mạng.

### Phase 4 — `utils.py` và `test_utils.py`

- Hiện thực seed/device/environment, TensorBoard writer và lưu/nạp artifacts.
- Hỗ trợ ghi nguyên tử, nạp model an toàn, kiểm tra điều kiện `reuse`, và resume có chủ đích từ `last_checkpoint.pth` tại ranh giới epoch sau khi kiểm tra phase config.
- Bảo đảm `reuse` không tự train; nếu thiếu riêng `test_metrics.json` thì đánh giá Test đúng một lượt rồi lưu.
- Kiểm tra schema `history` và `config.json` theo `00`.

**Cổng:** test utility đạt; lỗi thiếu artifact phải được báo rõ, không tự chuyển sang train.

### Phase 5 — `engine.py` và `test_engine.py`

- Hiện thực `train_step`, `evaluate_step`, vòng lặp train, chọn checkpoint theo Validation, restore best và resume.
- Ghi đủ sáu keys history, `epoch_time` gồm Train và Validation, TensorBoard tags và epoch offset.
- Nhận `train_mode_fn` thay vì phụ thuộc trực tiếp vào `ModelConfig`.

**Cổng:** test engine đạt với mô hình/dữ liệu đồ chơi; checkpoint best, resume và bước TensorBoard được xác minh.

### Phase 6 — `evaluate.py` và `test_evaluate.py`

- Hiện thực suy luận Test, classification metrics, confusion matrix, Wilson 95% CI, phân tích lỗi và cost metrics.
- Lưu `y_true`, `y_pred`, confidence tối đa; không lưu logits. Phân biệt `train_time_phase_s` và `train_time_total_s`.
- Tạo/lưu `test_metrics.json`; các hình xuất PNG 150 dpi vào `artifacts/figures/`.
- So sánh cả sáu runs; phân tích riêng ảnh hưởng của LR và batch size theo từng biến thể so với baseline. Không đưa McNemar, nhiều seed hay grid search toàn diện vào kết quả chính.

**Cổng:** test đánh giá đạt; bản ghi serialize được JSON và hình được tạo trong chế độ không màn hình.

### Phase 7 — Tích hợp Notebook

- Refactor `notebooks/pretrained_models.ipynb` làm orchestration cho workflow 7 bước.
- Tích hợp `RUN_MODE=train/reuse`, Drive/Colab nếu cần, và một config/artifact folder riêng cho mỗi run.
- Trong fine-tuning, nạp checkpoint ResNet-18 tốt nhất, dựng mô hình tương ứng, tạo optimizer mới và nối tiếp epoch.
- Thử pipeline ngắn trên dữ liệu/mô hình nhỏ trước khi chạy đầy đủ.

**Cổng:** Notebook chạy từ đầu đến cuối ở smoke test; `reuse` nạp cùng kiến trúc, history và dự đoán mà không huấn luyện lại.

### Phase 8 — Huấn luyện và đánh giá đầy đủ

| Thứ tự | Run | Phụ thuộc |
| :--- | :--- | :--- |
| 1 | `v0_vgg16` | Không |
| 2 | `v1_resnet18` | Không |
| 3 | `v1_resnet18_finetune` | `best_model.pth` của `v1_resnet18` |
| 4 | `v2_resnet18_lr1e-2` | Độc lập; chỉ thay LR |
| 5 | `v2_resnet18_lr1e-4` | Độc lập; chỉ thay LR |
| 6 | `v2_resnet18_bs64` | Độc lập; chỉ thay batch size |

- Chạy trên Colab GPU; ghi thiết bị và phiên bản thư viện vào `config.json`.
- Sao lưu artifacts về nơi lưu bền vững sau mỗi run.
- Đánh giá Test sau khi hoàn tất huấn luyện từng mô hình; không dùng Test để điều chỉnh cấu hình.

**Cổng:** mỗi run có checkpoint, config và history; sau đánh giá có thêm `test_metrics.json`; sáu runs cùng xuất hiện trong bảng/biểu đồ so sánh.

### Phase 9 — Báo cáo kết quả và nghiệm thu

- Tạo bảng so sánh cùng các biểu đồ đã định nghĩa ở `docs/define&designt/05_evaluation_results.md`.
- Diễn giải accuracy cùng khoảng tin cậy, thời gian/chi phí và giới hạn của một seed.
- Cập nhật README/tài liệu vận hành nếu có khác biệt thực tế so với thiết kế; mọi thay đổi phạm vi phải được phản ánh đồng thời trong `00` và các module liên quan.

**Cổng nghiệm thu:**

1. `pytest -m "not slow"` đạt; các test `slow` được chạy và ghi rõ nếu môi trường cho phép.
2. Notebook chạy đủ sáu runs và chế độ `reuse`.
3. Tập Test chỉ dùng cho đánh giá cuối.
4. Artifacts và bảng/biểu đồ khớp sáu runs đã chốt.
5. Các kết luận hyperparameter được giới hạn vào các biến thể learning rate/batch size đã chạy; không tuyên bố grid search toàn diện.

## 4. Quan hệ phụ thuộc

```text
Phase 1 ──► Phase 2 (model) ──┬──► Phase 5 (engine) ──┐
           Phase 3 (data)  ───┤                       ├──► Phase 7 ──► Phase 8 ──► Phase 9
           Phase 4 (utils) ───┘                       │
           Phase 6 (evaluate) ────────────────────────┘
```

Sau Phase 1, Phase 2–4 có thể triển khai song song. Phase 5 cần model và utilities; tích hợp Notebook cần các module chức năng hoàn tất.
