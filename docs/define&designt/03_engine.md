# Đặc tả Thiết kế Module Huấn luyện — `engine.py`

> **Tài liệu:** `docs/define&designt/03_engine.md` (Define & Design cho động cơ huấn luyện)
> **Module liên quan:** `going_modular/engine.py`
> **Bước thực hành tương ứng:** Step 6 — Train the Model (và phần chọn mô hình của Step 7)
> **Phụ thuộc:** `docs/define&designt/00_lab2_description.md` (tổng quan, cấu hình, `RUN_MODE`), `docs/define&designt/01_data.md` (DataLoader), `docs/define&designt/02_model.md` (mô hình, `set_train_mode`, `get_param_groups`)
> **Trạng thái:** Pre-implementation Design Specification (Đã chốt thiết kế chính thức).

---

## PHẦN I: DEFINE — Xác định Bài toán & Yêu cầu

### 1.1. Bối cảnh & Mục tiêu

Module này nằm ở giữa hệ thống:

* **`engine.py`** nhận mô hình và các `DataLoader`, chạy vòng lặp huấn luyện/kiểm định, chọn trọng số tốt nhất theo Validation, ghi log TensorBoard và trả về lịch sử huấn luyện.
* Các thao tác seed, device, TensorBoard writer và artifact I/O thuộc `utils.py`, được đặc tả độc lập tại `docs/define&designt/04_utils.md`.

Mục tiêu:

* Chuẩn hóa vòng lặp thành `train_step` và `evaluate_step` có thể kiểm thử độc lập.
* Giữ BatchNorm đã đóng băng đứng yên trong lúc huấn luyện (phối hợp với `set_train_mode` ở `02_model.md`).
* Cho phép huấn luyện hai giai đoạn (Feature Extraction rồi Fine-Tuning) với đường cong TensorBoard liền mạch.
* Phối hợp với `utils.py` để lưu checkpoint và tiếp tục train khi phiên bị gián đoạn.

### 1.2. Đầu vào / Đầu ra

| Module | Đầu vào | Đầu ra |
| :--- | :--- | :--- |
| `engine.py` | `nn.Module`, `DataLoader` (Train, Validation), hàm mất mát, optimizer, thiết bị, số epoch, đường dẫn lưu mô hình, TensorBoard writer tùy chọn | `history` (loss và accuracy theo epoch, learning rate); trọng số tốt nhất lưu trên đĩa |

### 1.3. Yêu cầu chức năng

| Mã | Yêu cầu |
| :--- | :--- |
| FR-1 | `train_step` thực hiện một epoch huấn luyện, trả về loss và accuracy trung bình **có trọng số theo số mẫu**. |
| FR-2 | `evaluate_step` đánh giá một epoch ở chế độ `eval`, không tính gradient, không thay đổi trọng số. |
| FR-3 | Vòng lặp huấn luyện giữ BatchNorm đã đóng băng ở chế độ `eval` thông qua hàm chuyển chế độ được truyền vào. |
| FR-4 | `train` chạy nhiều epoch, ghi `history` đầy đủ và chọn trọng số tốt nhất theo tập **Validation** (không dùng tập Test). |
| FR-5 | Sau huấn luyện có thể khôi phục trọng số tốt nhất vào mô hình để bước đánh giá dùng đúng mô hình đã chọn. |
| FR-6 | Hỗ trợ huấn luyện nối tiếp (Giai đoạn 2) với chỉ số epoch bắt đầu khác 0, để đường cong TensorBoard không bị chồng lên nhau. |
| FR-7 | Ghi log TensorBoard theo thời gian thực: loss, accuracy, learning rate cho từng epoch. |
| FR-8 | Phối hợp với `utils.py` để lưu resume checkpoint sau mỗi epoch và khôi phục trạng thái train khi được yêu cầu. |

### 1.4. Ràng buộc

* Hàm mất mát: `nn.CrossEntropyLoss`; optimizer: Adam (theo `00_lab2_description.md`).
* Cấu hình mặc định: `EPOCHS = 5`, `LR = 1e-3` (Giai đoạn 1); `EPOCHS_FT = 3`, `LR_FINETUNE = 1e-4` (Giai đoạn 2); `SEED = 42`.
* Mỗi run có thư mục artifacts riêng như quy định tại `00_lab2_description.md`.
* `engine.py` **không** phụ thuộc trực tiếp vào `ModelConfig`; chỉ nhận hàm chuyển chế độ qua tham số (xem mục 2.3).
* Ghi log sử dụng writer do orchestration tạo theo quy ước ở `04_utils.md`.

### 1.5. Tiêu chí chấp nhận

1. Sau `train_step`, chỉ các tham số `requires_grad=True` thay đổi; tham số đóng băng giữ nguyên giá trị.
2. Loss trung bình của một epoch bằng loss tính trên toàn bộ tập dữ liệu (không lệch do batch cuối nhỏ hơn).
3. BatchNorm đã đóng băng không đổi `running_mean` sau một epoch huấn luyện.
4. `history` có độ dài đúng bằng số epoch; `best_model.pth` tương ứng với epoch tốt nhất theo Validation.
5. Cùng `seed` cho cùng kết quả khởi tạo/chia dữ liệu trong cùng môi trường; lưu rồi nạp lại cho cùng đầu ra mô hình. Resume bảo đảm tiếp tục ở ranh giới epoch nhưng không cam kết cùng chuỗi RNG/batch như chạy liền mạch.
6. Ghi lỗi giữa chừng không làm hỏng tệp artifacts đã có.
7. Chức năng resume khôi phục chính xác trạng thái huấn luyện từ `last_checkpoint.pth`.
8. Bộ test ở mục 2.11 đều đạt.

---

## PHẦN II: DESIGN — Thiết kế Chi tiết

### 2.1. Vị trí trong kiến trúc hệ thống

```text
notebooks/pretrained_models.ipynb
   │  set_seed → get_device → data_setup → model_builder → (RUN_MODE?)
   │
   ├── "train" ──► engine.train(...) ──► artifacts/<exp>/best_model.pth
   │                    │                 + last_checkpoint.pth
   │                    │                 + history.json + config.json  (utils)
   │                    └──► runs/<exp>/  (TensorBoard, utils.create_writer)
   │
   └── "reuse" ──► utils.load_experiment(...) + utils.load_model(...)
                        │
                        ▼
             mô hình đã nạp ──► evaluate.py (suy luận trên test_dataloader)
```

### 2.2. Thiết kế tham số cấu hình

Tham số của hàm `train`:

| Tham số | Kiểu | Mặc định | Ý nghĩa |
| :--- | :--- | :--- | :--- |
| `model` | `nn.Module` | bắt buộc | Mô hình đã dựng, đã chuyển lên `device`. |
| `train_dataloader`, `val_dataloader` | `DataLoader` | bắt buộc | Hai luồng dữ liệu (`02`/`01`). |
| `loss_fn` | `nn.Module` | bắt buộc | `CrossEntropyLoss`. |
| `optimizer` | `Optimizer` | bắt buộc | Adam; Giai đoạn 2 dùng param groups. |
| `epochs` | `int` | bắt buộc | Tổng số epoch cần chạy trong phase hiện tại (`5` cho Feature Extraction, `3` cho Fine-Tuning). |
| `device` | `torch.device` | bắt buộc | Từ `get_device`. |
| `train_mode_fn` | `Optional[Callable]` | `None` | Hàm đưa mô hình vào chế độ train; `None` thì gọi `model.train()`. |
| `writer` | `Optional[SummaryWriter]` | `None` | `None` thì không ghi TensorBoard. |
| `start_epoch` | `int` | `0` | Offset epoch toàn cục chỉ dùng cho TensorBoard/báo cáo; không quyết định số epoch cần chạy. |
| `save_path` | `Optional[str]` | `None` | Đường dẫn lưu `best_model.pth`; `None` thì không lưu. |
| `monitor` | `str` | `"val_acc"` | Chỉ số dùng chọn mô hình tốt nhất (`val_acc` hoặc `val_loss`). |
| `restore_best` | `bool` | `True` | Nạp lại trọng số tốt nhất vào mô hình sau khi huấn luyện. |
| `checkpoint_dir` | `Optional[str]` | `None` | Thư mục lưu checkpoint; tự động lưu `last_checkpoint.pth` sau mỗi epoch. |
| `resume` | `bool` | `False` | Chỉ khi `True`, khôi phục phase từ `last_checkpoint.pth`; phải có checkpoint và config khớp. Mặc định không tự tiếp tục từ tệp cũ. |

Cấu trúc `config.json` (ghép từ ba tài liệu, lưu cùng artifacts để tái lập):

```json
{
  "model": {
    "arch": "resnet18", "num_classes": 10, "weights": null,
    "freeze_backbone": true, "unfreeze_last_n": 1,
    "dropout": 0.0, "freeze_bn_stats": true
  },
  "data": { "batch_size": 32, "val_size": 5000, "transform_model": "resnet18",
            "transform_weights": "IMAGENET1K_V1" },
  "train": { "epochs": 3, "lr_head": 1e-3, "lr_backbone": 1e-4,
             "monitor": "val_acc", "start_epoch": 5,
             "init_from": "artifacts/v1_resnet18" },
  "seed": 42
}
```

### 2.3. `train_step` và `evaluate_step`

**`train_step(model, dataloader, loss_fn, optimizer, device, train_mode_fn=None) -> Tuple[float, float]`**

```text
1. đưa mô hình vào chế độ train: train_mode_fn(model) nếu có, ngược lại model.train()
2. với mỗi batch (X, y):
     chuyển X, y lên device
     logits = model(X);  loss = loss_fn(logits, y)
     optimizer.zero_grad();  loss.backward();  optimizer.step()
     cộng dồn: loss_sum += loss.item() * len(y);  correct += (logits.argmax(1) == y).sum().item()
3. return loss_sum / số mẫu đã duyệt,  correct / số mẫu đã duyệt
```

**`evaluate_step(model, dataloader, loss_fn, device) -> Tuple[float, float]`**

```text
1. model.eval()
2. trong torch.inference_mode(): duyệt batch, cộng dồn loss và số dự đoán đúng như trên
3. return loss trung bình, accuracy
```

Quyết định thiết kế quan trọng:

* **Trung bình có trọng số theo số mẫu.** Batch cuối thường nhỏ hơn `batch_size`; nếu lấy trung bình cộng loss của các batch, kết quả sẽ lệch so với loss trên toàn tập.
* **`train_mode_fn` thay vì truyền `ModelConfig`.** `engine.py` không cần biết cấu hình mô hình. Notebook truyền `functools.partial(set_train_mode, cfg=model_cfg)`. Nhờ đó `engine.py` dùng lại được cho mô hình bất kỳ và dễ kiểm thử bằng mô hình đồ chơi.
* **`inference_mode` khi đánh giá** tiết kiệm bộ nhớ, không dựng đồ thị tính toán.
* `evaluate_step` cũng được `evaluate.py` dùng lại để tính loss và accuracy trên tập Test, tránh viết lại vòng lặp.

### 2.4. Hàm `train` (vòng lặp nhiều epoch)

**Chữ ký:** `train(model, train_dataloader, val_dataloader, loss_fn, optimizer, epochs, device, train_mode_fn=None, writer=None, start_epoch=0, save_path=None, monitor="val_acc", restore_best=True, checkpoint_dir=None, resume=False) -> Dict[str, List]`

**Luồng xử lý (kèm Resume Checkpoint):**

```text
1. validate đầu vào (epochs > 0, monitor hợp lệ, start_epoch >= 0)
2. nếu resume=True: yêu cầu last_checkpoint.pth và config phase khớp; nạp model_state,
   optimizer_state, phase_epochs_completed, history, best_score và best_epoch.
   Nếu không resume: phase_epochs_completed = 0; khởi tạo history rỗng; best = None
3. với mỗi phase_epoch từ phase_epochs_completed đến epochs - 1:
     bắt đầu đo thời gian epoch
     train_loss, train_acc = train_step(...)
     val_loss,   val_acc   = evaluate_step(...)
     nếu device là CUDA: đồng bộ CUDA trước khi dừng đồng hồ
     kết thúc đo thời gian epoch (epoch_time)
     ghi vào history (kèm lr của từng param group và epoch_time)
     global_epoch = start_epoch + phase_epoch + 1
     nếu writer: ghi scalar tại global_epoch
     nếu chỉ số monitor cải thiện thực sự:
       lưu state_dict vào save_path (best_model.pth), nhớ best_epoch
     nếu checkpoint_dir:
       lưu last_checkpoint.pth (model/optimizer state, phase_epochs_completed=phase_epoch + 1,
           best_score, best_epoch, history và phase_config) bằng atomic write
       lưu history.json nguyên tử sau mỗi epoch
     in một dòng tóm tắt epoch (loss, accuracy, thời gian)
4. lưu history cuối cùng nguyên tử; nếu restore_best và có save_path: nạp state_dict tốt nhất vào model
5. return history
```

`epochs` luôn là **tổng số epoch mục tiêu của phase**, không phải số epoch toàn cục. Ví dụ fine-tune có `epochs=3`, `start_epoch=5`: chạy đúng ba epoch phase; TensorBoard step là 6, 7, 8. Resume với một epoch đã hoàn tất sẽ chạy tiếp hai epoch còn lại. Checkpoint lưu số epoch đã hoàn tất trong phase, tách biệt với offset toàn cục. Resume chỉ hỗ trợ ranh giới epoch; không lưu/khôi phục RNG nên không đảm bảo cùng thứ tự sample/augmentation như một lần chạy liên tục.

**Schema `history`:**

| Khóa | Kiểu | Ý nghĩa |
| :--- | :--- | :--- |
| `train_loss`, `val_loss` | `List[float]` | Loss trung bình mỗi epoch |
| `train_acc`, `val_acc` | `List[float]` | Accuracy mỗi epoch, trong $[0, 1]$ |
| `lr` | `List[List[float]]` | Learning rate của từng param group mỗi epoch |
| `epoch_time` | `List[float]` | Thời gian thực thi mỗi epoch tính bằng giây (gồm cả train và val) |

**Chọn mô hình tốt nhất.**

* Mặc định theo `val_acc` (lớn hơn là tốt); với `val_loss` thì nhỏ hơn là tốt.
* Chỉ lưu khi cải thiện **thực sự** (so sánh chặt), nên khi hòa thì giữ epoch sớm hơn.
* Dùng tập Validation, **không bao giờ** dùng tập Test, để tránh rò rỉ dữ liệu (xem `01_data.md`).
* `restore_best=True` bảo đảm mô hình dùng ở Step 7 là mô hình tốt nhất, không phải mô hình ở epoch cuối.
* `resume=False` là mặc định. Khi bật resume, checkpoint phải tồn tại và phase config phải khớp; checkpoint không khớp hoặc epoch đã hoàn tất vượt `epochs` thì báo lỗi, không tự khởi động lại.

### 2.5. Chiến lược hai giai đoạn và `RUN_MODE`

**Hai giai đoạn huấn luyện** (cấu hình ở `02_model.md`, mục 2.5):

| | Giai đoạn 1: Feature Extraction | Giai đoạn 2: Fine-Tuning |
| :--- | :--- | :--- |
| Mô hình | `build_model` với `unfreeze_last_n=0` | Cùng mô hình, nạp `best_model.pth` của Giai đoạn 1, dựng lại với `unfreeze_last_n=N` |
| Optimizer | Adam mới, `lr=1e-3` cho head | Adam **mới** với `get_param_groups` (head `1e-3`, backbone `1e-4`) |
| Epochs | `EPOCHS = 5` | `EPOCHS_FT = 3` |
| `start_epoch` | `0` | `5` (số epoch Giai đoạn 1) |
| Thư mục artifacts | `artifacts/v1_resnet18/` | `artifacts/v1_resnet18_finetune/` |

Giai đoạn 2 tạo optimizer **mới** vì tập tham số trainable đã thay đổi (tham số mới mở khóa không có trạng thái Adam). Đặt `start_epoch = 5` để đường cong TensorBoard của hai giai đoạn nối liền nhau trên cùng trục.

**`RUN_MODE`** (điều phối trong notebook, dùng hàm của `utils.py`):

| Chế độ | Luồng |
| :--- | :--- |
| `"train"` | Notebook lưu `config.json` trước train; `engine.train` lưu `best_model.pth`, checkpoint và `history.json` sau mỗi epoch |
| `"reuse"` | Kiểm tra đủ ba artifacts huấn luyện; dựng mô hình bằng `weights=None`, nạp `best_model.pth`; dùng `test_metrics.json` nếu có, nếu thiếu thì đánh giá Test một lần và lưu |

Nếu `RUN_MODE = "reuse"` mà artifacts thiếu, hệ thống **báo lỗi rõ ràng** (`FileNotFoundError`), không âm thầm chuyển sang huấn luyện.

### 2.6. Thiết kế ghi log TensorBoard

| Tag | Giá trị | Thời điểm |
| :--- | :--- | :--- |
| `Loss/train`, `Loss/val` | loss trung bình của epoch | cuối mỗi epoch |
| `Accuracy/train`, `Accuracy/val` | accuracy của epoch | cuối mỗi epoch |
| `LR/group_<i>` | learning rate của param group `i` | cuối mỗi epoch |

* Bước ghi (step) = `start_epoch + phase_epoch + 1`, với `phase_epoch` từ $0$ đến `epochs - 1`.
* `create_writer(experiment_name)` ghi vào `runs/<experiment_name>`; tên thí nghiệm trùng tên thư mục artifacts để dễ đối chiếu.
* Cuối `train` gọi `writer.flush()`; việc `close()` do nơi tạo writer (notebook) chịu trách nhiệm.
* Ma trận nhầm lẫn và biểu đồ phân tích thuộc `evaluate.py`, không ghi ở `engine.py`.

### 2.7. Phối hợp với `utils.py`

Engine sở hữu logic quyết định *khi nào* lưu mô hình tốt nhất và resume checkpoint; `utils.py` sở hữu serialize/deserialize, schema validation và thao tác I/O an toàn. API của utilities được định nghĩa duy nhất tại `docs/define&designt/04_utils.md`.

| Hàm engine | Đầu vào | Đầu ra | Ngoại lệ |
| :--- | :--- | :--- | :--- |
| `train_step(model, dataloader, loss_fn, optimizer, device, train_mode_fn=None)` | như tên | `(loss, acc)` | Lỗi dữ liệu hoặc mô hình được truyền lên |
| `evaluate_step(model, dataloader, loss_fn, device)` | như tên | `(loss, acc)` | Lỗi dữ liệu hoặc mô hình được truyền lên |
| `train(...)` | xem mục 2.4 | `Dict[str, List]` | `ValueError` nếu `epochs <= 0`, `start_epoch < 0`, `monitor` không hợp lệ |

### 2.8. Tích hợp với các module khác

| Module | Cách phối hợp |
| :--- | :--- |
| `model_builder.py` | Notebook truyền `partial(set_train_mode, cfg=...)` làm `train_mode_fn`; dùng `get_param_groups` để dựng optimizer Giai đoạn 2. |
| `data_setup.py` | `train_dataloader` cho `train_step`, `val_dataloader` cho chọn mô hình; `test_dataloader` **không** đi vào `engine.train`. |
| `evaluate.py` | Dùng `run_inference` cho Test; nhận `history` từ utils để vẽ đường cong học. |
| `utils.py` | Checkpoint I/O và cập nhật history nguyên tử; config được ghi theo schema `00`. |
| Notebook | Ghi `config.json` nguyên tử trước khi train; điều phối `RUN_MODE`, hai giai đoạn, truyền `start_epoch`, đóng writer; dùng utilities theo `04_utils.md`. |

### 2.9. Kế hoạch kiểm thử

| Mã | Kiểm tra | Kỳ vọng |
| :--- | :--- | :--- |
| E1 | `train_step` trên mô hình đồ chơi và DataLoader nhiều batch | Trả loss/accuracy trung bình theo số mẫu; chỉ tham số trainable đổi. |
| E2 | `evaluate_step` trên mô hình đồ chơi | Không tạo gradient, không đổi weights, mô hình ở eval. |
| E3 | Train với batch cuối nhỏ và optimizer `lr=0` | Loss bằng loss tính trên toàn bộ tập, không phải trung bình không trọng số giữa các batch. |
| E4 | `train(epochs=3, start_epoch=5)` với writer giả | Chạy đúng 3 epoch; TensorBoard step là 6, 7, 8. |
| E5 | Scripted validation scores và trạng thái model thay đổi mỗi epoch | `best_model.pth` và `restore_best` khớp epoch tốt nhất theo monitor. |
| E6 | Giả lập gián đoạn sau một epoch rồi gọi lại `resume=True` | Tiếp tục từ phase epoch kế tiếp đến đúng tổng `epochs`; model, optimizer, history, best state được khôi phục. |
| E7 | Resume với checkpoint thiếu hoặc config phase không khớp | Báo lỗi rõ ràng, không tự bỏ qua checkpoint hay bắt đầu lại. |
| E8 | Kiểm tra history/checkpoint sau mỗi epoch và mô phỏng lỗi ghi tệp | History/checkpoint được cập nhật nguyên tử; file cũ không hỏng. |
| E9 | `train_mode_fn` giữ BatchNorm đóng băng ở eval; trường hợp `writer=None` | BN frozen không đổi running stats; train không yêu cầu writer. |
| E10 | Epoch timing trên CUDA nếu môi trường hỗ trợ | Đồng bộ trước/sau vùng đo; CPU không gọi CUDA synchronize. |

Chạy kiểm thử bằng `pytest tests/test_engine.py -v`; kiểm thử CUDA được bỏ qua có điều kiện khi không có CUDA.

### 2.10. Mở rộng

* **Scheduler learning rate:** thêm tham số `scheduler` tùy chọn vào `train` và gọi `scheduler.step()` cuối mỗi epoch; cột `lr` trong `history` đã sẵn sàng phản ánh thay đổi.
* **Early stopping:** thêm tham số `patience`; dựa trên cùng logic cải thiện của `monitor`.
* **Mixed precision (AMP):** bọc `forward`/`backward` trong `torch.autocast` và `GradScaler`; không ảnh hưởng giao diện hàm.
* **Nhiều worker:** nếu `num_workers > 0`, bổ sung `seed_worker` và `generator` cho `DataLoader` để giữ tái lập.

### 2.11. Quyết định thiết kế của nhóm so với chuẩn từ tài liệu

| Hạng mục | Có trong tài liệu chính thức | Quyết định thiết kế của nhóm |
| :--- | :--- | :--- |
| Mẫu `train_step` / `evaluate_step` / `train` | Có (Learn PyTorch ch.05) | |
| `torch.utils.tensorboard.SummaryWriter` | Có (PyTorch) | |
| `model.train()` / `model.eval()` do BatchNorm | Có (torchvision) | |
| Trung bình loss có trọng số theo số mẫu | | Có |
| `train_mode_fn` để tách engine khỏi `ModelConfig` | | Có |
| Chọn mô hình tốt nhất theo Validation và `restore_best` | | Có |
| `start_epoch` để nối đường cong hai giai đoạn | | Có |
| Cấu trúc history và chọn checkpoint tốt nhất theo Validation | | Có |
| Resume theo epoch và nối tiếp TensorBoard hai giai đoạn | | Có |

### 2.12. Tài liệu tham khảo

1. Daniel Bourke, *Learn PyTorch for Deep Learning*, chương 05 (Going Modular: `engine.py`) và chương 07 (Experiment Tracking): https://www.learnpytorch.io
2. PyTorch, `torch.utils.tensorboard`: https://docs.pytorch.org
3. PyTorch / torchvision, *Models and pre-trained weights* (hành vi train/eval của BatchNorm): https://docs.pytorch.org/vision/stable/models.html

---

## PHẦN III: NHẬT KÝ QUYẾT ĐỊNH THIẾT KẾ

Quyết định về seed/device, TensorBoard writer, artifacts và test utilities được ghi tại `docs/define&designt/04_utils.md`. Các quyết định thuộc riêng engine:

| Chủ đề | Quyết định |
| :--- | :--- |
| Chỉ số chọn checkpoint | `val_acc` mặc định; có hỗ trợ `val_loss`. |
| Hai giai đoạn | Optimizer mới khi unfreeze; `start_epoch=5` cho giai đoạn Fine-Tuning. |
| Early stopping/scheduler | Ngoài phạm vi; dùng số epoch cố định theo `00`. |
