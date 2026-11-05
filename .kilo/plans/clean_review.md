# Plan: Cleanup sau Phase 7 (dọn dẹp legacy API)

**File plan:** `D:\DEEP LEARNING\lab2\.kilo\plans\clean_review.md`

**Mục tiêu:** Gỡ các legacy API đã được chuyển caller sang API chuẩn, đồng bộ `__all__` về 14 symbols theo hợp đồng `00_lab2_description.md` mục 2.8.6, chuẩn hóa `save_model(model, path)`, làm atomic checkpoint write, tối ưu `_validate` không dựng model tạm, đồng bộ docs.

**Ràng buộc cứng (không đổi):**
- Không xóa API trước khi chuyển hết caller (notebook, tests, source).
- Không xóa artifacts hiện có, không tải dữ liệu/weights, không chạy training CIFAR-10.
- `__all__` phải đúng 14 symbols của hợp đồng, không ít, không nhiều.
- Giữ `verbose` nếu vẫn là tùy chọn output hợp lệ.

---

## Quyết định thiết kế (đã chốt từ clean.md)

| Vấn đề | Quyết định | Nguồn |
|--------|-----------|-------|
| Legacy APIs bị gỡ | `CIFAR10ModelV0`, `CIFAR10ModelV1`, `MODEL_NAMES`, legacy `build_model("v0"/"v1")`, `count_trainable_params`, `count_total_params`, `accuracy_fn`, `print_train_time`, `save_training_metadata`, `load_training_artifacts`, `eval_model`, `checkpoint_path` (kwarg), `frozen_backbone` (kwarg) | clean.md Bước 1,3 |
| `save_model` mới | `save_model(model, path)` — lưu state_dict vào file path trực tiếp, không còn `target_dir` + `model_name` | clean.md Bước 1 |
| `__all__` | Đúng 14 symbols theo `00_lab2_description.md` mục 2.8.6 | clean.md Bước 3 |
| `verbose` | Giữ nếu là tùy chọn output, không gỡ chỉ vì nằm nhóm legacy | clean.md Nguyên tắc 2 |
| `_validate` | Dùng metadata số block (ResNet-18: 4, VGG16: 5) thay vì `_dummy_model` | clean.md Bước 4 |
| Checkpoint atomic | `save_checkpoint` ghi temp + `os.replace` trong cùng thư mục; `best_model.pth` lưu trong engine cũng atomic | clean.md Bước 5 |
| Notebook fine-tune | Thay manual `for name, param...unfreeze layer4/fc` bằng `ModelConfig(unfreeze_last_n=1)` | clean.md Bước 2 |

---

## Danh sách tác vụ (theo thứ tự thực hiện)

### T0 — Baseline trước cleanup
- [ ] Chạy `pytest tests/ -m "not slow" -q` và ghi nhận kết quả/suite thời lượng làm baseline.
- [ ] Nếu baseline thất bại, tách lỗi cũ khỏi thay đổi cleanup.

### T1 — Ma trận caller và chốt hợp đồng
- [ ] Liệt kê toàn bộ caller của từng legacy API trong: `going_modular/*.py`, `tests/*.py`, `notebooks/pretrained_models.ipynb`, `docs/*.md`.
- [ ] Xác nhận mỗi API bị gỡ có API thay thế và không còn caller bắt buộc.
- [ ] Chốt `save_model(model, path)` và cập nhật callers/tests cùng thay đổi.
- [ ] Chốt 14 symbols của `__all__` (xem bảng dưới).

### T2 — Chuyển notebook sang API chuẩn (cell còn lại)
- [ ] Cell 21 (fine-tune): Thay manual unfreeze layer4/fc bằng `ModelConfig(arch='resnet18', weights=None, freeze_backbone=True, unfreeze_last_n=1)`, sau đó nạp `best_model.pth` của `v1_resnet18`.
- [ ] Cell 28-29 (fine-tune train): Thay `checkpoint_path=CHECKPOINT_FT, frozen_backbone=None` bằng `save_path=CHECKPOINT_FT, checkpoint_dir=str(ARTIFACT_FT), resume=False, expected_phase_config=CONFIG_FT, start_epoch=0, verbose=False`.
- [ ] Cell 29 (fine-tune reuse): Thay `utils.save_training_metadata(...)` bằng `save_config(...)` + `save_history(...)`; sửa cú pháp `load_model(..., device=DEVICE), device=DEVICE` (lỗi syntax) thành `load_model(..., device=DEVICE)`.
- [ ] Markdown Section 4.2: Cập nhật hướng dẫn không còn nhắc `frozen_backbone=model_v1.model`.

### T3 — Gỡ legacy API và siết exports
- [ ] `model_builder.py`: Xóa `CIFAR10ModelV0`, `CIFAR10ModelV1`, `MODEL_NAMES`, nhánh legacy trong `build_model` (chỉ giữ `build_model(cfg: ModelConfig)`), xóa `count_trainable_params`, `count_total_params`.
- [ ] `utils.py`: Xóa `accuracy_fn`, `print_train_time`, `save_training_metadata`, `load_training_artifacts`.
- [ ] `engine.py`: Xóa `eval_model`, xóa kwargs `checkpoint_path`, `frozen_backbone` (giữ `save_path`, `train_mode_fn`, `verbose`).
- [ ] `__init__.py`: Cập nhật imports và `__all__` về đúng 14 symbols (xem bảng dưới).
- [ ] `tests/test_engine.py`: Xóa hoặc cập nhật `test_backward_compat_checkpoint_path`, `test_backward_compat_frozen_backbone` (không còn legacy kwargs).

### T4 — Tối ưu validation, không dựng model tạm
- [ ] `model_builder.py`: Thay `_dummy_model(cfg.arch)` trong `_validate` bằng metadata số block từ registry (`ARCH_SPECS` có thêm `num_blocks`: resnet18=4, vgg16=5).
- [ ] `tests/test_model_builder.py`: Thêm test xác nhận `_validate` không gọi `_load_torchvision` khi cấu hình hợp lệ/lỗi; vẫn kiểm tra ngưỡng `unfreeze_last_n` đúng cho cả hai kiến trúc.
- [ ] Giữ coverage VGG16 (output/head, unfreeze blocks, parameter count) nhưng không dùng model mutable theo module scope.

### T5 — Checkpoint atomic write (thực, không mock toàn bộ)
- [ ] `utils.py`: `save_checkpoint` ghi file tạm cùng thư mục, đóng, rồi `os.replace`; dọn temp khi lỗi, raise lại exception.
- [ ] `engine.py`: Best-model `torch.save(model.state_dict(), save_p)` thay bằng ghi atomic (temp + `os.replace`).
- [ ] `tests/test_utils.py`: Thay U5 (mock `_atomic_write_json`) bằng fault injection thực: ép `os.replace` lỗi sau khi temp ghi, xác nhận file cũ nguyên vẹn, temp dọn, lỗi truyền lên.
- [ ] `tests/test_engine.py`: Thay E14 (mock `save_checkpoint`) bằng test lưu thật, inject lỗi ghi, xác nhận checkpoint cũ vẫn nạp được.
- [ ] Đồng bộ U5/E14 mô tả với tests thực tế.

### T6 — Đồng bộ docs và nghiệm thu
- [ ] Cập nhật `docs/plans/implementation_plan.md` chính sách legacy sau cleanup.
- [ ] Đồng bộ signature `save_model`, API `train`, 14 exports, validation metadata, persistence guarantee giữa code/docs/tests.
- [ ] Chạy kiểm chứng:
  ```powershell
  pytest tests\test_smoke.py -q
  pytest tests\test_model_builder.py tests\test_engine.py tests\test_utils.py -m "not slow" -q
  pytest tests\ -m "not slow" -q
  pytest tests\ -q
  git diff --check
  ```
- [ ] So sánh thời lượng suite nhanh với baseline; mục tiêu dưới 15 giây chỉ nếu đo thực tế đạt.

---

## Hợp đồng 14 exports (chốt)

Theo `00_lab2_description.md` mục 2.8.6, `__all__` phải chứa chính xác:

1. `ModelConfig`
2. `build_model`
3. `get_transforms`
4. `count_params`
5. `describe_model`
6. `set_train_mode`
7. `get_param_groups`
8. `create_train_val_test_dataloaders`
9. `train`
10. `evaluate_step`
11. `set_seed`
12. `get_device`
13. `run_inference`
14. `evaluate_predictions`

**Không được có thêm** `count_trainable_params`, `count_total_params`, `CIFAR10ModelV0`, `CIFAR10ModelV1`, `MODEL_NAMES`, `eval_model`, `save_training_metadata`, `load_training_artifacts`, `accuracy_fn`, `print_train_time`, `ARCH_SPECS`, `format_model_summary` trong `__all__`.

---

## Mức bảo đảm persistence (atomicity)

- **JSON artifacts** (`config.json`, `history.json`, `test_metrics.json`): đã atomic qua `_atomic_write_json` (temp + `os.replace`). ✅
- **Checkpoint** (`last_checkpoint.pth`): hiện dùng `torch.save` trực tiếp, **chưa atomic**. Cần làm atomic theo clean.md Bước 5.
- **Best model** (`best_model.pth`): hiện `engine.train` dùng `torch.save(model.state_dict(), save_p)` trực tiếp, **chưa atomic**. Cần làm atomic.

Nếu đặc tả không yêu cầu atomic checkpoint, không mở rộng code ngoài phạm vi; ghi rõ guarantee thực tế và kiểm thử đúng lỗi được truyền lên.

---

## Cổng nghiệm thu (cần hết mới hoàn tất)

- [ ] Notebook code không còn dùng API/kwargs legacy đã gỡ.
- [ ] Không còn định nghĩa hoặc export legacy API đã gỡ; imports công khai chính xác 14 symbols.
- [ ] `save_model(model, path)` và `train` signature thống nhất với API đã chốt; `verbose` giữ/bỏ theo vai trò hành vi.
- [ ] `_validate` không dựng model torchvision tạm; test vẫn bao phủ ResNet-18 và hành vi đặc thù VGG16.
- [ ] Test best-checkpoint phân biệt được weights giữa các epoch; test lỗi ghi chạm filesystem thật và kiểm tra atomicity contract.
- [ ] Tài liệu hiện hành không mâu thuẫn với code; suite chạy trong `.venv` và kết quả/thời lượng báo cáo chính xác.
- [ ] Không xóa/ghi đè artifacts huấn luyện có sẵn, không tải dữ liệu/pretrained weights, không chạy training.

---

## Câu hỏi cần xác nhận trước khi implement (nếu có)

1. **`save_model(model, path)`**: function này sẽ được đặt ở module nào (`utils.py` đã có `save_model` legacy với signature khác)? Cần giữ hàm cũ làm alias hay xóa sạch?
2. **`ARCH_SPECS` và `format_model_summary`**: đang trong `__all__` nhưng không thuộc 14 symbols hợp đồng. Gỡ hoàn toàn khỏi `__all__` hay giữ import trực tiếp từ submodule?
3. **Checkpoint atomic**: đặc tả có bắt buộc checkpoint phải atomic không, hay chỉ JSON artifacts là đủ?
4. **Test legacy kwargs**: `test_backward_compat_checkpoint_path` và `test_backward_compat_frozen_backbone` trong `test_engine.py` — xóa hoàn toàn hay rewrite để test API mới?

---

Khi các câu hỏi trên được trả lời, plan này sẵn sàng chuyển sang agent implementation để thực hiện.
