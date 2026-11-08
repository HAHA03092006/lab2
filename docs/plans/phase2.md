# Phase 2 — Checklist `model_builder.py` và tests

> **Nguồn chính:** [`implementation_plan.md`](./implementation_plan.md), Phase 2  
> **Đặc tả module:** [`02_model.md`](../define&designt/02_model.md)  
> **Đặc tả test:** [`02_test_model_builder.md`](../define&designt/tests/02_test_model_builder.md)  
> **Điều kiện đầu vào:** Phase 1 đạt theo [`phase1.md`](./phase1.md)  
> **Mục tiêu:** Xây API tạo mô hình cấu hình hóa cho VGG16/ResNet-18, kiểm chứng bằng test nhanh và cập nhật exports liên quan.  
> **Phạm vi:** `going_modular/model_builder.py`, `tests/test_model_builder.py` và exports model liên quan trong `going_modular/__init__.py`.

## 1. Xác nhận trước khi bắt đầu

- [x] Xác nhận Phase 1 đã đạt và môi trường pytest/smoke test hoạt động.
- [x] Kiểm tra trạng thái Git và đọc các thay đổi hiện có trong những file dự định sửa; không ghi đè thay đổi ngoài phạm vi.
- [x] Xác nhận test specs đang dùng là `docs/define&designt/tests/02_test_model_builder.md`; các mã test trong file này là căn cứ khi viết test.
- [x] Đối chiếu API mục tiêu với code hiện có và ghi nhận các call sites/exports legacy trước khi thay đổi chữ ký hoặc xóa API.
- [x] Xác nhận Phase 2 không yêu cầu tải ImageNet weights trong test nhanh; dùng `weights=None` trừ nhóm test được đánh dấu `slow`.

## 2. Thiết kế và triển khai `ModelConfig`

- [x] Tạo `ModelConfig` bằng `dataclass(frozen=True)` với đúng các trường và giá trị mặc định trong `02_model.md`:
  - [x] `arch`
  - [x] `num_classes`
  - [x] `weights`
  - [x] `freeze_backbone`
  - [x] `unfreeze_last_n`
  - [x] `dropout`
  - [x] `freeze_bn_stats`
- [x] Cấu hình chỉ chứa thông tin mô hình; không đưa learning rate, epochs, batch size hoặc seed vào `ModelConfig`.
- [x] Triển khai `to_dict()` và `from_dict()` để round-trip giữ nguyên cấu hình.
- [x] Từ chối khóa không được định nghĩa trong `from_dict()` bằng `TypeError`; không bỏ qua âm thầm.
- [x] Xác nhận cấu hình bất biến sau khi khởi tạo.
- [x] Kiểm tra giá trị mặc định và validation cấu hình được kiểm thử, không suy diễn từ giá trị ngầm định khác với spec.

## 3. Registry và dựng mô hình

- [x] Khai báo registry `ARCH_SPECS` cho `resnet18` và `vgg16`, mỗi mục cung cấp cách lấy/thay head và các khối backbone dùng cho unfreeze.
- [x] Dựng kiến trúc qua API `torchvision.models` hiện hành với weights enum cụ thể; không dùng `pretrained=True` hoặc `Weights.DEFAULT`.
- [x] Hỗ trợ `weights=None` để khởi tạo ngẫu nhiên hoặc dựng kiến trúc trước khi nạp checkpoint; không tải weights ngoài ý muốn.
- [x] Thay đúng head:
  - [x] ResNet-18: `fc`.
  - [x] VGG16: `classifier[6]`.
- [x] Đọc `in_features` từ head cũ thay vì ghi cứng.
- [x] Nếu `dropout > 0`, đặt `Dropout` trước `Linear` mới theo hợp đồng; `dropout=0` không thêm lớp dropout.
- [x] Bảo đảm đầu ra có `num_classes` lớp và mọi tham số head mới trainable.
- [x] Bảo đảm model được trả về trực tiếp theo kiến trúc torchvision, không bọc trong class wrapper mới làm đổi keys của `state_dict`.

## 4. Freeze, unfreeze và BatchNorm

- [x] Với `freeze_backbone=True`, đóng băng backbone và giữ head mới trainable.
- [x] Với `unfreeze_last_n > 0`, mở đúng N khối cuối theo từng kiến trúc:
  - [x] ResNet-18: `layer1`–`layer4`.
  - [x] VGG16: 5 khối convolution theo định nghĩa trong đặc tả.
- [x] Với `freeze_backbone=False` và `unfreeze_last_n=0`, mọi tham số trainable.
- [x] `build_model` báo `ValueError` cho kiến trúc không hỗ trợ, `unfreeze_last_n < 0`, giá trị lớn hơn số khối, cấu hình freeze/unfreeze mâu thuẫn và dropout ngoài `[0, 1)`.
- [x] Triển khai `set_train_mode(model, cfg)`:
  - [x] Đưa model về train mode.
  - [x] Khi `freeze_bn_stats=True`, đưa các BatchNorm có toàn bộ tham số đóng băng về eval mode.
  - [x] BN thuộc khối trainable vẫn ở train mode.
  - [x] Khi `freeze_bn_stats=False`, không ép BN frozen sang eval.
- [x] Kiểm tra running statistics của BN đóng băng không thay đổi sau forward train-mode.

## 5. Transform và nhóm tham số optimizer

- [x] Triển khai `get_transforms(cfg, weights=None)` để lấy preprocessing từ weights preset đã chọn.
- [x] Tham số `weights` truyền vào có thể override `cfg.weights`, để transform vẫn lấy được khi model được dựng với `weights=None`.
- [x] Nếu không xác định được weights từ đối số hoặc config, báo `ValueError`; không tự chọn preset mặc định.
- [x] Xác nhận transform xử lý ảnh CIFAR-10 về shape `(3, 224, 224)` và không tự thêm augmentation.
- [x] Triển khai `get_param_groups(model, cfg, lr_head, lr_backbone)`:
  - [x] Luôn đưa tham số trainable của head vào nhóm LR head.
  - [x] Chỉ tạo nhóm backbone khi có backbone trainable.
  - [x] Không tạo nhóm rỗng, không trùng hoặc bỏ sót tham số trainable.
  - [x] Fine-Tuning dùng LR riêng cho head và backbone theo tham số truyền vào.

## 6. Thống kê và mô tả kiến trúc

- [x] Triển khai `count_params(model)` trả về `{total, trainable, frozen}`.
- [x] Đối chiếu số lượng tham số kỳ vọng theo `02_model.md`:
  - [x] ResNet-18: total `11,181,642`; feature-extraction trainable `5,130`.
  - [x] VGG16: total `134,301,514`; feature-extraction trainable `40,970`.
- [x] Triển khai `describe_model` theo quyết định trong kế hoạch để phục vụ khảo sát kiến trúc Step 3, không phụ thuộc `torchinfo`.
- [x] Xác nhận summary thể hiện tối thiểu kiến trúc/head và tổng/trainable/frozen parameters; không tải pretrained weights chỉ để mô tả cấu trúc.

## 7. Triển khai `tests/test_model_builder.py`

- [x] Viết test theo các trường hợp và tiêu chí trong `02_test_model_builder.md`; không dùng test chỉ kiểm tra implementation detail.
- [x] Tạo test cho T1–T14c theo mã trong test spec, bao gồm:
  - [x] Shape đầu ra và head đúng kiến trúc.
  - [x] Freeze/unfreeze, số lượng tham số và full-training mode.
  - [x] Cấu hình lỗi và ngoại lệ rõ ràng.
  - [x] BatchNorm mode/running statistics và optimizer parameter groups.
  - [x] Tái lập khởi tạo, `ModelConfig` round-trip/frozen/unknown key.
  - [x] Transform preset, thiếu weights, kích thước đầu ra và override weights khi `cfg.weights=None`.
- [x] Dùng `pytest.raises` cho các đầu vào không hợp lệ theo loại lỗi đã đặc tả.
- [x] Đánh dấu S1–S2 bằng `@pytest.mark.slow`; test phải bỏ qua rõ ràng nếu không tải được ImageNet weights do môi trường/mạng.
- [x] Không tải CIFAR-10 hoặc weights ImageNet trong nhóm test nhanh.

## 8. API exports và tương thích trong giai đoạn chuyển tiếp

- [x] Cập nhật các exports model đã sẵn sàng trong `going_modular/__init__.py` theo hợp đồng mục 2.8.6 của `00`.
- [x] Giữ test module import trực tiếp từ `going_modular.model_builder` để cô lập module; kiểm tra exports package khi API được thêm.
- [x] Rà soát notebook và workspace call sites trước khi đổi/xóa các API legacy (`build_model` cũ, class wrappers, parameter counters).
- [x] Không xóa class/hàm legacy trong Phase 2 nếu còn call site cần chúng; thay thế hoặc deprecated chỉ sau khi có kế hoạch tương thích và call sites đã được xử lý.
- [x] Không sửa hành vi notebook ngoài phần cần thiết để giữ tương thích trong quá trình refactor; việc refactor notebook toàn diện thuộc Phase 7.

## 9. Lệnh kiểm thử và cổng nghiệm thu

- [x] Chạy toàn bộ test nhanh của model:
  - [x] `python -m pytest tests/test_model_builder.py -v -m "not slow"`
- [x] Chạy toàn bộ test module nếu môi trường cho phép tải ImageNet weights:
  - [x] `python -m pytest tests/test_model_builder.py -v`
- [x] Xác nhận các test nhanh bao phủ mã T1–T14c; test slow S1–S2 được đánh dấu riêng.
- [x] Kiểm tra smoke test Phase 1 còn đạt sau thay đổi exports.
- [x] Chạy `python -m pytest --collect-only` để xác nhận không lỗi collection/import.
- [x] Xác nhận test không cần GPU; nếu VGG16 forward quá nặng trên môi trường mục tiêu, ghi rõ test nào bị giới hạn thay vì âm thầm bỏ.
- [x] Xác nhận số tham số thực tế; nếu lệch khỏi giá trị đặc tả, điều tra nguyên nhân và cập nhật spec chỉ sau khi có quyết định rõ.
- [x] Chạy `git diff --check` và rà soát các thay đổi trước khi kết luận Phase 2.

## 10. Ngoài phạm vi Phase 2

- Refactor `data_setup.py`, `utils.py`, `engine.py` hoặc `evaluate.py`.
- Thực hiện toàn bộ refactor notebook hoặc chạy sáu thí nghiệm.
- Tải CIFAR-10 hoặc chạy huấn luyện để đánh giá chất lượng mô hình.
- Xóa artifacts/checkpoints cũ hoặc cam kết tương thích với state dict wrapper cũ.
- Xóa API legacy trước khi các call sites được chuyển đổi.

## 11. Cổng hoàn tất Phase 2

- [x] `ModelConfig`, registry, `build_model`, `get_transforms`, `set_train_mode`, `get_param_groups`, `count_params` và model summary đáp ứng hợp đồng thiết kế.
- [x] Test nhanh T1–T14c đạt; S1–S2 được đánh dấu `slow` và kết quả được ghi nhận theo điều kiện môi trường.
- [x] Số lượng tham số khớp giá trị kỳ vọng hoặc mọi sai khác đã được điều tra và xử lý có căn cứ.
- [x] Exports model trong `going_modular/__init__.py` được cập nhật mà không làm hỏng smoke test Phase 1.
- [x] Không còn lỗi collection, API mismatch ngoài kế hoạch hoặc thay đổi legacy chưa được rà soát.
- [x] `git diff --check` đạt; không xóa/ghi đè thay đổi hoặc artifacts ngoài phạm vi.

**Kết quả Phase 2:** ☐ Chưa đạt  ☒ Đạt  
**Ngày rà soát:** 2026-11-05  
**Ghi chú / quyết định còn mở:**  

---

Sau khi đạt cổng này, bắt đầu **Phase 3 — `data_setup.py` và tests** theo [`implementation_plan.md`](./implementation_plan.md).
