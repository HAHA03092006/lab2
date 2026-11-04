# Đặc tả Thiết kế Module Mô hình — `model_builder.py`

> **Tài liệu:** `docs/define&designt/02_model.md` (Define & Design cho kiến trúc mô hình chuyển giao)
> **Module liên quan:** `going_modular/model_builder.py`
> **Phụ thuộc:** `docs/define&designt/00_lab2_description.md` (tổng quan), `docs/define&designt/01_data.md` (tiền xử lý), `docs/define&designt/03_engine.md` (huấn luyện)
> **Trạng thái:** Pre-implementation Design Specification

---

## PHẦN I: DEFINE — Xác định Bài toán & Yêu cầu

### 1.1. Bối cảnh & Mục tiêu

Module `model_builder.py` chịu trách nhiệm **tạo ra mô hình sẵn sàng huấn luyện** từ một kiến trúc pre-trained trên ImageNet (Step 2, 3, 4 của đề bài). Module không chứa vòng lặp huấn luyện và không đọc dữ liệu.

Mục tiêu:

* Nạp VGG16 và ResNet-18 cùng trọng số ImageNet qua API của `torchvision.models`.
* Thay tầng phân loại cuối để khớp 10 lớp của CIFAR-10.
* Hỗ trợ hai chiến lược: **Feature Extraction** (đóng băng backbone) và **Fine-Tuning** (mở khóa N khối cuối).
* Tách toàn bộ lựa chọn thiết kế ra một cấu hình duy nhất (`ModelConfig`) để thí nghiệm tái lập được và đổi kiến trúc mà không sửa logic.

### 1.2. Đầu vào / Đầu ra

| | Mô tả |
| :--- | :--- |
| **Đầu vào** | `ModelConfig` (kiến trúc, số lớp, trọng số, chiến lược freeze, dropout) |
| **Đầu ra** | `nn.Module` có đầu ra kích thước `(B, num_classes)`; trạng thái `requires_grad` đã được thiết lập |
| **Đầu ra phụ** | Transform tiền xử lý đi kèm trọng số; thống kê tham số; nhóm tham số cho optimizer |

### 1.3. Yêu cầu chức năng

| Mã | Yêu cầu |
| :--- | :--- |
| FR-1 | Nạp được `vgg16` và `resnet18` với trọng số `IMAGENET1K_V1` hoặc khởi tạo ngẫu nhiên (`weights=None`). |
| FR-2 | Thay head đúng vị trí theo từng họ mạng, đầu ra `num_classes`. |
| FR-3 | Đóng băng toàn bộ backbone; head mới luôn trainable. |
| FR-4 | Mở khóa `N` khối cuối của backbone cho giai đoạn Fine-Tuning. |
| FR-5 | In và thống kê tham số `total / trainable / frozen` (Step 3). |
| FR-6 | Giữ BatchNorm đã đóng băng ở chế độ `eval` khi huấn luyện. |
| FR-7 | Tạo nhóm tham số với learning rate riêng cho head và backbone. |
| FR-8 | Cung cấp transform tiền xử lý đúng với bộ trọng số đã chọn. |
| FR-9 | Thêm kiến trúc mới chỉ bằng cách thêm một mục vào registry. |
| FR-10 | Cấu hình có thể chuyển qua lại với `dict`/JSON để lưu cùng artifacts. |

### 1.4. Ràng buộc

* `num_classes = 10` (CIFAR-10).
* Trọng số phải **ghi cụ thể** (`IMAGENET1K_V1`), không dùng `DEFAULT`, vì alias này có thể đổi giữa các phiên bản torchvision.
* Không dùng tham số `pretrained=True` (đã deprecated).
* Toàn bộ khởi tạo phải tái lập được khi cố định `SEED = 42` (xử lý ở `utils.py`).

### 1.5. Tiêu chí chấp nhận

1. Với cùng `ModelConfig`, `build_model` trả về mô hình có cùng số tham số trainable.
2. Sau Feature Extraction, chỉ các tham số của head là trainable.
3. Đầu ra mô hình có shape `(B, 10)` với đầu vào `(B, 3, 224, 224)`.
4. Bộ test ở mục 2.11 đều đạt.

---

## PHẦN II: DESIGN — Thiết kế Chi tiết

### 2.1. Vị trí trong kiến trúc hệ thống

```text
notebooks/pretrained_models.ipynb
        ├── tạo ModelConfig ──► model_builder.py ──► nn.Module ──► engine.py
        │                           │                    ▲
        │                           └── set_train_mode ──┘
        ├── lấy get_transforms(cfg, weights=...) ──► truyền vào data_setup.py
        └── lấy get_param_groups(...) ──► optimizer
```

### 2.2. Thiết kế `ModelConfig`

Dùng `dataclass(frozen=True)` để cấu hình bất biến và so sánh được.

| Tham số | Kiểu | Mặc định | Ý nghĩa |
| :--- | :--- | :--- | :--- |
| `arch` | `str` | `"resnet18"` | Khóa trong registry (`resnet18`, `vgg16`). |
| `num_classes` | `int` | `10` | Số lớp đầu ra. |
| `weights` | `str \| None` | `"IMAGENET1K_V1"` | Tên bộ trọng số; `None` là khởi tạo ngẫu nhiên. |
| `freeze_backbone` | `bool` | `True` | Đóng băng toàn bộ backbone trước khi thay head. |
| `unfreeze_last_n` | `int` | `0` | Số khối cuối của backbone được mở khóa (Fine-Tuning). |
| `dropout` | `float` | `0.0` | Dropout đặt trước `Linear` mới; `0` nghĩa là không dùng. |
| `freeze_bn_stats` | `bool` | `True` | Giữ BatchNorm đã đóng băng ở `eval` khi train. |

**Tách cấu hình mô hình và cấu hình huấn luyện.** `ModelConfig` chỉ mô tả mô hình. Các tham số `lr`, `epochs`, `batch_size`, `seed` thuộc cấu hình huấn luyện. Khi lưu artifacts, ghép hai phần vào một `config.json`:

```json
{
  "model": {
    "arch": "resnet18", "num_classes": 10, "weights": null,
    "freeze_backbone": true, "unfreeze_last_n": 1,
    "dropout": 0.0, "freeze_bn_stats": true
  },
  "data": { "batch_size": 32, "transform_weights": "IMAGENET1K_V1" },
  "train": { "lr_head": 1e-3, "lr_backbone": 1e-4, "epochs": 3, "start_epoch": 5,
             "init_from": "artifacts/v1_resnet18" },
  "seed": 42
}
```

`data.batch_size` là vị trí duy nhất của batch size trong schema hệ thống. `train.epochs` là số epoch chạy trong phase hiện tại; `train.start_epoch` chỉ là offset toàn cục cho log/báo cáo. Ví dụ này mô tả fine-tune: `model.weights=null` và `train.init_from` trỏ tới checkpoint baseline được nạp sau khi dựng kiến trúc.

### 2.3. Registry kiến trúc

Chỉ lưu những gì **khác nhau** giữa các họ mạng; phần còn lại dùng chung.

| Kiến trúc | Vị trí head | `in_features` | Danh sách khối để unfreeze |
| :--- | :--- | :--- | :--- |
| `resnet18` | `model.fc` | 512 | `layer1`, `layer2`, `layer3`, `layer4` (4 khối) |
| `vgg16` | `model.classifier[6]` | 4096 | 5 khối conv của `features`, mỗi khối kết thúc bằng `MaxPool2d` |

Mỗi mục registry gồm 3 hàm: `get_head`, `set_head`, `blocks`. `in_features` được đọc trực tiếp từ tầng cũ, không ghi cứng, để tránh sai khi đổi biến thể mạng.

**Với VGG16:** chỉ tầng `classifier[6]` bị thay. Hai tầng FC ẩn (`classifier[0]`, `classifier[3]`) giữ nguyên trạng thái bị đóng băng khi Feature Extraction và không nằm trong danh sách khối để unfreeze.

### 2.4. Luồng `build_model`

```text
1. validate(cfg)
2. model = get_model(cfg.arch, weights=cfg.weights)       # Step 2
3. nếu freeze_backbone: đặt requires_grad=False cho mọi tham số
4. set_head(model, num_classes, dropout)                  # Step 4: head mới trainable
5. nếu unfreeze_last_n > 0: mở khóa N khối cuối           # Fine-Tuning
6. return model
```

**Thứ tự freeze rồi mới thay head là có chủ đích.** Tầng mới tạo ra mặc định có `requires_grad=True`, nên không cần hàm "mở khóa head" riêng. Nếu thay head trước rồi freeze toàn bộ, head sẽ bị đóng băng theo và optimizer có thể nhận danh sách tham số rỗng.

### 2.5. Chiến lược hai giai đoạn

| | Giai đoạn 1: Feature Extraction | Giai đoạn 2: Fine-Tuning |
| :--- | :--- | :--- |
| Cấu hình | `freeze_backbone=True`, `unfreeze_last_n=0` | `freeze_backbone=True`, `unfreeze_last_n=N` (N ≥ 1) |
| Head mới | Trainable | Trainable |
| N khối cuối | Frozen | Trainable |
| Các khối còn lại | Frozen | Frozen |
| Learning rate | `LR = 1e-3` cho head | `LR_FINETUNE = 1e-4` cho backbone, head giữ `1e-3` |
| Epochs | 5 | 3 |

Quy ước cho trường hợp **huấn luyện toàn bộ mô hình**: `freeze_backbone=False` và `unfreeze_last_n=0`. Kết hợp `freeze_backbone=False` với `unfreeze_last_n>0` bị từ chối (xem mục 2.9).

**Số tham số kỳ vọng** (đối chiếu bằng `count_params` khi hiện thực hóa):

| Mô hình | Tổng | Trainable (Giai đoạn 1) |
| :--- | :--- | :--- |
| ResNet-18 (10 lớp) | 11,181,642 | 5,130 (`512 × 10 + 10`) |
| VGG16 (10 lớp) | 134,301,514 | 40,970 (`4096 × 10 + 10`) |

Với ResNet-18, `layer4` có khoảng 8.39 triệu tham số, nên `unfreeze_last_n=1` làm số tham số trainable tăng lên khoảng 8.4 triệu.

### 2.6. Xử lý BatchNorm

`requires_grad=False` **không** ngăn BatchNorm cập nhật running mean/variance khi mô hình ở chế độ `train()`. Nếu để nguyên, thống kê ImageNet của các tầng đóng băng sẽ bị ghi đè bằng thống kê CIFAR-10.

Thiết kế: hàm `set_train_mode(model, cfg)` thay cho `model.train()` trong `engine.py`:

1. Gọi `model.train()`.
2. Nếu `freeze_bn_stats=True`: với mỗi tầng BatchNorm mà **mọi tham số đều bị đóng băng**, gọi `.eval()`.

Hệ quả: khi Fine-Tuning với `unfreeze_last_n=1`, BN của `layer4` (trainable) ở chế độ train, còn BN của `bn1`, `layer1`–`layer3` ở chế độ eval. VGG16 không có BatchNorm nên hàm này không có tác dụng. Chế độ đánh giá vẫn dùng `model.eval()` bình thường.

### 2.7. Nhóm tham số cho optimizer

`get_param_groups(model, cfg, lr_head, lr_backbone)` trả về danh sách nhóm:

* Nhóm 1: tham số của head, learning rate `lr_head`.
* Nhóm 2 (chỉ khi có backbone trainable): các tham số trainable còn lại, learning rate `lr_backbone`.

Ở Giai đoạn 1, hàm chỉ trả về một nhóm, nên không có nhóm rỗng gây lỗi cho optimizer.

### 2.8. Tiền xử lý đi kèm trọng số

`get_transforms(cfg, weights=None)` trả về `weights.transforms()` của bộ trọng số đã chọn, thay vì tự khai báo mean/std. Nếu không truyền `weights`, hàm dùng `cfg.weights`; Notebook truyền `data.transform_weights` để giữ tiền xử lý độc lập với weights dùng khởi tạo mô hình.

* Notebook gọi hàm này rồi truyền transform vào `data_setup.py` (Dependency Injection); `model_builder.py` không phụ thuộc vào module dữ liệu.
* Transform này gồm Resize(256), CenterCrop(224) và chuẩn hóa ImageNet. Ảnh CIFAR-10 (32×32) bị phóng to trước khi cắt; cần đánh giá ảnh hưởng này trong `05_evaluation_results.md`.
* Transform này không chứa augmentation. Augmentation cho tập train (nếu dùng) do `data_setup.py` thêm riêng.
* Nếu cả `cfg.weights` và tham số `weights` đều không xác định weights, hàm báo `ValueError`.

### 2.9. Giao diện hàm và ngoại lệ

| Hàm | Đầu vào | Đầu ra | Ngoại lệ |
| :--- | :--- | :--- | :--- |
| `build_model(cfg)` | `ModelConfig` | `nn.Module` | `ValueError` nếu cấu hình không hợp lệ |
| `get_transforms(cfg, weights=None)` | `ModelConfig`, tên weights tùy chọn | transform | `ValueError` nếu không xác định được weights |
| `count_params(model)` | `nn.Module` | `{total, trainable, frozen}` | không |
| `set_train_mode(model, cfg)` | `nn.Module`, `ModelConfig` | `None` | không |
| `get_param_groups(model, cfg, lr_head, lr_backbone)` | như tên | `list[dict]` | không |
| `ModelConfig.to_dict()` / `from_dict(d)` | | `dict` / `ModelConfig` | `TypeError` nếu `d` có khóa lạ |

Các điều kiện kiểm tra trong `build_model` (đều báo `ValueError`):

* `arch` không có trong registry.
* `unfreeze_last_n < 0`.
* `freeze_backbone=False` mà `unfreeze_last_n > 0`.
* `dropout` ngoài khoảng `[0, 1)`.
* `unfreeze_last_n` lớn hơn số khối của kiến trúc (4 với ResNet-18, 5 với VGG16).

### 2.10. Tích hợp với các module khác

| Module | Cách dùng `model_builder` |
| :--- | :--- |
| `data_setup.py` | Nhận transform do Notebook lấy từ `get_transforms(cfg, weights=data.transform_weights)`; module dữ liệu không import `model_builder`. |
| `engine.py` | Gọi `set_train_mode(model, cfg)` đầu mỗi `train_step`; dùng `model.eval()` ở `evaluate_step`. |
| `utils.py` | Lưu và nạp `cfg.to_dict()` trong `config.json`; chế độ `reuse` dựng kiến trúc với `weights=None` rồi nạp `best_model.pth`, không tải lại torchvision weights. |
| Notebook | Dùng danh mục sáu runs duy nhất ở `00`; ba run khảo sát dùng ResNet-18 Feature Extraction và chỉ thay đúng hyperparameter đã chỉ định. `model_builder` không sở hữu danh sách thí nghiệm. |

**Lưu ý nguồn khởi tạo và chế độ `reuse`:** `model.weights` là weights torchvision được dùng trực tiếp để khởi tạo; `train.init_from` là checkpoint của run khác được nạp sau khi dựng kiến trúc. Với fine-tune từ baseline, dựng model với `weights=None`, nạp checkpoint baseline rồi mới mở khóa `layer4`. Trong `reuse`, dựng kiến trúc với `weights=None` trước khi nạp checkpoint của chính run đó. `data.transform_weights` vẫn giữ `IMAGENET1K_V1` để lấy preprocessing và mean/std; `state_dict` không chứa quy tắc preprocessing.

### 2.11. Kế hoạch kiểm thử

| Mã | Kiểm tra | Kỳ vọng |
| :--- | :--- | :--- |
| T1 | Dựng VGG16/ResNet-18 với `weights=None`, đầu vào giả lập | Logits có shape `(B, 10)`; tổng số tham số khớp giá trị kỳ vọng. |
| T2 | Feature Extraction cho từng kiến trúc | Chỉ head mới trainable; backbone frozen. |
| T3 | ResNet-18 Fine-Tuning với `unfreeze_last_n=1` | Head và `layer4` trainable; khối trước đóng băng; BN frozen ở eval. |
| T4 | `get_param_groups` ở Feature Extraction và Fine-Tuning | Không có nhóm rỗng; LR head/backbone đúng; tham số trainable xuất hiện đúng một lần. |
| T5 | Config lỗi: kiến trúc lạ, dropout sai, số khối unfreeze ngoài miền, kết hợp cờ mâu thuẫn | Báo `ValueError`. |
| T6 | `get_transforms` với weights được chọn và `weights=None` | Transform chỉ lấy từ weights được truyền/config; không có weights để xác định thì báo `ValueError`. |
| T7 | Round-trip `ModelConfig.to_dict/from_dict` và khóa lạ | Config giữ nguyên; khóa lạ bị từ chối. |

Chạy kiểm thử bằng `pytest tests/test_model_builder.py -v`; các kiểm thử cần tải ImageNet weights được đánh dấu `slow`.

### 2.12. Mở rộng

Thêm kiến trúc mới (ví dụ DenseNet, MobileNet) gồm 3 bước:

1. Viết `get_head`, `set_head`, `blocks` cho họ mạng đó.
2. Thêm một mục vào `ARCH_SPECS`.
3. Thêm test T1–T5 cho kiến trúc mới.

### 2.13. Quyết định thiết kế của nhóm so với chuẩn từ tài liệu

| Hạng mục | Có trong tài liệu chính thức | Quyết định thiết kế của nhóm |
| :--- | :--- | :--- |
| Nạp trọng số bằng `weights=` enum/chuỗi, `get_model` | Có (torchvision) | |
| Transform đi kèm trọng số (`weights.transforms()`) | Có (torchvision) | |
| `model.train()` / `model.eval()` do BatchNorm | Có (torchvision) | |
| Hai kịch bản Fine-Tuning và Feature Extraction | Có (PyTorch Tutorial) | |
| Cấu trúc package `going_modular` | Có (Learn PyTorch ch.05) | |
| `ModelConfig`, registry, `unfreeze_last_n` | | Có |
| Định dạng `config.json` | | Có |
| `set_train_mode` giữ BN đã freeze ở eval | | Có |
| Thứ tự freeze rồi mới thay head | | Có |

### 2.14. Tài liệu tham khảo

1. PyTorch / torchvision, *Models and pre-trained weights*: https://docs.pytorch.org/vision/stable/models.html
2. PyTorch Tutorials, *Transfer Learning for Computer Vision Tutorial*: https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
3. Daniel Bourke, *Learn PyTorch for Deep Learning*, chương 05 (Going Modular) và 06 (Transfer Learning): https://www.learnpytorch.io
4. Stanford CS231n, *Transfer Learning* (được trích trong tài liệu số 2).
