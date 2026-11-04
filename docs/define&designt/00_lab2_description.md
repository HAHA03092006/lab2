# Đặc tả Thiết kế Hệ thống Lab 2 — Ứng dụng Pre-trained Models

> **Chủ đề:** Practice 2 — Hands-on practice with pre-trained neural network architectures
> **Bài toán mục tiêu:** Phân loại ảnh trên tập dữ liệu CIFAR-10 ($60,000$ ảnh màu, $10$ lớp) bằng phương pháp Học chuyển giao (Transfer Learning).
> **Trạng thái tài liệu:** Pre-implementation Design Specification (Đặc tả thiết kế trước khi hiện thực hóa mã nguồn).
> **Vai trò của tài liệu này:** là **nguồn duy nhất** cho cấu hình thực nghiệm, schema `config.json`, danh mục thí nghiệm, chính sách artifacts và các quy ước chung. Các tài liệu `01` đến `05` trỏ về đây thay vì tự định nghĩa lại.

---

## PHẦN I: DEFINE — Xác định Bài toán & Yêu cầu Hệ thống

### 1.1. Xác định Bối cảnh & Mục tiêu (Problem Statement & Objectives)

Bài thực hành yêu cầu xây dựng một quy trình Học chuyển giao (Transfer Learning) hoàn chỉnh, tận dụng các mô hình thị giác máy tính đã được huấn luyện trên tập dữ liệu lớn ImageNet để giải quyết bài toán phân loại trên tập dữ liệu mục tiêu CIFAR-10.

* **Mục tiêu kiến thức:** Nắm vững cơ chế tải và khởi tạo các kiến trúc mạng nơ-ron huấn luyện sẵn (pre-trained models) từ thư viện `torchvision.models`.
* **Mục tiêu kỹ năng:** Xác định được phương pháp điều chỉnh (adapt) tầng phân loại cuối và chiến lược đóng băng (freeze) hoặc tinh chỉnh (fine-tune) các tầng trích xuất đặc trưng cho tác vụ cụ thể.

### 1.2. Xác định Yêu cầu Thực nghiệm (Experimental Requirements)

1. **Thử nghiệm đa kiến trúc:** Hệ thống phải hỗ trợ thử nghiệm và đối sánh ít nhất hai họ kiến trúc khác nhau (ví dụ: họ mạng tuần tự cổ điển **VGG** và họ mạng liên kết tắt **ResNet**).
2. **Điều chỉnh siêu tham số (Hyperparameters):** Cấu hình và thực nghiệm các giá trị khác nhau của tốc độ học (`learning_rate`) và kích thước lô (`batch_size`); các runs khảo sát được liệt kê tại mục 2.5. Không thực hiện grid search toàn diện.
3. **Giám sát quá trình huấn luyện:** Tích hợp công cụ **TensorBoard** vào vòng lặp huấn luyện để ghi nhận và trực quan hóa các chỉ số theo thời gian thực.

### 1.3. Xác định Quy trình 7 Bước Chuẩn từ Đề bài (7-Step Workflow Definition)

Hệ thống cần đáp ứng trọn vẹn 7 bước thực hành chuẩn của đề bài:

| Bước | Yêu cầu từ Đề bài (Steps) |
| :--- | :--- |
| **Step 1** | **Environment Setup:** Đảm bảo môi trường phát triển có đầy đủ `PyTorch` và `torchvision` (`pip install torch torchvision`). |
| **Step 2** | **Load a Pre-trained Model:** Sử dụng `torchvision.models` để nạp các kiến trúc mạng (ResNet, VGG, DenseNet, v.v.) cùng bộ trọng số đã huấn luyện trên tập dữ liệu ImageNet. |
| **Step 3** | **Explore the Model Architecture:** In và khảo sát cấu trúc mô hình để xác định tên các tầng (layers), kích thước chiều dữ liệu và số lượng tham số (parameters). |
| **Step 4** | **Adapt the Model for a Specific Task:** Thay thế tầng phân loại cuối cùng (classification layer) để khớp với số lượng lớp của tập dữ liệu mới. Thiết lập cơ chế đóng băng (freeze) các tầng phía trước khi trích xuất đặc trưng, hoặc mở khóa (unfreeze) một số tầng cuối khi tinh chỉnh (fine-tune). |
| **Step 5** | **Prepare the Data:** Tải và chuẩn bị tập dữ liệu, thiết kế bộ tiền xử lý ảnh bằng `torchvision.transforms` và đóng gói dữ liệu thành từng batch bằng `torch.utils.data.DataLoader`. |
| **Step 6** | **Train the Model:** Định nghĩa hàm mất mát (loss function), bộ tối ưu hóa (optimizer) và xây dựng vòng lặp huấn luyện mô hình trên tập dữ liệu. |
| **Step 7** | **Evaluate the Model:** Đánh giá hiệu năng tổng quát của mô hình trên tập dữ liệu kiểm thử độc lập (test dataset). |

**Độ phủ của thiết kế đối với 7 bước:**

| Bước | Nơi đáp ứng |
| :--- | :--- |
| 1 | `requirements.txt` (mục 2.7); `get_env_info` ghi phiên bản vào `config.json` (`04`) |
| 2 | `build_model` (`02`) |
| 3 | `count_params`, `describe_model`, `format_model_summary` (`02`) |
| 4 | `build_model`, registry kiến trúc, `unfreeze_last_n` (`02`) |
| 5 | `data_setup.py` (`01`) |
| 6 | `engine.py` (`03`) |
| 7 | `evaluate.py` (`05`) |

---

## PHẦN II: DESIGN — Thiết kế Kiến trúc & Cấu hình Hệ thống

### 2.1. Thiết kế Kiến trúc Phần mềm Dự kiến (Proposed System Architecture)

Để tránh việc viết toàn bộ mã nguồn rời rạc và trùng lặp trong một tệp Jupyter Notebook duy nhất, dự án được thiết kế theo kiến trúc **Modular Programming**. Toàn bộ logic nghiệp vụ sẽ được đóng gói thành một package độc lập mang tên `going_modular`, đi kèm hệ thống tài liệu đặc tả `docs/` và bộ kiểm thử tự động `tests/`.

#### Sơ đồ cấu trúc thư mục thiết kế

```text
toan58936-lab2/
├── .gitignore                        # Loại trừ data/, runs/, *.pth, __pycache__/ (JSON và hình trong artifacts/ vẫn được commit)
├── README.md                         # Hướng dẫn cài đặt, điều hướng tài liệu, cách tải trọng số (.pth)
├── requirements.txt                  # Danh sách các gói thư viện cần cài đặt (mục 2.7)
├── docs/                             # Hệ thống tài liệu đặc tả thiết kế (Define & Design Specs)
│   └── define&designt/              # Define & Design Specs
│       ├── 00_lab2_description.md   # Đặc tả tổng quan, cấu hình, danh mục thí nghiệm, quy ước chung
│       ├── 01_data.md               # Đặc tả thiết kế module xử lý dữ liệu
│       ├── 02_model.md               # Đặc tả thiết kế kiến trúc mô hình & Transfer Learning
│       ├── 03_engine.md              # Đặc tả thiết kế động cơ huấn luyện
│       ├── 04_utils.md               # Đặc tả thiết kế tiện ích và artifacts
│       ├── 05_evaluation_results.md  # Đặc tả thiết kế bộ công cụ đánh giá & biểu đồ
│       └── 06_implementation_plan.md # Kế hoạch hiện thực hóa
├── going_modular/                    # Package chứa các module mã nguồn cốt lõi
│   ├── __init__.py                   # Quản lý namespace và xuất các hàm API chính (mục 2.8)
│   ├── data_setup.py                 # Module tải, chia tách và tiền xử lý dữ liệu
│   ├── model_builder.py              # Module khởi tạo và tùy biến mô hình Pre-trained
│   ├── engine.py                     # Module thực thi vòng lặp huấn luyện và kiểm định
│   ├── evaluate.py                   # Module tính toán độ đo và vẽ biểu đồ phân tích
│   └── utils.py                      # Module tiện ích hệ thống (seed, device, I/O)
├── notebooks/                        # Không gian chạy thực nghiệm và trình bày báo cáo
│   └── pretrained_models.ipynb       # Notebook điều phối luồng thực nghiệm 7 bước
├── tests/                            # Bộ Unit Tests kiểm chứng từng module trước khi tích hợp
│   ├── conftest.py                   # Marker `slow`, backend matplotlib `Agg`, đường dẫn import
│   ├── test_data_setup.py
│   ├── test_model_builder.py
│   ├── test_engine.py
│   ├── test_utils.py
│   └── test_evaluate.py
├── data/                             # CIFAR-10 tải về (không commit)
├── artifacts/                        # Kết quả sinh ra khi chạy (mục 2.6)
│   ├── figures/                      # Biểu đồ xuất ra từ quá trình đánh giá
│   └── <tên thí nghiệm>/             # Mỗi thí nghiệm một thư mục (mục 2.5)
└── runs/                             # Event logs phục vụ TensorBoard (không commit)
```

#### Thiết kế Phân rã Trách nhiệm các Module (Module Responsibilities)

* **`data_setup.py`:** Giải quyết bài toán tương thích kích thước và màu sắc giữa CIFAR-10 và ImageNet, phân chia tập dữ liệu đảm bảo cân bằng lớp. Nhận transforms từ Notebook theo Dependency Injection; chỉ dùng `get_pretrained_transform` làm fallback độc lập.
* **`model_builder.py`:** Factory khởi tạo các kiến trúc mạng (`VGG16`, `ResNet-18`), thay lớp đầu ra, thiết lập trạng thái đóng băng, cung cấp tiền xử lý đi kèm trọng số và công cụ khảo sát kiến trúc (Step 3).
* **`engine.py`:** Chuẩn hóa vòng lặp `train_step` và `evaluate_step`, quản lý cập nhật trọng số, xử lý hành vi của tầng BatchNorm, đo thời gian mỗi epoch và ghi log sang TensorBoard.
* **`evaluate.py`:** Suy luận một lượt trên toàn bộ tập Test, xây dựng ma trận nhầm lẫn, độ đo, khoảng tin cậy, đo chi phí và trực quan hóa kết quả.
* **`utils.py`:** Hạt giống ngẫu nhiên, thiết bị tính toán, thông tin môi trường, TensorBoard writer và cơ chế lưu/nạp an toàn các tệp artifacts.

### 2.2. Thiết kế Tham số Cấu hình Thực nghiệm (Planned Configuration Design)

Để quá trình thực nghiệm trong Notebook diễn ra nhất quán và có tính tái lập cao, hệ thống được thiết kế xoay quanh một bộ tham số cấu hình tập trung:

| Nhóm cấu hình | Tên tham số dự kiến | Giá trị thiết kế | Mục đích thiết kế | Khóa trong `config.json` |
| :--- | :--- | :--- | :--- | :--- |
| **Dữ liệu & Tái lập** | `SEED` | `42` | Cố định mọi nguồn ngẫu nhiên; cho kết quả tái lập **trên cùng môi trường** (phần cứng và phiên bản thư viện). | `seed` |
| | `VAL_SIZE` | `5000` | Trích xuất $10\%$ từ tập Train gốc ($50,000$ mẫu) làm tập Validation chọn mô hình. | `data.val_size` |
| | `BATCH_SIZE` | `32` | Cân bằng giữa tốc độ huấn luyện và dung lượng bộ nhớ VRAM trên GPU. | `data.batch_size` |
| **Giai đoạn 1: Feature Extraction** | `EPOCHS` | `5` | Số vòng lặp huấn luyện riêng tầng Classifier mới khi đóng băng toàn bộ Backbone. | `train.epochs` |
| | `LR` | $1 \times 10^{-3}$ | Tốc độ học của **head** (tầng Linear mới) với bộ tối ưu hóa Adam; head luôn dùng `LR` ở mọi giai đoạn. | `train.lr_head` |
| **Giai đoạn 2: Fine-Tuning** | `UNFREEZE_LAST_N` | `1` | Số khối cuối của backbone được mở khóa (với ResNet-18 là `layer4`). | `model.unfreeze_last_n` |
| | `EPOCHS_FT` | `3` | Số vòng lặp tinh chỉnh bổ sung sau khi mở khóa các tầng cuối của Backbone. | `train.epochs` |
| | `LR_FINETUNE` | $1 \times 10^{-4}$ | Tốc độ học của **các khối backbone được mở khóa**, thấp hơn `LR` $10$ lần để tránh làm hỏng các đặc trưng ImageNet đã học. | `train.lr_backbone` |

### 2.3. Thiết kế Chế độ Thực thi (`RUN_MODE` Design)

Nhằm giải quyết vấn đề thời gian huấn luyện lâu trên các mạng lớn và giới hạn thời gian kết nối của môi trường đám mây (như Google Colab), luồng thực thi được thiết kế hỗ trợ 2 chế độ, **chọn riêng cho từng thí nghiệm**:

1. **Chế độ `"train"`:** Thực thi toàn bộ pipeline huấn luyện, lưu trữ trọng số tốt nhất cùng lịch sử huấn luyện xuống ổ đĩa.
2. **Chế độ `"reuse"`:** Bỏ qua bước huấn luyện, khôi phục trực tiếp trạng thái mô hình và lịch sử từ ổ đĩa để phục vụ nhanh cho việc phân tích và chấm điểm.

Điều kiện của `"reuse"`: thư mục thí nghiệm phải có đủ `best_model.pth`, `history.json`, `config.json`. Thiếu tệp huấn luyện thì báo `FileNotFoundError` nêu tên tệp thiếu và hướng dẫn tải trọng số trong `README.md`; hệ thống **không** tự chuyển sang huấn luyện. Nếu chỉ thiếu `test_metrics.json`, hệ thống nạp mô hình rồi chạy đánh giá Test một lần, lưu kết quả và dùng lại ở các lần sau.

### 2.4. Schema `config.json` (nguồn duy nhất)

```json
{
  "experiment": "v1_resnet18_finetune",
  "seed": 42,
  "env": { "device": "Tesla T4", "torch": "2.x", "torchvision": "0.x" },
  "model": { "arch": "resnet18", "num_classes": 10, "weights": null,
             "freeze_backbone": true, "unfreeze_last_n": 1, "dropout": 0.0, "freeze_bn_stats": true },
  "data":  { "batch_size": 32, "val_size": 5000, "num_workers": 0,
             "transform_model": "resnet18", "transform_weights": "IMAGENET1K_V1" },
  "train": { "epochs": 3, "lr_head": 1e-3, "lr_backbone": 1e-4, "monitor": "val_acc",
             "start_epoch": 5, "init_from": "artifacts/v1_resnet18" }
}
```

Trong schema này, `train.epochs` là **số epoch cần chạy trong phase hiện tại**; `train.start_epoch` là offset epoch toàn cục chỉ dùng cho TensorBoard/báo cáo. Fine-tune chạy 3 epoch với `start_epoch=5`, tương ứng bước toàn cục 6–8. Epoch đã hoàn tất trong phase được lưu riêng trong resume checkpoint.
* `model.weights`: nguồn torchvision dùng trực tiếp khi khởi tạo kiến trúc; `null` nếu run khởi tạo từ một checkpoint thí nghiệm.
* `data.transform_weights`: nguồn weights dùng để lấy preprocessing và mean/std; vẫn là `IMAGENET1K_V1` ở fine-tune dù `model.weights=null`.
* `train.init_from`: đường dẫn/thí nghiệm checkpoint khởi tạo run hiện tại; `null` nếu mô hình được khởi tạo trực tiếp từ `model.weights`.
* `env`: thiết bị và phiên bản thư viện lúc huấn luyện, để số liệu thời gian có ngữ cảnh.
* `lr_backbone` chỉ có tác dụng khi `unfreeze_last_n > 0`.
* Trong fine-tune, Notebook dựng ResNet-18 với `weights=None`, nạp `best_model.pth` của `init_from`, rồi mở khóa `layer4`; không tải lại ImageNet weights. Trong `reuse`, kiến trúc cũng được dựng với `weights=None` trước khi nạp run checkpoint. Transform luôn lấy theo `data.transform_weights`; hai nguồn này độc lập.

### 2.5. Danh mục thí nghiệm

| Thí nghiệm | Kiến trúc | Chiến lược | Epochs | `lr_head` / `lr_backbone` | Batch | `init_from` | Mục đích |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `v0_vgg16` | VGG16 | Feature Extraction | 5 | $10^{-3}$ / — | 32 | — | Đối sánh kiến trúc |
| `v1_resnet18` | ResNet-18 | Feature Extraction | 5 | $10^{-3}$ / — | 32 | — | Mốc chuẩn (baseline) |
| `v1_resnet18_finetune` | ResNet-18 | Fine-Tuning (`unfreeze_last_n=1`) | 3 (tổng 8) | $10^{-3}$ / $10^{-4}$ | 32 | `v1_resnet18` | Tác động của Fine-Tuning |
| `v2_resnet18_lr1e-2` | ResNet-18 | Feature Extraction | 5 | $10^{-2}$ / — | 32 | — | Khảo sát LR cao |
| `v2_resnet18_lr1e-4` | ResNet-18 | Feature Extraction | 5 | $10^{-4}$ / — | 32 | — | Khảo sát LR thấp |
| `v2_resnet18_bs64` | ResNet-18 | Feature Extraction | 5 | $10^{-3}$ / — | 64 | — | Khảo sát batch size |

**Quy ước đặt tên:** `v0` là mốc VGG16; `v1` là ResNet-18 và biến thể fine-tune; `v2` là các biến thể khảo sát hyperparameter.

**Nguyên tắc khảo sát:** các run `v2` xuất phát từ cùng cấu hình ResNet-18 Feature Extraction (`v1_resnet18`): giữ seed, dữ liệu, kiến trúc và số epoch cố định, mỗi run chỉ thay đổi một yếu tố. So sánh hai run `v2_resnet18_lr*` với baseline để phân tích learning rate; so sánh `v2_resnet18_bs64` với baseline để phân tích batch size. Đây là khảo sát có kiểm soát, không phải grid search toàn diện. Fine-tune vẫn là một run riêng để khảo sát chiến lược thích nghi.

**Nguyên tắc đánh giá:** chọn checkpoint bằng accuracy trên **Validation**, không dùng Test; chỉ đánh giá Test cuối cùng theo quy trình ở `05`.

### 2.6. Chính sách Artifacts và phân phối

Mỗi thư mục `artifacts/<thí nghiệm>/` gồm:

| Tệp | Nội dung | Commit vào Git? |
| :--- | :--- | :--- |
| `config.json` | Cấu hình theo schema mục 2.4 | Có |
| `history.json` | Lịch sử huấn luyện (`03`) | Có |
| `test_metrics.json` | Kết quả Test, chi phí, `y_true`, `y_pred`, confidence (`05`); sinh sau bước đánh giá | Có |
| `best_model.pth` | `state_dict` của mô hình tốt nhất | **Không** (`*.pth` bị loại trong `.gitignore`) |
| `last_checkpoint.pth` | Trạng thái phase gần nhất (model/optimizer, `phase_epochs_completed`, best score/epoch, history, phase config) phục vụ resume | **Không** (`*.pth` bị loại trong `.gitignore`) |

* `artifacts/figures/` được commit để đưa trực tiếp vào báo cáo.
* **Phân phối trọng số:** các tệp `.pth` (VGG16 cỡ $512$ MiB, vượt giới hạn $100$ MB của GitHub; ResNet-18 cỡ $43$ MiB mỗi tệp) được đóng gói và phát hành ngoài Git (GitHub Release hoặc Google Drive). `README.md` ghi liên kết tải và cách giải nén vào `artifacts/`.
* Nhờ `test_metrics.json` được commit, người chấm xem được kết quả Test ngay cả khi chưa tải `.pth`.
* Điều kiện đủ cho `reuse` gồm ba tệp huấn luyện (`best_model.pth`, `history.json`, `config.json`); nếu `test_metrics.json` chưa có, `reuse` đánh giá Test một lần rồi lưu. Bản ghi metrics giữ `y_true`, `y_pred` và confidence tối đa theo mẫu để có thể phân tích lỗi mà không lưu logits.

### 2.7. `requirements.txt`

Tối thiểu: `torch`, `torchvision` (từ phiên bản có `get_model`, tức $\ge 0.14$), `tensorboard`, `matplotlib`, `numpy`, `pillow`, `pytest`. Ghim phiên bản cụ thể sau khi chạy thành công lần đầu. Step 3 dùng `describe_model` tự viết nên không cần thêm `torchinfo`.

### 2.8. Quy ước chung giữa các module

1. **Giao thức hạt giống:** gọi `set_seed(SEED)` **ngay trước mỗi thí nghiệm** (trước `build_model`, trước khi tạo loader và trước khi huấn luyện). Loader được **tạo riêng cho từng thí nghiệm** và Train DataLoader nhận `torch.Generator` riêng được seed bằng `SEED`, để thứ tự shuffle không bị ảnh hưởng bởi các thao tác ngẫu nhiên khác.
2. **Tiền xử lý qua Dependency Injection:** Notebook lấy transform chuẩn từ `model_builder.get_transforms(cfg, weights=data.transform_weights)`, thêm augmentation train nếu cần, rồi truyền `train_transform` và `eval_transform` vào `data_setup`. `data_setup` giữ `get_pretrained_transform(...)` độc lập làm fallback mặc định; không import ngược `model_builder`.
3. **Tập Test:** chỉ dùng đánh giá cuối; chọn checkpoint tốt nhất bằng Validation. Các hyperparameter được chọn trước mỗi run; không điều chỉnh dựa trên kết quả Test.
4. **Giai đoạn 2:** dựng mô hình từ checkpoint `init_from` bằng cấu hình kiến trúc `weights=None`, dùng optimizer mới; trong `reuse`, tạo bản sao cấu hình với `weights=None` trước khi dựng kiến trúc và nạp checkpoint.
5. **Đặt tên test:** `test_<mã>_<mô tả>`; mã có tiền tố riêng theo tài liệu (`T` mô hình, `D` dữ liệu, `E`/`U`/`I` huấn luyện và tiện ích, `V` đánh giá), test chậm dùng `S`.
6. **`going_modular/__init__.py`:** xuất `ModelConfig`, `build_model`, `get_transforms`, `count_params`, `describe_model`, `set_train_mode`, `get_param_groups`, `create_train_val_test_dataloaders`, `train`, `evaluate_step`, `set_seed`, `get_device`, `run_inference`, `evaluate_predictions`.

### 2.9. Lộ trình Đặc tả Chi tiết (Documentation Roadmap)

Từ bản thiết kế tổng thể này, từng thành phần của hệ thống được đặc tả chi tiết theo đúng cấu trúc **DEFINE & DESIGN** qua chuỗi tài liệu tiếp theo:

1. **`docs/define&designt/01_data.md`:** Define & Design cho pipeline dữ liệu (`data_setup.py`).
2. **`docs/define&designt/02_model.md`:** Define & Design cho kiến trúc mô hình chuyển giao (`model_builder.py`).
3. **`docs/define&designt/03_engine.md`:** Define & Design cho động cơ huấn luyện (`engine.py`).
4. **`docs/define&designt/04_utils.md`:** Define & Design cho tiện ích và lưu/nạp artifacts (`utils.py`).
5. **`docs/define&designt/05_evaluation_results.md`:** Define & Design cho bộ công cụ đánh giá và phân tích thực nghiệm (`evaluate.py`).
6. **`docs/define&designt/06_implementation_plan.md`:** Kế hoạch hiện thực hóa theo từng giai đoạn.