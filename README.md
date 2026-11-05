# CIFAR-10 Image Classification — Lab 2 (PyTorch)

> **Môn học:** Deep Learning  
> **Chủ đề:** Hands-on practice with pre-trained neural network architectures (Transfer Learning & Fine-tuning)  
> **Tập dữ liệu:** CIFAR-10 (60.000 ảnh màu 32×32, 10 lớp)

Dự án này tập trung vào thực hành ứng dụng các kiến trúc mạng nơ-ron học sâu đã được huấn luyện trước (**Pre-trained Models** trên ImageNet) bằng PyTorch. Toàn bộ luồng thực nghiệm, huấn luyện, phân tích và đánh giá được thiết kế tập trung duy nhất tại **Jupyter Notebook**, kết hợp cùng gói thư viện tái sử dụng tinh gọn **`going_modular/`**.

---

## 🎯 Kiến trúc và Mô hình Thực nghiệm

| Mô hình | ID | Backbone Pretrained | Cơ chế huấn luyện | Ghi chú |
|---|---|---|---|---|
| **VGG16** | `v0` | `vgg16(weights='IMAGENET1K_V1')` | Feature Extraction (Đóng băng feature extractor, chỉ train classifier mới) | Kiến trúc CNN cổ điển sâu, nhiều tham số |
| **ResNet-18** | `v1` | `resnet18(weights='IMAGENET1K_V1')` | Feature Extraction & Fine-tuning | Mạng Residual hiện đại, nhẹ, hội tụ nhanh |

---

## 📁 Cấu trúc Thư mục Dự án

```
lab2/
│
├── notebooks/
│   └── pretrained_models.ipynb   # 🎯 NOTEBOOK CHÍNH (Entry point duy nhất)
│
├── going_modular/                     # Thư viện hỗ trợ mô-đun hoá (modules)
│   ├── __init__.py                    # Export các hàm và lớp tiện ích
│   ├── data_setup.py                  # Chia train/val/test & DataLoader với ImageNet transform
│   ├── model_builder.py               # Xây dựng và tùy biến mô hình Pre-trained
│   ├── engine.py                      # Vòng lặp huấn luyện và kiểm định
│   ├── evaluate.py                    # Confusion matrix, per-class accuracy, biểu đồ trực quan
│   └── utils.py                       # seed, device, lưu/nạp checkpoint (.pth)
│
├── data/                              # Tự động tải CIFAR-10 (~170 MB) khi chạy lần đầu (không commit)
├── artifacts/                         # Kết quả sinh ra khi chạy (config/history/metrics commit, *.pth không commit)
│   ├── figures/                       # Biểu đồ xuất ra từ quá trình đánh giá
│   └── <tên thí nghiệm>/              # Mỗi thí nghiệm một thư mục
├── runs/                              # Thư mục log TensorBoard (không commit)
├── tests/                             # Bộ kiểm thử tự động (conftest.py + test module)
│   ├── test_data_setup.py
│   ├── test_model_builder.py
│   ├── test_engine.py
│   ├── test_utils.py
│   ├── test_evaluate.py
│   └── test_smoke.py
├── docs/                              # Hệ thống tài liệu đặc tả thiết kế
│   ├── define&designt/               # Define & Design Specs (00–05)
│   │   └── tests/                    # Bộ đặc tả kiểm thử (00_test_plan.md – 05_test_evaluate.md)
│   ├── plans/                        # Kế hoạch triển khai theo phase (phase0.md, phase1.md, implementation_plan.md)
│   └── analysts/                     # Báo cáo phân tích (spec_vs_code_analyst.md, Spec_Review_Report.md)
├── requirements.txt                   # Danh sách thư viện phụ thuộc
└── README.md                          # Tài liệu hướng dẫn dự án
```

> [!NOTE]
> Dự án được tối giản hóa cho môi trường học tập/thực hành: **không dùng scripts CLI rời rạc**, toàn bộ luồng chạy và tham số được cấu hình inline trong notebook giúp dễ quan sát, đối chiếu và báo cáo kết quả.

---

## 🚀 Khởi tạo môi trường

Các lệnh dưới đây dành cho Windows PowerShell, chạy từ thư mục gốc của repository. Dự án yêu cầu Python 3.10 trở lên.

### Windows PowerShell

```powershell
# Tạo môi trường ảo (chỉ cần chạy một lần)
python -m venv .venv

# Kích hoạt môi trường ảo trong PowerShell
.venv\Scripts\Activate.ps1

# Nếu PowerShell chặn script kích hoạt, có thể bỏ qua kích hoạt và dùng
# trực tiếp .venv\Scripts\python.exe trong các lệnh bên dưới.

python --version
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Nếu cần dùng phiên bản Python cụ thể đã cài đặt, có thể tạo môi trường bằng Python Launcher, ví dụ:

```powershell
py -3.10 -m venv .venv
```

> Nếu sử dụng GPU NVIDIA trên máy cá nhân, hãy chọn lệnh cài PyTorch phù hợp với hệ điều hành/CUDA từ [trang cài đặt PyTorch](https://pytorch.org/get-started/locally/). Sau đó xác nhận `torch.cuda.is_available()` trả về `True`. Trên Google Colab, chọn GPU trong **Runtime → Change runtime type**.

## 🧪 Chạy kiểm thử

Chạy từ thư mục gốc repository. Trên Windows, có thể dùng trực tiếp Python trong `.venv`, không phụ thuộc việc môi trường đã được activate hay chưa.

```powershell
# Smoke test nhanh để xác nhận package import được
.venv\Scripts\python.exe -m pytest tests\test_smoke.py -v

# Bộ test nhanh, bỏ qua các test được đánh dấu slow
.venv\Scripts\python.exe -m pytest tests\ -m "not slow" -q

# Toàn bộ test, bao gồm cả các test slow
.venv\Scripts\python.exe -m pytest tests\ -v
```

Lệnh cuối bao gồm các test `slow`: một số test tải CIFAR-10 hoặc ImageNet weights và cần kết nối mạng. Nếu mạng/dữ liệu không sẵn sàng, một số test dữ liệu có thể được skip; hãy đọc phần tổng kết của pytest để phân biệt `passed`, `skipped`, `failed` và `deselected`.

Có thể chạy riêng nhóm test chậm hoặc một module:

```powershell
.venv\Scripts\python.exe -m pytest tests\ -m slow -v
.venv\Scripts\python.exe -m pytest tests\test_model_builder.py -m "not slow" -q
.venv\Scripts\python.exe -m pytest tests\test_engine.py -q
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest tests/ -m "not slow" -q
python -m pytest tests/ -v  # toàn bộ test, có cả nhóm slow
```

## 📓 Mở Notebook và chạy thực nghiệm

Notebook chính là [notebooks/pretrained_models.ipynb](notebooks/pretrained_models.ipynb).

### Chạy trên máy cá nhân

1. Mở notebook trong VS Code (cần extension Jupyter) hoặc JupyterLab.
2. Chọn kernel `.venv` của repository.
3. Chạy các cell theo thứ tự. Lần đầu, notebook có thể tải CIFAR-10 và pretrained weights; cần đủ dung lượng lưu trữ và kết nối mạng.

### Chạy trên Google Colab

1. Đặt repository sao cho thư mục `going_modular/` nằm tại `MyDrive/Lab2/going_modular/`.
2. Mở `MyDrive/Lab2/notebooks/pretrained_models.ipynb` và chọn GPU nếu có.
3. Chạy cell đầu để mount Drive. Notebook dùng `MyDrive/Lab2` làm project/artifact root và `/content/cifar10_data` làm thư mục dữ liệu tạm.
4. Chạy các cell tuần tự; giữ nguyên runtime trong khi chạy và kiểm tra artifacts đã xuất hiện trên Drive trước khi đóng phiên.

### Trạng thái các run

Notebook hiện tại có `RUN_MODE = "train"` hoặc `"reuse"` và thực hiện VGG16, ResNet-18 cùng một thí nghiệm fine-tuning ResNet-18. **Nó chưa triển khai đầy đủ sáu cấu hình Phase 8** trong `docs/plans/implementation_plan.md`: chưa có bộ chọn `RUN_ID`/`RESUME` và ba run khảo sát `v2` chưa được nối vào notebook. Vì vậy, không xem một lần chạy notebook hiện tại là đã hoàn tất Phase 8; cần tích hợp các run còn thiếu và kiểm chứng resume trước khi tuyên bố đủ sáu run.

`reuse` dùng artifacts đã lưu để bỏ qua huấn luyện; chỉ chọn chế độ này khi các artifacts cần thiết đã tồn tại. Không xóa hay ghi đè artifacts cũ để bắt đầu lại nếu chưa sao lưu và xác nhận đúng thư mục run.

Không commit dataset, TensorBoard logs hoặc checkpoint `.pth` lên GitHub. Config, history, metrics và hình kết quả được quản lý theo quy ước artifacts của dự án.

### 4. Giám sát bằng TensorBoard

Trong quá trình huấn luyện ở Section 4, quá trình học (Loss & Accuracy trên tập Train và Validation) sẽ được tự động ghi log vào thư mục `runs/`. Để mở giao diện trực quan:

```bash
# Trong terminal (đã kích hoạt .venv):
tensorboard --logdir=runs
```

Mở trình duyệt truy cập: [http://localhost:6006](http://localhost:6006)

---

## 📖 Nội dung Chi tiết trong Notebook (`pretrained_models.ipynb`)

Notebook được cấu trúc chặt chẽ thành **8 Section** tương ứng đầy đủ các bước của bài lab:

1. **Section 0 — Setup & Config:** Tự động nhận diện đường dẫn `ROOT`, import thư viện và thiết lập inline hyperparameters (`SEED=42`, `BATCH_SIZE=32`, `LR=1e-3`, `EPOCHS=5`, `VAL_SIZE=5000`).
2. **Section 1 — Load & Visualize Data:** Tải CIFAR-10, phân chia tập (45.000 train / 5.000 val / 10.000 test), kiểm tra kích thước và vẽ lưới ảnh minh họa kèm nhãn.
3. **Section 2 — Khám phá kiến trúc Pre-trained:** Soi chi tiết cấu trúc mạng VGG16, ResNet-18 qua `describe_model()`; phân tích số lượng tham số tổng vs tham số trainable.
4. **Section 3 — Kiểm tra model đã adapt:** Classifier 10 lớp được cấu hình trong `model_builder.py`; notebook xác nhận output shape `[2, 10]` và giải thích chiến lược đóng băng (freezing).
5. **Section 4 — Train & Monitor với TensorBoard:** Huấn luyện `v0` (VGG16) và `v1` (ResNet-18) với `engine.train()`; chọn checkpoint tốt nhất theo validation accuracy, lưu weights/history/config vào `artifacts/`, log vào TensorBoard; hỗ trợ nạp lại bằng `RUN_MODE="reuse"`.
6. **Section 5 — So sánh Models:** So sánh đường cong Loss/Accuracy qua các epoch, so sánh thời gian huấn luyện và hiệu quả hội tụ giữa VGG16 và ResNet-18.
7. **Section 6 — Thử nghiệm Fine-tuning:** Mở đóng băng (unfreeze) một số block cuối của ResNet-18, huấn luyện bổ sung với learning rate nhỏ (`1e-4`) và so sánh đối chứng giữa Frozen vs Fine-tuned.
8. **Section 7 — Đánh giá trên tập Test:** Đánh giá khách quan mô hình tốt nhất trên tập Test (10.000 ảnh chưa từng thấy); vẽ ma trận nhầm lẫn (**Confusion Matrix**), thống kê độ chính xác theo từng lớp (**Per-class Accuracy**) và tìm ra các cặp nhãn hay nhầm lẫn nhất.

---

## ⚙️ Quy ước Tiền xử lý Dữ liệu (Transforms)

Tiền xử lý được lấy trực tiếp từ weights tương ứng trong torchvision: `VGG16_Weights.IMAGENET1K_V1.transforms()` cho VGG16 và `ResNet18_Weights.IMAGENET1K_V1.transforms()` cho ResNet-18. Với hai weights hiện đang dùng, cấu hình là:

| Thành phần | Thông số | Mô tả |
|---|---|---|
| **Resize** | Cạnh ngắn lên `256` | Phóng to ảnh, giữ nguyên tỷ lệ khung hình |
| **Center crop** | `(224, 224)` | Cắt vùng giữa ảnh theo pipeline của weights |
| **ToTensor** | Chuyển đổi | Chuyển pixel về tensor và dải `[0.0, 1.0]` |
| **Normalize** | Mean/std lấy từ weights | Chuẩn hóa theo cấu hình ImageNet-1K đi kèm weights |

`get_pretrained_transform(model_name="vgg16")` và `get_pretrained_transform(model_name="resnet18")` chọn preset riêng cho từng kiến trúc. `create_train_val_test_dataloaders(transform_model=...)` dùng lựa chọn này nếu không truyền transform tùy chỉnh. Hai preset hiện tại có cùng cấu hình, nên notebook dùng chung DataLoader cho cả hai model. Khi bật augmentation cho tập train, `RandomHorizontalFlip` được áp dụng trước pipeline của weights; validation và test không augmentation.

---

## 🛠️ Xử lý Sự cố Thường gặp

| Hiện tượng | Nguyên nhân | Hướng khắc phục |
|---|---|---|
| `No module named 'going_modular'` | Kernel không nhận đúng thư mục cha | Section 0 trong notebook đã có cell tự động thêm `ROOT` vào `sys.path`. Đảm bảo đã chạy Section 0. |
| Tải Pretrained Weights rất chậm | Trọng số mạng (đặc biệt là VGG16 ~528MB) lớn | Quá trình tải chỉ diễn ra một lần duy nhất (được cache tại thư mục `~/.cache/torch/hub/checkpoints`). Hãy kiên nhẫn đợi tải xong. |
| `CUDA out of memory` | Bộ nhớ GPU không đủ khi xử lý batch 224×224 | Giảm `BATCH_SIZE` xuống `16` trong cell cấu hình ở Section 0. |
| Cảnh báo BatchNorm khi backbone bị freeze | PyTorch giữ eval mode cho BatchNorm | `set_train_mode(model, cfg)` trong `engine.py` giữ trạng thái `eval()` cho BatchNorm của phần đóng băng khi train. |