# Phân Tích & Kế Hoạch Thực Nghiệm Lab 2: Transfer Learning (CIFAR-10)

- **Người thực hiện:** Sơn 
- **Môn học:** Deep Learning - Practice 2
- **Mục tiêu:** Đánh giá, theo dõi tiến độ thực nghiệm và so sánh các mô hình Pre-trained Networks (PyTorch) trên tập dữ liệu CIFAR-10.

---

## 1. Tóm Tắt Mục Tiêu & Yêu Cầu Bài Lab
Dựa trên đề bài Practice 2, dự án tập trung hoàn thiện các nội dung chính:
1. **Thực nghiệm với nhiều mô hình Pre-trained:** So sánh các kiến trúc như VGG-16, ResNet-18/50, và DenseNet-121 từ `torchvision.models`.
2. **Tinh chỉnh Hyperparameters:** Khảo sát sự thay đổi hiệu năng khi thay đổi Learning Rate, Batch Size, Optimizer (Adam vs SGD).
3. **Theo dõi tiến trình bằng TensorBoard:** Ghi lại biểu đồ Loss/Accuracy trực quan trong từng epoch.
4. **Kỹ thuật Transfer Learning:**
   - **Feature Extraction:** Đóng đóng băng (Freeze) các lớp backbone, chỉ train lại classifier (head).
   - **Fine-Tuning:** Mở đóng băng (Unfreeze) một vài lớp cuối cùng để tinh chỉnh trọng số.

---

## 2. Bảng Theo Dõi & So Sánh Mô Hình Thực Nghiệm (Experiments Matrix)

| STT | Kiến trúc (Model) | Kỹ thuật (Method) | Batch Size | Learning Rate | Freeze Backbone | Accuracy (Test) | Ghi chú / Trạng thái |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---|
| 1 | **ResNet-18** | Feature Extraction | 32 | 0.001 | Có (True) | *Chờ cập nhật* | Baseline model chuẩn bị chạy |
| 2 | **VGG-16** | Feature Extraction | 32 | 0.001 | Có (True) | *Chờ cập nhật* | Tốn VRAM hơn, cần theo dõi |
| 3 | **DenseNet-121** | Feature Extraction | 64 | 0.001 | Có (True) | *Chờ cập nhật* | Đang cấu hình |
| 4 | **ResNet-18** | Fine-Tuning | 32 | 0.0001 | Không (Lớp cuối)| *Chờ cập nhật* | Unfreeze `layer4` |

---

## 3. Quy Trình Kỹ Thuật (Technical Steps & Pipeline)

1. **Chuẩn bị Dữ liệu (Data Pipeline):**
   - Tải tập dữ liệu CIFAR-10 (10 lớp, ảnh $32 \times 32$).
   - Áp dụng `torchvision.transforms`: Resize lên $224 \times 224$ (nếu dùng chuẩn ImageNet), Normalize theo mean/std của ImageNet.
   - Chia DataLoader cho Train / Test.

2. **Cấu Trúc Mô Hình (Model Adaptation):**
   - Khởi tạo model với `weights=ResNet18_Weights.DEFAULT`.
   - Đóng băng `param.requires_grad = False` cho toàn bộ các lớp trích xuất đặc trưng.
   - Thay thế lớp `fc` cuối cùng phù hợp với **10 lớp đầu ra** của CIFAR-10.

3. **Huấn Luyện & Ghi Log (Training & TensorBoard):**
   - Hàm tổn thất: `nn.CrossEntropyLoss()`.
   - Thuật toán tối ưu: `optim.Adam` (hoặc `optim.SGD` với momentum).
   - Đưa chỉ số `Loss` và `Accuracy` của từng Epoch vào `SummaryWriter` để trực quan hóa trên TensorBoard.
