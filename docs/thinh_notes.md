# Đóng góp tài liệu - Huỳnh Quang Thịnh

## 1. Giới thiệu

Tài liệu này tổng hợp chức năng chính của các thành phần trong thư mục `going_modular` thuộc dự án Deep Learning, đồng thời mô tả cách các thành phần phối hợp trong quá trình huấn luyện và đánh giá mô hình.

## 2. Chức năng các file

- `__init__.py`: Khởi tạo package và hỗ trợ import các thành phần từ going_modular.
- `data_setup.py`: Chuẩn bị dữ liệu, chia tập dữ liệu và tạo DataLoader.
- `model_builder.py`: Xây dựng mô hình Deep Learning và cấu hình các thành phần của mô hình.
- `engine.py`: Thực hiện quá trình huấn luyện và đánh giá mô hình trong từng epoch.
- `evaluate.py`: Hỗ trợ dự đoán, tính toán các chỉ số và trực quan hóa kết quả.
- `utils.py`: Chứa các hàm tiện ích phục vụ cấu hình, lưu mô hình, quản lý kết quả và ghi log.

## 3. Mục đích đóng góp

- Bổ sung tài liệu giúp thành viên dễ hiểu cấu trúc dự án.
- Giúp người mới nhanh chóng nắm được chức năng của từng file.
- Không thay đổi mã nguồn Python hiện có.

## 4. Quy trình chuẩn bị dữ liệu - data_setup.py

File `data_setup.py` chịu trách nhiệm chuẩn bị dữ liệu đầu vào cho mô hình.

Các chức năng chính:

- Tải bộ dữ liệu CIFAR-10 thông qua torchvision.
- Chia dữ liệu huấn luyện thành hai tập train và validation.
- Sử dụng phương pháp Stratified Split để giữ tỷ lệ các lớp tương đối cân bằng giữa hai tập.
- Tạo các DataLoader để cung cấp dữ liệu theo từng batch.
- Áp dụng preprocessing phù hợp với mô hình pretrained như ResNet18 và VGG16.

Điểm đáng chú ý:

- Tập train có thể sử dụng RandomHorizontalFlip để tăng tính đa dạng dữ liệu.
- Tập validation và test sử dụng transform đánh giá, không áp dụng augmentation ngẫu nhiên.
- Việc sử dụng seed giúp quá trình chia dữ liệu có thể tái lập.

Kết quả đầu ra gồm:

- train_dataloader
- val_dataloader
- test_dataloader
- class_names

## 5. Xây dựng mô hình - model_builder.py

File `model_builder.py` chịu trách nhiệm xây dựng kiến trúc mạng và cấu hình quá trình học.

Dự án hỗ trợ hai kiến trúc chính:

- ResNet18: Mạng CNN sử dụng các khối residual để hỗ trợ học đặc trưng.
- VGG16: Mạng CNN sử dụng nhiều lớp tích chập để trích xuất đặc trưng hình ảnh.

Cấu trúc mô hình:

- Backbone: Mạng pretrained dùng để trích xuất đặc trưng.
- Head: Lớp phân loại mới được thay thế để phù hợp với số lượng lớp của CIFAR-10.

Một số cấu hình quan trọng trong ModelConfig:

- arch: Lựa chọn kiến trúc ResNet18 hoặc VGG16.
- num_classes: Số lớp đầu ra, mặc định là 10.
- weights: Trọng số pretrained được sử dụng để khởi tạo mô hình.
- freeze_backbone: Đóng băng các tham số của backbone.
- unfreeze_last_n: Mở khóa một số khối cuối để Fine-Tuning.
- dropout: Cấu hình Dropout nhằm hỗ trợ hạn chế overfitting.

Ngoài ra, file còn có các chức năng:

- build_model(): Xây dựng mô hình từ cấu hình.
- get_transforms(): Lấy preprocessing tương ứng với trọng số pretrained.
- set_train_mode(): Thiết lập chế độ huấn luyện và xử lý BatchNorm.
- get_param_groups(): Tách nhóm tham số backbone và head để sử dụng learning rate riêng.

## 6. Quy trình huấn luyện - engine.py

File `engine.py` quản lý quá trình huấn luyện và đánh giá mô hình theo từng epoch.

Quy trình cơ bản:

1. Đưa dữ liệu đầu vào lên thiết bị tính toán.
2. Đưa dữ liệu qua mô hình để thực hiện Forward Pass.
3. Tính toán Loss giữa kết quả dự đoán và nhãn thực tế.
4. Sử dụng Backpropagation để tính gradient.
5. Optimizer cập nhật trọng số mô hình.
6. Đánh giá kết quả trên tập validation.

Hai hàm quan trọng:

- train_step(): Thực hiện một epoch huấn luyện, cập nhật trọng số và tính train loss, train accuracy.
- evaluate_step(): Đánh giá mô hình mà không cập nhật trọng số.

Hàm train() quản lý toàn bộ quá trình nhiều epoch.

Ngoài ra, engine còn hỗ trợ:

- Lưu lịch sử huấn luyện gồm loss, accuracy, learning rate và thời gian mỗi epoch.
- Theo dõi kết quả bằng TensorBoard.
- Chọn checkpoint tốt nhất dựa trên val_acc hoặc val_loss.
- Lưu checkpoint để có thể tiếp tục huấn luyện khi cần.

## 7. Đánh giá mô hình - evaluate.py

Sau khi huấn luyện, mô hình cần được đánh giá trên dữ liệu để xác định chất lượng dự đoán.

Các chức năng chính:

- run_inference(): Chạy suy luận trên DataLoader và thu thập kết quả.
- predict_probs(): Tính xác suất dự đoán thông qua Softmax.
- predict_dataloader(): Dự đoán nhãn cho các mẫu trong DataLoader.
- compute_confusion_matrix(): Xây dựng ma trận nhầm lẫn giữa nhãn thực tế và nhãn dự đoán.
- classification_metrics(): Tính toán các chỉ số đánh giá.

Các chỉ số quan trọng:

- Accuracy: Tỷ lệ dự đoán đúng trên tổng số mẫu.
- Precision: Mức độ chính xác của các dự đoán thuộc một lớp.
- Recall: Khả năng tìm ra các mẫu thực tế thuộc một lớp.
- F1-score: Trung bình điều hòa giữa Precision và Recall.

File còn hỗ trợ trực quan hóa:

- Confusion Matrix.
- Accuracy theo từng lớp.
- Các mẫu dự đoán sai.
- Biểu đồ Loss và Accuracy trong quá trình huấn luyện.

## 8. Quản lý và lưu kết quả - utils.py

File `utils.py` cung cấp các hàm dùng chung để hỗ trợ quá trình thực nghiệm.

Các nhóm chức năng chính:

**Cấu hình môi trường:**

- set_seed(): Thiết lập random seed để tăng khả năng tái lập kết quả.
- get_device(): Xác định thiết bị chạy mô hình như CUDA, MPS hoặc CPU.
- get_env_info(): Thu thập thông tin phiên bản thư viện và thiết bị.

**Lưu và tải mô hình:**

- save_model(): Lưu trọng số mô hình.
- load_model(): Tải trọng số đã lưu vào mô hình.
- save_checkpoint(): Lưu trạng thái để tiếp tục quá trình huấn luyện.
- load_checkpoint(): Khôi phục trạng thái huấn luyện.

**Quản lý kết quả:**

- save_config(): Lưu cấu hình thực nghiệm vào config.json.
- save_history(): Lưu lịch sử huấn luyện vào history.json.
- save_test_metrics(): Lưu các chỉ số đánh giá trên tập test.
- load_experiment(): Đọc lại lịch sử và cấu hình thực nghiệm.

**Theo dõi quá trình thực nghiệm:**

- create_writer(): Tạo TensorBoard SummaryWriter để ghi lại các thông số huấn luyện.
- make_session_id(): Tạo mã định danh cho một phiên thực nghiệm.
- make_eval_id(): Tạo mã định danh cho một lần đánh giá.
- file_sha256(): Tính mã băm SHA-256 của file.

## 9. Sơ đồ tổng quan hoạt động

Quy trình hoạt động của các thành phần trong dự án:

```text
        DATASET CIFAR-10
                |
                v
       data_setup.py
     Chuẩn bị và chia dữ liệu
                |
                v
       model_builder.py
      Xây dựng mô hình CNN
                |
                v
           engine.py
       Huấn luyện mô hình
       Forward Pass
       Tính toán Loss
       Backpropagation
                |
                v
         evaluate.py
       Đánh giá kết quả
       Accuracy, Precision
       Recall, F1-score
                |
                v
           utils.py
       Lưu mô hình, lịch sử
       và cấu hình thực nghiệm
```

Các file không hoạt động độc lập hoàn toàn mà phối hợp với nhau để tạo thành quy trình xử lý dữ liệu, huấn luyện, đánh giá và quản lý kết quả của mô hình Deep Learning.

## 10. Kết luận

Thông qua việc tìm hiểu cấu trúc `going_modular`, có thể thấy dự án được tổ chức theo hướng chia nhỏ chức năng thành các module riêng biệt.

Cách tổ chức này giúp mã nguồn dễ quản lý, dễ bảo trì và thuận tiện khi phát triển hoặc mở rộng mô hình Deep Learning.

**Người đóng góp:** Huỳnh Quang Thịnh
