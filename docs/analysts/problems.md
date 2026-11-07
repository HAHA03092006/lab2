Các vấn đề cụ thể
data_setup luôn lấy preprocessing từ preset ImageNet.
get_pretrained_transform() chọn VGG16_Weights.IMAGENET1K_V1 hoặc ResNet18_Weights.IMAGENET1K_V1 rồi gọi weights.transforms(). Vì vậy, dù model dùng weights=None, DataLoader vẫn có thể resize và chuẩn hóa ảnh theo preset ImageNet. Đây không nhất thiết là sai, nhưng phải là lựa chọn có chủ đích.

Notebook hiện cố định transform sang VGG16.
Cell DataLoader truyền transform_model="vgg16". Vì vậy, khi chọn run ResNet-18, model có thể nhận preprocessing dành cho VGG16. Đây là lệch cấu hình cần sửa khi nối DataLoader với RUN_ID.

model_builder.get_transforms() có hành vi khác data_setup.
Hàm get_transforms() mặc định lấy weights từ ModelConfig. Nếu cfg.weights=None và không truyền weights riêng, hàm báo lỗi thay vì tự chọn preset. Trong khi đó, data_setup vẫn dùng preset ImageNet theo kiến trúc. Cùng một model config vì thế có thể dẫn tới cách xử lý transform khác nhau tùy API được gọi.

Chế độ reuse vẫn cần transform phù hợp.
Khi reuse, model có thể được dựng với weights=None để tránh tải weights ImageNet mới, rồi nạp checkpoint đã lưu. Nhưng ảnh đầu vào vẫn phải được tiền xử lý nhất quán với cách model được huấn luyện. Việc không nạp weights ImageNet lúc dựng model không có nghĩa là bỏ transform đã dùng khi train.

Cấu hình hiện tại chưa lưu rõ transform được dùng.
data_setup nhận transform_model, nhưng không nhận trực tiếp một transform_weights từ run spec. Do đó, chỉ nhìn ModelConfig.weights chưa đủ để biết chính xác preprocessing đã áp dụng; nhất là với reuse hoặc khởi tạo từ đầu.



Có một điểm nhỏ nên ghi nhận để xử lý khi hoàn thiện cấu hình run: notebook không truyền rõ freeze_bn_stats vào ModelConfig và cũng chưa ghi trường này trong RUN_CONFIG, nên hiện đang ngầm dùng mặc định True. Luồng hiện tại vẫn hoạt động, nhưng lưu giá trị rõ ràng sẽ giúp cấu hình run dễ đọc và tái lập hơn.