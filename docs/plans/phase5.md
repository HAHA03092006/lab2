# Phase 5 — Danh sách kiểm tra thực thi được khuyên dùng (kế hoạch phiên làm việc)

> Điểm đến cuối cùng dự kiến: `docs/plans/phase5.md`. Kế hoạch phiên làm việc này không sửa đổi kho lưu trữ.

## Phạm vi và điều kiện tiên quyết của Phase 5

- [x] Xác nhận cổng Phase 2 và Phase 4 đã hoàn tất trước khi triển khai; ghi chép bất kỳ blocker nào thay vì tránh vòng quanh phụ thuộc chưa ổn định.
- [x] Xác nhận `train_step`/`evaluate_step` sử dụng tích lũy batch-loss có trọng số theo mẫu, và trình ghi checkpoint/history của Phase 4 cung cấp ranh giới ghi nguyên tử đáng tin cậy.
- [x] Giới hạn Phase 5 ở `going_modular/engine.py`, các test/exports của nó, và chỉ các call sites hoặc docs cần thiết để giải quyết tương thích API.

## 1. Tối ưu đối public API trước khi thay đổi nó

- [x] Kiểm kê các chữ ký hiện tại và tất cả các điểm gọi, bao gồm `going_modular/__init__.py` và notebook.
- [x] Giải quyết thứ tự tham số `optimizer` trước `loss_fn` hiện có so với ví dụ thiết kế/test, và đối số `checkpoint_path`/`frozen_backbone` của notebook so với `save_path`/`train_mode_fn` trong thiết kế.
- [x] Chọn một hợp đồng chính thống; giữ người gọi hiện tại khi có thể hoặc di chuyển chúng rõ ràng. Thêm độ bao quát cho quyết định tương thích để không hoãn bất kỳ sai khớp nào đến quá trình tích hợp notebook.
- [x] Tài liệu ghi nhận việc resume là tiếp tục ở ranh giới epoch với khớp phase config; `start_epoch` là offset toàn cục cho log, không phải là thay thế cho việc khôi phục trạng thái checkpoint.

## 2. Triển khai và xác minh các hàm bước

- [x] `train_step`: chế độ train (hoặc `train_mode_fn` được tiêm), cập nhật chỉ tham số có thể train, và loss/accuracy được trung bình theo số lượng mẫu trên các batch không đều.
- [x] `evaluate_step`: hành vi không gradient/đánh giá, loss có trọng số theo mẫu và accuracy chính xác; để lại mô hình trong trạng thái đánh giá đã tài liệu hóa.
- [x] Giữ `train_mode_fn` tùy chọn; khi được cung cấp, gọi nó để khiển các chế độ như BatchNorm đóng băng thay vì ghi đè nó bằng `model.train()`.

## 3. Triển khai vòng lặp epoch và hợp đồng lưu trữ

- [x] Xác thực `epochs`, `start_epoch`, `monitor` và đầu vào resume; từ chối giá trị không hợp lệ bằng `ValueError` và từ chối sai khớp phase config mà không khởi động lại im lặng.
- [x] Ghi tất cả sáu chuỗi history: `train_loss`, `val_loss`, `train_acc`, `val_acc`, `lr`, `epoch_time`; chèn đúng một mục cho mỗi epoch đã hoàn tất. Ghi LR theo từng nhóm optimizer và thời gian cả epoch train-plus-validation.
- [x] Hỗ trợ chọn checkpoint tốt nhất theo `val_acc` (tối đa hóa) và `val_loss` (tối thiểu hóa); xác minh việc xử lý trường hợp bằng nhau/epoch đầu tiên tuân theo một chính sách xác định đã tài liệu hóa.
- [x] Triển khai `restore_best`: khi được bật, khôi phục model tốt nhất đã chọn cho đánh giá tiếp theo; khi tắt, để lại trọng số epoch cuối cùng trong bộ nhớ.
- [x] Ở mỗi epoch đã hoàn tất, lưu `last_checkpoint.pth` và history thông qua ranh giới ghi nguyên tử đã xác minh. Đảm bảo checkpoint cuối đại diện cho epoch mới nhất đã hoàn tất ngay cả khi `restore_best` sau đó thay đổi trọng số trong bộ nhớ.
- [x] Resume chỉ từ last checkpoint hợp lệ với khớp config; khôi phục model/optimizer/history và tiếp tục từ epoch tiếp theo. Giữ nguyên chính sách hiện tại rằng trạng thái RNG không được khôi phục trừ khi đặc tả thay đổi.

## 4. TensorBoard và số thứ tự epoch

- [x] Phát các thẻ chuẩn: `Loss/train`, `Loss/val`, `Accuracy/train`, `Accuracy/val`, và `LR/group_<index>` cho mỗi nhóm optimizer.
- [x] Sử dụng bước epoch nhất quán bắt đầu từ 1 cộng offset `start_epoch` (trường hợp `epochs=2, start_epoch=5` ghi tại bước 6 và 7).
- [x] Xem `writer=None` là một no-op được hỗ trợ; việc đóng/flush quyền sở hữu vẫn thuộc về người gọi trừ khi API được thiết lập quy định khác.

## 5. Test: giữ E1–E12 và mở rộng độ bao quát

- [x] Sử dụng định nghĩa chi tiết E1–E12 trong `docs/define&designt/tests/03_test_engine.md`; khi ghi chú thiết kế engine cũ chỉ liệt kê E1–E10, hãy xem chi tiết E1–E12 là chính thống và không bỏ E11/E12.
- [x] Triển khai E1–E12 với các fixture/mock CPU nhỏ và `tmp_path`: loss có trọng số theo batch không đều, trọng số frozen/trainable, chế độ đánh giá, chế độ train được tiêm/BatchNorm, history sáu khóa, lưu/khôi phục tốt nhất, xác thực đối số, thẻ TensorBoard chính xác/offset, và `writer=None`.
- [x] Thêm E13 — vòng đời resume: config khớp khôi phục epoch mới nhất đã hoàn tất và nối history từ epoch tiếp theo; config không khớp báo lỗi rõ ràng và không ghi đè artifacts.
- [x] Thêm E14 — ngắt gián đoạn nguyên tử: tiêm lỗi ghi cho checkpoint và history riêng lẻ; mỗi artifact đã cam kết trước đó vẫn đọc được/không đổi và không có artifact bất kỳ nào được chấp nhận.
- [x] Thêm E15 — `monitor="val_loss"`: các giá trị val_loss được viết kịch bản chọn epoch có loss nhỏ nhất và `restore_best` khôi phục trọng số của epoch đó.
- [x] Thêm E16 — thời gian/history: khẳng định `epoch_time` được ghi một lần mỗi epoch và bao gồm cả train plus validation; khẳng định mỗi mục lịch sử LR phản ánh tất cả các nhóm optimizer.
- [x] Giữ các test engine xác định và nhanh; tránh tải pretrained, tập dữ liệu thực và các khẳng định cụ thể phần cứng.

## 6. Exports, tài liệu và kiểm tra tương thích

- [x] Cập nhật chỉ các exports engine trong `going_modular/__init__.py` theo yêu cầu của API đã thống nhất.
- [x] Cập nhật `docs/define&designt/tests/03_test_engine.md` để chỉ định E13–E16 và làm rõ rằng chi tiết E1–E12 thế thế danh sách cũ E1–E10.
- [x] Chạy lại kiểm tra tồn tại điểm gọi; cập nhật người gọi chỉ khi được yêu cầu bởi hợp đồng đã chọn và đảm bảo các đối số notebook vẫn ánh xạ đúng.

## Lệnh kiểm thử

Chạy từ thư mục gốc của kho lưu trữ theo trình tự này:

```powershell
python -m pytest tests/test_engine.py -v
python -m pytest tests/test_utils.py tests/test_engine.py -v
python -m pytest -q
```

- [x] Lệnh đầu tiên vượt qua tất cả E1–E16.
- [x] Lệnh thứ hai xác nhận ranh giới persist Phase 4 vẫn tương thích với việc sử dụng checkpoint/history của engine.
- [x] Toàn bộ bộ test vượt qua mà không có suy giảm; điều tra các thất bại thay vì yếu giả các khẳng định.

## Cổng hoàn tất Phase 5

- [x] Không đánh dấu Phase 5 là hoàn tất trước khi các điều kiện tiên quyết Phase 2/4 đã thỏa mãn, tất cả E1–E16 đạt, toàn bộ bộ test đạt, checkpoint tốt nhất và checkpoint mới nhất có ý nghĩa riêng biệt đã được xác minh, resume khớp config hoạt động, mismatch và ghi bị gián đoạn thất bại an toàn, và sự tương thích API/điểm gọi đã được giải quyết.
- [x] Ghi lại API/chữ ký và kết quả test thu được vào `docs/plans/phase5.md` sau này; bỏ tích hợp end-to-end notebook cho giai đoạn được kế hoạch khác trừ khi sự tương thích yêu cầu cập nhật tối thiểu ở đây.
