# Kế hoạch: Tách artifacts, outputs đánh giá và training logs

## Mục tiêu

Tách rõ ba loại dữ liệu:

```text
<LAB2_ROOT>/
├── artifacts/<RUN_ID>/                         # trạng thái và kết quả train để resume/reuse
├── outputs/YYYY-MM-DD/<RUN_ID>/<EVAL_ID>/       # kết quả đánh giá/ảnh theo ngày
└── logs/<RUN_ID>/<SESSION_ID>/                  # TensorBoard cho từng phiên train/resume
```

Trong Colab, `LAB2_ROOT=/content/drive/MyDrive/lab2`; chạy local thì `LAB2_ROOT=CODE_ROOT`.

Giữ `config.json`, `history.json`, `best_model.pth`, `last_checkpoint.pth` trong `artifacts/<RUN_ID>/`. `history.json` tiếp tục là hồ sơ train/validation cần cho đồ thị, phân tích run và reuse. Chuyển `test_metrics.json` cùng các hình/báo cáo đánh giá sang `outputs`; không lưu ảnh đánh giá trong `artifacts`.

Ngày thư mục dùng timezone `Asia/Ho_Chi_Minh`; `EVAL_ID` và `SESSION_ID` là timestamp duy nhất, có timezone, để các lần đánh giá/log trong cùng ngày không ghi đè nhau. Không đổi semantics engine hay chạy huấn luyện thật trong giai đoạn triển khai đường dẫn.

## Checklist triển khai

### A. Chốt contract đường dẫn và danh mục file

- [ ] Định nghĩa một root cho ba nhánh `ARTIFACT_ROOT`, `OUTPUT_ROOT`, `LOG_ROOT`, dùng nhất quán ở local và Colab.
- [ ] Giữ `artifacts/<RUN_ID>/`: `config.json`, `history.json`, `best_model.pth`, `last_checkpoint.pth`.
- [ ] Chuyển `test_metrics.json`, confusion matrix, per-class charts, ảnh dự đoán và các báo cáo đánh giá vào `outputs/YYYY-MM-DD/<RUN_ID>/<EVAL_ID>/`.
- [ ] Đặt TensorBoard events ở `logs/<RUN_ID>/<SESSION_ID>/`; một phiên train/resume có writer riêng, không append/ghi đè log phiên trước.
- [ ] Ghi metadata đánh giá gồm `RUN_ID`, timestamp/timezone, checkpoint identity, evaluation protocol/test-data identity và đường dẫn tương đối của các file output.

### B. Tách metrics khỏi training artifacts

- [ ] Cập nhật notebook để tạo output directory theo ngày và evaluation ID trước khi lưu metrics/ảnh.
- [ ] Chuyển caller của `utils.save_test_metrics()` / `load_test_metrics()` sang output directory; giữ tính tương thích của `artifacts_exist()` với đúng các file cần cho train/reuse, không yêu cầu metrics đánh giá.
- [ ] Thiết kế cache lookup dựa trên checkpoint đã đánh giá và protocol/dataset identity, không chỉ dựa vào `RUN_ID`. Ưu tiên SHA-256 của `best_model.pth` cùng fingerprint cấu hình đánh giá; không dùng cache nếu metadata thiếu hoặc không khớp.
- [ ] Kiểm tra cache trước inference. Khi cache hợp lệ, dùng record đã nạp; khi không có cache hợp lệ, chạy inference một lần, ghi metrics và metadata vào evaluation directory mới.
- [ ] Tái sử dụng predictions/result của inference hiện tại cho metrics và các biểu đồ cần chúng; tránh các cell gọi `predict_dataloader()` lặp lại trên cùng Test set.
- [ ] Không tạo output giả khi inference/lưu file thất bại; lỗi phải được báo và không đánh dấu evaluation thành công.

### C. Tổ chức training logging theo phiên

- [ ] Dùng `utils.create_writer()` hoặc một helper chung để tạo writer dưới `logs/<RUN_ID>/<SESSION_ID>/`, thay vì tạo `SummaryWriter` trực tiếp tại một thư mục cố định theo RUN_ID.
- [ ] Lưu `SESSION_ID`/thời điểm bắt đầu vào config hoặc metadata phiên để truy vết train/resume; không thay đổi thẻ scalar mà `engine.py` đang ghi.
- [ ] Đảm bảo writer luôn được đóng bằng `try/finally`, kể cả khi train phát sinh lỗi.
- [ ] Giữ nguyên checkpoint/resume và history path trong `artifacts/<RUN_ID>/`; không chuyển logging sang engine.

### D. Lưu hình ảnh và kết quả đánh giá

- [ ] Truyền `save_path` rõ ràng cho các plot cần bàn giao; lưu PNG ở evaluation directory theo ngày/run/session.
- [ ] Tiếp tục hiển thị hình trong notebook nếu cần, đồng thời lưu bản PNG; dùng độ phân giải hiện quy định.
- [ ] Khi dùng cache metrics, không giả định cache tự chứa ảnh. Hoặc giữ ảnh trong evaluation record được cache tham chiếu, hoặc tạo ảnh mới từ predictions đã lưu mà không inference lại.
- [ ] Hiển thị/in đường dẫn kết quả cuối để người dùng biết file được lưu ở đâu.

### E. Đồng bộ tài liệu, ignore và hướng dẫn

- [ ] `docs/define&designt/00_lab2_description.md`: sửa sơ đồ thư mục và bảng danh mục; phân biệt artifacts train, dated evaluation outputs và logs.
- [ ] `docs/define&designt/04_utils.md`: cập nhật `save_test_metrics`/`load_test_metrics`, `create_writer`, điều kiện artifacts tồn tại và quy tắc cache.
- [ ] `docs/define&designt/05_evaluation_results.md`: chuyển metrics/figures khỏi `artifacts/figures`, định nghĩa date/EVAL_ID, metadata và cache.
- [ ] README và các checklist phase liên quan: cập nhật đường dẫn Colab/Drive/local, quy trình tìm output và tình trạng xác minh.
- [ ] Rà soát `.gitignore`: checkpoints và logs không commit; quyết định rõ có commit evaluation JSON/PNG hay không và chỉ ignore thư mục generated phù hợp.

### F. Kiểm thử

- [ ] Unit test path builder với thời gian cố định: ngày, RUN_ID, EVAL_ID tạo đúng đường dẫn và hai evaluation cùng ngày không ghi đè.
- [ ] Kiểm tra `artifacts_exist()` chỉ dựa vào artifacts train; thiếu `test_metrics.json` không làm run train/reuse bị xem là thiếu checkpoint.
- [ ] Kiểm tra save/load metrics đi vào output directory và không tạo file khi metrics rỗng.
- [ ] Kiểm tra cache khớp checkpoint + protocol thì không gọi inference; cache sai/thiếu metadata thì chạy inference và ghi record mới.
- [ ] Kiểm tra notebook tái sử dụng một inference result cho metrics/plots; plot tạo PNG không rỗng tại output path.
- [ ] Kiểm tra writer sinh TensorBoard event trong session directory riêng và được đóng cả khi thành công/lỗi.
- [ ] Kiểm tra notebook JSON/cell syntax; chạy test tập trung rồi suite liên quan. Không tải CIFAR-10/weights và không train thật.

## File dự kiến

- `notebooks/pretrained_models.ipynb`
- `going_modular/utils.py` (chỉ nếu cần chỉnh API/path helper/cache helper)
- `going_modular/evaluate.py` (chỉ nếu cần chuyển orchestration dùng chung ra khỏi notebook; API plot đã nhận `save_path`)
- `tests/test_utils.py`
- `tests/test_evaluate.py`
- test notebook/path orchestration nếu có thể kiểm thử phần đó ngoài notebook
- `.gitignore`, `README.md`
- `docs/define&designt/00_lab2_description.md`
- `docs/define&designt/04_utils.md`
- `docs/define&designt/05_evaluation_results.md`
- checklist phase hiện hành có nhắc đến vị trí artifacts/outputs/logs

## Tiêu chí nghiệm thu

- Training/resume/reuse vẫn đọc và ghi đúng artifacts hiện hữu; history/checkpoint/config không bị chuyển hoặc mất.
- Output đánh giá mới nằm ngoài artifacts, có ngày/run/evaluation ID, không ghi đè output cũ.
- Metrics cache chỉ được dùng khi xác nhận cùng checkpoint và evaluation protocol; cache hit không chạy inference Test lại.
- Một evaluation không lặp inference chỉ để tạo nhiều biểu đồ; các hình cần bàn giao được lưu và notebook in đường dẫn.
- Mỗi phiên train/resume có TensorBoard event directory riêng; writer được đóng đúng cách.
- Tài liệu, `.gitignore`, local và Colab layout thống nhất. Không tuyên bố đã nghiệm thu Drive/Colab nếu chưa chạy kiểm tra thật.
