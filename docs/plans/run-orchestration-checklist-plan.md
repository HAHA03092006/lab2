# Kế hoạch: Notebook điều phối một RUN_ID mỗi phiên

## Mục tiêu và giới hạn

Notebook điều phối chọn `RUN_ID`, chế độ `RUN_MODE` (`train`/`reuse`), `RESUME` và quản lý artifact (config/history/checkpoint/metrics theo `RUN_ID`). `going_modular/engine.py` phụ trách vòng lặp train/validation, chọn best model và cơ chế checkpoint/resume.

Ưu tiên dùng API engine hiện có. Trong giai đoạn triển khai, nếu đối chiếu với hợp đồng tại `docs/define&designt/03_engine.md` thấy thiếu hành vi huấn luyện cốt lõi mà notebook không thể xử lý đúng ở tầng gọi API, cho phép sửa engine tối thiểu, đồng thời cập nhật đặc tả engine (`docs/define&designt/03_engine.md`) và test engine tương ứng. Không tạo tài liệu orchestration mới, không mở rộng engine thành bộ điều phối nhiều run, không chạy huấn luyện thật hoặc nghiệm thu Colab/Drive trong checklist này.

## Checklist triển khai

### A. Chốt hợp đồng run và kiểm tra đầu vào

- [ ] `RUN_ID` resolve đúng một mục trong `RUN_SPECS`; kiểm tra đầy đủ sáu spec và dependency của fine-tune.
- [ ] Quy định rõ `RUN_MODE=train|reuse` và `RESUME`: resume chỉ áp dụng cho train; báo lỗi sớm nếu mode, spec hoặc cấu hình không hợp lệ.
- [ ] Tạo cấu hình hiệu lực cho run đã chọn (model/weights/freeze, epochs, batch size, learning rate/optimizer, param groups, seed, dependency); ghi cấu hình vào đúng thư mục run.
- [ ] Kiểm tra trạng thái artifact trước khi chạy: train mới không ghi đè run đã có; resume cần checkpoint tương thích; reuse cần đủ config/history/best checkpoint. Không tự chuyển reuse lỗi thành train.

### B. Thay luồng notebook cố định bằng luồng một run

- [ ] Chỉ dựng model cho run được chọn; sửa import/namespace các symbol model đang dùng và khởi tạo `DEVICE` trước các cell train/evaluate.
- [ ] Dùng transform phù hợp kiến trúc được chọn; truyền seed/generator và `download=False` tường minh cho DataLoader, giữ nguyên quy trình stage dữ liệu hiện có.
- [ ] Hợp nhất hoặc gate các cell VGG16, ResNet-18 baseline và fine-tune đang chạy cố định; chạy notebook không được vô tình train nhiều model.
- [ ] Truyền đúng optimizer, epochs, log directory, artifact directory, config kỳ vọng, `RESUME` và epoch offset vào các API engine hiện có; notebook không tự triển khai vòng lặp train/validation hay chọn best model.
- [ ] Với fine-tune, yêu cầu baseline checkpoint, nạp đúng checkpoint và thiết lập optimizer parameter groups theo spec; không thay đổi semantics engine.
- [ ] Đối chiếu cách notebook gọi `engine.train()` với hợp đồng train/resume/checkpoint trong `docs/define&designt/03_engine.md`; không chuyển logic chọn `RUN_ID`, mode hoặc artifact ownership vào engine.

### C. Reuse, đánh giá và so sánh

- [ ] Reuse nạp model/history/config từ artifact của run đã chọn mà không train lại.
- [ ] Chỉ đánh giá test khi run đã hoàn thành; lưu và nạp `test_metrics.json` theo RUN_ID, tránh tính lại nếu kết quả đã tồn tại và hợp lệ.
- [ ] Sửa các cell bảng/biểu đồ/dự đoán đang phụ thuộc mọi model trong cùng kernel: đọc artifacts của các run cần so sánh hoặc chỉ hiển thị run có kết quả. Thiếu artifact phải được báo rõ, không làm hỏng/đổi trạng thái run đang chọn.

### D. Đồng bộ tài liệu thiết kế hiện có

- [ ] `docs/plans/phase8.md` — làm checklist triển khai notebook một-run; đánh dấu rõ code/config trong phạm vi, còn chạy huấn luyện thật và cổng Colab/Drive ngoài phạm vi cho đến khi thực hiện.
- [ ] `docs/plans/implementation_plan.md` — đồng bộ tên phase, phạm vi và sơ đồ phụ thuộc; tách tích hợp one-run khỏi việc thực thi đủ sáu thí nghiệm.
- [ ] `docs/plans/phase7.md` — phân biệt phần thiết kế/việc đã hoàn tất với tích hợp notebook thực tế; không ghi tích hợp là xong nếu notebook còn luồng cố định.
- [ ] `docs/plans/clean.md` — sửa hoặc gỡ chỉ dẫn notebook stale (cell/API cũ); không mở rộng sang cleanup engine/API.
- [ ] Nếu triển khai cần đổi hành vi engine: cập nhật `docs/define&designt/03_engine.md` và test engine tương ứng; nếu API hiện tại đáp ứng thì không đổi engine.
- [ ] Không tạo tài liệu orchestration mới; giữ `phase0.md` nếu hợp đồng hiện có vẫn khớp.
- [ ] Đây là cập nhật tài liệu: không chạy test suite trong bước này.

## Tiêu chí nghiệm thu

- Mỗi giá trị RUN_ID chọn đúng một spec, model, đường dẫn và nhánh xử lý; không có cell nào tự chạy thêm run khác.
- Có thể kiểm tra bằng fixtures/artifacts giả: train mới bị chặn khi artifact tồn tại, resume/reuse nhận đúng paths, reuse không gọi train, fine-tune yêu cầu baseline.
- Notebook JSON hợp lệ và code cells biên dịch được; kiểm tra có mục tiêu cho resolve spec, artifact guards, config và metrics load/save.
- Engine được kiểm chứng với đúng hợp đồng cần dùng: nếu không sửa engine, xác nhận các API train/resume hiện tại đáp ứng; nếu sửa, chạy test engine tập trung và đồng bộ đặc tả.
- Test run/flow không tải dữ liệu, không tải pretrained weights và không huấn luyện thật. Full suite chỉ chạy nếu notebook-side thay đổi yêu cầu; không dùng nó thay cho kiểm thử các nhánh trên.
- Không mở rộng engine thành bộ điều phối sáu run; không tuyên bố đã xác minh môi trường Colab/Drive.
- Bước này chỉ cập nhật tài liệu: không chạy test suite.
