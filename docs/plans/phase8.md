# Phase 8 — Chuẩn hóa cấu hình Notebook, GitHub và Google Drive

## Mục tiêu

Điều chỉnh và kiểm chứng notebook để sẵn sàng cho quy trình Phase 8 thực nghiệm trên Colab:

- Notebook và mã `going_modular/` được lấy từ phiên bản GitHub đã xác định.
- CIFAR-10 được quản lý trên Google Drive; có thể copy/giải nén vào Colab runtime để đọc nhanh.
- Checkpoint và kết quả cần lưu lâu dài được ghi về Drive, tách biệt theo từng run.
- Cấu hình có thể chọn đúng một trong sáu run và xác định rõ chế độ train/reuse/resume.

**Phạm vi Phase 8 này chỉ là cấu hình, chỉnh sửa và kiểm chứng luồng notebook/Drive. Không chạy huấn luyện sáu run.** Việc thực hiện thực nghiệm thật được để sang phase/task tiếp theo.

## Bố trí tài nguyên mục tiêu

```text
GitHub (branch/commit đã chốt)
└── lab2/
    ├── going_modular/
    ├── notebooks/pretrained_models.ipynb
    ├── requirements.txt
    └── ...

Colab runtime (tạm thời)
├── /content/lab2/                 # Clone từ GitHub; import code từ đây
├── /content/cifar10_data/         # Bản làm việc của CIFAR-10
└── /content/runs/                 # TensorBoard logs tạm hoặc đồng bộ theo chính sách

Google Drive (lưu bền vững)
└── MyDrive/Lab2/
    ├── data/
│   ├── cifar-10-python.tar.gz     # Hoặc thư mục cifar-10-batches-py/
    └── artifacts/
        ├── v0_vgg16/
        ├── v1_resnet18/
        ├── v1_resnet18_finetune/
        ├── v2_resnet18_lr1e-2/
        ├── v2_resnet18_lr1e-4/
        ├── v2_resnet18_bs64/
        └── figures/
```

Trong mỗi thư mục run, quy trình huấn luyện sau này dự kiến lưu `config.json`, `history.json`, `best_model.pth`, `last_checkpoint.pth`, và `test_metrics.json` sau khi run hoàn tất. Phase này chỉ kiểm tra cấu hình thư mục/quyền truy cập, không tạo checkpoint hay metrics thực nghiệm.

## 1. Xác lập baseline và nguồn cấu hình

- [ ] Ghi nhận trạng thái hiện tại của `notebooks/pretrained_models.ipynb`, `going_modular/`, `README.md` và Phase 7 trước khi sửa.
- [ ] Chốt repository/branch/commit GitHub làm nguồn code cho lần chuẩn bị Colab; cấu hình phải cho phép xem/ghi lại revision đã clone.
- [ ] Quy định notebook và thông số run nằm trong nội dung được quản lý phiên bản Git, không dùng bản notebook chỉnh sửa riêng trên Drive làm nguồn cấu hình không truy vết được.
- [ ] Không sửa hoặc xóa dataset/artifacts đang có trong repository/Drive.

## 2. Tách đường dẫn code, dữ liệu và kết quả

**File chính:** `notebooks/pretrained_models.ipynb` (cell setup/paths); cập nhật `README.md` nếu hướng dẫn hiện hành không còn khớp.

- [ ] Thay giả định hiện tại rằng `going_modular/` phải tồn tại trong `MyDrive/Lab2` bằng bootstrap lấy repository từ GitHub vào thư mục runtime, ví dụ `/content/lab2`.
- [ ] Import `going_modular` từ checkout trong runtime; kiểm tra repository, branch/commit và đường dẫn import để tránh vô tình nạp package khác từ Drive.
- [x] Đặt các biến có vai trò riêng và tên rõ ràng, ví dụ:
  - `CODE_ROOT`: checkout GitHub trong Colab runtime.
  - `DRIVE_ROOT`: `/content/drive/MyDrive/Lab2`.
  - `DRIVE_DATA_ROOT`: thư mục dữ liệu CIFAR-10 trên Drive.
  - `DATA_DIR`: thư mục dữ liệu mà DataLoader đọc.
  - `ARTIFACT_ROOT`: thư mục artifacts trên Drive.
  - `RUNS_ROOT`: nơi lưu TensorBoard logs theo chính sách đã chốt.
- [ ] Mount Drive trước khi truy cập dữ liệu/artifacts; báo lỗi rõ nếu Drive không mounted hoặc thư mục đích không đọc/ghi được.
- [x] Không dùng một biến `ROOT` chung cho source code, dataset, artifacts và logs.

## 3. Cấu hình CIFAR-10 trên Drive

**File liên quan:** notebook cell setup/dữ liệu; `going_modular/data_setup.py` chỉ thay đổi nếu kiểm thử chứng minh thật sự cần đổi hợp đồng.

- [ ] Chốt vị trí nguồn dữ liệu trên Drive, ví dụ `MyDrive/Lab2/data/`.
- [ ] Hỗ trợ cấu hình rõ ràng cho dataset đã upload:
  - archive `cifar-10-python.tar.gz`; hoặc
  - thư mục đã giải nén `cifar-10-batches-py/` với các batch và `batches.meta`.
- [ ] Kiểm tra cấu trúc/danh sách file cần thiết trước khi tạo DataLoader; thiếu hoặc sai đường dẫn phải báo lỗi hướng dẫn cách sửa, không tự chuyển sang tải dữ liệu từ mạng.
- [ ] Có thể copy archive/thư mục từ Drive vào `/content/cifar10_data` để giảm I/O trực tiếp từ Drive; không sửa hoặc xóa bản gốc trên Drive.
- [ ] Khi tạo DataLoader cho Phase này, truyền `data_dir` tường minh và `download=False` sau khi đã xác nhận dữ liệu hợp lệ.
- [ ] Giữ seed, stratified split và quy tắc train/validation/test hiện tại; việc cấu hình dữ liệu không được làm thay đổi nội dung split ngoài hợp đồng.
- [ ] Ghi rõ trong README quy trình upload dataset, cấu trúc thư mục mong đợi và sự khác nhau giữa bản nguồn Drive với bản runtime.

## 4. Cấu hình artifacts và logs trên Drive

**File chính:** notebook cells cấu hình run/lưu kết quả.

- [ ] Đặt `ARTIFACT_ROOT` trên Drive, tách khỏi `CODE_ROOT`, `DATA_DIR` và `RUNS_ROOT`.
- [ ] Mỗi run có thư mục riêng theo đúng `RUN_ID`; không để các run dùng chung hoặc ghi đè artifacts.
- [ ] Kiểm tra Drive có thể tạo/ghi/xóa một probe file riêng do Phase tạo, dọn đúng probe sau khi kiểm tra; phát lỗi nếu thất bại.
- [ ] Khai báo cấu trúc và ý nghĩa artifacts dự kiến:
  - `config.json`: cấu hình run và môi trường.
  - `history.json`: metric train/validation theo epoch.
  - `best_model.pth`: weights được chọn theo validation.
  - `last_checkpoint.pth`: trạng thái resume tại ranh giới epoch.
  - `test_metrics.json`: metrics/predictions sau đánh giá cuối.
- [ ] Xác định logs TensorBoard lưu trực tiếp trên Drive hay ghi ở runtime rồi đồng bộ sau run; không để hướng dẫn mơ hồ giữa hai chính sách.
- [ ] Không tạo artifacts huấn luyện giả trong Phase này; chỉ xác minh đường dẫn và quyền truy cập.

## 5. Khai báo và kiểm tra cấu hình sáu run

**File chính:** notebook cell cấu hình run; thông số chuẩn theo `docs/plans/implementation_plan.md` và đặc tả cấu hình tổng thể.

- [x] Khai báo danh sách allowlist `RUN_ID` gồm đúng sáu giá trị:
  - `v0_vgg16`
  - `v1_resnet18`
  - `v1_resnet18_finetune`
  - `v2_resnet18_lr1e-2`
  - `v2_resnet18_lr1e-4`
  - `v2_resnet18_bs64`
- [x] Mỗi ID ánh xạ tường minh sang kiến trúc, weights khi khởi tạo, freeze/unfreeze, epochs, batch size, LR theo nhóm tham số, seed và dependency nếu có.
- [x] Với `v1_resnet18_finetune`, khai báo nguồn khởi tạo là `best_model.pth` của `v1_resnet18`; optimizer là cấu hình mới cho phase fine-tune.
- [x] Cấu hình chỉ resolve và hiển thị run được chọn; không khởi chạy `engine.train`, inference Test hoặc vòng lặp lần lượt qua cả sáu run.
- [x] Validate ID không hợp lệ, tham số thiếu/sai, thư mục run và dependency trước khi người dùng chuyển sang bước thực nghiệm.
- [x] Mỗi lần chuẩn bị chỉ chọn một run; hiển thị rõ tham số, source checkpoint (nếu có), data path và artifact path trước khi cho phép tiếp tục.

## 6. Khai báo rõ `RUN_MODE` và `RESUME` cho phase huấn luyện sau

- [x] Hỗ trợ `RUN_MODE="train"` và `RUN_MODE="reuse"` với validate giá trị.
- [x] Khai báo `RESUME` chỉ có ý nghĩa khi `RUN_MODE="train"`:
  - `RESUME=False`: run mới; nếu artifacts đã có thì dừng, không tự xóa/ghi đè.
  - `RESUME=True`: yêu cầu `last_checkpoint.pth` tồn tại, đọc được và phase config khớp.
  - `RUN_MODE="reuse"`: không train; yêu cầu artifacts hoàn tất theo hợp đồng.
- [x] Với run fine-tune, kiểm tra dependency baseline tồn tại trước khi cho phép bước thực nghiệm tiếp theo.
- [x] Trong Phase 8, chỉ kiểm thử resolve/validate cấu hình; không thực thi nhánh train, resume, reuse inference hay đánh giá Test.
- [ ] Tách hoặc đặt cổng rõ cho cell thực nghiệm để chạy tuần tự phần setup/config không vô tình bắt đầu huấn luyện.

## 7. Sửa lỗi và đối chiếu API notebook

**Files:** `notebooks/pretrained_models.ipynb`, các module `going_modular/` chỉ khi thật sự cần.

- [x] Sửa code cell import đang có `try/except` rỗng và các lỗi cú pháp khác được phát hiện khi compile cell.
- [ ] Đối chiếu mọi lời gọi notebook với API hiện tại: `build_model`, `train`, `save/load config/history/checkpoint`, `run_inference` và các hàm evaluate.
- [ ] Thay các lời gọi legacy còn tồn tại trước khi notebook được dùng làm entry point cho phase huấn luyện.
- [ ] Giữ đánh giá Test sau khi run hoàn tất; không đưa Test vào quá trình chọn cấu hình/checkpoint.
- [ ] Không refactor ngoài phạm vi cấu hình notebook/Drive nếu không có lỗi trực tiếp ngăn cản Phase 8.

## 8. Kiểm chứng notebook ở chế độ setup-only

- [x] Parse notebook JSON và xác nhận schema notebook hợp lệ.
- [x] Compile toàn bộ code cells sau khi xử lý riêng IPython magic nếu có; không còn lỗi syntax.
- [x] Chạy smoke test package và suite nhanh hiện hành bằng interpreter dự án:
  ```powershell
  .venv\Scripts\python.exe -m pytest tests\test_smoke.py -q
  .venv\Scripts\python.exe -m pytest tests\ -m "not slow" -q
  ```
- [ ] Kiểm thử resolve cả sáu `RUN_ID` bằng cấu hình giả/đường dẫn tạm; xác nhận từng run có đúng thông số và không gọi train/evaluate.
- [ ] Trên Colab, xác nhận clone đúng revision GitHub, import từ checkout, mount Drive, nhận diện CIFAR-10 đã upload/stage, ghi được probe và resolve artifact path.
- [ ] Nếu kiểm tra DataLoader, chỉ lấy một batch từ dữ liệu đã có với `download=False`; không tải CIFAR-10 hoặc pretrained weights trong kiểm chứng này.
- [ ] Kiểm tra cấu hình/log output không lộ nhầm đường dẫn và không tạo/ghi đè checkpoint hoặc metrics thực nghiệm.
- [ ] Ghi lại revision code, lệnh kiểm thử và kết quả thực tế; không đánh dấu hoàn tất nếu chỉ mới sửa notebook mà chưa chạy các cổng trên.

## Ngoài phạm vi Phase 8

- Chạy huấn luyện bất kỳ run nào, kể cả smoke training trên tập nhỏ.
- Resume run thật, đánh giá trên tập Test, chọn/tinh chỉnh hyperparameter dựa trên metrics.
- Tạo `best_model.pth`, `last_checkpoint.pth`, `test_metrics.json`, bảng/biểu đồ kết quả thực nghiệm.
- Tuyên bố hoàn tất sáu runs hoặc nghiệm thu kết quả Phase 8 thực nghiệm.
- Thay đổi kiến trúc mô hình, schema dữ liệu/artifacts hoặc chiến lược thí nghiệm đã chốt.
- Xóa hoặc ghi đè dataset/artifacts đã tồn tại trên Drive.

## Cổng nghiệm thu Phase 8 — Notebook configuration

- [ ] Notebook lấy code từ GitHub revision đã xác định; không cần package copy thủ công trên Drive.
- [ ] CIFAR-10 được xác thực từ Drive và dùng từ thư mục được cấu hình; không có fallback download ngoài ý muốn.
- [ ] Artifacts/log paths nằm riêng trên Drive, partition theo từng `RUN_ID`; quyền ghi đã được kiểm chứng.
- [ ] Sáu cấu hình resolve đúng; mode/resume semantics được validate nhưng không kích hoạt huấn luyện trong phase này.
- [ ] Notebook JSON và Python cells hợp lệ; API calls khớp code; setup-only smoke checks pass.
- [ ] Không có run nào được huấn luyện/đánh giá và không có checkpoint/metrics thực nghiệm mới được tạo.

## Phase tiếp theo — Thực nghiệm sáu run

Sau khi Phase 8 cấu hình notebook được nghiệm thu, lập hoặc thực hiện phase riêng cho train/resume/reuse và đánh giá đủ sáu run theo thứ tự/phụ thuộc trong `docs/plans/implementation_plan.md`. Chỉ ở phase đó mới tạo checkpoint, history, test metrics, bảng so sánh và báo cáo kết quả.

> **Lưu ý đồng bộ kế hoạch:** `implementation_plan.md` hiện đang gọi Phase 8 là “Huấn luyện và đánh giá đầy đủ (6 runs)”. Khi áp dụng checklist này, cần đổi roadmap để Phase 8 là cấu hình notebook/Drive và chuyển nội dung huấn luyện sáu run sang phase kế tiếp; không được để hai tài liệu mang hai phạm vi Phase 8 khác nhau.
