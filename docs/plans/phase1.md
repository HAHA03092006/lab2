# Phase 1 — Checklist hạ tầng kiểm thử và cấu hình dự án

> **Nguồn chính:** [`implementation_plan.md`](./implementation_plan.md), Phase 1
> **Điều kiện đầu vào:** Phase 0 đạt theo [`phase0.md`](./phase0.md)
> **Mục tiêu:** Chuẩn bị môi trường và nền kiểm thử tối thiểu để bắt đầu các phase module.
> **Phạm vi:** Dependency manifest, ignore rules, test scaffold, smoke test và hướng dẫn dự án. Chưa refactor module hay chạy các thí nghiệm đầy đủ.

## 1. Xác nhận phạm vi và trạng thái ban đầu

- [x] Xác nhận Phase 0 đã đạt; không đưa các quyết định thiết kế còn mở vào cấu hình một cách ngầm định.
- [x] Kiểm tra trạng thái Git trước khi thay đổi; giữ nguyên các chỉnh sửa có sẵn ngoài phạm vi Phase 1.
- [x] Xác nhận các test specs nằm tại `docs/define&designt/tests/`; đây là tài liệu đặc tả, chưa phải test pytest có thể chạy.
- [x] Ghi nhận trạng thái ban đầu của `requirements.txt`, `.gitignore`, `README.md`, thư mục `tests/` và khả năng import package.

## 2. Cập nhật dependency manifest và môi trường

- [x] Bổ sung các dependency bắt buộc còn thiếu trong `requirements.txt`:
  - [x] `pytest`
  - [x] `numpy`
  - [x] `Pillow`
- [x] Giữ `torchinfo` trong manifest ở Phase 1; chỉ gỡ sau khi Phase 7 đã thay thế toàn bộ call sites và xác minh không còn sử dụng.
- [x] Giữ các dependency khác nếu còn call site trong notebook hoặc package; không xóa chỉ vì chúng không nằm trong danh sách tối thiểu của thiết kế.
- [x] Cài đặt/cập nhật dependency trong đúng môi trường Python của dự án theo quy trình đã chọn; không cài vào interpreter khác với môi trường dùng để chạy pytest.
- [x] Xác minh import/version các package cần cho test scaffold và Matplotlib backend.
- [x] Ghi lại lệnh cài đặt và các điều kiện đặc biệt nếu PyTorch/CUDA cần được cài theo hướng dẫn tương ứng với runtime.

## 3. Rà soát `.gitignore` và chính sách lưu artifacts

- [x] Đảm bảo `data/`, `runs/`, cache Python, `.pytest_cache/`, môi trường ảo và các checkpoint (`*.pth`, `*.pt`, `*.ckpt`) được loại khỏi Git.
- [x] Đối chiếu quy tắc `artifacts/` với `00_lab2_description.md`: không ignore toàn bộ artifacts nếu JSON kết quả và figures cần được commit.
- [x] Cấu hình ignore để checkpoint vẫn bị loại, trong khi các file được phép lưu như `config.json`, `history.json`, `test_metrics.json` và `artifacts/figures/` không bị loại nhầm.
- [x] Kiểm tra bằng `git check-ignore` các trường hợp đại diện: dataset, TensorBoard event, checkpoint, JSON metrics và figure.
- [x] Không xóa hoặc sửa artifacts hiện có để làm sạch trạng thái Git; chỉ chỉnh quy tắc ignore và báo cáo trạng thái tracked/untracked nếu phát hiện dữ liệu cũ.

## 4. Tạo test scaffold tối thiểu

- [x] Tạo `tests/` tại repository root.
- [x] Tạo `tests/conftest.py`:
  - [x] Đăng ký marker `slow` cho test cần dữ liệu thật, mạng hoặc pretrained weights.
  - [x] Đặt Matplotlib backend không tương tác (`Agg`) trước khi test import plotting modules.
  - [x] Đảm bảo package được import ổn định khi pytest chạy từ repository root; tránh thêm sửa `sys.path` không cần thiết nếu cấu hình package đã đủ.
- [x] Tạo `tests/test_smoke.py` với kiểm tra nhanh, không cần mạng:
  - [x] Import `going_modular` và các submodule hiện được package cung cấp.
  - [x] Kiểm tra dependency/import tối thiểu và một hành vi đơn giản, ổn định của package.
  - [x] Không buộc smoke test xác nhận các API mục tiêu chưa được triển khai ở Phase 2–6.
- [x] Không tạo các file test module rỗng chỉ để pytest báo “collected”; test module được thêm cùng phase triển khai tương ứng theo đặc tả `docs/define&designt/tests/`.
- [x] Đảm bảo test scaffold không tải CIFAR-10, không tải ImageNet weights và không cần GPU.

## 5. Rà soát README và hướng dẫn dự án

- [x] Đồng bộ cấu trúc thư mục trong README với tên/đường dẫn hiện tại, đặc biệt:
  - [x] Notebook chính là `notebooks/pretrained_models.ipynb`.
  - [x] Kế hoạch nằm trong `docs/plans/`.
  - [x] Test specs nằm trong `docs/define&designt/tests/`.
- [x] Cập nhật hướng dẫn cài dependency và chạy smoke test theo môi trường đã chọn.
- [x] Rà soát các mô tả README có thể gây hiểu nhầm so với thiết kế đã chốt, như DenseNet hoặc danh sách runs cũ, chế độ `train/reuse`, nơi lưu artifacts và chính sách không ghi đè.
- [x] Không mô tả tính năng chưa triển khai (ví dụ `RUN_ID`, resume, metrics reuse) như thể đang hoạt động; ghi rõ là mục tiêu của các phase sau nếu README nhắc tới chúng.
- [x] Giữ README trong phạm vi Phase 1: hướng dẫn cấu trúc/môi trường, không biến bước này thành viết báo cáo kết quả thực nghiệm.

## 6. Kiểm tra và bằng chứng nghiệm thu

- [x] Chạy smoke test bằng interpreter của môi trường dự án:
  - [x] `python -m pytest tests/test_smoke.py -v`
- [x] Xác nhận pytest thu thập test mà không lỗi import:
  - [x] `python -m pytest --collect-only`
- [x] Xác nhận smoke test không cần network/GPU và chạy ổn định từ repository root.
- [x] Xác nhận các đường dẫn dependency, README và test specs được cập nhật trỏ đến file/thư mục đang tồn tại.
- [x] Kiểm tra `git status --short` và `git check-ignore` để chắc chắn data, logs, checkpoints không bị thêm nhầm; các JSON/figures dự kiến commit không bị ignore.
- [x] Chạy `git diff --check` trên các thay đổi của Phase 1.

## 7. Ngoài phạm vi Phase 1

- Refactor `model_builder.py`, `data_setup.py`, `utils.py`, `engine.py` hoặc `evaluate.py`.
- Triển khai đầy đủ các test D/T/E/U/V; các test này thuộc phase module tương ứng.
- Thay thế API legacy hoặc thay đổi behavior của notebook.
- Chạy CIFAR-10 thật, tải pretrained weights hoặc huấn luyện bất kỳ run nào.
- Xóa artifacts cũ hay checkpoint cũ.

## 8. Cổng hoàn tất Phase 1

- [x] Dependency manifest đã phản ánh yêu cầu kiểm thử/hệ thống; môi trường dự án cài được dependency cần thiết.
- [x] `.gitignore` khớp chính sách artifacts: không commit dữ liệu/logs/checkpoints, không vô tình loại JSON/figures cần lưu.
- [x] `tests/conftest.py` và smoke test tồn tại, chạy thành công không cần mạng/GPU.
- [x] `pytest --collect-only` thành công; không yêu cầu toàn bộ test module đã có ở giai đoạn này.
- [x] README và đường dẫn tài liệu không hướng dẫn theo notebook/cấu trúc cũ hoặc tuyên bố tính năng chưa triển khai là đã hoạt động.
- [x] Thay đổi không xóa/ghi đè dữ liệu artifacts hiện có; `git diff --check` đạt.

**Kết quả Phase 1:** ☐ Chưa đạt  ☒ Đạt
**Ngày rà soát:** 2026-11-05
**Ghi chú / quyết định còn mở:** Không còn mục nào chặn Phase 2. Các module sẽ được triển khai theo thứ tự trong `implementation_plan.md`, bắt đầu từ Phase 2 (`model_builder.py`).

---

Sau khi đạt cổng này, bắt đầu **Phase 2 — `model_builder.py` và test của module** theo [`implementation_plan.md`](./implementation_plan.md).
