# Phase 4 — Checklist `utils.py` và tests

> **Nguồn chính:** Phase 4 trong `docs/plans/implementation_plan.md`  
> **Đặc tả module:** `docs/define&designt/04_utils.md`  
> **Đặc tả test:** `docs/define&designt/tests/04_test_utils.md`  
> **Điều kiện đầu vào:** Phase 3 đã đạt theo xác nhận của người dùng.  
> **Phạm vi dự kiến:** `going_modular/utils.py`, `tests/test_utils.py` và exports utility trong `going_modular/__init__.py`.

## 1. Xác nhận trước khi triển khai

- [x] Rà soát trạng thái Git và thay đổi hiện có trong các file thuộc Phase 4; không ghi đè thay đổi ngoài phạm vi.
- [x] Rà soát call sites hiện tại trong Notebook/package trước khi đổi chữ ký hoặc hành vi của utility legacy.
- [x] Dùng test spec chi tiết `04_test_utils.md` làm chuẩn cho mã test U1–U11.
- [x] Ghi nhận khác biệt đánh số U8–U10 giữa test spec và mục 2.8 của tài liệu thiết kế; giữ đúng mã theo test spec và vẫn kiểm tra config mismatch khi resume.
- [x] Không mở rộng Phase 4 sang tích hợp Notebook, engine hoặc chạy thí nghiệm.

## 2. Seed, device và thông tin môi trường

- [x] Hoàn thiện `set_seed` để seed Python, NumPy, PyTorch CPU và CUDA khi khả dụng.
- [x] Hỗ trợ cờ deterministic theo đặc tả; không cam kết tái lập tuyệt đối giữa phần cứng/phiên bản thư viện khác nhau.
- [x] Hoàn thiện `get_device(prefer=...)` trả về `torch.device`; hỗ trợ lựa chọn tự động và các backend được đặc tả.
- [x] Báo lỗi rõ ràng với preference không hợp lệ hoặc thiết bị được yêu cầu nhưng không khả dụng.
- [x] Bổ sung `get_env_info` theo schema phục vụ ghi nhận phiên bản/thông tin runtime.
- [x] Bổ sung `create_writer` tạo TensorBoard writer tại đúng thư mục experiment; bảo đảm writer có thể flush/close.

## 3. Lưu/nạp cấu hình, history và model

- [x] Tách `save_config` và `save_history`; validate schema trước khi ghi/nạp.
- [x] Ghi JSON UTF-8, giữ Unicode (`ensure_ascii=False`) và không tạo kết quả thành công giả khi lỗi I/O.
- [x] Thực hiện atomic write bằng file tạm cùng thư mục và atomic replace; khi ghi lỗi phải giữ nguyên file đích cũ và dọn file tạm.
- [x] Lưu đủ sáu history keys theo hợp đồng hệ thống: `train_loss`, `val_loss`, `train_acc`, `val_acc`, `lr`, `epoch_time`.
- [x] `save_model`/`load_model` xử lý state dict theo API mới, hỗ trợ `map_location` và nạp với `weights_only=True`.
- [x] Không fallback sang unsafe load khi `weights_only=True` thất bại; lỗi phải được báo rõ.
- [x] Bảo đảm lưu/nạp model giữ nguyên state và hành vi dự đoán.

## 4. Artifacts, checkpoint resume và test metrics

- [x] Bổ sung `artifacts_exist` kiểm tra bộ artifacts huấn luyện bắt buộc: `best_model.pth`, `history.json`, `config.json`; metrics không bắt buộc trước khi đánh giá.
- [x] Báo lỗi rõ ràng cho artifacts thiếu, JSON/schema không hợp lệ hoặc payload không an toàn; không dùng defaults im lặng.
- [x] Bổ sung `save_checkpoint`/`load_checkpoint` để lưu/nạp trạng thái cần thiết cho resume tại ranh giới epoch: model, optimizer, epoch, best score/epoch và history theo đặc tả.
- [x] Kiểm tra phase config khi resume; từ chối checkpoint có cấu hình không khớp, không tự khởi động lại từ đầu.
- [x] Bổ sung `save_test_metrics`/`load_test_metrics`; không ghi metrics rỗng hoặc giả trước khi đã đánh giá.

## 5. Tương thích API và package exports

- [x] Giữ các hàm legacy đang được Notebook sử dụng hoặc cung cấp tương thích có chủ đích; không xóa/đổi chữ ký trước khi rà soát call sites.
- [x] Đảm bảo các lời gọi hiện hữu như `set_seed(seed)`, `get_device()`, metadata/artifact helpers vẫn hoạt động trong giai đoạn chuyển tiếp.
- [x] Cập nhật các import và `__all__` utility trong `going_modular/__init__.py` cho API Phase 4 đã sẵn sàng.
- [x] Không sửa Notebook hoặc chuyển Notebook sang API mới trong Phase 4; tích hợp đó thuộc Phase 7.

## 6. Tạo `tests/test_utils.py`

- [x] Viết test theo các hành vi và mã của test spec chi tiết:
  - [x] **U1:** seed Python/NumPy/PyTorch tái lập.
  - [x] **U2:** chọn CPU và xử lý preference/thiết bị không hợp lệ hoặc không khả dụng.
  - [x] **U3:** round-trip config/history JSON có Unicode và schema hợp lệ.
  - [x] **U4:** round-trip state dict model.
  - [x] **U5:** atomic-write failure giữ nguyên file đích và dọn file tạm.
  - [x] **U6:** phát hiện artifacts bắt buộc đầy đủ/thiếu.
  - [x] **U7:** lỗi rõ ràng với file thiếu hoặc schema/payload sai.
  - [x] **U8:** từ chối payload `.pth` không an toàn; xác nhận không fallback sang unsafe load.
  - [x] **U9:** writer tạo event file và ghi/flush được.
  - [x] **U10:** resume checkpoint round-trip; thêm trường hợp config phase mismatch bị từ chối như subcase, không đổi mã test.
  - [x] **U11:** metrics round-trip và không sinh metrics placeholder trước đánh giá.
- [x] Dùng thư mục tạm và monkeypatch/mock để test lỗi ghi, device không khả dụng, không phụ thuộc GPU/mạng.
- [x] Xác nhận kiểm thử không để lại artifacts trong repository.

## 7. Kiểm thử và cổng nghiệm thu

- [x] Chạy bộ test Phase 4:
  - [x] `python -m pytest tests/test_utils.py -v`
- [x] Chạy smoke test package:
  - [x] `python -m pytest tests/test_smoke.py -v`
- [x] Chạy collection test:
  - [x] `python -m pytest --collect-only`
- [x] Xác nhận U1–U11 được thu thập và đạt, gồm atomic-write cleanup, unsafe-load rejection và config mismatch.
- [x] Xác nhận legacy import/call sites cần giữ vẫn tương thích.
- [x] Chạy `git diff --check` và rà soát diff, không bao gồm thay đổi Notebook/engine.

## 8. Ngoài phạm vi Phase 4

- Sửa vòng lặp huấn luyện/checkpoint selection trong `engine.py` (Phase 5).
- Tích hợp API mới vào Notebook hoặc triển khai RUN_MODE/resume end-to-end (Phase 7).
- Đồng bộ artifacts lên Drive/cloud, mã hóa/chữ ký artifacts hoặc bảo đảm tiếp tục RNG chính xác giữa runtime/phần cứng khác nhau.
- Huấn luyện hoặc đánh giá các runs CIFAR-10.
- Xóa API legacy khi vẫn còn call sites.

## 9. Cổng hoàn tất Phase 4

- [x] Các API utility thuộc Phase 4 đáp ứng hợp đồng thiết kế và xử lý lỗi rõ ràng.
- [x] U1–U11 đạt; trường hợp config mismatch được xác minh mà không làm lệch mã test spec.
- [x] Atomic write, kiểm tra artifacts và nạp an toàn được kiểm chứng bằng test.
- [x] Package exports và legacy call sites cần duy trì được kiểm tra.
- [x] Smoke test, collection và `git diff --check` đạt.
- [x] Không có thay đổi ngoài phạm vi Phase 4.

**Kết quả Phase 4:** ☐ Chưa đạt  ☒ Đạt  
**Ngày rà soát:** 2026-11-05  
**Ghi chú / quyết định còn mở:**  

---

Sau khi đạt cổng này, tiếp tục **Phase 5 — `engine.py` và `test_engine.py`** theo `docs/plans/implementation_plan.md`.
