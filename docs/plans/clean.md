# Kế hoạch dọn dẹp sau Phase 7

## Mục tiêu

Loại bỏ các API tương thích cũ sau khi đã chuyển các caller còn hoạt động sang API chuẩn; đồng bộ hợp đồng package và tài liệu; giảm chi phí khởi tạo model trong validation/test nhưng vẫn giữ coverage cho VGG16 và ResNet-18; kiểm chứng đúng hành vi lưu checkpoint khi lỗi ghi. Không chạy thực nghiệm CIFAR-10, không tải pretrained weights và không xóa artifacts hiện có.

## Nguyên tắc thực hiện

- Thực hiện theo thứ tự các bước dưới đây; không xóa API trước khi chuyển và kiểm tra callers.
- Dùng `docs/define&designt/00_lab2_description.md` làm chuẩn cho 14 flat exports.
- Giữ `verbose` nếu đây là tùy chọn output được dùng; không loại bỏ chỉ vì nó nằm trong nhóm kwargs tương thích.
- Chỉ cập nhật tài liệu quy định hành vi hiện tại. Giữ tài liệu phase cũ như lịch sử, trừ khi nội dung gây hiểu nhầm về trạng thái hiện hành.
- Không gộp vào đợt này các cải tiến test dữ liệu hoặc reliability không cần thiết cho cleanup.
- Attachment notebook trong hội thoại là snapshot chỉ đọc; thay đổi notebook phải thực hiện trên `notebooks/pretrained_models.ipynb` trong repository.

## Bước 0 — Chốt baseline và phạm vi

**File liên quan:** toàn bộ `tests/`, `.venv\Scripts\python.exe`.

1. Chạy suite nhanh trước khi sửa:
   ```powershell
   .venv\Scripts\python.exe -m pytest tests\ -m "not slow" -q
   ```
2. Ghi nhận kết quả và thời lượng mới làm baseline (kết quả trước đó là 122 passed, 5 deselected, 72.70 giây).
3. Nếu baseline thất bại, tách lỗi có sẵn khỏi thay đổi cleanup trước khi tiếp tục.

**Cổng hoàn tất:** Có baseline mới từ đúng Python môi trường dự án; không coi kết quả lần chạy cũ là xác nhận cho trạng thái hiện tại.

## Bước 1 — Lập ma trận caller và chốt hợp đồng

**Đọc/đối chiếu:** `going_modular/`, `tests/`, executable code cells và markdown của `notebooks/pretrained_models.ipynb`, `docs/define&designt/00_lab2_description.md`, `docs/define&designt/02_model.md`, `03_engine.md`, `04_utils.md`, `docs/plans/implementation_plan.md`.

1. Liệt kê mỗi API/alias dự kiến bỏ cùng toàn bộ caller trong source, tests, notebook và tài liệu:
   - Model builder: `CIFAR10ModelV0`, `CIFAR10ModelV1`, `MODEL_NAMES`, dạng `build_model("v0"|"v1", ...)`, các counter alias cũ nếu còn.
   - Utils: `accuracy_fn`, `print_train_time`, `save_training_metadata`, `load_training_artifacts`; xác định các helper nào khác thực sự legacy.
   - Engine: `eval_model`, `checkpoint_path`, `frozen_backbone`.
2. Chốt `save_model` theo hợp đồng đặc tả `save_model(model, path)`. Cập nhật callers/tests và tài liệu cùng một thay đổi; không giữ song song hai signature nếu mục tiêu là gỡ API cũ.
3. Xác minh phạm vi export: hợp đồng 14 symbol đã liệt kê trong mục 2.8.6 của tài liệu tổng thể là tập chính xác, không phải danh sách tối thiểu.
4. Chốt hợp đồng atomicity từ đặc tả đang có: JSON writer đã có temp + replace; `save_checkpoint` và best-model write hiện lưu trực tiếp. Xác định rõ checkpoint nào bắt buộc atomic trước khi viết implementation/test. Không tuyên bố atomicity nếu chưa thực hiện.

**Cổng hoàn tất:** Mỗi API bị xóa có API thay thế và không còn caller bắt buộc; signature, exports và mức bảo đảm lưu file đã thống nhất với tài liệu chuẩn.

## Bước 2 — Chuyển notebook sang API chuẩn

**File:** `notebooks/pretrained_models.ipynb`.

1. Sửa flow fine-tuning, tập trung cell 21, 28, 29:
   - Thay `checkpoint_path` bằng `save_path`.
   - Thay phụ thuộc `frozen_backbone` bằng `train_mode_fn` phù hợp hoặc callback canonical hiện được engine hỗ trợ.
   - Thay `save_training_metadata` bằng các API lưu riêng `save_history` và `save_config`; giữ nguyên schema, thư mục artifacts và metadata thời lượng cần thiết.
   - Thay phần mở khóa thủ công `layer4`/head bằng `ModelConfig.unfreeze_last_n` khi đúng với kiến trúc và ý nghĩa cấu hình đã chốt; không làm thay đổi phạm vi tham số được fine-tune.
   - Sửa markdown còn hướng dẫn `frozen_backbone`.
2. Kiểm tra kỹ biểu thức `load_model` của cell reuse đã được ghi nhận là có thể sai cú pháp/đối số; chỉ sửa nếu xác nhận lỗi trên notebook hiện tại.
3. Giữ nguyên RUN_ID, train/resume/reuse, seed, optimizer, artifact paths và logic đánh giá; không chạy training hoặc tải dữ liệu.

**Kiểm chứng:** Notebook parse được bằng `json`; compile từng code cell (bỏ qua magic/IPython syntax nếu có); kiểm tra source code cells và hướng dẫn hiện hành không còn lời gọi đến API sẽ xóa.

**Cổng hoàn tất:** Notebook sử dụng API chuẩn và giữ nguyên hành vi lưu/nạp artifacts cùng cấu hình fine-tuning.

## Bước 3 — Gỡ API cũ và siết public exports

**Files:** `going_modular/model_builder.py`, `utils.py`, `engine.py`, `__init__.py`; tests và docs bị ảnh hưởng.

1. Sau khi Bước 2 qua cổng, xóa wrappers/aliases và nhánh legacy đã được xác nhận không còn caller.
2. Giữ các hàm canonical và hành vi không thuộc cleanup; giữ `verbose` nếu vẫn là tùy chọn hợp lệ.
3. Chuẩn hóa `save_model(model, path)` theo đặc tả; cập nhật test round-trip và mọi caller.
4. Đưa `going_modular.__all__` và imports về đúng 14 exports của hợp đồng.
5. Sửa smoke test để so sánh `set(going_modular.__all__)` với chính xác tập 14 tên; kiểm tra từng tên import được. Bỏ kỳ vọng flat imports cũ không thuộc hợp đồng, nhưng không cấm import trực tiếp từ submodule.
6. Cập nhật test engine/utils kiểm tra aliases đã bị xóa; cập nhật tài liệu hiện hành về API.

**Kiểm chứng:** Chạy smoke test, tìm kiếm toàn workspace các tên legacy đã gỡ, xác nhận chỉ còn nhắc đến trong tài liệu lịch sử nếu phù hợp.

**Cổng hoàn tất:** Không còn caller thực thi; `__all__` chính xác; tests và tài liệu dùng signature/API chuẩn.

## Bước 4 — Tối ưu validation và test model có chọn lọc

**Files:** `going_modular/model_builder.py`, `tests/test_model_builder.py`.

1. Thay `_validate` hiện dựng `_dummy_model` chỉ để đếm block bằng metadata số block không cấp phát model trong registry (ResNet-18: 4; VGG16: 5), hoặc giải pháp metadata tương đương.
2. Dùng cùng metadata trong validate `unfreeze_last_n`; tránh hai nguồn dữ liệu số block có thể lệch nhau.
3. Thêm test đảm bảo validate cấu hình hợp lệ/lỗi không gọi `_load_torchvision`/không dựng model; vẫn kiểm tra ngưỡng unfreeze đúng cho hai kiến trúc.
4. Giảm số lần dựng VGG: giữ coverage đại diện cho output/head, VGG-specific unfreeze blocks và parameter count; chuyển logic dùng chung sang ResNet nếu không làm mất kiểm tra hành vi riêng của VGG.
5. Không dùng chung model mutable theo module scope; mỗi test cần model riêng nếu thay đổi mode, gradients hoặc parameters. Không loại bỏ VGG test chỉ dựa trên thời gian chạy.

**Cổng hoàn tất:** Model-builder tests giữ các hành vi đặc thù của cả hai kiến trúc; validate không khởi tạo model tạm; tập test nhanh giảm thời gian đo được mà không làm suy giảm mục tiêu coverage.

## Bước 5 — Làm test hành vi checkpoint có khả năng phát hiện lỗi

**Files:** `going_modular/utils.py`, `going_modular/engine.py`, `tests/test_utils.py`, `tests/test_engine.py`, các đặc tả test liên quan.

1. Làm cho E8/E9/E15 phân biệt được epoch tốt nhất và epoch cuối: dùng training step/fixture deterministic làm thay đổi tham số theo epoch; xác nhận checkpoint được chọn và `restore_best` khôi phục đúng weights. Không dùng `lr=0` cho assertion cần quan sát thay đổi weights.
2. Thay E14 đang mock toàn bộ `save_checkpoint` bằng test tại biên lưu thật, phù hợp với mục tiêu test (lỗi được truyền lên và không tạo trạng thái thành công giả).
3. Thay U5 đang mock `_atomic_write_json` bằng fault injection tại thao tác thực tế:
   - Với JSON: ép `os.replace` lỗi sau khi temp file đã được ghi; xác nhận file cũ nguyên vẹn, temp được dọn, lỗi được truyền lên.
   - Nếu cần kiểm tra lỗi giữa lúc ghi, giả lập `Path.write_text` ghi một phần rồi raise; kiểm tra cùng các điều kiện bảo toàn.
4. Nếu hợp đồng ở Bước 1 yêu cầu atomic cho resume checkpoint/best model, triển khai ghi file tạm cùng thư mục, đóng file rồi `os.replace`; dọn temp cụ thể khi lỗi và phát lại exception. Thêm kiểm thử failure injection cho chính đường ghi đó, chứng minh file checkpoint cũ vẫn nạp được.
5. Nếu đặc tả không yêu cầu atomic checkpoint, không mở rộng code ngoài phạm vi; ghi rõ guarantee thực tế và kiểm thử đúng lỗi được truyền lên. Không dùng mock thay cho thao tác cần kiểm chứng.
6. Đồng bộ mô tả U5/E14 và test plan/spec để chúng khớp tests thực tế.

**Cổng hoàn tất:** Test sẽ thất bại nếu chọn sai epoch, không restore best, nuốt lỗi ghi, làm hỏng file cũ, hoặc không dọn temp theo guarantee đã chốt.

## Bước 6 — Đồng bộ tài liệu và nghiệm thu

**Files:** `docs/plans/implementation_plan.md`, các đặc tả module/test liên quan; notebook và tests.

1. Cập nhật chính sách legacy trong tài liệu kế hoạch để phản ánh cleanup đã hoàn tất; không sửa các phase cũ trừ khi chúng mô tả sai trạng thái hiện tại.
2. Đồng bộ signature `save_model`, API `train`, chính xác 14 exports, validation metadata và persistence guarantee giữa code/docs/tests.
3. Chạy kiểm chứng theo thứ tự:
   ```powershell
   .venv\Scripts\python.exe -m pytest tests\test_smoke.py -q
   .venv\Scripts\python.exe -m pytest tests\test_model_builder.py tests\test_engine.py tests\test_utils.py -m "not slow" -q
   .venv\Scripts\python.exe -m pytest tests\ -m "not slow" -q
   .venv\Scripts\python.exe -m pytest tests\ -q
   git diff --check
   ```
4. Nếu không chạy được suite đầy đủ do mạng/weights, báo tách bạch lỗi môi trường; không biến skip rộng thành pass. Không chạy notebook training để nghiệm thu cleanup.
5. So sánh thời lượng suite nhanh với baseline mới ở Bước 0; mục tiêu dưới 15 giây chỉ được tuyên bố đạt nếu đo thực tế đạt, không dùng ước lượng.

## Tiêu chí nghiệm thu

- Notebook code hiện hành không còn dùng các API/kwargs legacy đã gỡ.
- Không còn định nghĩa hoặc export các legacy API đã được chốt loại bỏ; imports công khai chính xác 14 symbols.
- `save_model` và `train` signatures thống nhất với API đã chốt; `verbose` chỉ giữ/bỏ theo vai trò hành vi, không theo tên.
- Validation không dựng model torchvision tạm; test vẫn bao phủ ResNet-18 và hành vi đặc thù VGG16.
- Test best-checkpoint phân biệt được weights giữa các epoch; test lỗi ghi chạm vào thao tác filesystem thật và kiểm tra đúng atomicity contract.
- Tài liệu hiện hành không mâu thuẫn với code; suite chạy trong `.venv` và kết quả/thời lượng được báo cáo chính xác.
- Không xóa/ghi đè artifacts huấn luyện có sẵn, không tải dữ liệu hoặc pretrained weights, không thực hiện training.
