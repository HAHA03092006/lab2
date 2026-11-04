# Đặc tả Kiểm thử — `model_builder.py`

> Nguồn: `docs/define&designt/02_model.md` mục 2.11 (Kế hoạch kiểm thử `tests/test_model_builder.py`).

## 1. Mục tiêu

Bộ test kiểm chứng `model_builder.py` **độc lập** trước khi tích hợp với `engine.py`. Test **không** đánh giá độ chính xác của mô hình (việc của `05_evaluation_results.md`). Test chỉ kiểm tra **tính đúng đắn của cấu trúc và trạng thái** mô hình: đúng hình dạng đầu ra, đúng tham số nào được huấn luyện, cấu hình sai bị chặn, BatchNorm cư xử đúng, kết quả tái lập được.

## 2. Nguyên tắc

1. **Nhanh, không cần mạng:** hầu hết test dùng `weights=None` (khởi tạo ngẫu nhiên). Cấu trúc mô hình giống hệt bản pre-trained nên vẫn kiểm tra được hình dạng và trạng thái.
2. **Kiểm tra hành vi quan sát được:** so sánh `requires_grad`, `shape`, cờ `training`, giá trị tensor, thay vì kiểm tra cách cài đặt bên trong.
3. **Mỗi test một ý chính.**
4. **Test lỗi dùng `pytest.raises`:** cấu hình sai phải báo `ValueError` hoặc `TypeError` rõ ràng, không được âm thầm chạy tiếp.
5. **Tham số hóa theo kiến trúc:** các test chung chạy lặp cho cả `resnet18` và `vgg16` bằng `@pytest.mark.parametrize`.

## 3. Sáu nhóm kiểm thử

| Nhóm | Khía cạnh | Câu hỏi cần trả lời | Test |
| :--- | :--- | :--- | :--- |
| A | Cấu trúc và hình dạng | Mô hình có đầu ra và head đúng không? | T1, T2, T2b |
| B | Trạng thái đóng băng | Đúng tham số nào trainable, đúng số lượng không? | T3, T4, T5, T5b, T5c |
| C | Kiểm tra cấu hình | Cấu hình sai có bị chặn bằng lỗi rõ ràng không? | T6, T7, T8, T8b |
| D | BatchNorm và optimizer | BN đã freeze có thật sự đứng yên? Nhóm tham số có đúng không? | T9, T10, T10b, T10c, T11 |
| E | Tái lập, lưu/nạp cấu hình, tiền xử lý | Cùng cấu hình và seed có cho cùng kết quả? Cấu hình lưu rồi nạp lại có giữ nguyên? | T12, T13, T13b, T13c, T14, T14b, T14c |
| F | Trọng số ImageNet thật (chậm) | Trọng số pre-trained có thực sự được nạp và freeze đúng? | S1, S2 |

## 4. Mô tả chi tiết từng test

### Nhóm A: Cấu trúc và hình dạng

| Mã | Hàm test | Cách kiểm thử (ví dụ) | Kỳ vọng | Lỗi bắt được |
| :--- | :--- | :--- | :--- | :--- |
| T1 | `test_t1_output_shape` | Dựng mô hình, chuyển `eval()`, đưa tensor ngẫu nhiên `(2, 3, 224, 224)` qua mô hình trong `torch.no_grad()`. Lặp cho ResNet-18 và VGG16. | Đầu ra có shape `(2, 10)` | Quên thay head (ra 1000 lớp); `in_features` của `Linear` mới sai khiến forward lỗi. |
| T2 | `test_t2_head_replaced` | Dựng với `num_classes=7` (cố ý khác 10), lấy head qua `get_head` của registry. | Head là `nn.Linear` và `out_features == 7` | Ghi cứng số 10 trong code; thay nhầm vị trí head (ví dụ VGG phải là `classifier[6]`). |
| T2b | `test_t2b_head_with_dropout` | Dựng với `dropout=0.3`. | Head là `nn.Sequential`: phần tử 0 là `Dropout(p=0.3)`, phần tử 1 là `Linear` có `out_features == 10` | Tham số `dropout` bị bỏ qua; thứ tự Dropout và Linear bị đảo. |

### Nhóm B: Trạng thái đóng băng

| Mã | Hàm test | Cách kiểm thử (ví dụ) | Kỳ vọng | Lỗi bắt được |
| :--- | :--- | :--- | :--- | :--- |
| T3 | `test_t3_only_head_trainable` | Xác định tập tham số của head bằng `id(p)`. Duyệt toàn bộ `model.parameters()`. | Tham số thuộc head có `requires_grad=True`; mọi tham số còn lại có `requires_grad=False` | Thay head trước rồi mới freeze (head bị đóng băng theo); quên freeze một phần backbone. |
| T4 | `test_t4_param_counts` | Gọi `count_params` ở Giai đoạn 1. | ResNet-18: total 11,181,642, trainable 5,130. VGG16: total 134,301,514, trainable 40,970. `frozen = total - trainable` | Đếm sai; head sai kích thước; trainable lớn bất thường do freeze thiếu. |
| T5 | `test_t5_unfreeze_last_n` | Với `n` thuộc `{1, 2}` và từng kiến trúc, lấy danh sách khối qua `blocks` của registry. | `n` khối cuối: mọi tham số `requires_grad=True`. Các khối trước đó: không tham số nào `requires_grad=True` | Lỗi lệch một (`[-n:]` viết thành `[-n-1:]`); mở nhầm khối; VGG chia khối sai. |
| T5b | `test_t5b_unfreeze_increases_trainable` | So sánh `trainable` của ResNet-18 giữa `unfreeze_last_n=0` và `unfreeze_last_n=1`. | Giai đoạn 2 có nhiều tham số trainable hơn | Cờ `unfreeze_last_n` không có tác dụng. |
| T5c | `test_full_training_mode_all_trainable` | Dựng với `freeze_backbone=False`. | `trainable == total` | Chế độ huấn luyện toàn bộ mô hình vẫn bị freeze một phần. |

### Nhóm C: Kiểm tra cấu hình

| Mã | Hàm test | Cách kiểm thử (ví dụ) | Kỳ vọng | Lỗi bắt được |
| :--- | :--- | :--- | :--- | :--- |
| T6 | `test_t6_unfreeze_more_than_blocks` | `unfreeze_last_n=5` với ResNet-18 (có 4 khối) và `6` với VGG16 (có 5 khối). | `ValueError` | Slice vượt phạm vi bị bỏ qua âm thầm, mô hình mở khóa toàn bộ mà không báo. |
| T7 | `test_t7_unfreeze_without_freeze` | `freeze_backbone=False` kèm `unfreeze_last_n=1`. | `ValueError` | Cấu hình mâu thuẫn không bị phát hiện. |
| T8 | `test_t8_unknown_arch` | `arch="not_a_model"`. | `ValueError` | Báo `KeyError` khó hiểu thay vì thông báo liệt kê các kiến trúc hợp lệ. |
| T8b | `test_t8b_invalid_values` | `unfreeze_last_n=-1`; `dropout=1.0`. | Cả hai đều `ValueError` | Giá trị ngoài miền hợp lệ lọt qua và gây lỗi khó đoán ở bước sau. |

### Nhóm D: BatchNorm và optimizer

| Mã | Hàm test | Cách kiểm thử (ví dụ) | Kỳ vọng | Lỗi bắt được |
| :--- | :--- | :--- | :--- | :--- |
| T9 | `test_t9_bn_eval_when_fully_frozen` | ResNet-18 ở Giai đoạn 1; gọi `set_train_mode`; lấy mọi `BatchNorm2d`. Kiểm tra danh sách BN không rỗng. | `model.training == True`, nhưng mọi BN có `training == False` | `set_train_mode` quên đưa BN về eval. Việc kiểm tra danh sách không rỗng tránh trường hợp test "đúng" chỉ vì không có BN nào. |
| T10 | `test_t10_bn_train_only_for_unfrozen_blocks` | ResNet-18 với `unfreeze_last_n=1`; gọi `set_train_mode`. | BN của `bn1`, `layer1` đến `layer3` ở eval; BN của `layer4` ở train | Đưa nhầm mọi BN về eval (làm Fine-Tuning không học được thống kê mới), hoặc để nguyên mọi BN ở train. |
| T10b | `test_t10b_freeze_bn_stats_disabled` | `freeze_bn_stats=False`; gọi `set_train_mode`. | Mọi BN ở chế độ train | Cờ `freeze_bn_stats` bị bỏ qua. |
| T10c | `test_t10c_running_stats_unchanged_when_frozen` | Sao chép `bn1.running_mean`, gọi `set_train_mode`, chạy forward một batch `(4, 3, 64, 64)`, so sánh lại. | `torch.equal` đúng: thống kê không đổi | Kiểm tra hành vi thật, không chỉ cờ `training`. |
| T11 | `test_t11_groups_stage1`, `test_t11_groups_stage2` | Gọi `get_param_groups(model, cfg, 1e-3, 1e-4)` ở hai giai đoạn. Gom `id` của mọi tham số trong các nhóm. | Giai đoạn 1: một nhóm, `lr=1e-3`. Giai đoạn 2: hai nhóm, `lr=[1e-3, 1e-4]`. Không tham số nào xuất hiện hai lần; tổng số phần tử các nhóm bằng `trainable` | Nhóm rỗng; tham số trùng giữa hai nhóm (optimizer báo lỗi); tham số trainable bị sót. |

### Nhóm E: Tái lập, lưu/nạp cấu hình, tiền xử lý

| Mã | Hàm test | Cách kiểm thử (ví dụ) | Kỳ vọng | Lỗi bắt được |
| :--- | :--- | :--- | :--- | :--- |
| T12 | `test_t12_reproducible_with_same_seed` | `torch.manual_seed(42)`, dựng mô hình; đặt lại seed, dựng lần hai; so sánh từng tensor trong `state_dict()` bằng `torch.equal`. | Hai mô hình giống hệt | Có nguồn ngẫu nhiên không được kiểm soát khi dựng mô hình. |
| T13 | `test_t13_config_roundtrip` | `ModelConfig.from_dict(cfg.to_dict())` rồi so sánh với `cfg`. | Hai cấu hình bằng nhau | Mất hoặc đổi giá trị khi lưu/nạp `config.json`. |
| T13b | `test_t13b_unknown_key_rejected` | `from_dict({"arch": "resnet18", "foo": 1})`. | `TypeError` | Khóa lạ hoặc gõ sai tên khóa trong `config.json` bị bỏ qua âm thầm. |
| T13c | `test_t13c_config_is_frozen` | Gán `cfg.arch = "vgg16"` sau khi tạo. | Báo lỗi (dataclass `frozen`) | Cấu hình bị sửa giữa chừng, làm artifacts ghi lại sai thí nghiệm đã chạy. |
| T14 | `test_t14_transforms_without_weights` | Gọi `get_transforms(ModelConfig(weights=None))` mà không truyền weights riêng. | `ValueError` | Trả về transform sai hoặc `None` khi không có trọng số đi kèm. |
| T14b | `test_t14b_transforms_output` | Tạo ảnh PIL đen 32×32 (kích thước CIFAR-10), áp transform với `weights="IMAGENET1K_V1"`. | Tensor có shape `(3, 224, 224)` | Tiền xử lý không đưa ảnh CIFAR-10 về kích thước mà mô hình yêu cầu. |
| T14c | `test_t14c_transform_weights_override` | Dùng `ModelConfig(weights=None)` và truyền `weights="IMAGENET1K_V1"` vào `get_transforms`. | Trả về transform ImageNet hợp lệ; không tải model weights | Chế độ reuse không thể tạo đúng transform độc lập với trọng số dùng khởi tạo mô hình. |

### Nhóm F: Trọng số ImageNet thật (đánh dấu `slow`)

| Mã | Hàm test | Cách kiểm thử (ví dụ) | Kỳ vọng | Lỗi bắt được |
| :--- | :--- | :--- | :--- | :--- |
| S1 | `test_slow_pretrained_only_head_trainable` | Nạp ResNet-18 với `IMAGENET1K_V1`, lặp lại kiểm tra của T3 và T4. | Chỉ head trainable; trainable bằng 5,130 | Lỗi chỉ xuất hiện khi nạp trọng số thật. |
| S2 | `test_slow_pretrained_backbone_differs_from_random` | So sánh `conv1.weight` của mô hình pre-trained với mô hình `weights=None`. | Hai tensor khác nhau | Trọng số ImageNet không thực sự được nạp (âm thầm rơi về khởi tạo ngẫu nhiên). |

Nếu không tải được trọng số (mất mạng, bị chặn), fixture bắt `OSError` và **bỏ qua** test thay vì báo lỗi.

## 5. Hai ví dụ đọc test chi tiết

**Ví dụ 1: T3 (chỉ head được huấn luyện)**

```python
model = build_model(ModelConfig(arch="resnet18", weights=None))
ids = {id(p) for p in model.fc.parameters()}      # tập tham số của head

for p in model.parameters():
    if id(p) in ids:
        assert p.requires_grad                    # head phải học được
    else:
        assert not p.requires_grad                # backbone phải bị đóng băng
```

Test dùng `id(p)` để phân biệt tham số của head với phần còn lại. Nếu `build_model` thay head rồi mới freeze, vòng lặp sẽ thấy head có `requires_grad=False` và test thất bại ngay ở nhánh đầu.

**Ví dụ 2: T10c (BatchNorm đã freeze không cập nhật thống kê)**

```python
cfg = ModelConfig(arch="resnet18", weights=None)
model = build_model(cfg)
before = model.bn1.running_mean.clone()           # chụp thống kê trước

set_train_mode(model, cfg)
model(torch.randn(4, 3, 64, 64))                  # forward ở chế độ train

assert torch.equal(before, model.bn1.running_mean)  # thống kê không đổi
```

Nếu dùng `model.train()` thay cho `set_train_mode`, `running_mean` sẽ bị cập nhật sau forward và test thất bại. Đây là cách duy nhất chứng minh được rằng `requires_grad=False` một mình **không** đủ để giữ BatchNorm đứng yên.

## 6. Cách chạy và ánh xạ tiêu chí chấp nhận

```bash
pytest tests/test_model_builder.py -v                 # tất cả
pytest tests/test_model_builder.py -v -m "not slow"   # bỏ qua test tải trọng số
```

Marker `slow` cần được đăng ký trong `tests/conftest.py`:

```python
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: tests that download ImageNet weights")
```

| Tiêu chí (mục 1.5) | Test chứng minh |
| :--- | :--- |
| 1. Cùng cấu hình cho cùng số tham số trainable | T4, T12 |
| 2. Sau Feature Extraction chỉ head trainable | T3, S1 |
| 3. Đầu ra có shape `(B, 10)` | T1 |
| 4. Toàn bộ test đạt | Mọi test ở trên |

**Giới hạn:** Không kiểm tra độ chính xác, tốc độ hay chạy trên GPU. Mỗi lần dựng VGG16 cần khoảng 0.5 GB RAM; máy yếu chạy các test `vgg16` sẽ chậm. Các con số ở T4 được tính tay; cần đối chiếu bằng `count_params` ở lần chạy đầu tiên và sửa mục 2.5 nếu lệch.
