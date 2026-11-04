# Đặc tả Thiết kế Module Dữ liệu — `data_setup.py`

> **Tài liệu:** `docs/define&designt/01_data.md` (Define & Design cho pipeline dữ liệu)
> **Module liên quan:** `going_modular/data_setup.py`
> **Bước thực hành tương ứng:** Step 5 — Prepare the Data
> **Phụ thuộc:** `docs/define&designt/00_lab2_description.md` (tổng quan), `docs/define&designt/02_model.md` (tiền xử lý đi kèm trọng số), `docs/define&designt/03_engine.md` (huấn luyện), `docs/define&designt/04_utils.md` (seed/device)
> **Trạng thái:** Pre-implementation Design Specification

---

## PHẦN I: DEFINE — Xác định Bài toán & Yêu cầu

### 1.1. Bối cảnh & Mục tiêu

Module `data_setup.py` chịu trách nhiệm **tạo ra ba `DataLoader` (Train, Validation, Test) sẵn sàng cho mô hình pre-trained** từ tập CIFAR-10 (Step 5 của đề bài). Module không chứa mô hình và không chứa vòng lặp huấn luyện.

Mục tiêu:

* Tải CIFAR-10 qua `torchvision.datasets.CIFAR10` và đưa ảnh về dạng đầu vào mà VGG16 và ResNet-18 (ImageNet) yêu cầu.
* Tách tập Validation độc lập từ tập Train gốc, chia phân tầng và tái lập được.
* Cách ly phép biến đổi giữa các tập mà không nhân bản dữ liệu trên RAM.

**Đặc điểm tập dữ liệu đầu vào (CIFAR-10):**

| Thuộc tính | Giá trị |
| :--- | :--- |
| Quy mô | $60,000$ ảnh màu RGB, $10$ lớp cân bằng tuyệt đối ($6,000$ ảnh/lớp) |
| Tên lớp | `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck` |
| Phân chia gốc | Chỉ có 2 tập: Train (`train=True`, $50,000$ ảnh) và Test (`train=False`, $10,000$ ảnh) |
| Kích thước ảnh | $32 \times 32$ pixels |

**Bốn bài toán cốt lõi cần giải quyết:**

1. **Tương thích không gian và phân phối.** Mô hình pre-trained yêu cầu tensor $3 \times 224 \times 224$ chuẩn hóa theo phân phối màu ImageNet. Đưa trực tiếp ảnh $32 \times 32$ chưa chuẩn hóa vào sẽ làm kích thước feature map ở các tầng sâu bị suy biến và làm sai lệch trọng số pre-trained.
2. **Thiếu tập Validation.** CIFAR-10 không có sẵn tập này. Nếu dùng tập Test để theo dõi từng epoch và lưu mô hình tốt nhất, thông tin của tập Test sẽ rò rỉ vào quá trình chọn mô hình (**Data Leakage**). Vì vậy phải trích tập Validation độc lập từ tập Train gốc.
3. **Cân bằng lớp khi chia (Stratified Split).** Khi tách Train gốc ($50,000$) thành Train mới ($45,000$) và Validation ($5,000$), `random_split` thuần túy sẽ làm lệch tỷ lệ 10 lớp. Hệ thống yêu cầu chia phân tầng để mỗi lớp có đúng $4,500$ mẫu Train và $500$ mẫu Validation.
4. **Cách ly phép biến đổi và tối ưu bộ nhớ.** Tập Train cần Data Augmentation (lật ảnh ngẫu nhiên); tập Validation và Test tuyệt đối không được biến đổi ngẫu nhiên. Hai luồng biến đổi phải tách biệt mà **không tải hay nhân bản hai lần** tập $50,000$ ảnh trên RAM.

### 1.2. Đầu vào / Đầu ra

| | Mô tả |
| :--- | :--- |
| **Đầu vào** | Thư mục dữ liệu (`data_dir`), `batch_size`, `val_size`, `seed`, transform tùy chọn (`train_transform`, `eval_transform`), tên mô hình để lấy tiền xử lý (`transform_model`) |
| **Đầu ra** | `train_dataloader`, `val_dataloader`, `test_dataloader` (mỗi batch có shape `[B, 3, 224, 224]`, kiểu `float32`) và `class_names` (danh sách 10 tên lớp) |
| **Đầu ra phụ** | Bộ tiền xử lý chuẩn ImageNet (`get_pretrained_transform`); chỉ mục chia phân tầng (`stratified_split_indices`); công thức giải chuẩn hóa để hiển thị ảnh |

### 1.3. Yêu cầu chức năng

| Mã | Yêu cầu |
| :--- | :--- |
| FR-1 | Tải được CIFAR-10 Train và Test với `transform=None` (ảnh PIL $32 \times 32$). |
| FR-2 | Chia Train gốc thành Train/Validation **phân tầng**, tái lập 100% theo `seed`. |
| FR-3 | Tiền xử lý tự động đồng bộ với bộ trọng số pre-trained, không gõ cứng mean/std. |
| FR-4 | Augmentation (lật ngang ngẫu nhiên) chỉ áp dụng cho tập Train; Validation và Test chỉ có phép biến đổi xác định. |
| FR-5 | Không nhân bản dữ liệu ảnh trên RAM khi dùng nhiều transform khác nhau. |
| FR-6 | Trả về 3 `DataLoader` và `class_names`; `shuffle=True` chỉ cho Train. |
| FR-7 | `pin_memory=torch.cuda.is_available()`; `num_workers` cấu hình được. Chính sách này dựa trên khả năng CUDA của runtime, không dựa vào device được yêu cầu cho model. |
| FR-8 | Có công thức giải chuẩn hóa để hiển thị ảnh từ tensor đã chuẩn hóa. |
| FR-9 | Kiểm tra đầu vào hợp lệ và báo `ValueError` rõ ràng. |
| FR-10 | Các tham số `seed`, `val_size`, `batch_size` có thể ghi vào `config.json` để tái lập thí nghiệm. |

### 1.4. Ràng buộc

* Tập dữ liệu là CIFAR-10 từ `torchvision.datasets`, chỉ có hai tập gốc; `VAL_SIZE = 5000`, `SEED = 42`, `BATCH_SIZE = 32` (theo `00_lab2_description.md`).
* Thuật toán chia phân tầng viết bằng **PyTorch thuần**, không phụ thuộc `scikit-learn`.
* Bộ tiền xử lý dựa trên bộ trọng số `IMAGENET1K_V1`, cùng bộ trọng số mà `model_builder.py` nạp.
* Tập dữ liệu gốc chỉ được tải **một lần** vào bộ nhớ cho Train và Validation.
* `num_workers` mặc định `0` để chạy ổn định trên Google Colab và Windows.
* Phạm vi cam kết và kiểm thử của `stratified_split_indices` là CIFAR-10 với 10 lớp cân bằng và `val_size=5000`; hỗ trợ nhãn lệch hoặc lớp hiếm ngoài cấu hình này không thuộc yêu cầu hiện tại.

### 1.5. Tiêu chí chấp nhận

1. Giao của `train_indices` và `val_indices` rỗng; hợp của chúng bao phủ đủ $50,000$ chỉ mục.
2. Tập Validation $5,000$ mẫu có đúng $500$ mẫu cho mỗi lớp từ $0$ đến $9$; tập Train có đúng $4,500$ mẫu mỗi lớp.
3. Mỗi batch từ `DataLoader` có shape `(batch_size, 3, 224, 224)` kiểu `torch.float32`.
4. Cùng `seed` cho cùng kết quả chia; `seed` khác cho kết quả chia khác.
5. Tập Validation và Test không chịu biến đổi ngẫu nhiên.
6. Bộ test ở mục 2.11 đều đạt.

---

## PHẦN II: DESIGN — Thiết kế Chi tiết

### 2.1. Vị trí trong kiến trúc hệ thống

```text
notebooks/pretrained_models.ipynb
        │  gọi create_train_val_test_dataloaders(...)
        ▼
going_modular/data_setup.py ──► (train_dl, val_dl, test_dl, class_names)
        │                              │
        │                              ├──► engine.py   (train_step / evaluate_step)
        │                              └──► evaluate.py (suy luận trên test_dl, confusion matrix)
        │
Notebook ── model_builder.get_transforms(cfg, weights=data.transform_weights) ──► truyền transforms vào data_setup
                                                   └── fallback nội bộ: get_pretrained_transform
```

### 2.2. Kiến trúc pipeline dữ liệu

Luồng dữ liệu từ ổ đĩa đến batch đưa vào mô hình gồm 4 tầng xử lý tuần tự:

```text
[1. Raw CIFAR-10 Dataset (transform=None)]
       ├── Tập Train gốc (50,000 ảnh PIL 32x32)
       └── Tập Test gốc  (10,000 ảnh PIL 32x32)
                 │
                 ▼
[2. Stratified Index Splitter (seed=42)]
       ├── train_indices (45,000 chỉ mục: 4,500 mẫu/lớp)
       └── val_indices   (5,000 chỉ mục:  500 mẫu/lớp)
                 │
                 ▼
[3. Transform Wrapper (_TransformSubset)]
       ├── Train Subset (raw_train + train_indices + train_transform)
       ├── Val Subset   (raw_train + val_indices   + eval_transform)
       └── Test Subset  (raw_test  + all_indices   + eval_transform)
                 │
                 ▼
[4. PyTorch DataLoaders (batch_size=32, pin_memory)]
       ├── train_dataloader (shuffle=True, generator riêng theo seed) -> Tensor [B, 3, 224, 224]
       ├── val_dataloader   (shuffle=False) -> Tensor [B, 3, 224, 224]
       └── test_dataloader  (shuffle=False) -> Tensor [B, 3, 224, 224]
```

### 2.3. Thiết kế tham số cấu hình

Các tham số của `create_train_val_test_dataloaders`:

| Tham số | Kiểu | Mặc định | Ý nghĩa |
| :--- | :--- | :--- | :--- |
| `batch_size` | `int` | `32` | Kích thước lô; cân bằng tốc độ và VRAM. |
| `data_dir` | `str` | `"data"` | Thư mục lưu CIFAR-10 (đã loại khỏi Git qua `.gitignore`). |
| `train_transform` | `Optional[Callable]` | `None` | Transform cho Train; `None` thì dùng fallback `get_pretrained_transform(augment=True)`. |
| `eval_transform` | `Optional[Callable]` | `None` | Transform cho Validation và Test; `None` thì dùng fallback `get_pretrained_transform(augment=False)`. |
| `val_size` | `int` | `5000` | Số mẫu Validation trích từ Train gốc ($10\%$). |
| `seed` | `int` | `42` | Hạt giống cho phép chia phân tầng. |
| `num_workers` | `int` | `0` | Số tiến trình nạp dữ liệu. |
| `generator` | `Optional[torch.Generator]` | `None` | Generator riêng cho thứ tự shuffle của Train; Notebook truyền generator được seed theo từng run. |
| `download` | `bool` | `True` | Tự tải CIFAR-10 nếu chưa có. |
| `transform_model` | `Literal["vgg16", "resnet18"]` | `"vgg16"` | Chọn bộ trọng số để lấy tiền xử lý mặc định. |

### 2.4. Bộ tiền xử lý chuẩn ImageNet — `get_pretrained_transform`

**Mục tiêu:** tạo chuỗi biến đổi ảnh (`transforms.Compose`) tự động đồng bộ với cấu hình tiền xử lý của bộ trọng số được chọn, tránh gõ cứng hằng số.

**Thuật toán xử lý bên trong:**

1. Truy xuất bộ trọng số `VGG16_Weights.IMAGENET1K_V1` hoặc `ResNet18_Weights.IMAGENET1K_V1` từ `torchvision.models`.
2. Nếu `augment=True`, chèn `transforms.RandomHorizontalFlip()` vào **đầu** danh sách biến đổi (khi ảnh còn là PIL).
3. Nối tiếp bằng `weights.transforms()` của torchvision, thực hiện lần lượt: phóng cạnh ngắn lên $256$ pixels (nội suy Bilinear) $\rightarrow$ `CenterCrop` $224 \times 224$ $\rightarrow$ chuyển sang tensor thực $[0.0, 1.0]$ $\rightarrow$ chuẩn hóa theo từng kênh màu với $\text{mean} = [0.485, 0.456, 0.406]$ và $\text{std} = [0.229, 0.224, 0.225]$:

$$X_{\text{norm}}^{(c)} = \frac{X^{(c)} - \text{mean}^{(c)}}{\text{std}^{(c)}}$$

**Lưu ý:** ảnh CIFAR-10 ($32 \times 32$) bị phóng to $8$ lần trước khi cắt, nên ảnh đầu vào mô hình khá mờ. Cần đánh giá ảnh hưởng này khi phân tích kết quả ở `05_evaluation_results.md`.

### 2.5. Thuật toán chia phân tầng — `stratified_split_indices`

**Mục tiêu:** chia tập chỉ mục thành `train_indices` và `val_indices` bằng PyTorch thuần, giữ nguyên tỷ lệ phân bố của mọi lớp.

**Thuật toán phân bổ và xáo trộn:**

1. **Nhóm chỉ mục theo lớp:** gom các vị trí có cùng nhãn $c$ vào từ điển `class_indices[c]`.
2. **Tính hạn ngạch cho từng lớp:**
   $$q_c = \frac{\text{val\_size} \times N_c}{N}$$
   Gán số mẫu nguyên ban đầu cho mỗi lớp: $v_c = \min(\lfloor q_c \rfloor, N_c - 1)$.
3. **Cân bằng phần dư:** nếu $\sum v_c < \text{val\_size}$, sắp xếp các lớp theo phần lẻ $(q_c - \lfloor q_c \rfloor)$ giảm dần và cộng thêm $1$ mẫu cho từng lớp cho đến khi tổng bằng đúng `val_size`. Khi các phần lẻ bằng nhau, ưu tiên lớp có chỉ số nhỏ hơn (quy tắc phá hòa này giúp thuật toán xác định; cần giữ nhất quán khi hiện thực hóa).
4. **Lấy mẫu ngẫu nhiên tái lập:** dùng `torch.Generator().manual_seed(seed)` và `torch.randperm` để xáo trộn chỉ mục từng lớp; $v_c$ phần tử đầu vào `val_indices`, phần còn lại vào `train_indices`.

**Ví dụ minh họa phần dư.** Nhãn có số mẫu theo lớp là $[4, 3, 3]$ ($N = 10$), `val_size = 3`:

| Lớp | $N_c$ | $q_c$ | $\lfloor q_c \rfloor$ | Phần lẻ | $v_c$ cuối |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | 4 | 1.2 | 1 | 0.2 | 1 |
| 1 | 3 | 0.9 | 0 | 0.9 | 1 |
| 2 | 3 | 0.9 | 0 | 0.9 | 1 |

Tổng ban đầu bằng $1 < 3$, còn thiếu $2$ mẫu; hai lớp có phần lẻ lớn nhất (lớp 1 và lớp 2) mỗi lớp được cộng thêm $1$. Kết quả: mỗi lớp có đúng $1$ mẫu Validation.

**Với CIFAR-10:** $q_c = 5000 \times 5000 / 50000 = 500$ là số nguyên cho mọi lớp, nên không có phần dư; mỗi lớp có $500$ mẫu Validation và $4,500$ mẫu Train.

**Tính độc lập với seed toàn cục:** phép chia dùng `torch.Generator` riêng nên kết quả chia **không phụ thuộc** trạng thái bộ sinh số ngẫu nhiên toàn cục. Train DataLoader cũng nhận generator riêng để thứ tự shuffle không phụ thuộc việc khởi tạo model hay các thao tác ngẫu nhiên khác.

### 2.6. Lớp wrapper cách ly biến đổi — `_TransformSubset`

**Mục tiêu:** lớp kế thừa `torch.utils.data.Dataset`, đóng vai trò Adapter quanh tập dữ liệu gốc, cho phép gắn `transform` khác nhau lên các tập con chỉ mục khác nhau mà **không nhân bản ảnh trên RAM**.

| Thành phần | Mô tả |
| :--- | :--- |
| Thuộc tính `dataset` | Dataset gốc với `transform=None` (dùng chung giữa các tập con) |
| Thuộc tính `indices` | Danh sách chỉ mục của tập con |
| Thuộc tính `transform` | Hàm biến đổi riêng của tập con |
| `__len__()` | Trả về `len(self.indices)` |
| `__getitem__(index)` | Lấy `(image, target)` từ `self.dataset[self.indices[index]]`, áp dụng `self.transform(image)` ngay lúc truy xuất (on-the-fly), trả về `(tensor, target)` |

Train và Validation cùng trỏ đến **một** đối tượng `raw_train_data`; chúng chỉ khác `indices` và `transform`. Đây là cách thỏa yêu cầu "không tải hai lần".

### 2.7. Hàm factory khởi tạo DataLoader — `create_train_val_test_dataloaders`

**Chữ ký hàm:**

```python
create_train_val_test_dataloaders(
    batch_size: int = 32,
    data_dir: str = "data",
    train_transform: Optional[Callable] = None,
    eval_transform: Optional[Callable] = None,
    val_size: int = 5000,
    seed: int = 42,
    num_workers: int = 0,
    generator: Optional[torch.Generator] = None,
    download: bool = True,
    transform_model: Literal["vgg16", "resnet18"] = "vgg16",
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]
```

**Luồng tích hợp:**

```text
1. nếu train_transform / eval_transform là None: tạo mặc định qua get_pretrained_transform
2. raw_train_data = CIFAR10(train=True,  transform=None)
   raw_test_data  = CIFAR10(train=False, transform=None)
3. train_idx, val_idx = stratified_split_indices(raw_train_data.targets, val_size, seed)
4. bọc 3 tập vào _TransformSubset:
     train = (raw_train, train_idx, train_transform)
     val   = (raw_train, val_idx,   eval_transform)
     test  = (raw_test,  tất cả,    eval_transform)
5. tạo 3 DataLoader với pin_memory = torch.cuda.is_available(); truyền generator riêng vào Train DataLoader
     shuffle=True cho Train; shuffle=False cho Validation và Test
6. return train_dl, val_dl, test_dl, class_names
```

### 2.8. Công thức giải chuẩn hóa để hiển thị ảnh

Tensor đã chuẩn hóa không hiển thị trực tiếp được. Để vẽ ảnh (ví dụ ở `evaluate.py` hoặc notebook), đảo phép chuẩn hóa rồi cắt về $[0, 1]$:

$$X_{\text{vis}} = \text{clamp}\left( X_{\text{norm}} \odot \text{std} + \text{mean},\; 0.0,\; 1.0 \right)$$

### 2.9. Giao diện hàm và ngoại lệ

| Hàm | Đầu vào | Đầu ra | Ngoại lệ |
| :--- | :--- | :--- | :--- |
| `get_pretrained_transform(augment=False, model_name="vgg16")` | `bool`, `Literal["vgg16", "resnet18"]` | `transforms.Compose` | `ValueError` nếu `model_name` không được hỗ trợ |
| `stratified_split_indices(labels, val_size, seed=42)` | `Sequence[int]`, `int`, `int` | `Tuple[List[int], List[int]]` (train, val) | `ValueError` (xem bên dưới) |
| `_TransformSubset(dataset, indices, transform)` | Dataset, danh sách chỉ mục, hàm biến đổi | `Dataset` | không (lớp nội bộ) |
| `create_train_val_test_dataloaders(...)` | xem mục 2.7 | 3 `DataLoader` và `List[str]` | `ValueError` truyền từ hàm con; lỗi tải dữ liệu do torchvision |

`stratified_split_indices` báo `ValueError` khi:

* `val_size <= 0`.
* `val_size >= N` (N là tổng số mẫu).
* Có lớp chứa ít hơn $2$ mẫu.
* `val_size` không chừa lại ít nhất $1$ mẫu huấn luyện cho mỗi lớp (kiểm tra sau bước cân bằng phần dư).

### 2.10. Tích hợp với các module khác

| Module | Cách phối hợp |
| :--- | :--- |
| `model_builder.py` | Cung cấp transform chuẩn qua `get_transforms(cfg, weights=data.transform_weights)`. Notebook truyền transform vào `create_train_val_test_dataloaders`; `data_setup.py` không import `model_builder`, tránh phụ thuộc vòng. Hàm `get_pretrained_transform` của `data_setup.py` là fallback độc lập khi không truyền transform. |
| `engine.py` | `train_dataloader` cho `train_step`; `val_dataloader` để chọn `best_model.pth` mỗi epoch; `test_dataloader` **không** được dùng để chọn mô hình. |
| `evaluate.py` | Dùng `test_dataloader` và `class_names` để suy luận, tính ma trận nhầm lẫn; dùng công thức giải chuẩn hóa để hiển thị ảnh. |
| `utils.py` | `set_seed(42)` cố định bộ sinh ngẫu nhiên toàn cục (ảnh hưởng augmentation); phép chia phân tầng và shuffle dùng generator riêng. `pin_memory` bật khi `torch.cuda.is_available()`, không phụ thuộc device được chọn để chạy model. |
| Notebook | Gọi `set_seed(SEED)`, tạo `torch.Generator().manual_seed(SEED)` riêng và loader mới cho mỗi run. Lấy transform từ `model_builder.get_transforms(cfg, weights=data.transform_weights)` và truyền vào factory; dùng fallback nội bộ của data module nếu transform không được truyền. |

### 2.11. Kế hoạch kiểm thử

| Mã | Kiểm tra | Kỳ vọng |
| :--- | :--- | :--- |
| D1 | Split CIFAR-10 giả lập cân bằng 10 lớp với `val_size=5000`, `seed=42` | Train/val rời nhau, phủ đủ chỉ mục; mỗi lớp có 4,500/500 mẫu; cùng seed cho cùng kết quả. |
| D2 | Dùng seed khác với cùng nhãn | Split thay đổi nhưng quota lớp giữ nguyên. |
| D3 | `val_size` sai, nhãn không đủ mẫu/lớp hoặc đầu vào rỗng | Báo `ValueError` rõ ràng. |
| D4 | `_TransformSubset` trên cùng dataset gốc với hai transforms | Transform đúng cho từng subset, dataset gốc không bị nhân bản. |
| D5 | Mock CIFAR-10 và lấy batch từ ba loaders | Shape/dtype đúng; chỉ Train shuffle/augmentation; Validation/Test xác định; Train có generator riêng. |
| D6 | Kiểm tra cấu hình loader | `pin_memory` bằng `torch.cuda.is_available()`; mặc định `num_workers=0`. |
| D7 | Fallback transform không có model hỗ trợ | Transform có kích thước/chuẩn hóa đúng; model không hỗ trợ báo `ValueError`. |

Chạy kiểm thử bằng `pytest tests/test_data_setup.py -v`; các kiểm thử không tải dữ liệu thật, còn kiểm thử CIFAR-10/download nếu có được đánh dấu `slow`.

### 2.12. Mở rộng

Phạm vi hiện thực và kiểm thử hiện tại giới hạn ở CIFAR-10. Mở rộng sang dữ liệu nhãn lệch, tập khác (ví dụ CIFAR-100) hoặc mô hình khác cần thiết kế và kiểm thử riêng:

1. Thay lớp dataset trong bước 2 của mục 2.7 và cập nhật `val_size` cho phù hợp; thuật toán chia phân tầng giữ nguyên vì chỉ phụ thuộc nhãn.
2. Với mô hình mới, thêm tên mô hình vào `get_pretrained_transform` và `transform_model`.
3. Thêm nhóm test D1–D2 và S1 cho tập dữ liệu mới.

### 2.13. Quyết định thiết kế của nhóm so với chuẩn từ tài liệu

| Hạng mục | Có trong tài liệu chính thức | Quyết định thiết kế của nhóm |
| :--- | :--- | :--- |
| `torchvision.datasets.CIFAR10`, `DataLoader` | Có (PyTorch/torchvision) | |
| Tiền xử lý đi kèm trọng số (`weights.transforms()`) | Có (torchvision) | |
| Cấu trúc package `going_modular`, `data_setup.py` | Có (Learn PyTorch ch.05) | |
| Chia Train gốc thành Train/Validation | | Có |
| Thuật toán chia phân tầng bằng PyTorch thuần | | Có |
| Quy tắc phá hòa theo chỉ số lớp nhỏ hơn | | Có |
| `_TransformSubset` (một dataset gốc, nhiều transform) | | Có |
| Chèn `RandomHorizontalFlip` trước `weights.transforms()` | | Có |

### 2.14. Tài liệu tham khảo

1. PyTorch / torchvision, *Models and pre-trained weights* (tiền xử lý đi kèm trọng số): https://docs.pytorch.org/vision/stable/models.html
2. PyTorch / torchvision, *Datasets* (`CIFAR10`): https://docs.pytorch.org/vision/stable/datasets.html
3. Daniel Bourke, *Learn PyTorch for Deep Learning*, chương 05 (Going Modular): https://www.learnpytorch.io
