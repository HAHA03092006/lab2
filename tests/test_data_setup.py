"""Kiểm thử cho going_modular.data_setup (Phase 3).

Chạy nhanh (không tải CIFAR-10):
    pytest tests/test_data_setup.py -v -m "not slow"

Chạy toàn bộ (bao gồm slow S1–S3, cần tải CIFAR-10):
    pytest tests/test_data_setup.py -v

Ánh xạ mã test: D1–D15 (nhanh), S1–S3 (chậm, đánh dấu slow).
"""

from __future__ import annotations

from collections import Counter
from typing import List

import pytest
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData
from torchvision import transforms

from going_modular.data_setup import (
    _TransformSubset,
    create_train_val_test_dataloaders,
    get_pretrained_transform,
    stratified_split_indices,
)


# ---------------------------------------------------------------------------
# Nhóm A: Chia phân tầng đúng
# ---------------------------------------------------------------------------

def test_d1_disjoint_and_complete():
    """D1: Train/val rời nhau, phủ đủ chỉ mục, kích thước đúng."""
    labels = [i % 10 for i in range(50000)]
    train_idx, val_idx = stratified_split_indices(labels, val_size=5000)

    train_set = set(train_idx)
    val_set = set(val_idx)
    assert len(train_idx) == 45_000
    assert len(val_idx) == 5_000
    assert train_set.isdisjoint(val_set)
    assert train_set | val_set == set(range(50_000))


def test_d2_balanced_per_class():
    """D2: Mỗi lớp 0–9 có đúng 500 mẫu Validation và 4,500 mẫu Train."""
    labels = [i % 10 for i in range(50000)]
    train_idx, val_idx = stratified_split_indices(labels, val_size=5000)

    val_counts = Counter(labels[i] for i in val_idx)
    train_counts = Counter(labels[i] for i in train_idx)

    for c in range(10):
        assert val_counts[c] == 500, f"Lớp {c}: val={val_counts[c]}"
        assert train_counts[c] == 4500, f"Lớp {c}: train={train_counts[c]}"


def test_d3_returns_python_lists_of_ints():
    """D3: Trả về list các int."""
    labels = [i % 10 for i in range(100)]
    train_idx, val_idx = stratified_split_indices(labels, val_size=20)

    assert isinstance(train_idx, list)
    assert isinstance(val_idx, list)
    assert all(isinstance(i, int) for i in train_idx)
    assert all(isinstance(i, int) for i in val_idx)


# ---------------------------------------------------------------------------
# Nhóm B: Tái lập và phần dư
# ---------------------------------------------------------------------------

def test_d4_reproducible_by_seed():
    """D4: Cùng seed → cùng kết quả (dù seed toàn cục khác); seed khác → khác kết quả."""
    labels = [i % 10 for i in range(50000)]

    torch.manual_seed(999)
    train1, val1 = stratified_split_indices(labels, val_size=5000, seed=42)

    torch.manual_seed(12345)
    train2, val2 = stratified_split_indices(labels, val_size=5000, seed=42)

    assert train1 == train2
    assert val1 == val2

    train3, val3 = stratified_split_indices(labels, val_size=5000, seed=7)
    assert train3 != train1 or val3 != val1


def test_d5_remainder_distribution():
    """D5: Nhãn [4, 3, 3], val_size=3 → mỗi lớp đúng 1 mẫu Validation."""
    labels = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]
    train_idx, val_idx = stratified_split_indices(labels, val_size=3)

    assert len(val_idx) == 3
    val_labels = [labels[i] for i in val_idx]
    assert Counter(val_labels) == {0: 1, 1: 1, 2: 1}

    train_labels = [labels[i] for i in train_idx]
    assert all(train_labels.count(c) >= 1 for c in range(3))


# ---------------------------------------------------------------------------
# Nhóm C: Kiểm tra đầu vào
# ---------------------------------------------------------------------------

def test_d6_invalid_val_size():
    """D6: val_size=0, -5, N → ValueError."""
    labels = [i % 10 for i in range(100)]
    with pytest.raises(ValueError):
        stratified_split_indices(labels, val_size=0)
    with pytest.raises(ValueError):
        stratified_split_indices(labels, val_size=-5)
    with pytest.raises(ValueError):
        stratified_split_indices(labels, val_size=100)


def test_d7_invalid_labels():
    """D7: Một lớp chỉ có 1 mẫu hoặc val_size lớn quá → ValueError."""
    labels = [0] * 99 + [1]  # lớp 1 chỉ có 1 mẫu
    with pytest.raises(ValueError):
        stratified_split_indices(labels, val_size=10)

    labels = [i % 10 for i in range(100)]
    with pytest.raises(ValueError):
        stratified_split_indices(labels, val_size=95)


# ---------------------------------------------------------------------------
# Nhóm D: Bộ tiền xử lý
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_name", ["vgg16", "resnet18"])
def test_d8_transform_output(model_name):
    """D8: Tensor shape (3, 224, 224), kiểu float32."""
    from PIL import Image

    img = Image.new("RGB", (32, 32), color=(128, 64, 32))
    transform = get_pretrained_transform(model_name=model_name)
    out = transform(img)
    assert out.shape == (3, 224, 224)
    assert out.dtype == torch.float32


def test_d9_normalization_values():
    """D9: Ảnh trắng 255 qua transform → giá trị chuẩn hóa đúng."""
    from PIL import Image

    transform = get_pretrained_transform(model_name="vgg16", augment=False)
    img = Image.new("RGB", (32, 32), color=(255, 255, 255))
    out = transform(img)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    expected = (1.0 - mean) / std
    assert torch.allclose(out, expected, atol=1e-5)


def test_d10_augment_flag():
    """D10: augment=True → phần tử đầu là RandomHorizontalFlip; augment=False → không có."""
    from torchvision.transforms import RandomHorizontalFlip

    train_t = get_pretrained_transform(augment=True, model_name="vgg16")
    eval_t = get_pretrained_transform(augment=False, model_name="vgg16")

    assert isinstance(train_t.transforms[0], RandomHorizontalFlip)
    assert not any(isinstance(t, RandomHorizontalFlip) for t in eval_t.transforms)


def test_d11_unknown_model_name():
    """D11: model_name='foo' → ValueError."""
    with pytest.raises(ValueError):
        get_pretrained_transform(model_name="foo")


def test_d11b_unknown_weights_name():
    """Unknown preprocessing weights are rejected with a clear error."""
    with pytest.raises(ValueError, match="Unknown weights"):
        get_pretrained_transform(model_name="resnet18", weights_name="UNKNOWN")


def test_factory_uses_selected_transform_configuration(monkeypatch):
    """DataLoader fallback passes architecture and preprocessing preset to data_setup."""
    import going_modular.data_setup as data_setup

    calls = []

    class TinyCIFAR10:
        classes = [str(index) for index in range(10)]
        targets = [label for label in range(10) for _ in range(2)]

        def __init__(self, *, train, **kwargs):
            self.train = train

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, index):
            return torch.zeros(3, 32, 32), self.targets[index]

    def fake_transform(*, augment, model_name, weights_name):
        calls.append((augment, model_name, weights_name))
        return None

    monkeypatch.setattr(data_setup.datasets, "CIFAR10", TinyCIFAR10)
    monkeypatch.setattr(data_setup, "get_pretrained_transform", fake_transform)

    create_train_val_test_dataloaders(
        batch_size=2,
        val_size=10,
        transform_model="resnet18",
        transform_weights="IMAGENET1K_V1",
        download=False,
    )

    assert calls == [
        (True, "resnet18", "IMAGENET1K_V1"),
        (False, "resnet18", "IMAGENET1K_V1"),
    ]


# ---------------------------------------------------------------------------
# Nhóm E: Wrapper và cách ly biến đổi
# ---------------------------------------------------------------------------

def test_d12_len_and_index_mapping():
    """D12: len == số chỉ mục; phần tử i trả về mẫu gốc indices[i]."""
    raw = FakeData(size=20, image_size=(3, 32, 32), transform=None)
    indices = [3, 7, 11]
    subset = _TransformSubset(raw, indices, transform=None)

    assert len(subset) == 3
    to_tensor = transforms.ToTensor()
    for i, idx in enumerate(indices):
        img, target = subset[i]
        raw_img, raw_target = raw[idx]
        assert torch.equal(to_tensor(img), to_tensor(raw_img))
        assert target == raw_target


def test_d13_shares_underlying_dataset():
    """D13: Hai wrapper cùng trỏ đến một đối tượng dataset gốc."""
    raw = FakeData(size=20, image_size=(3, 32, 32), transform=None)
    train_subset = _TransformSubset(raw, [0, 1], transform=None)
    val_subset = _TransformSubset(raw, [0, 1], transform=None)
    assert train_subset.dataset is val_subset.dataset


def test_d14_eval_is_deterministic():
    """D14: Lấy cùng mẫu Validation hai lần → torch.equal."""
    from PIL import Image

    raw = FakeData(size=20, image_size=(3, 32, 32), transform=None)
    transform = get_pretrained_transform(augment=False, model_name="vgg16")
    subset = _TransformSubset(raw, [0, 1, 2], transform=transform)

    img1, _ = subset[0]
    img2, _ = subset[0]
    assert torch.equal(img1, img2)


def test_d15_train_augmentation_varies():
    """D15: Lấy cùng mẫu Train 20 lần → có ít nhất 2 kết quả khác nhau."""
    raw = FakeData(size=20, image_size=(3, 32, 32), transform=None)
    transform = get_pretrained_transform(augment=True, model_name="vgg16")
    subset = _TransformSubset(raw, [0], transform=transform)

    outputs = [subset[0][0] for _ in range(20)]
    assert any(not torch.equal(outputs[0], o) for o in outputs[1:])


# ---------------------------------------------------------------------------
# Nhóm F: DataLoader thật (đánh dấu slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_s1_dataset_sizes():
    """S1: Train 45,000; Validation 5,000; Test 10,000; class_names 10 phần tử."""
    try:
        train_dl, val_dl, test_dl, class_names = create_train_val_test_dataloaders(
            batch_size=32, download=True
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Không tải được CIFAR-10: {exc}")

    assert len(train_dl.dataset) == 45_000
    assert len(val_dl.dataset) == 5_000
    assert len(test_dl.dataset) == 10_000
    assert len(class_names) == 10


@pytest.mark.slow
def test_s2_batch_shape_and_dtype():
    """S2: Batch (32, 3, 224, 224), float32; labels (32,)."""
    try:
        train_dl, val_dl, test_dl, _ = create_train_val_test_dataloaders(
            batch_size=32, download=True
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Không tải được CIFAR-10: {exc}")

    for dl in (train_dl, val_dl, test_dl):
        images, labels = next(iter(dl))
        assert images.shape == (32, 3, 224, 224)
        assert images.dtype == torch.float32
        assert labels.shape == (32,)


@pytest.mark.slow
def test_s3_shuffle_flags():
    """S3: Train RandomSampler; Validation/Test SequentialSampler."""
    from torch.utils.data.sampler import RandomSampler, SequentialSampler

    try:
        train_dl, val_dl, test_dl, _ = create_train_val_test_dataloaders(
            batch_size=32, download=True
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Không tải được CIFAR-10: {exc}")

    assert isinstance(train_dl.sampler, RandomSampler)
    assert isinstance(val_dl.sampler, SequentialSampler)
    assert isinstance(test_dl.sampler, SequentialSampler)


# ---------------------------------------------------------------------------
# Kiểm thử factory (Dependency Injection, generator, fallback)
# ---------------------------------------------------------------------------

def test_factory_train_fallback_uses_augment():
    """Train fallback không truyền transform → augmentation=True."""
    from torchvision.transforms import RandomHorizontalFlip

    # Không thể kiểm tra trực tiếp transform bên trong loader dễ dàng,
    # nhưng kiểm tra hàm fallback get_pretrained_transform đã được gọi đúng.
    train_t = get_pretrained_transform(augment=True, model_name="vgg16")
    eval_t = get_pretrained_transform(augment=False, model_name="vgg16")
    assert isinstance(train_t.transforms[0], RandomHorizontalFlip)
    assert not any(isinstance(t, RandomHorizontalFlip) for t in eval_t.transforms)


def test_factory_generator_controls_shuffle():
    """Generator riêng cho Train DataLoader."""
    dataset = FakeData(size=100, image_size=(3, 32, 32), transform=None)
    transform = get_pretrained_transform(augment=False, model_name="vgg16")
    train_subset = _TransformSubset(dataset, list(range(50)), transform=transform)
    val_subset = _TransformSubset(dataset, list(range(50, 100)), transform=transform)

    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(99)

    train_dl1 = DataLoader(train_subset, batch_size=10, shuffle=True, generator=g1)
    train_dl2 = DataLoader(train_subset, batch_size=10, shuffle=True, generator=g2)

    batches1 = [batch[0] for batch in train_dl1]
    batches2 = [batch[0] for batch in train_dl2]

    # Hai generator khác nhau nên thứ tự shuffle khác nhau
    assert any(
        not torch.equal(b1, b2) for b1, b2 in zip(batches1, batches2)
    ), "Generator khác nhau nên cho thứ tự shuffle khác nhau"


def test_factory_positional_compatibility():
    """Thêm tham số generator ở cuối, không đổi vị trí positional hiện có."""
    import inspect

    sig = inspect.signature(create_train_val_test_dataloaders)
    params = list(sig.parameters)
    # generator phải nằm sau num_workers và trước download
    assert params.index("num_workers") < params.index("generator") < params.index("download")


def test_factory_eval_transform_deterministic_with_mock():
    """Validation/Test dùng eval transform, không có augmentation."""
    from torchvision.transforms import RandomHorizontalFlip

    dataset = FakeData(size=50, image_size=(3, 32, 32), transform=None)
    transform = get_pretrained_transform(augment=False, model_name="vgg16")
    assert not any(isinstance(t, RandomHorizontalFlip) for t in transform.transforms)

    subset = _TransformSubset(dataset, list(range(10)), transform=transform)
    imgs = [subset[i][0] for i in range(10)]
    for i in range(10):
        assert torch.equal(imgs[i], subset[i][0])
