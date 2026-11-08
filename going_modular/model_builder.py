
from __future__ import annotations

from dataclasses import dataclass, fields
from torch import nn
from torchvision import models
from torchvision.models import VGG16_Weights, ResNet18_Weights


def build_model(cfg: ModelConfig) -> nn.Module:
    """Xây dựng mô hình torchvision trực tiếp từ ModelConfig (không wrapper)."""
    return _build_model_from_config(cfg)


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:

    
    arch: str = "resnet18"
    num_classes: int = 10
    weights: str | None = "IMAGENET1K_V1"
    freeze_backbone: bool = True
    unfreeze_last_n: int = 0
    dropout: float = 0.0
    freeze_bn_stats: bool = True

    def to_dict(self) -> dict:
        """Chuyển config thành dict (phục vụ lưu config.json)."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        """Tạo ModelConfig từ dict; từ chối khóa không xác định bằng TypeError."""
        valid_names = {f.name for f in fields(cls)}
        unknown = set(d) - valid_names
        if unknown:
            raise TypeError(
                f"Khóa không được định nghĩa trong ModelConfig: {sorted(unknown)}"
            )
        return cls(**d)


# ---------------------------------------------------------------------------
# Architecture registry
# ---------------------------------------------------------------------------

def _vgg16_conv_blocks(model: nn.Module) -> list[nn.Sequential]:
    """Chia VGG16 features thành 5 khối conv, mỗi khối kết thúc bằng MaxPool2d."""
    blocks: list[nn.Sequential] = []
    current: list[nn.Module] = []
    for module in model.features:
        current.append(module)
        if isinstance(module, nn.MaxPool2d):
            blocks.append(nn.Sequential(*current))
            current = []
    return blocks


ARCH_SPECS: dict[str, dict] = {
    "resnet18": {
        "get_head": lambda m: m.fc,
        "set_head": lambda m, h: setattr(m, "fc", h),
        "blocks": lambda m: [m.layer1, m.layer2, m.layer3, m.layer4],
        "num_blocks": 4,
    },
    "vgg16": {
        "get_head": lambda m: m.classifier[6],
        "set_head": lambda m, h: setattr(m.classifier, "6", h),
        "blocks": _vgg16_conv_blocks,
        "num_blocks": 5,
    },
}

_WEIGHT_ENUMS: dict[str, type] = {
    "resnet18": ResNet18_Weights,
    "vgg16": VGG16_Weights,
}


def _resolve_weights(arch: str, weights_name: str | None):
    """Ánh xạ tên trọng số (str) thành torchvision WeightsEnum member."""
    if weights_name is None:
        return None
    enum_cls = _WEIGHT_ENUMS.get(arch)
    if enum_cls is None:
        raise ValueError(f"Không hỗ trợ kiến trúc '{arch}' để giải quyết trọng số.")
    try:
        return enum_cls[weights_name]
    except (KeyError, AttributeError) as exc:
        raise ValueError(
            f"Không xác định được trọng số '{weights_name}' cho kiến trúc '{arch}'."
        ) from exc


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(cfg: ModelConfig) -> None:
    if cfg.arch not in ARCH_SPECS:
        raise ValueError(
            f"Kiến trúc không hỗ trợ: '{cfg.arch}'. "
            f"Các kiến trúc hợp lệ: {sorted(ARCH_SPECS)}."
        )
    if cfg.unfreeze_last_n < 0:
        raise ValueError(f"unfreeze_last_n không được âm (đã nhận {cfg.unfreeze_last_n}).")
    if not (0.0 <= cfg.dropout < 1.0):
        raise ValueError(
            f"dropout phải nằm trong [0, 1) (đã nhận {cfg.dropout})."
        )
    num_blocks = ARCH_SPECS[cfg.arch]["num_blocks"]
    if cfg.unfreeze_last_n > num_blocks:
        raise ValueError(
            f"unfreeze_last_n={cfg.unfreeze_last_n} vượt quá số khối của "
            f"'{cfg.arch}' ({num_blocks})."
        )
    if not cfg.freeze_backbone and cfg.unfreeze_last_n > 0:
        raise ValueError(
            "Không thể kết hợp freeze_backbone=False với unfreeze_last_n>0. "
            "Dùng freeze_backbone=True, unfreeze_last_n=N cho Fine-Tuning."
        )


def _load_torchvision(arch: str, weights_name: str | None) -> nn.Module:
    """Nạp kiến trúc torchvision với weights enum cụ thể (không dùng pretrained=True/DEFAULT)."""
    if arch == "resnet18":
        return models.resnet18(weights=_resolve_weights(arch, weights_name))
    if arch == "vgg16":
        return models.vgg16(weights=_resolve_weights(arch, weights_name))
    raise ValueError(f"Kiến trúc không hỗ trợ: '{arch}'.")


# ---------------------------------------------------------------------------
# build_model (new API)
# ---------------------------------------------------------------------------

def _build_model_from_config(cfg: ModelConfig) -> nn.Module:
    """Xây dựng mô hình từ ModelConfig theo hợp đồng thiết kế.

    Thứ tự: validate → nạp kiến trúc → freeze backbone → thay head → unfreeze N khối cuối.
    """
    _validate(cfg)

    model = _load_torchvision(cfg.arch, cfg.weights)

    spec = ARCH_SPECS[cfg.arch]

    # Bước 1: đóng băng toàn bộ backbone nếu được yêu cầu.
    if cfg.freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Bước 2: thay head (mới tạo mặc định requires_grad=True).
    old_head = spec["get_head"](model)
    in_features = old_head.in_features

    if cfg.dropout > 0:
        new_head = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(in_features, cfg.num_classes),
        )
    else:
        new_head = nn.Linear(in_features, cfg.num_classes)

    spec["set_head"](model, new_head)

    # Bước 3: mở khóa N khối cuối (Fine-Tuning).
    if cfg.unfreeze_last_n > 0:
        blocks = spec["blocks"](model)
        for block in blocks[-cfg.unfreeze_last_n:]:
            for param in block.parameters():
                param.requires_grad = True

    return model


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_transforms(cfg: ModelConfig, weights: str | None = None):

    weights_name = weights if weights is not None else cfg.weights
    if weights_name is None:
        raise ValueError(
            "Không xác định được trọng số để lấy transform. "
            "Truyền weights vào get_transforms hoặc đặt cfg.weights."
        )
    from .data_setup import get_pretrained_transform

    return get_pretrained_transform(
        model_name=cfg.arch,
        weights_name=weights_name,
    )


# ---------------------------------------------------------------------------
# Train mode (BatchNorm handling)
# ---------------------------------------------------------------------------

def set_train_mode(model: nn.Module, cfg: ModelConfig) -> None:

    model.train()
    if cfg.freeze_bn_stats:
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                all_frozen = all(
                    not p.requires_grad for p in module.parameters(recurse=False)
                )
                if all_frozen:
                    module.eval()


# ---------------------------------------------------------------------------
# Parameter groups for optimizer
# ---------------------------------------------------------------------------

def _head_param_ids(model: nn.Module, cfg: ModelConfig) -> set[int]:
    head = ARCH_SPECS[cfg.arch]["get_head"](model)
    return {id(p) for p in head.parameters()}


def get_param_groups(
    model: nn.Module,
    cfg: ModelConfig,
    lr_head: float,
    lr_backbone: float,
) -> list[dict]:
    """Tạo nhóm tham số cho optimizer với LR riêng cho head và backbone.

    Group 1: tham số trainable của head (lr_head).
    Group 2 (chỉ khi có backbone trainable): tham số trainable còn lại (lr_backbone).
    Không tạo nhóm rỗng, không trùng, không bỏ sót.
    """
    head_ids = _head_param_ids(model, cfg)

    head_params = []
    backbone_params = []

    for p in model.parameters():
        if not p.requires_grad:
            continue
        if id(p) in head_ids:
            head_params.append(p)
        else:
            backbone_params.append(p)

    groups: list[dict] = []
    if head_params:
        groups.append({"params": head_params, "lr": lr_head})
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr_backbone})

    return groups


# ---------------------------------------------------------------------------
# Parameter statistics
# ---------------------------------------------------------------------------

def count_params(model: nn.Module) -> dict:
    """Trả về {total, trainable, frozen} cho mô hình."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


# ---------------------------------------------------------------------------
# Model summary
# ---------------------------------------------------------------------------

def describe_model(model: nn.Module, cfg: ModelConfig | None = None) -> str:
    """Tóm tắt kiến trúc: head, tổng/trainable/frozen parameters.

    Không phụ thuộc torchinfo, không tải pretrained weights chỉ để mô tả.
    """
    stats = count_params(model)
    lines = [
        f"Model: {model.__class__.__name__}",
        f"Total parameters: {stats['total']:,}",
        f"Trainable parameters: {stats['trainable']:,}",
        f"Frozen parameters: {stats['frozen']:,}",
    ]
    if cfg is not None:
        lines.insert(1, f"Architecture: {cfg.arch}")
        lines.insert(2, f"Classes: {cfg.num_classes}")
    return "\n".join(lines)



