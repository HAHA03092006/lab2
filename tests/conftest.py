"""Cấu hình chung cho bộ kiểm thử Lab 2.

* Đăng ký marker ``slow`` cho test cần dữ liệu thật, mạng hoặc pretrained weights.
* Đặt Matplotlib backend ``Agg`` (không tương tác) trước khi các module vẽ được import.
* Đảm bảo package ``going_modular`` được import ổn định khi pytest chạy từ repository root.
"""

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: tests that download CIFAR-10 or ImageNet weights")
