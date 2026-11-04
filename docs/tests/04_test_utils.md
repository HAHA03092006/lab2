# Đặc tả Kiểm thử — `utils.py`

> Nguồn: `docs/define&designt/04_utils.md` mục 2.8 (Kế hoạch kiểm thử `tests/test_utils.py`).

## 1. Mục tiêu

Kiểm chứng `utils.py` — module cung cấp các tiện ích dùng chung cho toàn pipeline (seed, device, TensorBoard writer, lưu/nạp model và artifacts).

## 2. Bảng tóm tắt mã test

| Mã | Kiểm thử | Kỳ vọng |
| :--- | :--- | :--- |
| U1 | Thiết lập cùng seed hai lần, sinh số Python/NumPy/PyTorch | Chuỗi số lặp lại trên cùng môi trường |
| U2 | Chọn `cpu`, thiết bị không hợp lệ và thiết bị không khả dụng | Trả `torch.device("cpu")`; input không hỗ trợ báo `ValueError` |
| U3 | Lưu/nạp history và config có Unicode | Round-trip giữ nguyên dữ liệu |
| U4 | Lưu/nạp `state_dict` vào model cùng kiến trúc | Trọng số và đầu ra bằng nhau |
| U5 | Giả lập lỗi giữa atomic write trên file đã tồn tại | File đích còn nguyên; không còn file tạm |
| U6 | Kiểm tra `artifacts_exist` khi thiếu lần lượt từng artifact bắt buộc | Chỉ `True` khi đủ checkpoint tốt nhất, history và config |
| U7 | Nạp artifacts từ thư mục thiếu file/schema sai | Báo lỗi cụ thể, không trả kết quả mặc định |
| U8 | Nạp payload `.pth` chứa đối tượng Python tùy ý | Bị từ chối bởi chế độ nạp an toàn |
| U9 | Tạo writer, ghi scalar, flush | Có event file dưới `runs/<experiment>` |
| U10 | Lưu/nạp resume checkpoint | Khôi phục model, optimizer, epoch, score, history chính xác |
| U11 | Lưu metrics JSON và đọc lại | JSON serialize/round-trip thành công; không tạo metrics rỗng khi chưa evaluate |

## 3. Cách chạy

```bash
pytest tests/test_utils.py -v
```

## 4. Ngoài phạm vi

* Lưu và quản lý checkpoint bên ngoài filesystem của run (ví dụ tự động đồng bộ cloud).
* Mã hóa/định danh artifact và xác thực chữ ký.
* Đồng bộ RNG toàn trạng thái để tiếp tục đúng từng sample giữa các phiên bản phần cứng/phần mềm khác nhau.
