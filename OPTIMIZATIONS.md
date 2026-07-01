# Tối ưu hóa Huấn luyện Mô hình OCR trên GPU (L40 / RTX 3080 Ti)

Tài liệu này ghi chú lại các kỹ thuật tối ưu hóa đã được áp dụng để tăng tốc độ huấn luyện và tận dụng tối đa phần cứng GPU.

## 1. Automatic Mixed Precision (AMP) với `bfloat16`
- **Vấn đề:** Huấn luyện với độ chính xác mặc định (FP32) tốn nhiều VRAM và băng thông bộ nhớ, làm chậm quá trình tính toán.
- **Giải pháp:** Sử dụng AMP (`torch.cuda.amp.autocast`) kết hợp với `bfloat16`.
- **Lợi ích:**
  - Giảm một nửa dung lượng VRAM cần thiết cho các tensor (từ 32-bit xuống 16-bit).
  - Tăng tốc độ tính toán đáng kể trên các GPU kiến trúc mới (như Ada Lovelace trên L40 hoặc Ampere trên 3080 Ti) nhờ Tensor Cores.
  - `bfloat16` giữ nguyên dải giá trị (dynamic range) như FP32, giúp tránh hiện tượng tràn số (overflow/underflow) thường gặp với `float16`, làm cho quá trình huấn luyện ổn định hơn mà không cần `GradScaler`.

## 2. `torch.compile` (PyTorch 2.0+)
- **Vấn đề:** PyTorch thông thường thực thi theo kiểu eager (từng phép toán một), gây overhead do việc gọi kernel liên tục từ CPU xuống GPU.
- **Giải pháp:** Áp dụng `torch.compile(model)` trước khi bắt đầu vòng lặp huấn luyện.
- **Lợi ích:**
  - Tối ưu hóa đồ thị tính toán (kernel fusion), gộp nhiều phép toán nhỏ thành một kernel lớn, giảm thiểu chi phí chuyển đổi (overhead) và tăng tốc độ xử lý của GPU.

## 3. Loại bỏ Đồng bộ hóa CPU-GPU (CPU-GPU Sync Bottleneck)
- **Vấn đề:** Trong vòng lặp huấn luyện cũ, biến tính độ chính xác được lấy giá trị về CPU ở mỗi batch bằng lệnh `.item()` (ví dụ: `correct += (preds == labels).sum().item()`). Lệnh `.item()` buộc CPU phải chờ GPU hoàn thành tính toán (synchronization point), làm gián đoạn luồng công việc và khiến GPU bị "đói" dữ liệu (GPU utilization thấp, khoảng 10-20%).
- **Giải pháp:** 
  - Tích lũy độ chính xác (`correct`, `total`) trực tiếp trên GPU dưới dạng Tensor: `correct_t += (preds == labels).sum()` và `total_t += labels.numel()`.
  - Chỉ gọi `.item()` một lần sau mỗi 50 batch để in log tiến độ.
- **Lợi ích:**
  - Giúp GPU hoạt động liên tục (asynchronous execution) mà không bị chặn bởi CPU. GPU utilization có thể đạt mức cao (90-100%), tăng số iter/s lên rất nhiều lần.

## 4. Điều chỉnh siêu tham số (Hyperparameters Tuning)
- **Số luồng đọc dữ liệu (`num_workers`):**
  - Đã thêm cấu hình `num_workers` vào `configs/recognition.yaml` và tham số dòng lệnh.
  - Tăng `num_workers` (ví dụ: 4 hoặc 8) giúp CPU đọc dữ liệu từ ổ cứng và chuẩn bị batch nhanh hơn, cung cấp dữ liệu kịp thời cho GPU.
- **Learning Rate (Tránh bùng nổ gradient):**
  - **Vấn đề:** Ở các epoch đầu, loss có thể tăng đột biến (NaN hoặc loss rất cao) do gradient bùng nổ, làm cho việc học bị hỏng (CER/Accuracy không cải thiện).
  - **Giải pháp:** Giảm `learning_rate` ban đầu từ `0.001` xuống `0.0003` trong file cấu hình. Kết hợp với việc mô hình có các `warmup_epochs`, điều này giúp quá trình hội tụ mượt mà và ổn định hơn.

## Kết quả
Nhờ áp dụng các biện pháp trên, tốc độ huấn luyện (`it/s`) tăng lên đáng kể và tận dụng được toàn bộ sức mạnh của GPU (GPU utilization cao thay vì chỉ 10-20% như ban đầu).
