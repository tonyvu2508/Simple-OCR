# Các Hướng Cải Tiến Mô Hình Hybrid OCR (Tốc Độ & Kiến Trúc)

*Ngày lập tài liệu: 2026-07-01 17:00 (Local Time)*

Tài liệu này đề xuất các giải pháp nâng cấp kiến trúc và tối ưu hiệu năng (tốc độ, tài nguyên) cho mô hình Hybrid OCR (ConvNeXt + Transformer Decoder) dựa trên các nguyên lý tối ưu hóa tiên tiến của báo cáo kỹ thuật GLM-OCR.

---

## 1. Cơ Chế Dự Đoán Đa Token (Multi-Token Prediction - MTP)
*   **Hạn chế hiện tại:** Bộ giải mã Transformer Decoder đang thực hiện giải mã tự hồi quy truyền thống (Greedy Decoding) — mỗi bước chạy forward chỉ sinh ra **1 token duy nhất** ($t$).
*   **Đề xuất cải tiến:** 
    *   Tích hợp thêm các đầu dự đoán phụ (MTP heads) dùng chung tham số để mô hình dự đoán đồng thời $k$ token tiếp theo (ví dụ: $k=2$ hoặc $3$) ở mỗi bước giải mã.
    *   Huấn luyện mô hình với hàm mất mát MTP bổ sung.
*   **Hiệu quả dự kiến:** Giảm số bước chạy forward của Decoder đi từ 2 đến 3 lần, tăng tốc độ suy luận của mô hình nhận diện lên **gấp 1.5 - 2 lần** mà không làm tăng kích thước mô hình.

---

## 2. Nhận Dạng Song Song Hóa Vùng Chữ Tối Đa (Max Parallel Region Recognition)
*   **Hạn chế hiện tại:** Mô hình đang chạy nhận dạng với kích thước lô cố định (`batch_size=16`). Trên các GPU có băng thông rộng (như MPS của Mac Mini hoặc CUDA của RTX 4090), hiệu năng phần cứng chưa được khai thác triệt để.
*   **Đề xuất cải tiến:**
    *   **Dynamic Batching:** Tự động tối đa hóa kích thước batch (ví dụ: nâng lên `64` hoặc `128`) dựa trên giới hạn VRAM thực tế của phần cứng.
    *   **Bucketing & Padding Minimization:** Phân nhóm các ảnh crop có chiều rộng tương đồng vào cùng một batch trước khi đệm (padding) viền đen. Điều này giúp giảm thiểu việc tính toán trên các pixel đệm vô ích, tối ưu hóa tốc độ xử lý của GPU.

---

## 3. Học Tăng Cường (RLHF/GRPO) Cho Dữ Liệu Cấu Trúc
*   **Hạn chế hiện tại:** Việc ghép các vùng chữ đơn lẻ thành cấu trúc bảng biểu/dòng cột của phiếu đấu giá đang sử dụng thuật toán heuristic hình học (dễ bị sai lệch nếu tọa độ box lệch vài pixel).
*   **Đề xuất cải tiến:**
    *   Áp dụng thuật toán học tăng cường **GRPO (Group Score Policy Optimization)** để phạt các lỗi định dạng và thưởng cho các kết quả trích xuất cấu trúc Markdown/JSON hoàn chỉnh.
    *   Huấn luyện mô hình sinh trực tiếp văn bản có cấu trúc theo dòng thay vì nhận diện từng từ rời rạc.

---

## 4. Tích Hợp Mô-đun Phân Tích Bố Cục (Layout Parser) Tự Động
*   **Hạn chế hiện tại:** Pipeline đang gửi toàn bộ các box phát hiện được sang mô hình nhận dạng mà không phân loại trước, gây lãng phí tính toán.
*   **Đề xuất cải tiến:**
    *   Tích hợp bộ phân tích bố cục chuyên sâu (như **PP-DocLayout-V3** hoặc **YOLOv8-Layout**).
    *   Phân loại trước các vùng ảnh: 
        *   Các vùng chứa chữ in (Printed): Gửi sang mô hình Hybrid OCR siêu nhẹ (Lite model) để chạy cực nhanh.
        *   Các vùng chứa chữ viết tay (Handwritten): Gửi sang mô hình Hybrid OCR cấu hình sâu (Deep model) để đảm bảo độ chính xác.
