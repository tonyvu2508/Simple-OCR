# Các Hướng Cải Thiện Độ Chính Xác Cho Hybrid OCR (Accuracy Optimization)

Tài liệu này tổng hợp các giải pháp khoa học dữ liệu và kỹ thuật mô hình nhằm tối ưu hóa độ chính xác nhận diện ký tự (giảm thiểu CER/WER), đặc biệt đối với chữ viết tay tiếng Nhật (Handwritten Japanese) trên các phiếu đấu giá xe.

---

## 1. Nâng Cấp Chất Lượng Dữ Liệu Huấn Luyện (Data-Centric OCR)
Dữ liệu huấn luyện quyết định phần lớn độ chính xác của mô hình nhận diện chữ viết tay.

*   **Tăng Cường Sinh Dữ Liệu Giả Lập (Advanced Synthetic Augmentations):**
    *   *Mô phỏng chất lượng ảnh scan thực tế:* Tích hợp thêm các bộ lọc làm nhòe ảnh (Blurring), nhiễu hạt (Gaussian Noise), bóng đổ (Shadows), và biến dạng phối cảnh (Perspective Transform) vào script `synthetic_data.py`.
    *   *Mô phỏng nét bút viết tay thực tế:* Bổ sung hiệu ứng đứt quãng của nét mực (ink bleeding) và độ mờ của bút chì/bút bi nhạt màu.
    *   *Mở rộng phông chữ:* Sử dụng thêm ít nhất 20-30 bộ phông chữ viết tay tiếng Nhật (Japanese Handwritten Fonts) khác nhau trong giai đoạn Pre-train.
*   **Mở Rộng Dữ Liệu Thực Tế (Real-World Dataset Expansion):**
    *   Khai thác công cụ gán nhãn tự động bằng **GLM-OCR** (`--labeler glmocr`) vừa được tích hợp để trích xuất dữ liệu tự động từ hàng trăm trang PDF thật.
    *   Thực hiện hậu kiểm (labels checking) thủ công để đảm bảo 100% độ chính xác của tập dữ liệu fine-tune thực tế trước khi đưa vào huấn luyện Stage 2.

---

## 2. Giải Phóng Bộ Mã Hóa Ảnh (Unfreeze Encoder) Khi Fine-Tuning
*   **Hạn chế hiện tại:** Nếu đóng băng (freeze) bộ mã hóa ConvNeXt Encoder, mô hình nhận dạng chỉ sử dụng đặc trưng học từ tập dữ liệu tổng quát ImageNet, dẫn đến việc không nhạy bén với nét viết tay nhỏ/mờ của tiếng Nhật.
*   **Giải pháp cải tiến:**
    *   Bật tùy chọn `--unfreeze-encoder` trong quá trình Fine-tuning để mở khóa toàn bộ trọng số của ConvNeXt.
    *   Sử dụng tốc độ học cực nhỏ (`lr=1e-6` hoặc `lr=5e-7`) để cập nhật nhẹ nhàng các trọng số của Encoder mà không làm phá hủy các đặc trưng tốt đã học, giúp mô hình thích nghi sâu sắc với chất lượng ảnh quét thực tế.

---

## 3. Điều Chỉnh Khoảng Đệm Tránh Mất Nét Chữ (Layout-Guided Padding)
*   **Hạn chế hiện tại:** Nếu mô hình phát hiện vùng chữ (Detector) cắt quá sát rìa, chữ có thể bị mất một phần nét ngoài (ví dụ: chữ `日` bị cắt sát trông giống chữ `口`). Đầu vào bị lỗi hình học sẽ khiến mô hình nhận dạng dự đoán sai hoàn toàn.
*   **Giải pháp cải tiến:**
    *   Tăng tham số khoảng đệm rìa `crop_padding` khi cắt dòng chữ (ví dụ: nâng từ `5px` lên `8px` hoặc `10px`).
    *   Sử dụng mô hình phát hiện chuyên dụng cho tài liệu như **Surya** để đảm bảo bounding box bao phủ trọn vẹn 100% các ký tự dòng chữ.

---

## 4. Tích Hợp Bộ Sửa Lỗi Chính Tả Ngôn Ngữ (Language Model Post-Correction)
*   **Lý do:** Đối với các nét chữ viết tay bị mất nét hoặc bị nhòe hoàn toàn, mô hình thị giác (Vision) đơn thuần rất dễ đoán sai.
*   **Giải pháp cải tiến:**
    *   Tích hợp một mô hình ngôn ngữ tiếng Nhật nhỏ (như RoBERTa-Japanese) làm bộ hậu lọc.
    *   Bộ hậu lọc này sẽ nhận diện ngữ cảnh của dòng chữ và sửa các lỗi chính tả phổ biến trong văn cảnh đấu giá xe (ví dụ: tự động sửa cụm từ sai chính tả `"冷黒"` thành từ đúng ngữ cảnh `"冷房"` - hệ thống điều hòa).
