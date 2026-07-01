# Tổng quan Dự án: Hybrid OCR (Nhận diện chữ viết tiếng Nhật)

Dự án này là hệ thống Nhận diện Ký tự Quang học (OCR) lai (Hybrid OCR), được thiết kế đặc biệt để nhận diện chữ tiếng Nhật từ các tài liệu đấu giá, sử dụng các kiến trúc mạng nơ-ron hiện đại.

## 1. Kiến trúc Mô hình (Model Architecture)
- **Encoder:** Sử dụng **ConvNeXt** để trích xuất đặc trưng hình ảnh (image feature extraction).
- **Decoder:** Sử dụng mô hình chuỗi **Transformer (Seq2Seq)** để sinh ra chuỗi ký tự dự đoán dựa trên các đặc trưng được trích xuất.
- **Phương pháp huấn luyện:** Sử dụng kỹ thuật *Teacher Forcing* trong quá trình huấn luyện và *Greedy Decoding* trong quá trình suy luận (evaluation).

## 2. Cấu trúc Thư mục Chính
- `src/hybrid_ocr/`: Mã nguồn chính của mô hình (Encoder, Decoder, Dataset).
- `src/hybrid_ocr/train/`: Chứa các script liên quan đến huấn luyện (vd: `train_recognizer.py`).
- `configs/`: Chứa các file cấu hình huấn luyện. File cấu hình chính là [`configs/recognition.yaml`](file:///Volumes/SpaceX/WorkSpace/python/Simple-OCR/configs/recognition.yaml).

## 3. Tài liệu Tham khảo Kỹ thuật
Để theo dõi các hướng dẫn và kỹ thuật đã áp dụng cho dự án, vui lòng tham khảo các tài liệu sau:

*   🚀 **[Tối ưu hóa Huấn luyện GPU (OPTIMIZATIONS.md)](file:///Volumes/SpaceX/WorkSpace/python/Simple-OCR/OPTIMIZATIONS.md)**
    *   Tài liệu ghi chú các kỹ thuật tối ưu hóa để tận dụng 100% sức mạnh GPU (L40, 3080 Ti), bao gồm:
        *   `bfloat16` AMP (Automatic Mixed Precision).
        *   `torch.compile()` cho kernel fusion (PyTorch 2.0+).
        *   Loại bỏ độ trễ đồng bộ hóa CPU-GPU (CPU-GPU Sync Bottleneck).
        *   Điều chỉnh Hyperparameters (`num_workers`, `learning_rate`).

*   ☁️ **[Hướng dẫn chạy trên RunPod (RUNPOD_TRAINING.md)](file:///Volumes/SpaceX/WorkSpace/python/Simple-OCR/RUNPOD_TRAINING.md)**
    *   Hướng dẫn thiết lập môi trường, cài đặt thư viện (`requirements_hybrid.txt`), chuẩn bị dữ liệu và các lệnh terminal để bắt đầu quá trình pretrain trên máy chủ RunPod.
