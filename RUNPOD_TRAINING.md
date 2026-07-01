# Hướng Dẫn Huấn Luyện Hybrid OCR (RunPod / Ubuntu)

Tài liệu này tổng hợp toàn bộ các lệnh cần thiết để thiết lập môi trường, sinh dữ liệu và huấn luyện mô hình ConvNeXt + Transformer (Hybrid OCR) từ đầu trên các server GPU như RunPod.

---

## 1. Cài Đặt Môi Trường Ban Đầu

Khi khởi tạo xong Pod (khuyên dùng các mẫu có sẵn PyTorch 2.x), chạy các lệnh sau tại Terminal để kéo code và cài đặt thư viện:

```bash
# 1. Kéo code mới nhất từ Github
git clone https://github.com/tonyvu2508/Simple-OCR.git
cd Simple-OCR

# 2. Tạo và kích hoạt môi trường ảo (Virtual Environment)
python -m venv venv
source venv/bin/activate

# 3. Cài đặt các thư viện bắt buộc
pip install -r requirements_hybrid.txt

# 4. Cài đặt phông chữ tiếng Nhật (BẮT BUỘC TRÊN RUNPOD/UBUNTU)
# Để Pillow có thể vẽ được chữ tiếng Nhật khi sinh dữ liệu tổng hợp
apt-get update && apt-get install -y fonts-noto-cjk fonts-ipafont-gothic
```

---

## 2. Sinh Dữ Liệu Tổng Hợp (Synthetic Data)

Mô hình Transformer cần hàng trăm ngàn ảnh mẫu để học được cấu trúc chữ tiếng Nhật. Chạy lệnh sau để tạo thư mục dữ liệu:

```bash
# Sinh tập Train (Ví dụ: 270,000 mẫu) - Sẽ tốn khoảng 5-10 phút
python -m src.hybrid_ocr.dataset.synthetic_data --num-samples 300000 --output-dir data/synth_train

# Sinh tập Validation (Ví dụ: 30,000 mẫu)
python -m src.hybrid_ocr.dataset.synthetic_data --num-samples 30000 --output-dir data/synth_val
```

*Dữ liệu sinh ra sẽ được tự động lưu trong thư mục `data/` với định dạng tương thích (gồm folder `images/` và tệp `annotations.json`).*

---

## 3. Huấn Luyện (Pre-training)

### Lệnh chạy lần đầu (Train from scratch)
Chạy lệnh sau để bắt đầu học từ số 0. Cấu hình mặc định (Batch size 512) yêu cầu GPU có tối thiểu 16GB-24GB VRAM (như RTX 3090, 4090, L4).

```bash
python -m src.hybrid_ocr.train.train_recognizer \
    --config configs/recognition.yaml \
    --train-data data/synth_train \
    --val-data data/synth_val \
    --stage pretrain \
    --num-workers 4
```

### Lệnh chạy tiếp tục (Resume Training)
Nếu bạn bị đứt kết nối, lỗi máy chủ, hoặc đổi GPU, bạn có thể chạy tiếp tục bằng cách thêm cờ `--checkpoint`:

```bash
python -m src.hybrid_ocr.train.train_recognizer \
    --config configs/recognition.yaml \
    --train-data data/synth_train \
    --val-data data/synth_val \
    --stage pretrain \
    --checkpoint runs/recognition/model_last.pt \
    --num-workers 4
```

> [!TIP]
> **Tối ưu hóa Tốc độ với `num_workers`:**
> - Bạn có thể cấu hình nhanh tham số này trực tiếp trong file cấu hình tại dòng `num_workers: 4` ở phần `# Common` của tệp `configs/recognition.yaml`.
> - Trên máy cấu hình yếu hoặc Docker giới hạn (như một số GPU giá rẻ), hãy để `num_workers: 0` để tránh lỗi `Bus Error`.
> - Trên các GPU mạnh mẽ có lượng bộ nhớ chia sẻ lớn (như L40, RTX 4090 trên RunPod), hãy tăng lên `num_workers: 4` hoặc `num_workers: 8` để nạp dữ liệu song song từ CPU lên GPU, giúp tăng tốc độ huấn luyện lên gấp nhiều lần!
> - *(Bạn vẫn có thể ghi đè giá trị này lúc chạy lệnh bằng cách thêm cờ `--num-workers <số_worker>` vào câu lệnh)*

*Trong quá trình học, những checkpoint tốt nhất (CER thấp nhất) sẽ được tự động lưu tại `models/recognition/model_best.pt`.*

---

## 4. Kiểm Thử Trực Quan Mô Hình (Inference)

Để kiểm tra xem mô hình đang đọc được chữ như thế nào (thay vì chỉ nhìn vào các con số Loss), bạn có thể tạo một script nhanh:

Tạo tệp `eval_test.py`:
```python
import torch
from src.hybrid_ocr.recognition.model import HybridOCR
from src.hybrid_ocr.dataset.vocabulary import Vocabulary
from src.hybrid_ocr.dataset.dataset import OCRDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
vocab = Vocabulary.load("runs/recognition/vocab.json")
model = HybridOCR.load_checkpoint("models/recognition/model_best.pt", vocab_size=vocab.size, device=device)
model.eval()

# Load vài chục ảnh từ tập Val để test
dataset = OCRDataset("data/synth_val", vocab, is_train=False)

print(f"{'NHÃN ĐÚNG':<30} | {'MÔ HÌNH ĐOÁN':<30}")
print("-" * 65)

with torch.no_grad():
    for i in range(10): # Lấy 10 ảnh đầu
        image = dataset[i]["image"].unsqueeze(0).to(device)
        preds = model.predict(image, vocab, decoding="greedy")
        print(f"{dataset.samples[i]['label']:<30} | {preds[0]['text']:<30}")
```

Sau đó chạy:
```bash
python eval_test.py
```
