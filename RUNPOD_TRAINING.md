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
    --num-workers 8
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
    --num-workers 8
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

---

## 5. Huấn Luyện Fine-tuning trên Dữ Liệu Thực Tế

Sau khi mô hình đã được huấn luyện Pre-train đạt độ chính xác tốt trên dữ liệu tổng hợp (synthetic), bạn tiến hành các bước sau để fine-tune mô hình trên dữ liệu thực tế cắt ra từ các file PDF thật.

### Bước 1: Trích xuất và Tự động Gán Nhãn Dữ Liệu Thật
Sử dụng script [extract_fine_tune_data.py](file:///Volumes/SpaceX/WorkSpace/python/Simple-OCR/scratch/extract_fine_tune_data.py) để tự động hóa việc đọc PDF, cắt ảnh box (deskewed) và gán nhãn tự động bằng bộ nhận diện PaddleOCR (vô cùng chính xác và nhanh chóng):

```bash
# Trích xuất dữ liệu từ trang PDF thật (ví dụ: lấy 5 trang đầu)
python -m scratch.extract_fine_tune_data \
    --pdf pdfs/2026年6月25日-JU愛知-2163-通常車-151-200.pdf \
    --output-dir data/real_fine_tune \
    --max-pages 5
```
*Dữ liệu ảnh crop sẽ lưu tại `data/real_fine_tune/images/` và nhãn lưu tại `data/real_fine_tune/labels.json`.*

### Bước 2: Hậu Kiểm / Chỉnh Sửa Nhãn (Tùy chọn)
Mở tệp `data/real_fine_tune/labels.json` để kiểm tra nhanh. Do được gán nhãn tự động từ mô hình OCR mạnh của Paddle, tỷ lệ chính xác rất cao. Bạn chỉ cần điều chỉnh lại một số ít chữ viết tay quá mờ hoặc bị lỗi nhận diện trước khi bắt đầu huấn luyện.

### Bước 3: Huấn Luyện Fine-tuning
> [!IMPORTANT]
> **ĐỒNG BỘ TỆP TỪ VỰNG (VOCABULARY SYNC):**
> Trước khi khởi chạy lệnh fine-tune với thư mục output riêng biệt (`--output runs/finetune`), bạn **bắt buộc** phải sao chép tệp `vocab.json` từ thư mục pre-train sang thư mục đầu ra mới.
> Nếu không thực hiện, script sẽ tự động sinh lại một bộ từ vựng mới làm thay đổi toàn bộ index ký tự, khiến mô hình bị lỗi nghiêm trọng (Accuracy cực thấp và CER > 100%).
> 
> Chạy lệnh sau để đồng bộ trước:
> ```bash
> mkdir -p runs/finetune && cp runs/recognition/vocab.json runs/finetune/vocab.json
> ```

Chạy lệnh huấn luyện với tham số `--stage finetune` và trỏ `--checkpoint` tới mô hình tốt nhất thu được ở giai đoạn pre-train:

```bash
python -m src.hybrid_ocr.train.train_recognizer \
    --config configs/recognition.yaml \
    --train-data data/real_fine_tune \
    --val-data data/real_fine_tune \
    --stage finetune \
    --checkpoint runs/recognition/model_best.pt \
    --output runs/finetune \
    --num-workers 8
```
*(Lưu ý: Bằng việc thêm cờ `--output runs/finetune`, các checkpoint trong quá trình fine-tune sẽ được lưu riêng biệt tại thư mục `runs/finetune/` và không đè lên các checkpoint của giai đoạn pre-train trước đó. Có thể giảm `learning_rate` xuống nhỏ hơn nữa trong cấu hình `finetune` tại `configs/recognition.yaml` để quá trình học chuyển tiếp diễn ra mượt mà).*

### Bước 4: Tinh Chỉnh Sâu Hơn (Stage 2 Fine-tuning - Mở khóa Encoder)
Sau khi kết thúc 100 epoch với Encoder được đóng băng, nếu bạn muốn tinh chỉnh sâu hơn cả bộ trích xuất ảnh ConvNeXt để đạt độ chính xác tối đa, hãy chạy lượt fine-tune ngắn (ví dụ: 20 epoch) từ checkpoint tốt nhất của Stage 1 bằng cách sử dụng các tham số ghi đè dòng lệnh sau:

```bash
python -m src.hybrid_ocr.train.train_recognizer \
    --config configs/recognition.yaml \
    --train-data data/real_fine_tune \
    --val-data data/real_fine_tune \
    --stage finetune \
    --checkpoint runs/finetune/model_best.pt \
    --output runs/finetune_stage2 \
    --num-workers 8 \
    --unfreeze-encoder \
    --lr 0.000001 \
    --epochs 20
```

**Chi tiết các cờ linh hoạt được bổ sung:**
*   `--unfreeze-encoder`: Ép buộc mở khóa toàn bộ trọng số của bộ xương ConvNeXt Encoder (mặc định cấu hình fine-tune của yaml là đóng băng).
*   `--lr 0.000001` (`1e-6`): Đặt tốc độ học siêu nhỏ để tránh phá hủy các trọng số tốt đã học.
*   `--epochs 20`: Chỉ chạy thêm 20 epoch ngắn.
*   `--output runs/finetune_stage2`: Lưu checkpoint kết quả cuối cùng ra một thư mục riêng biệt.

---

## 6. Chạy Suy Diễn / Dự Đoán (Inference Pipeline)

Script `src/hybrid_ocr/pipeline.py` cung cấp đường ống đầu-cuối từ đọc tệp PDF, cắt dòng chữ, nhận diện và lưu kết quả cấu trúc hóa. Bạn có thể sử dụng các tham số sau để tối ưu hóa tốc độ và chất lượng:

### Lệnh chạy mẫu:
```bash
python -m src.hybrid_ocr.pipeline \
    --pdf pdfs/2026年6月25日-JU愛知-2163-通常車-151-200.pdf \
    --yolo-model yolov8s.pt \
    --rec-model runs/finetune_stage2/model_best.pt \
    --output runs/inference_test \
    --pages 0-1 \
    --use-clahe \
    --detector surya
```

### Các cờ tùy chọn cấu hình:

| Tham số | Giá trị | Mô tả |
| :--- | :--- | :--- |
| `--detector` | `paddle` (mặc định) \| `yolo` \| `surya` | Chọn mô hình phát hiện khung chữ. `surya` cho chất lượng tài liệu tốt nhất, `yolo` cho tốc độ nhanh nhất trên Mac Mini (MPS). |
| `--rec-model` | Đường dẫn `.pt` \| `.onnx` \| `mangaocr` \| `glmocr` | Mô hình nhận dạng chữ. Bạn có thể truyền checkpoint PyTorch gốc, tệp ONNX hoặc gõ chữ `mangaocr`/`glmocr` để dùng mô hình nhận dạng tương ứng. |
| `--use-clahe` | Không cần giá trị (cờ bật) | Kích hoạt bộ lọc cân bằng ánh sáng cục bộ CLAHE giúp cải thiện độ tương phản cho nét chữ viết tay mờ/nhạt màu. |
| `--pages` | Ví dụ: `0-4`, `0` hoặc `all` | Chỉ định các trang cần xử lý (bắt đầu từ 0). |
| `--output` | Đường dẫn thư mục | Thư mục lưu tệp kết quả JSON và ảnh vẽ Bounding Box. |



