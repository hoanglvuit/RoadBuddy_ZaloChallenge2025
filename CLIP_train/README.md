## CLIP_train

Thư mục này chứa toàn bộ pipeline fine-tune CLIP cho bài toán chọn frame/video phù hợp với câu hỏi (caption), bao gồm:

- **Khai thác frame dương/âm từ video**
- **Chuẩn bị dataset cho CLIP**
- **Huấn luyện (fine-tune) CLIP với positive/negative frames**
- **Đánh giá bằng DINOv2**
- **Suy luận (chọn top-4 frames liên quan nhất với caption)**  

---

### 1. Cấu trúc thư mục

- **`in_video_negative_mining.py`**:  
  - Đọc `train/train.json` trong folder dataset gốc.  
  - Dùng `support_frames` để trích xuất **true frames**.  
  - Chia timeline video thành các đoạn, chọn frame ngẫu nhiên từng đoạn và lọc bằng cosine similarity với true frames để lấy **negative frames**.  
  - Lưu kết quả vào:
    - `dataset/frames_true/<id>/...png`
    - `dataset/frames_neg/<id>/...png`

- **`dataset.py`**:  
  - Hàm **`load_data`**:
    - Đọc JSON QA (field `data`), ghép caption theo `setting`:
      - `1`: chỉ `question`
      - `2`: chỉ `choices`
      - `3`: `question + choices`
    - Tạo:
      - **`train_list`**: `pos_path`, `neg_path`, `caption`
      - **`val_list`**: `caption`, `video_path`, `pos_path`
  - Lớp **`PosNegDataset`**: load list ảnh từ thư mục `frames_true` / `frames_neg`.  
  - **`custom_collate_fn`**: gom batch theo dạng list (caption, pos_imgs, neg_imgs).

- **`model.py`**:  
  - Hàm **`load_model_to_train`**: load `CLIPModel` + `CLIPProcessor` (mặc định `openai/clip-vit-large-patch14`).

- **`loss.py`**: các biến thể loss cho CLIP:
  - **`loss_setting = 1`**: BCE với **positive + negative** (label pos=1, neg=0).  
  - **`loss_setting = 2`**: BCE **chỉ positive**.  
  - **`loss_setting = 3`**: **contrastive loss chuẩn CLIP** (cross-entropy trên similarity matrix).

- **`train.py`**:  
  - Dùng `load_data` + `PosNegDataset` để tạo dataloader.  
  - Fine-tune CLIP:
    - Optimizer: `AdamW(lr=1e-5)`  
    - Scheduler: `get_cosine_schedule_with_warmup`  
    - Gradient accumulation: `accumulation_steps = 16`, `batch_size = 1`.  
    - `loss_setting` có thể chọn qua argument.  
  - Lưu model + processor vào `saved_models/...`.  
  - Nếu có `val_list`: dùng DINOv2 (`facebook/dinov2-large`) + hàm `sim_m` để đánh giá top-4 frames.

- **`evaluate.py`**:  
  - Hàm **`sim_m`**:  
    - Encode ảnh bằng DINOv2, lấy CLS embedding, chuẩn hóa.  
    - Nếu với threshold `m` (ví dụ 0.95) mỗi true frame đều có ít nhất một top frame similarity ≥ m → trả về 1, ngược lại 0.

- **`predict.py`**:  
  - Hàm **`read_video_frames`**: đọc video, trích frame theo `fps_target`.  
  - Hàm **`get_top4_frames`**:
    - Embed caption + toàn bộ frame bằng CLIP.  
    - Tính similarity text–image.  
    - Chia timeline video thành 4 đoạn, chọn frame **điểm cao nhất mỗi đoạn**, resize 1920×1080.  
    - Trả về `top_frames`, `top_scores`.

- **`utils.py`**:  
  - Hàm **`set_seed`**: cố định seed cho Python, NumPy, PyTorch, CUDA để reproducible.

- **`saved_models/`**: nơi lưu checkpoint CLIP đã fine-tune (`model.save_pretrained`, `processor.save_pretrained`).

- **`dataset/`**: thư mục output chứa frames được sinh bởi `in_video_negative_mining.py`
  - `frames_true/<id>/...png`
  - `frames_neg/<id>/...png`

---

### 2. Chuẩn bị dữ liệu

- **Input dataset gốc** (ví dụ thư mục `../dataset`):
  - `train/train.json` có cấu trúc:
    - `data[i].id`
    - `data[i].question`
    - `data[i].choices` (list string)
    - `data[i].video_path`
    - `data[i].support_frames` (các timestamp tính bằng giây)
  - `train/videos/...mp4` tương ứng với `video_path`.

- **Sinh frames true + negative**:

```bash
python in_video_negative_mining.py \
  --input_folder ../dataset \
  --output_folder dataset
```

Kết quả: `dataset/frames_true` và `dataset/frames_neg` được dùng bởi `dataset.py`.

---

### 3. Huấn luyện (fine-tune CLIP)

Ví dụ lệnh chạy mặc định (từ trong thư mục `CLIP_train`):

```bash
python train.py \
  --json_path ../dataset/train/train.json \
  --image_path dataset \
  --video_path ../dataset/train/videos \
  --caption_setting 1 \
  --output_path saved_models/clip-finetuned-posneg \
  --ratio 0.2 \
  --model_name openai/clip-vit-large-patch14 \
  --loss_setting 1
```

- **Giải thích các tham số chính**:
  - **`--json_path`**: đường dẫn tới file JSON train.  
  - **`--image_path`**: thư mục chứa `frames_true` và `frames_neg`.  
  - **`--video_path`**: thư mục chứa video (dùng cho phần eval với DINOv2).  
  - **`--caption_setting`**:
    - `1`: chỉ câu hỏi  
    - `2`: chỉ đáp án lựa chọn  
    - `3`: câu hỏi + đáp án  
  - **`--ratio`**: tỉ lệ val (ví dụ 0.2 → 80% train, 20% val).  
  - **`--loss_setting`**: lựa chọn công thức loss (1/2/3 như mô tả ở trên).  

Model sau khi train sẽ được lưu tại `output_path`.

---

### 4. Đánh giá

Trong `train.py`, nếu `val_ratio > 0` (không set về 0 trong `load_data`), pipeline sẽ:

- Dùng CLIP đã fine-tune để lấy **top-4 frames** cho từng sample val (qua `get_top4_frames`).  
- Dùng DINOv2 để tính các **Sim m Score** cho các ngưỡng: 0.95, 0.96, 0.97, 0.98, 0.99 bằng hàm `sim_m`.  
- In ra trung bình theo từng ngưỡng.

Nếu không có val, code có ví dụ test tay với một video trong public test.

---

### 5. Suy luận (Inference)

Để dùng model đã fine-tune cho inference:

1. Load lại `CLIPModel` + `CLIPProcessor` từ thư mục trong `saved_models/...`.  
2. Gọi lại hàm **`get_top4_frames`** trong `predict.py`:

```python
from transformers import CLIPModel, CLIPProcessor
from predict import get_top4_frames
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("saved_models/clip-finetuned-posneg").to(device)
processor = CLIPProcessor.from_pretrained("saved_models/clip-finetuned-posneg")

top_frames, top_scores = get_top4_frames(
    video_path="path/to/video.mp4",
    caption="Câu hỏi cần truy vấn",
    model=model,
    processor=processor,
    device=device,
)
```

`top_frames` là list 4 ảnh PIL (đã resize 1920×1080), `top_scores` là điểm similarity tương ứng.

---

### 6. Ghi chú

- Code được thiết kế ưu tiên **tính reproducible** (`set_seed`) và **tận dụng GPU** nếu có (`cuda`).  
- Khi chạy trên máy có VRAM thấp, có thể cần:
  - Giảm `batch_size` trong `get_top4_frames`.  
  - Giảm `num_segments` hoặc threshold trong `in_video_negative_mining.py`.  
  - Điều chỉnh `accumulation_steps` trong `train.py`.


