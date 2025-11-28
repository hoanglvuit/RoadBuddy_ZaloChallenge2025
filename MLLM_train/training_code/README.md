# Tóm tắt giải pháp nhóm

> Lưu ý: Do hạn chế thời gian và cấu hình giữa các thư viện khi chạy trên driver và CUDA (của BTC) còn hơi cũ, nhóm không cài đặt môi trường training trong Docker, chỉ triển khai **inference**. BTC có thể thuê từ vastai (vì toàn gpu có driver mới, dễ sử dụng)

---

## Stage 1: Chọn Frame từ Video

1. **Đọc video**:
   - Tốc độ: **15 fps** (giảm số lượng frame để tăng tốc xử lý).

2. **Chia video thành các phần**:
   - Video được chia thành **4 phần**.
   - Phần đầu: **3 frame**.
   - Các phần còn lại: chia đều frame.

3. **Chọn frame liên quan nhất đến câu hỏi**:
   - Sử dụng **CLIP-large**.
   - Phải **finetune** để đạt hiệu quả cao.
   - Quá trình finetune được trình bày trong **`clip_train` folder**.

---

## Stage 2: Tạo Kết Quả Cuối

1. Sử dụng **4 frame đã chọn** từ Stage 1.
2. Kết hợp với **câu hỏi và các lựa chọn** (choices).
3. Mô hình **Qwen3VL-8B** được dùng để đưa ra kết quả cuối cùng.
4. Quá trình training Qwen3VL được trình bày trong **`qwenvl_train` folder**.

---
