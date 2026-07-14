# Tài liệu dự án — Phát hiện Mỏi cơ từ tín hiệu sEMG

Bộ tài liệu tiếng Việt giúp đọc hiểu toàn bộ dự án: từ bài toán, dữ liệu, đặc
trưng, mô hình, chỉ số đánh giá, cho tới cách đọc từng biểu đồ trên ứng dụng
Streamlit và cách trình bày kết quả demo với mentor.

## Mục lục (đọc theo thứ tự)

| # | Tài liệu | Nội dung |
|---|---|---|
| 1 | [01-tong-quan.md](01-tong-quan.md) | Bài toán mỏi cơ, mục tiêu, pipeline tổng thể, cấu trúc mã nguồn |
| 2 | [02-du-lieu-va-nhan.md](02-du-lieu-va-nhan.md) | Dữ liệu sEMG, định dạng file, cách gán nhãn, đơn vị mẫu, chia train/test |
| 3 | [03-dac-trung-14.md](03-dac-trung-14.md) | Ý nghĩa và công thức của 14 đặc trưng thời gian & tần số |
| 4 | [04-chon-dac-trung-va-mo-hinh.md](04-chon-dac-trung-va-mo-hinh.md) | Thuật toán chọn đặc trưng mRMR và 4 mô hình phân loại |
| 5 | [05-chi-so-danh-gia.md](05-chi-so-danh-gia.md) | Confusion matrix, Accuracy/Precision/Recall/F1/AUC, cross-validation |
| 6 | [06-huong-dan-doc-streamlit.md](06-huong-dan-doc-streamlit.md) | Giải thích từng tab và từng biểu đồ trên ứng dụng demo |
| 7 | [07-giai-thich-ket-qua-cho-mentor.md](07-giai-thich-ket-qua-cho-mentor.md) | Diễn giải kết quả demo, điểm mạnh/hạn chế, kịch bản trình bày & Q&A |
| 9 | [09-so-do-kenh-va-mfcv.md](09-so-do-kenh-va-mfcv.md) | Sơ đồ ma trận điện cực 64 kênh (13×5, IED 8mm) và câu hỏi có tính được MFCV (vận tốc dẫn truyền sợi cơ) từ EMG không |

## Tóm tắt nhanh (1 phút)

- **Bài toán:** với mỗi kênh tín hiệu điện cơ, phân loại cơ đang **bình thường
  (Normal)** hay **mỏi (Fatigue)**.
- **Dữ liệu:** sEMG 64 kênh, 2000 Hz; nhãn suy ra từ tên file.
- **Cách làm:** trích 14 đặc trưng/kênh (mRMR chỉ để xếp hạng/báo cáo top-3) →
  phân loại bằng KNN(1-NN)/KNN(5-NN)/SVM(linear)/LDA (cả 14 đặc trưng) → đánh giá
  trên một đối tượng giữ riêng (subject 6), cộng thêm leave-one-subject-out đầy đủ
  qua cả 10 subject để có số liệu trung bình đáng tin cậy hơn.
- **Kết quả:** trên subject 6 (fold demo), KNN(1-NN) tốt nhất (F1 ≈ 0.98, AUC ≈
  0.99); trung bình qua cả 10 fold LOSO, F1 thực tế thấp hơn (≈ 0.81–0.84 ± 0.09–0.11).
- **Nguồn:** **convert từ code MATLAB gốc** (`code-matlab/`, 7 file `.m`); hai bài
  báo trong `papers/` (ICACE 2019, BME 2024) chỉ mang tính **tham khảo**.

> Thư mục `docs/` chỉ lưu cục bộ (được liệt kê trong `.gitignore`).
