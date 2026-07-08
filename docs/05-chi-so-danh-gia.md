# 05 — Các chỉ số đánh giá

Mọi chỉ số được tính **từ confusion matrix thực tế** trên tập test (không có giá
trị nào bị gán cứng). Cài đặt ở `src/evaluate.py`. Quy ước **lớp dương (positive)
= Fatigue (1)**.

## 5.1. Confusion matrix (ma trận nhầm lẫn)

Bảng 2×2 đếm số dự đoán đúng/sai:

|  | Dự đoán **Normal** | Dự đoán **Fatigue** |
|---|---|---|
| **Thực tế Normal** | TN (đúng) | FP (báo mỏi nhầm) |
| **Thực tế Fatigue** | FN (bỏ sót mỏi) | TP (đúng) |

- **TP** (True Positive): cơ mỏi, đoán đúng là mỏi.
- **TN** (True Negative): cơ bình thường, đoán đúng bình thường.
- **FP** (False Positive): cơ bình thường nhưng báo mỏi (báo động nhầm).
- **FN** (False Negative): cơ mỏi nhưng bỏ sót (nguy hiểm hơn trong ứng dụng thực).

Trên Streamlit, confusion matrix hiển thị dạng heatmap: ô trên đường chéo (TN, TP)
càng đậm/càng lớn thì mô hình càng tốt.

## 5.2. Các chỉ số dẫn xuất

| Chỉ số | Công thức | Trả lời câu hỏi | Khi nào quan trọng |
|---|---|---|---|
| **Accuracy** (độ chính xác) | (TP+TN) / tổng | Tỉ lệ dự đoán đúng nói chung. | Khi hai lớp cân bằng. Dễ gây hiểu lầm nếu lệch lớp. |
| **Precision** (độ chuẩn xác) | TP / (TP+FP) | Trong số ca **báo mỏi**, bao nhiêu % đúng thật? | Khi báo động nhầm tốn kém. |
| **Recall** (độ nhạy) | TP / (TP+FN) | Trong số ca **mỏi thật**, phát hiện được bao nhiêu %? | Khi bỏ sót mỏi là nguy hiểm. |
| **F1-score** | 2·P·R / (P+R) | Trung bình điều hoà của Precision và Recall. | Chỉ số cân bằng tổng hợp, dùng khi lớp lệch. |

> Vì tập test lệch lớp (Normal 318 vs Fatigue 128), **F1** phản ánh chất lượng
> tốt hơn Accuracy đơn thuần. Nên nhìn F1 và AUC làm chỉ số chính.

## 5.3. ROC và AUC

- **Đường ROC (Receiver Operating Characteristic):** vẽ quan hệ giữa **True
  Positive Rate** (recall) và **False Positive Rate** khi thay đổi ngưỡng quyết
  định. Đường càng "phình" lên góc trên bên trái càng tốt.
- **AUC (Area Under Curve):** diện tích dưới đường ROC, từ 0.5 (đoán mò) đến 1.0
  (hoàn hảo). AUC đo khả năng **xếp hạng** một ca Fatigue cao hơn một ca Normal,
  **không phụ thuộc ngưỡng** — nên rất tiện để so sánh mô hình.

Cách đọc nhanh: AUC ≈ 0.9–1.0 là rất tốt; ≈ 0.7–0.9 khá; ≈ 0.5 vô dụng.

## 5.4. Cross-validation (CV-Acc)

Cột **CV-Acc** là độ chính xác ước lượng bằng **StratifiedKFold** *trên tập train*
(chia train thành nhiều phần, luân phiên giữ lại một phần để kiểm tra). Nó cho biết
mô hình học **ổn định** đến đâu trên chính dữ liệu huấn luyện.

> Lưu ý: CV-Acc (trên train) và Accuracy (trên subject 9 giữ riêng) đo hai thứ
> khác nhau. CV-Acc cao nhưng test thấp → mô hình **khó tổng quát** sang người mới.
> Ví dụ trong demo, Decision Tree có CV-Acc ≈ 0.98 nhưng test chỉ ≈ 0.81 —
> dấu hiệu học vẹt (overfit) trên đặc điểm riêng của nhóm train.

## 5.5. Bảng kết quả demo (test trên subject 9)

| Mô hình | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| SVM | 0.951 | 0.853 | 1.000 | 0.921 | 0.977 |
| KNN | 0.933 | 0.938 | 0.820 | 0.875 | 0.899 |
| **LDA** | **0.957** | 0.876 | 0.992 | **0.930** | **0.996** |
| Decision Tree | 0.805 | 0.652 | 0.688 | 0.669 | 0.770 |

## 5.6. Đối chiếu với kết quả gốc của bài báo BME 2024

Bảng 2 trong bài báo (test theo leave-one-subject-out, 10 đối tượng):

| Chỉ số | SVM (RBF) | SVM (Linear) | LDA | KNN |
|---|---|---|---|---|
| F1 | 0.747 | 0.895 | 0.909 | **0.954** |
| Accuracy | 0.786 | 0.888 | 0.904 | **0.953** |
| Precision | 0.630 | 0.953 | 0.964 | **0.974** |
| Recall | 0.917 | 0.843 | 0.861 | **0.935** |

- Trong bài báo, **KNN tốt nhất** (F1 0.954, AUC 0.95); LDA và SVM tuyến tính theo sau.
- Demo cho **xu hướng tương đồng** (nhóm SVM/LDA/KNN đều mạnh), xác nhận pipeline
  chuyển đổi hợp lý. Con số tuyệt đối khác vì demo chỉ dùng **5/10 đối tượng** và
  **bỏ bước wavelet denoising** (xem [tài liệu 07](07-giai-thich-ket-qua-cho-mentor.md)).
- Giá trị `config.PAPER_REFERENCE` trong code (`KNN_F1=0.9541`, `KNN_AUC=0.95`) lấy
  đúng từ Bảng 2 và Hình 2 của bài báo. (Phần Abstract của bài ghi 95.12% — lệch
  nhẹ so với bảng 95.41%; ta dùng số trong bảng.)

Diễn giải chi tiết ở [tài liệu 07](07-giai-thich-ket-qua-cho-mentor.md).
