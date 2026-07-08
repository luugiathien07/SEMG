# 05 — Các chỉ số đánh giá (Evaluation Metrics)

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
- **FP** (False Positive — Lỗi Loại I): cơ bình thường nhưng báo mỏi (báo động nhầm).
- **FN** (False Negative — Lỗi Loại II): cơ mỏi nhưng bỏ sót (nguy hiểm hơn trong ứng dụng thực).

> Trong bối cảnh sEMG, **Lỗi Loại II (FN)** mang rủi ro cao nhất vì bỏ sót trạng
> thái mỏi cơ có thể dẫn đến chấn thương thể thao hoặc sai lệch chẩn đoán y khoa.

Trên Streamlit, confusion matrix hiển thị dạng heatmap: ô trên đường chéo (TN, TP)
càng đậm/càng lớn thì mô hình càng tốt.

## 5.2. Các chỉ số hiệu năng dẫn xuất (Derived Performance Metrics)

| Chỉ số | Công thức | Trả lời câu hỏi | Khi nào quan trọng |
|---|---|---|---|
| **Accuracy** (độ chính xác tổng thể) | (TP+TN) / (TP+TN+FP+FN) | Tỉ lệ dự đoán đúng nói chung. | Khi hai lớp cân bằng. Dễ gây hiểu lầm nếu lệch lớp. |
| **Precision** (độ chuẩn xác / PPV) | TP / (TP+FP) | Trong số ca **báo mỏi**, bao nhiêu % đúng thật? | Khi báo động nhầm tốn kém (giảm FP). |
| **Recall** (độ nhạy / Sensitivity / TPR) | TP / (TP+FN) | Trong số ca **mỏi thật**, phát hiện được bao nhiêu %? | Khi bỏ sót mỏi là nguy hiểm (giảm FN). |
| **F1-Score** (trung bình điều hòa) | 2·P·R / (P+R) | Cân bằng tổng hợp giữa Precision và Recall. | Chỉ số chính khi lớp lệch — phản ánh chất lượng tốt hơn Accuracy. |

> Vì tập test lệch lớp (Normal 318 vs Fatigue 128), **F1-Score** và **AUC** được
> chọn làm **chỉ số đánh giá chính (Primary Metrics)** thay vì Accuracy đơn thuần.

## 5.3. ROC và AUC (Threshold-Independent Evaluation)

- **Đường ROC (Receiver Operating Characteristic):** vẽ quan hệ giữa **True
  Positive Rate** (Recall) và **False Positive Rate** (FPR = FP / (FP+TN)) khi
  thay đổi ngưỡng quyết định. Đường càng "phình" lên góc trên bên trái càng tốt.
- **AUC (Area Under Curve):** diện tích dưới đường ROC. Đo khả năng **xếp hạng**
  một ca Fatigue cao hơn một ca Normal, **không phụ thuộc ngưỡng** — tiện để so sánh
  mô hình với nhau.

Thang đánh giá chuẩn quốc tế:

| Khoảng AUC | Đánh giá |
|---|---|
| 0.9 – 1.0 | Xuất sắc (Outstanding) |
| 0.8 – 0.9 | Tốt (Excellent) |
| 0.7 – 0.8 | Chấp nhận được (Acceptable) |
| 0.5 | Đoán ngẫu nhiên (Random guess) |

Trong hệ thống, hàm `_scores()` tự động chọn `predict_proba` hoặc `decision_function`
tuỳ thuật toán để tính xác suất lớp dương cho đường ROC.

## 5.4. Cross-validation (CV-Acc)

Cột **CV-Acc** là độ chính xác ước lượng bằng **StratifiedKFold** (k=9, cấu hình
tại `config.CV_FOLDS`) *trên tập train*. Stratified đảm bảo tỷ lệ Normal/Fatigue
được duy trì đều trong mỗi fold.

> CV-Acc (trên train) và Accuracy (trên subject 9 giữ riêng) đo hai thứ khác nhau.
> CV-Acc cao nhưng test thấp → mô hình **khó tổng quát** sang người mới (dấu hiệu
> overfit). Ví dụ: DecisionTree có CV-Acc ≈ 0.982 nhưng test chỉ ≈ 0.800.

## 5.5. Phương pháp kiểm định — Leave-One-Subject-Out (LOSO)

Tín hiệu sEMG mang tính cá thể hoá rất cao. Nếu trộn dữ liệu cùng một người vào
cả train lẫn test sẽ gây **rò rỉ dữ liệu (Data Leakage)**, cho điểm ảo cao.

Hệ thống tách bạch hoàn toàn theo đối tượng:
- **Train:** Subject {5, 7, 8, 11} → 824 mẫu (Normal 506, Fatigue 318).
- **Test:** Subject 9 → 446 mẫu (Normal 318, Fatigue 128).

Cách chia này đo đúng **khả năng tổng quát hoá cho người dùng mới** — tiêu chuẩn
bắt buộc trong nghiên cứu sEMG quốc tế (BME 2024 cũng dùng LOSO).

## 5.6. Bảng kết quả demo (test trên Subject 9)

| Mô hình | Feature Set | Accuracy | Precision | Recall | F1 | CV-Acc | AUC |
|---|---|---|---|---|---|---|---|
| SVM | Top-3 mRMR | 0.951 | 0.853 | **1.000** | 0.921 | 0.904 | 0.977 |
| KNN | All 14 | 0.933 | **0.938** | 0.820 | 0.875 | 0.990 | 0.899 |
| **LDA** | All 14 | **0.957** | 0.876 | 0.992 | **0.930** | 0.988 | **0.996** |
| DecisionTree | All 14 | 0.800 | 0.654 | 0.648 | 0.651 | 0.982 | 0.755 |

**Phân tích kết quả:**

- **LDA là mô hình tốt nhất tổng thể:** F1 cao nhất (0.930), AUC cao nhất (0.996 —
  mức *Outstanding*), Accuracy cao nhất (0.957).
- **SVM đạt Recall = 1.000:** không bỏ sót bất kỳ ca mỏi nào, nhưng Precision thấp
  hơn (0.853) — tức có một số ca bình thường bị báo nhầm là mỏi.
- **KNN có Precision cao nhất (0.938):** khi máy báo mỏi thì gần như chắc chắn đúng,
  nhưng Recall chỉ 0.820 — bỏ sót 18% ca mỏi thật.
- **DecisionTree có dấu hiệu overfit rõ rệt:** CV-Acc = 0.982 (rất cao trên train)
  nhưng test chỉ 0.800. F1 = 0.651 và AUC = 0.755 — mức chỉ *Acceptable*.

**Top-3 đặc trưng mRMR (dùng cho SVM):** `Spectral_Entropy`, `Spectral_STD`, `Max`.

## 5.7. Đối chiếu với kết quả gốc của bài báo BME 2024

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

## 5.8. Tổng kết chiến lược đánh giá

| Thành phần | Vai trò | Nguồn triển khai |
|---|---|---|
| Confusion Matrix | Nền tảng đếm TP/TN/FP/FN thực tế | `sklearn.metrics.confusion_matrix` |
| F1-Score | **Chỉ số chính** — cân bằng Precision & Recall khi lớp lệch | `sklearn.metrics.f1_score` |
| AUC | **Chỉ số chính** — so sánh mô hình không phụ thuộc ngưỡng | `sklearn.metrics.roc_auc_score` |
| CV-Acc | Phát hiện overfit trên tập train | `sklearn.model_selection.cross_val_predict` |
| LOSO Split | Đảm bảo tổng quát hoá cho người dùng mới, chống data leakage | `config.TRAIN_SUBJECTS / TEST_SUBJECT` |
