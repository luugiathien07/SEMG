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

> Vì tập test lệch lớp (Normal 320 vs Fatigue 128, subject 6 hiện tại), **F1-Score**
> và **AUC** được chọn làm **chỉ số đánh giá chính (Primary Metrics)** thay vì
> Accuracy đơn thuần.

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

> CV-Acc (trên train) và Accuracy (trên subject 6 giữ riêng) đo hai thứ khác nhau.
> CV-Acc cao nhưng test thấp → mô hình **khó tổng quát** sang người mới (dấu hiệu
> overfit).

## 5.5. Phương pháp kiểm định — Leave-One-Subject-Out (LOSO)

Tín hiệu sEMG mang tính cá thể hoá rất cao. Nếu trộn dữ liệu cùng một người vào
cả train lẫn test sẽ gây **rò rỉ dữ liệu (Data Leakage)**, cho điểm ảo cao.

Hệ thống tách bạch hoàn toàn theo đối tượng. Có **hai cách chạy LOSO** trong dự án:

**(a) Single-split cố định** (`config.TRAIN_SUBJECTS`/`TEST_SUBJECT`, dùng cho tab
Classification của Streamlit và `python -m src.pipeline`):
- **Train:** Subject {5, 7, 8, 9, 10, 11, 12, 13, 14} → 3878 mẫu (Normal 2754, Fatigue 1124).
- **Test:** Subject 6 → 448 mẫu (Normal 320, Fatigue 128).

**(b) LOSO đầy đủ** (`scripts/run_loso.py` → `results/loso_results.txt`): lặp lại
(a) **10 lần**, mỗi lần giữ 1 subject khác nhau làm test — cho bức tranh đầy đủ về
độ ổn định của mô hình qua nhiều subject, thay vì chỉ 1 con số của cách (a). Xem
mục 5.6b.

Cách chia này đo đúng **khả năng tổng quát hoá cho người dùng mới** — tiêu chuẩn
bắt buộc trong nghiên cứu sEMG quốc tế (BME 2024 cũng dùng LOSO).

## 5.6. Bảng kết quả demo

### 5.6a. Single-split (test trên Subject 6, `python -m src.pipeline`)

| Mô hình | Feature Set | Accuracy | Precision | Recall | F1 | CV-Acc | AUC |
|---|---|---|---|---|---|---|---|
| **KNN(1NN)** | All 14 | **0.991** | **0.992** | 0.977 | **0.984** | 0.996 | 0.987 |
| KNN(5NN) | All 14 | 0.987 | 0.992 | 0.961 | 0.976 | 0.993 | 0.995 |
| SVM(linear) | All 14 | 0.900 | 0.740 | **1.000** | 0.850 | 0.934 | 0.996 |
| LDA | All 14 | 0.940 | 0.839 | 0.977 | 0.903 | 0.929 | **0.997** |

**Phân tích kết quả:**

- **KNN(1-NN) tốt nhất trên fold này:** F1 cao nhất (0.984), Accuracy cao nhất
  (0.991), Precision cao nhất (0.992).
- **SVM đạt Recall = 1.000:** không bỏ sót bất kỳ ca mỏi nào, nhưng Precision thấp
  hơn (0.740) — báo nhầm khá nhiều ca bình thường thành mỏi.
- **LDA có AUC cao nhất (0.997):** khả năng xếp hạng lớp Fatigue tốt nhất, dù
  ngưỡng 0.5 mặc định cho F1 thấp hơn KNN.
- **Lưu ý quan trọng:** subject 6 là fold **thuận lợi nhất** trong 10 fold LOSO
  (xem 5.6b) — không nên dùng riêng bảng này để kết luận về khả năng tổng quát
  của mô hình. Chi tiết diễn giải ở [tài liệu 07](07-giai-thich-ket-qua-cho-mentor.md).

**Top-3 đặc trưng mRMR** (chỉ để báo cáo, không giới hạn đặc trưng của model nào):
`MAV`, `Spectral_Max`, `Kurtosis`.

### 5.6b. LOSO đầy đủ — trung bình & pooled qua cả 10 subject

Chạy `python -m scripts.run_loso` (nhãn theo ngưỡng %MVC > 60), kết quả đầy đủ ở
`results/loso_results.txt` / `results/new_results.txt`:

| Mô hình | TB F1 ± SD (10 fold) | TB Acc | TB AUC | Pooled F1 | Pooled AUC |
|---|---|---|---|---|---|
| KNN(1NN) | 0.842 ± 0.089 | 0.910 | 0.894 | 0.847 | 0.895 |
| KNN(5NN) | 0.827 ± 0.112 | 0.906 | 0.939 | 0.835 | 0.935 |
| SVM(linear) | 0.811 ± 0.110 | 0.895 | 0.964 | 0.824 | 0.954 |
| LDA | 0.826 ± 0.103 | 0.906 | 0.979 | 0.836 | 0.965 |

- **TB F1 ± SD:** trung bình cộng F1 của 10 fold (mỗi subject trọng số bằng nhau).
- **Pooled F1/AUC:** gộp dự đoán của cả 10 fold thành 1 confusion matrix rồi tính
  1 lần (mỗi mẫu trọng số bằng nhau).
- F1 từng fold dao động rộng: thấp nhất subject 13 (KNN 1NN F1=0.699), cao nhất
  subject 6 (F1=0.984) — đúng là fold được chọn cho bảng 5.6a. Đây là lý do bảng
  5.6b (trung bình/pooled) đáng tin cậy hơn bảng 5.6a khi muốn báo cáo hiệu năng
  tổng quát của mô hình.

## 5.7. Đối chiếu với kết quả gốc của bài báo BME 2024

Bảng 2 trong bài báo (test theo leave-one-subject-out, 10 đối tượng):

| Chỉ số | SVM (RBF) | SVM (Linear) | LDA | KNN |
|---|---|---|---|---|
| F1 | 0.747 | 0.895 | 0.909 | **0.954** |
| Accuracy | 0.786 | 0.888 | 0.904 | **0.953** |
| Precision | 0.630 | 0.953 | 0.964 | **0.974** |
| Recall | 0.917 | 0.843 | 0.861 | **0.935** |

- Trong bài báo, **KNN tốt nhất** (F1 0.954, AUC 0.95); LDA và SVM tuyến tính theo sau.
- Bảng 2 của bài chỉ có **1 con số mỗi model**, không có SD hay bảng per-subject —
  nhiều khả năng đó là kết quả của **một fold LOSO cụ thể**, không phải trung bình
  qua nhiều fold (xem phân tích ở [tài liệu 07](07-giai-thich-ket-qua-cho-mentor.md)).
- So sánh công bằng nhất là dùng **bảng 5.6b (trung bình/pooled LOSO)** của demo,
  không phải bảng 5.6a (1 fold thuận lợi) — theo 5.6b, demo **thấp hơn** bài báo
  (F1 trung bình ≈ 0.84 vs 0.9541).
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
| LOSO Split (single) | Đảm bảo tổng quát hoá cho người dùng mới, chống data leakage | `config.TRAIN_SUBJECTS / TEST_SUBJECT` |
| LOSO đầy đủ (10 fold) | Trung bình ± SD và pooled qua mọi subject — chống kết luận sai từ 1 fold thuận lợi | `src/loso.py`, `scripts/run_loso.py` |
