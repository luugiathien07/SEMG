# 01 — Tổng quan dự án

## 1.1. Định hướng dự án

Đây là dự án **chuyển đổi (convert) pipeline MATLAB gốc sang Python**, kèm giao diện
trực quan Streamlit. **Code MATLAB (7 file `.m` trong `code-matlab/`) là nguồn tham
chiếu chính** — mục tiêu là giữ đúng logic xử lý số liệu của code, chỉ **sửa các lỗi
rõ ràng** (xem [tài liệu 07](07-giai-thich-ket-qua-cho-mentor.md)). Hai bài báo trong
`papers/` **chỉ mang tính tham khảo** để hiểu ý nghĩa bài toán và đối chiếu kết quả;
khi code MATLAB và bài báo khác nhau, **ưu tiên bám theo code MATLAB**.

## 1.2. Mỏi cơ và tín hiệu sEMG (bối cảnh)

**sEMG (surface Electromyography)** là tín hiệu điện sinh ra khi cơ co, đo bằng
điện cực dán trên bề mặt da. **Mỏi cơ (muscle fatigue)** là trạng thái cơ suy giảm
khả năng sinh lực sau khi hoạt động kéo dài. Dấu vết đặc trưng nhất trên sEMG là
**phổ công suất dịch về vùng tần số thấp** — tần số trung vị (MDF) và tần số trung
bình (MNF) **giảm dần theo thời gian** — kèm biến đổi ở biên độ và hình dạng tín hiệu.

**Bài toán:** phân loại nhị phân từng kênh EMG là **Normal (0)** hay **Fatigue (1)**.

Ứng dụng: phục hồi chức năng, thể thao (tránh chấn thương do quá tải), công thái học.

## 1.3. Hai bài báo tham khảo

Dự án tham chiếu hai bài báo cùng nhóm tác giả (dùng chung bộ dữ liệu):

| | **ICACE 2019** | **BME 2024** |
|---|---|---|
| Đặc trưng | 2 (RMS, MAV) | 14 (8 thời gian + 6 tần số) |
| Chọn đặc trưng | Không | mRMR |
| Mô hình | SVM | SVM, LDA, KNN |
| Kết quả chính | Acc 82.81%, F1 0.807 | KNN: F1 0.9541, AUC 0.95 |

Code MATLAB được convert **triển khai theo bài BME 2024** (14 đặc trưng + mRMR +
nhiều mô hình), nên demo cũng theo hướng này. Pipeline đang chạy (`src/loso.py`,
dùng chung cho cả `python -m src.pipeline` và Streamlit) giữ **4 mô hình
paper-faithful**: KNN(1-NN), KNN(5-NN), SVM (kernel tuyến tính), LDA — tất cả
dùng **cả 14 đặc trưng**, chuẩn hoá StandardScaler, không SMOTE, không tinh
chỉnh ngưỡng. Có một bộ mô hình khác (`src/models.py` + `src/evaluate.py`: SVM
top-3 mRMR, KNN/LDA/DecisionTree + SMOTE + GridSearchCV tinh chỉnh ngưỡng)
**không còn được `app.py` gọi tới** — xem mục 1.4.

## 1.4. Bản đồ: file MATLAB → module Python

Đây là dự án convert nên mỗi phần Python bám theo một phần code MATLAB:

| Code MATLAB gốc | Module Python | Vai trò |
|---|---|---|
| `Feature_Extraction.m`, `Import.m` | `feature_extraction.py`, `data_loader.py` | Đọc CSV, loại kênh lỗi, tính 14 đặc trưng/kênh |
| `Feature_Selection.m` | `feature_selection.py` | mRMR (`fscmrmr` → MRMR-FCQ), chỉ để **báo cáo top-3**, không lọc đặc trưng đưa vào model |
| `SVMClassification.m` | `loso.py` (`SVM(linear)`) | SVM kernel tuyến tính, **cả 14 đặc trưng** |
| `KNNClassification.m` | `loso.py` (`KNN(1NN)`, `KNN(5NN)`) | KNN k=1 (đúng bản gốc) + biến thể k=5, 14 đặc trưng |
| `LDAClassification.m` | `loso.py` (`LDA`) | LDA, 14 đặc trưng |
| `RFClassification.m` | *(không dùng trong pipeline hiện tại)* | Bản Decision Tree (`fitctree`) từng có ở `models.py`, đã ngừng gọi từ `app.py` |
| (chỉ số trong các file trên) | `loso.py` / `pipeline.py` | Confusion matrix, Accuracy/Precision/Recall/F1, ROC/AUC |

→ Pipeline đang chạy (CLI + Streamlit) giữ **4 mô hình**: KNN(1-NN), KNN(5-NN),
SVM(linear), LDA — tất cả dùng **cả 14 đặc trưng**, không SMOTE, không tinh
chỉnh ngưỡng (`src/loso.py::build_loso_models`). Bộ mô hình cũ hơn có SMOTE +
Decision Tree/RandomForest/LogisticRegression + GridSearchCV (`src/models.py`,
`src/evaluate.py`) vẫn còn trong code nhưng **không được `app.py` gọi tới nữa**
— chỉ dùng trong test (`tests/test_evaluate.py`).

## 1.5. Pipeline tổng thể

```
Dữ liệu thô CSV (64 kênh, 2000 Hz)
        │
        ▼
[1] Đọc & gán nhãn        →  data_loader.py
        │  (mỗi kênh của mỗi file = 1 mẫu; loại kênh lỗi toàn 0)
        ▼
[2] Trích 14 đặc trưng     →  feature_extraction.py
        │  (8 miền thời gian + 6 miền tần số cho mỗi kênh)
        ▼
[3] Xếp hạng mRMR (báo cáo) →  feature_selection.py
        │  (chỉ fit trên tập train; top-3 hiển thị ở tab Feature Selection,
        │   không lọc bớt đặc trưng đưa vào model)
        ▼
[4] Huấn luyện & phân loại →  loso.py (build_loso_models)
        │  (KNN 1-NN / KNN 5-NN / SVM linear / LDA — cả 14 đặc trưng,
        │   chuẩn hoá, không SMOTE, không tinh chỉnh ngưỡng)
        ▼
[5] Đánh giá               →  pipeline.py
        │  (leave-one-subject-out: train {5,7,8,9,10,11,12,13,14}, test 6;
        │   chỉ số từ confusion matrix, ROC/AUC)
        ▼
Bảng kết quả + biểu đồ     →  pipeline.py (CLI) & app.py (Streamlit)
```

> **Về bước tiền xử lý (lọc nhiễu).** Sơ đồ trong bài báo có wavelet denoising +
> bandpass + notch, **nhưng code MATLAB gốc (`Feature_Extraction.m`) không triển
> khai bước này** — nó đọc thẳng CSV rồi tính đặc trưng. Demo **giữ đúng như code
> MATLAB**: không lọc lại (dữ liệu vốn đã lọc phần cứng 20–400 Hz).

## 1.6. Cấu trúc mã nguồn Python

| File | Vai trò |
|---|---|
| `src/config.py` | Cấu hình: đường dẫn, `Fs=2000`, danh sách 14 đặc trưng, subject train/test, số fold. |
| `src/data_loader.py` | Đọc CSV, phân tích tên file → (subject, điều kiện, nhãn), mask loại kênh lỗi. |
| `src/feature_extraction.py` | 14 đặc trưng cho một kênh; PSD, MDF, MNF, spectral entropy. |
| `src/feature_selection.py` | mRMR (MRMR-FCQ): xếp hạng đặc trưng (chỉ để báo cáo top-3). |
| `src/loso.py` | Bảng đặc trưng đầy đủ (10 subject), 4 mô hình paper-faithful (KNN 1NN/5NN, SVM linear, LDA), leave-one-subject-out qua từng subject. Dùng chung bởi CLI, Streamlit và `scripts/run_loso.py`. |
| `src/pipeline.py` | Điều phối split train/test cố định (`config.TRAIN_SUBJECTS`/`TEST_SUBJECT`), cache đặc trưng, in báo cáo. |
| `src/models.py`, `src/evaluate.py` | Bộ mô hình cũ hơn (SMOTE, Decision Tree/RandomForest/LogisticRegression, GridSearchCV) — không còn được `app.py` gọi, chỉ dùng trong test. |
| `app.py` | Ứng dụng Streamlit **6 tab** (thêm tab Predict/Inference so với bản đầu). |

## 1.7. Cách chạy nhanh

```bash
pip install -r requirements.txt
python -m src.pipeline      # in bảng kết quả
streamlit run app.py        # demo trực quan
```

> Chi tiết dữ liệu ở [02](02-du-lieu-va-nhan.md), đặc trưng ở
> [03](03-dac-trung-14.md), mô hình ở [04](04-chon-dac-trung-va-mo-hinh.md).
