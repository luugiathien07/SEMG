# sEMG AI Platform — POC Vinmec

## Chạy nhanh (POC demo)

```bash
# 1. Kích hoạt môi trường Python
source ~/.venv/bin/activate       # hoặc: conda activate <env>

# 2. Cài dependencies (chỉ cần làm 1 lần)
pip install -r requirements.txt

# 3. Chạy landing page POC
streamlit run poc_vinmec.py --server.port 8502
```

Mở trình duyệt tại: **http://localhost:8502**

### Cấu trúc demo (3 trang — chọn trên sidebar)

| Trang | Nội dung |
|---|---|
| 📋 Giới thiệu | Bài toán, pipeline, bảng kết quả mô hình |
| ⚡ Demo UC1 | Giám sát mỏi cơ real-time — 64 kênh · Nhấn "Bắt đầu mô phỏng" |
| 📈 Demo UC2 | Theo dõi tiến trình phục hồi cơ qua các phiên |

> **Lần đầu chạy UC1** sẽ mất ~1–2 phút để huấn luyện mô hình và ghép dữ liệu.
> Từ lần thứ hai trở đi kết quả được cache lại, khởi động gần như tức thì.

---

### Các app khác trong project

```bash
# App phân tích đầy đủ (5 tab: tổng quan, tín hiệu, đặc trưng, mRMR, phân loại)
streamlit run app.py

# App real-time UC1 + UC2 độc lập (không có trang giới thiệu)
streamlit run app_realtime_session.py
```

---

# Phát hiện Mỏi cơ từ tín hiệu sEMG

Demo học máy phát hiện **trạng thái mỏi cơ** từ tín hiệu điện cơ bề mặt
(surface EMG). Với mỗi kênh tín hiệu điện cơ, hệ thống phân loại nhị phân:
cơ đang ở trạng thái **bình thường — Normal (0)** hay **mỏi — Fatigue (1)**.

## 1. Bài toán

Mỏi cơ (muscle fatigue) là hiện tượng suy giảm khả năng sinh lực của cơ khi
hoạt động kéo dài. Trên tín hiệu điện cơ, dấu vết đặc trưng và bền vững nhất của
mỏi cơ là **phổ công suất dịch về vùng tần số thấp** (tần số trung vị MDF và tần
số trung bình MNF giảm dần theo thời gian), kèm biến đổi ở biên độ và hình dạng
tín hiệu. Việc phát hiện sớm mỏi cơ có ý nghĩa trong phục hồi chức năng, thể thao,
và công thái học (ergonomics).

**Dữ liệu.** Tín hiệu sEMG đo trên cơ nhị đầu (biceps) bằng ma trận điện cực
**64 kênh**, tần số lấy mẫu **2000 Hz**. Đối tượng thực hiện các mức co cơ theo
% MVC (Maximal Voluntary Contraction) khác nhau, kèm bài tập gây mỏi. Mỗi lần đo
lưu thành một file CSV; mỗi kênh của một file là **một mẫu** cho bộ phân loại.

**Gán nhãn.** Suy ra từ số %MVC nhúng trong tên file điều kiện đo
`Sujet_{id}_{điều_kiện}_emg.csv` (ví dụ "10" trong `10_ap_fatigue`, "70" trong
`fatigue_70`): trên ngưỡng sinh lý `FATIGUE_MVC_THRESHOLD = 60` → **Fatigue**;
từ ngưỡng đó trở xuống → **Normal** (`src/config.py`). Quy tắc này thay thế quy
tắc cũ dựa theo từ khoá `fatigue` trong tên file — quy tắc cũ gán nhãn sai file
`10_ap_fatigue` (chỉ co cơ 10% MVC, cường độ rất nhẹ) thành Fatigue chỉ vì tên
file có chữ "fatigue", trong khi bản thân đoạn co cơ đó không đủ mạnh để gây mỏi
thật sự.

## 2. Phương pháp

Pipeline gồm 4 bước:

1. **Trích xuất đặc trưng** — 14 đặc trưng cho mỗi kênh, tính trên toàn bộ chiều
   dài tín hiệu:
   - *Miền thời gian (8):* RMS, MAV, Skewness, Kurtosis, Max, Min, STD, Mean.
   - *Miền tần số (6):* Spectral Min/Max/STD (trên phổ công suất PSD),
     MDF (tần số trung vị), MNF (tần số trung bình), Spectral Entropy.
2. **Chọn đặc trưng** — thuật toán **mRMR** (Minimum Redundancy Maximum Relevance)
   xếp hạng đặc trưng theo mức liên quan với nhãn và loại bỏ dư thừa; lấy **top-3**.
   Việc chọn đặc trưng chỉ fit trên tập train để tránh rò rỉ dữ liệu.
3. **Huấn luyện & phân loại** — 4 mô hình: **SVM** (kernel tuyến tính, dùng top-3 đặc trưng), **KNN**, **LDA**,
   **Decision Tree** (dùng cả 14 đặc trưng).
4. **Đánh giá** — kiểu **leave-one-subject-out**: huấn luyện trên toàn bộ các đối
   tượng còn lại và kiểm thử trên một đối tượng hoàn toàn tách biệt (mặc định
   **subject 6**), nhằm đo khả năng tổng quát hoá cho người mới. Mọi chỉ số
   (Accuracy, Precision, Recall, F1, AUC) đều tính từ confusion matrix thực tế
   trên tập test.

## 3. Codebase

```
sEMG-demo/
├── app.py                   # Ứng dụng demo Streamlit (5 tab trực quan hoá)
├── requirements.txt
├── src/
│   ├── config.py            # đường dẫn, Fs, danh sách 14 đặc trưng, cách chia train/test
│   ├── data_loader.py       # đọc CSV, phân tích tên file → nhãn, loại kênh lỗi
│   ├── feature_extraction.py# 14 đặc trưng + PSD, MDF/MNF, spectral entropy
│   ├── feature_selection.py # mRMR (MRMR-FCQ)
│   ├── models.py            # định nghĩa SVM / KNN / LDA / DecisionTree
│   ├── evaluate.py          # confusion matrix, các chỉ số, cross-validation, ROC/AUC
│   └── pipeline.py          # điều phối toàn bộ: load → extract → select → train → evaluate
├── cache/                   # cache đặc trưng đã trích xuất (features.parquet)
├── dataset/                 # dữ liệu thô CSV
├── code-matlab/             # code MATLAB tham chiếu của nhóm tác giả gốc
└── plans/                   # tài liệu đặc tả (PROJECT_SPEC.md)
```

Từng module tách bạch một nhiệm vụ và có thể dùng độc lập; `pipeline.py` ghép
chúng lại và cache kết quả trích xuất đặc trưng (lần chạy sau nhanh hơn).

## 4. Cài đặt & chạy

```bash
pip install -r requirements.txt
```

Chạy từ **thư mục gốc dự án**:

```bash
# In bảng kết quả ra terminal (build lại cache đặc trưng)
python -m src.pipeline

# Mở ứng dụng demo trực quan (5 tab)
streamlit run app.py
```

Lần chạy đầu sẽ trích xuất đặc trưng từ toàn bộ CSV và lưu vào
`cache/features.parquet`; xoá file này để buộc build lại.

### Ứng dụng demo (5 tab)
1. **Tổng quan** — số đối tượng/file, số kênh hợp lệ vs kênh lỗi, phân bố lớp.
2. **Tín hiệu & PSD** — chọn file + kênh để xem dạng sóng theo thời gian và phổ công suất.
3. **Đặc trưng** — boxplot so sánh phân bố từng đặc trưng giữa Normal và Fatigue.
4. **Chọn đặc trưng** — bảng xếp hạng mRMR, làm nổi bật top-3 được chọn.
5. **Phân loại** — bảng Accuracy/Precision/Recall/F1/AUC, confusion matrix, đường ROC.

## 5. Kết quả

Dữ liệu đầy đủ **40 file EMG / 10 đối tượng** (2468 mẫu kênh). Đánh giá
leave-one-subject-out trên **subject 6 giữ riêng** (256 mẫu: 192 Normal, 64
Fatigue), huấn luyện trên 9 đối tượng còn lại. Các mô hình bám sát bài báo: chuẩn
hoá đặc trưng rồi phân loại trên **cả 14 đặc trưng**, KNN dùng **1 láng giềng gần
nhất** — không SMOTE, không tinh chỉnh ngưỡng (ngưỡng 0.5 mặc định). Nhãn theo tiêu
chí %MVC > 60. mRMR top-3 chọn ra: **MAV, Spectral Max, Kurtosis**.

| Mô hình | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| **KNN (1-NN)** | **0.988** | 0.955 | 1.000 | **0.977** | 0.992 |
| KNN (5-NN) | 0.992 | 0.970 | 1.000 | 0.985 | 1.000 |
| SVM (linear) | 0.812 | 0.571 | 1.000 | 0.727 | 0.994 |
| LDA | 0.836 | 0.606 | 0.984 | 0.750 | 0.990 |

KNN(1-NN) đạt **F1 = 0.977, AUC = 0.992** trên subject 6, tái lập con số headline
của bài báo (BME 2024: KNN F1 ≈ 0.9541, AUC ≈ 0.95). Lưu ý con số của bài báo
tương ứng với **một fold LOSO thuận lợi**, không phải trung bình gộp. Trên toàn bộ
LOSO:

- **Trung bình 7 fold đủ 2 lớp:** KNN F1 ≈ **0.90 ± 0.11**, AUC ≈ 0.93–0.95.
- **Pooled (gộp mọi dự đoán):** KNN F1 ≈ **0.876** (Fatigue), 0.959 (Normal-conv),
  AUC ≈ 0.91–0.93; LDA có AUC pooled cao nhất ≈ 0.96.
- 3 đối tượng (11, 13, 14) chỉ có một lớp nên fold của họ không tính được F1.

Xem báo cáo đầy đủ per-fold + pooled (2 cách gán nhãn) bằng
`python -m scripts.run_loso` (ghi ra `loso_results.txt`); đánh giá nhanh một đối
tượng: `python -m scripts.eval_held_out_subject --subject 6`.
