# 06 — Hướng dẫn đọc ứng dụng Streamlit

Chạy `streamlit run app.py` rồi mở trình duyệt. Ứng dụng có **6 tab**. Mẹo chung:
di chuột (hover) vào bất kỳ biểu đồ nào để xem giá trị chi tiết (tooltip nền đen,
chữ trắng, cỡ lớn).

---

## Tab 1 — 📊 Overview (Tổng quan)

**Mục đích:** nắm nhanh quy mô và cân bằng của dữ liệu.

- **4 ô số (metrics):**
  - *Subjects* — số đối tượng.
  - *Files* — số file CSV.
  - *Valid channel-samples* — tổng số mẫu hợp lệ (kênh × file, sau khi loại kênh lỗi).
  - *Dropped (dead) channels* — số kênh lỗi đã bị loại.
- **Biểu đồ "Class distribution":** số mẫu của mỗi lớp Normal (xanh) và Fatigue
  (đỏ). Cho thấy dữ liệu **lệch lớp** (Normal nhiều hơn) — lý do ta ưu tiên nhìn F1/AUC.
- **Biểu đồ "Samples per subject":** số mẫu từng lớp theo mỗi đối tượng (nay cả
  10 subject đều có đủ hai lớp). Giúp thấy quy mô dữ liệu từng subject, liên quan
  việc chọn subject 6 (`config.TEST_SUBJECT`) làm test.
- **Bảng "Per-file summary":** mỗi file có bao nhiêu kênh hợp lệ, thuộc lớp gì.

---

## Tab 2 — 📈 Signal & PSD (Tín hiệu & phổ)

**Mục đích:** nhìn tận mắt một tín hiệu EMG thô và phổ của nó.

- Chọn **File** rồi chọn **Channel** (chỉ hiện kênh hợp lệ).
- **Biểu đồ trên (Time-domain signal):** dạng sóng biên độ theo thời gian (giây).
  Tín hiệu dài được lấy mẫu thưa để vẽ mượt.
- **Biểu đồ dưới (Power spectral density):** phổ công suất theo tần số (Hz). Nhìn
  vào đây để hình dung năng lượng tập trung ở dải tần nào.
- **Bảng "14 features":** giá trị 14 đặc trưng tính riêng cho kênh đang chọn — đối
  chiếu trực tiếp giữa hình dạng tín hiệu và con số đặc trưng.

> Gợi ý trình diễn: mở một kênh file `..._fatigue_70...` và một kênh file `..._20...`
> để so sánh biên độ và hình dạng phổ giữa mỏi và bình thường.

---

## Tab 3 — 🎯 Features (Phân phối đặc trưng)

**Mục đích:** xem đặc trưng nào **tách hai lớp tốt**.

- Ô **"Select all features"**: tick để hiện cả 14 đặc trưng; bỏ tick thì mặc định
  hiện top-3. Vẫn có thể tự thêm/bớt trong ô multiselect.
- Mỗi đặc trưng là một **boxplot** (nền đen) so sánh Normal (xanh) vs Fatigue (đỏ).

**Cách đọc boxplot:** hộp là khoảng giữa (25%–75%), vạch giữa là trung vị, râu là
vùng dữ liệu điển hình. **Hai hộp Normal và Fatigue càng ít chồng lấn → đặc trưng
đó phân biệt hai lớp càng tốt.** Đây là bằng chứng trực quan cho việc mRMR chọn
đặc trưng nào.

---

## Tab 4 — 🏅 Feature Selection (Chọn đặc trưng mRMR)

**Mục đích:** minh bạch việc chọn đặc trưng.

- **Biểu đồ cột:** chiều cao cột = độ liên quan (mutual information) của mỗi đặc
  trưng với nhãn; thứ tự trục X = **thứ hạng mRMR** (trái = tốt nhất). Cột **xanh
  lá** là top-3 được chọn; **xám** là còn lại.
- **Bảng bên dưới:** rank, tên đặc trưng, điểm relevance, và cột `selected` (đánh
  dấu top-3).

> Điểm nhấn: top-3 hiện tại là **MAV, Spectral_Max, Kurtosis** — xem giải
> thích ở [tài liệu 04](04-chon-dac-trung-va-mo-hinh.md). Đây chỉ để **báo cáo**;
> không giới hạn đặc trưng của model nào ở tab Classification (mọi model dùng cả
> 14 đặc trưng).

---

## Tab 5 — 🤖 Classification (Kết quả phân loại)

**Mục đích:** so sánh các mô hình trên đối tượng giữ riêng — hiện tại là **subject
6** (`config.TEST_SUBJECT`), train trên subject {5,7,8,9,10,11,12,13,14}.

- **Ô chọn Models:** bật/tắt 4 mô hình muốn so sánh — **KNN(1NN), KNN(5NN),
  SVM(linear), LDA** (không còn Decision Tree trong pipeline hiện tại).
- **Bảng chỉ số:** Accuracy / Precision / Recall / F1 / CV-Acc / AUC cho từng mô
  hình; cột **F1 được tô màu xanh** đậm dần theo giá trị (dễ so sánh). Dưới bảng
  có chú thích rằng số của bài báo BME 2024 là **một fold LOSO thuận lợi**, không
  phải trung bình — xem [tài liệu 05](05-chi-so-danh-gia.md) mục 5.6 để so sánh
  đúng cách (trung bình/pooled qua 10 subject, không chỉ 1 fold).
- **Bảng "Phương pháp huấn luyện":** liệt kê siêu tham số thật sự dùng cho mỗi
  model và ngưỡng quyết định (mặc định 0.5, không tinh chỉnh).
- **Confusion matrices:** heatmap 2×2 cho mỗi mô hình (đã phóng to cả hình lẫn số).
  Nhìn hai ô đường chéo (TN góc trên-trái, TP góc dưới-phải) — càng lớn càng tốt;
  hai ô còn lại là lỗi (FP, FN).
- **ROC curves:** mỗi mô hình một đường; đường nét đứt xám là mức "đoán mò" (AUC
  0.5). Đường càng sát góc trên-trái và AUC càng gần 1 → càng tốt.

Ý nghĩa từng chỉ số xem [tài liệu 05](05-chi-so-danh-gia.md); cách diễn giải kết
quả để trình bày xem [tài liệu 07](07-giai-thich-ket-qua-cho-mentor.md).

---

## Tab 6 — 🔮 Predict / Inference (Đưa tín hiệu vào → dự đoán)

**Mục đích:** demo trực tiếp — chọn 1 file + 1 kênh **của subject test (subject
6)**, chạy cả 4 model đã huấn luyện để dự đoán Normal/Fatigue, đối chiếu với nhãn
thật suy từ tên file. Chỉ dùng dữ liệu subject test vì đây là phép thử công bằng
cho khả năng tổng quát hoá (dữ liệu subject train mô hình đã thấy khi huấn luyện).

- **① Tín hiệu EMG · Đặc trưng miền thời gian:** dạng sóng thô kèm chú thích vị
  trí Mean, dải ±1 STD, Max/Min trực tiếp trên hình.
- **② Phổ công suất (PSD) · Đặc trưng miền tần số:** đường PSD kèm chú thích MDF/MNF.
- **③ Vector 14 đặc trưng → Kết quả dự đoán:** bảng 14 đặc trưng (có tooltip giải
  thích từng đặc trưng) cạnh bảng dự đoán của 4 model (nhãn dự đoán, P(Fatigue),
  đúng/sai so với nhãn thật).
- **④ Thống kê dự đoán 64 kênh:** chạy cả 4 model trên **toàn bộ kênh hợp lệ** của
  file đang chọn, đếm số kênh dự đoán Mỏi/Không mỏi và % khớp nhãn thật — cho thấy
  các model có nhất quán trên toàn file hay chỉ đúng ở một vài kênh lẻ tẻ.
