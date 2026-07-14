# 07 — Diễn giải kết quả demo (trình bày với mentor)

Tài liệu này tổng hợp cách **kể câu chuyện kết quả** một cách trung thực, kèm điểm
mạnh, hạn chế và các câu hỏi có thể gặp.

## 7.1. Thông điệp chính (một câu)

> "Pipeline hoạt động đúng và chạy được đầu-cuối trên dữ liệu thật (10 đối tượng,
> 4326 mẫu kênh): với 14 đặc trưng phổ + biên độ và các bộ phân loại
> KNN/SVM/LDA, ta phân biệt được kênh EMG mỏi vs bình thường trên một đối tượng
> **chưa từng thấy**. Trên fold test hiện tại (subject 6) F1 ≈ 0.85–0.98 tuỳ
> model; trung bình qua cả 10 fold LOSO, F1 thực tế thấp hơn — khoảng 0.81–0.84
> ± 0.09–0.11 (xem 7.2b)."

## 7.2. Bảng kết quả và cách đọc

### 7.2a. Single-split — test trên subject 6 (`python -m src.pipeline`)

| Mô hình | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| **KNN(1-NN)** | **0.991** | **0.992** | 0.977 | **0.984** | 0.987 |
| KNN(5-NN) | 0.987 | 0.992 | 0.961 | 0.976 | 0.995 |
| SVM(linear) | 0.900 | 0.740 | **1.000** | 0.850 | 0.996 |
| LDA | 0.940 | 0.839 | 0.977 | 0.903 | **0.997** |

Những điểm nên nói:

- **KNN(1-NN) tốt nhất trên fold này.** F1 0.984, Accuracy 0.991 — gần như hoàn hảo.
- **SVM: Recall 1.0 nhưng Precision 0.74** → mô hình "thà báo nhầm còn hơn bỏ sót".
  Trong bối cảnh cảnh báo mỏi cơ (bỏ sót nguy hiểm hơn báo nhầm), đây là đánh đổi
  hợp lý dù Precision thấp hơn KNN/LDA.
- **LDA có AUC cao nhất (0.997)** — khả năng xếp hạng lớp Fatigue tốt nhất về mặt
  threshold-independent, dù F1 ở ngưỡng 0.5 thấp hơn KNN.
- **Quan trọng — đừng dừng lại ở bảng này:** subject 6 là fold **thuận lợi nhất**
  trong 10 fold LOSO có thể chọn (xem 7.2b). Một bảng số đẹp trên 1 subject không
  chứng minh mô hình tổng quát tốt cho mọi người dùng mới.

### 7.2b. LOSO đầy đủ (10 fold) — bức tranh thực tế hơn

`python -m scripts.run_loso` → `results/loso_results.txt`:

| Mô hình | TB F1 ± SD (10 fold) | Pooled F1 | F1 thấp nhất | F1 cao nhất |
|---|---|---|---|---|
| KNN(1NN) | 0.842 ± 0.089 | 0.847 | 0.699 (subj 13) | 0.984 (subj 6) |
| KNN(5NN) | 0.827 ± 0.112 | 0.835 | 0.633 (subj 13) | 0.981 (subj 11) |
| SVM(linear) | 0.811 ± 0.110 | 0.824 | 0.578 (subj 9) | 0.992 (subj 11) |
| LDA | 0.826 ± 0.103 | 0.836 | 0.659 (subj 12) | 1.000 (subj 11) |

- F1 dao động rất rộng theo subject test — không có mô hình nào ổn định tuyệt đối
  qua mọi subject.
- **TB F1 ± SD** = trung bình cộng 10 F1 riêng lẻ (mỗi subject trọng số bằng
  nhau). **Pooled F1** = gộp toàn bộ dự đoán của 10 fold thành 1 confusion matrix
  rồi tính 1 lần (mỗi mẫu trọng số bằng nhau). Hai cách gần nhau ở đây vì số mẫu
  giữa các subject không lệch quá nhiều.
- **So với bài báo BME 2024** (KNN F1 = 0.9541, AUC = 0.95): con số này khớp gần
  đúng với fold subject-6 thuận lợi của demo (F1=0.984), nhưng **cao hơn hẳn**
  trung bình LOSO thật (F1 ≈ 0.84). Bảng 2 của bài báo chỉ có 1 số/model, không
  có SD hay bảng per-subject — nhiều khả năng đó cũng chỉ là **một fold LOSO
  thuận lợi**, không phải trung bình qua nhiều fold (xem 7.5 và 7.8).

## 7.3. Vì sao top-3 mRMR khác bài báo?

Bài báo BME 2024 báo cáo top-3 mRMR là **MAV, Skewness, Mean**. Demo hiện chọn
**MAV, Spectral_Max, Kurtosis** (khớp 1/3 với bài báo — MAV). Giải thích trung thực:

- Cách chia train/test khác (subject 6 làm test thay vì subject dùng trong bài
  báo), và cài đặt mRMR (MRMR-FCQ tự viết) khác `fscmrmr` của MATLAB → thứ hạng
  đặc trưng đổi là bình thường.
- Chính bài báo cũng lưu ý mRMR *"chỉ mang tính chỉ báo"* (indicative): đặc trưng
  mRMR chọn không đảm bảo là tối ưu cho mọi mô hình. Đó là lý do trong demo hiện
  tại **cả 4 model đều dùng đủ 14 đặc trưng**, top-3 mRMR chỉ để minh hoạ/báo cáo.
- MDF/MNF (chỉ dấu mỏi kinh điển) vẫn nằm trong 14 đặc trưng và vẫn được mọi mô
  hình khai thác — chúng chỉ không lọt top-3 của mRMR trên tập dữ liệu này.

## 7.4. Điểm mạnh của bản demo

1. **Convert trung thành code MATLAB gốc:** giữ đúng 14 đặc trưng, thứ tự đặc trưng,
   mRMR, và các mô hình KNN/SVM/LDA cùng siêu tham số gần với 7 file `.m` gốc.
2. **Sửa các lỗi rõ ràng của code MATLAB** (theo `PROJECT_SPEC.md`): không gán cứng
   chỉ số TP/FP mà tính từ confusion matrix thật; sửa lỗi luỹ thừa PSD `^4`→`^2`;
   bỏ dead code / lỗi cú pháp.
3. **Chạy được đầu-cuối trên dữ liệu thật** (nay đủ 10/10 đối tượng), có cache đặc
   trưng và giao diện trực quan 6 tab (kể cả tab Predict/Inference chạy trực tiếp
   trên tín hiệu).
4. **Đánh giá nghiêm túc và minh bạch về giới hạn:** không chỉ báo cáo 1 fold —
   có thêm **leave-one-subject-out đầy đủ qua cả 10 subject** (`scripts/run_loso.py`),
   báo cáo cả trung bình ± SD và pooled, thay vì chỉ chọn fold đẹp nhất để trình bày.
   Chọn đặc trưng (mRMR) chỉ fit trên train.

## 7.5. Hạn chế (nên chủ động nêu trước khi mentor hỏi)

1. **Dữ liệu vẫn ít theo số subject:** 10 đối tượng, 69 file, 4326 mẫu kênh. Với
   LOSO, mỗi fold test chỉ dựa trên 1 subject — nên F1 dao động khá rộng giữa các
   fold (0.70–0.98 tuỳ subject, xem 7.2b), độ tin cậy của con số trung bình vẫn
   còn hạn chế vì N=10 fold là nhỏ về mặt thống kê.
2. **Không có bước lọc nhiễu (wavelet denoising):** demo **giữ đúng như code MATLAB
   `Feature_Extraction.m`** — đọc thẳng CSV rồi tính đặc trưng, không lọc lại (dữ
   liệu đã lọc phần cứng 20–400 Hz). Sơ đồ trong bài báo *có* mô tả wavelet denoising
   + bandpass + notch nhưng code gốc không triển khai; nếu muốn bám bài báo hơn thì
   đây là bước có thể bổ sung.
3. **Gán nhãn từ tên file:** theo ngưỡng sinh lý %MVC > 60 (mục 2.4). Vì cả hai lớp
   đều trải nhiều mức lực (Normal có cả 90%... không, hiện `90` đã là Fatigue; nhưng
   Normal vẫn có `10_ap_fatigue` sau mỏi) nên mô hình **không thể "ăn gian" chỉ bằng
   biên độ/lực**; song ranh giới lực–mỏi chưa được cô lập hoàn toàn.
4. **Kết quả single-split (subject 6) không đại diện cho khả năng tổng quát:** đây
   là fold thuận lợi nhất trong 10 fold LOSO — nên **luôn đi kèm bảng LOSO đầy đủ**
   (7.2b) khi báo cáo, không chỉ dùng bảng 7.2a.
5. **Mẫu = toàn bộ tín hiệu một kênh:** chưa dùng cửa sổ trượt theo thời gian, nên
   chưa theo dõi được **diễn tiến mỏi** trong một lần đo.

## 7.6. Hướng phát triển tiếp

- Thu thập thêm đối tượng để tăng N của LOSO, giảm phương sai giữa các fold và có
  kết luận thống kê vững hơn về khả năng tổng quát.
- Thêm **cửa sổ trượt** (ví dụ 1–2 giây) để quan sát MDF/MNF giảm dần theo thời gian.
- Kiểm soát yếu tố lực: so sánh trong cùng mức %MVC để cô lập ảnh hưởng của mỏi.
- Thử các cách chia LOSO/k-fold theo subject khác nhau nhiều lần để ước lượng độ
  ổn định thay vì chỉ 1 lần chạy 10-fold.

## 7.7. Kịch bản demo gợi ý (5–7 phút)

1. **Bài toán & dữ liệu** (tab Overview): giải thích mỏi cơ, quy mô (10 đối tượng,
   4326 mẫu), cân bằng lớp.
2. **Tín hiệu thật** (tab Signal & PSD): mở 1 kênh mỏi vs 1 kênh bình thường, chỉ
   ra khác biệt biên độ/phổ.
3. **Đặc trưng phân biệt** (tab Features): mở boxplot vài đặc trưng, chỉ hộp ít
   chồng lấn.
4. **Chọn đặc trưng** (tab Feature Selection): giải thích mRMR và top-3 (MAV,
   Spectral_Max, Kurtosis).
5. **Kết quả** (tab Classification): bảng chỉ số + confusion matrix + ROC trên
   subject 6; **chủ động nói thêm** đây là 1 fold, kèm số liệu LOSO trung bình
   (7.2b) để không gây hiểu lầm là con số tổng quát.
6. **Demo trực tiếp** (tab Predict/Inference): chọn 1 file/kênh của subject test,
   bấm Dự đoán — cho thấy pipeline chạy được đầu-cuối trên tín hiệu thật.
7. **Chốt:** điểm mạnh, hạn chế (mục 7.5) và hướng phát triển (mục 7.6).

## 7.8. Câu hỏi mentor có thể hỏi & gợi ý trả lời

- *"Vì sao không trộn ngẫu nhiên train/test?"* → Sẽ rò rỉ: các kênh của cùng một
  người rất giống nhau, lọt vào cả train lẫn test làm điểm ảo cao. Leave-one-
  subject-out đo đúng khả năng dùng cho **người mới**.
- *"Kết quả (F1≈0.98) có phải quá tốt/đáng ngờ, sao lại cao hơn bài báo?"* → Đây
  là kết quả trên **một cấu hình test cố định** (subject 6) — dùng cho mục đích
  demo. Khi chạy leave-one-subject-out đầy đủ qua cả 10 subject, F1 trung bình
  chỉ khoảng 0.84 ± 0.09, dao động 0.70–0.98 tuỳ subject test. Không nên dùng con
  số 1-fold để kết luận mô hình tốt/kém hơn bài báo — nên dùng trung bình/pooled
  LOSO (mục 7.2b).
- *"Vì sao kết quả dao động mạnh giữa các subject?"* → Dataset chỉ có 10 subject;
  mỗi fold LOSO test trên đúng 1 subject (300–450 mẫu). Với số subject ít, một
  subject có đặc điểm tín hiệu khác biệt (nhiễu, cách co cơ...) đã đủ làm F1 lệch
  hẳn so với các fold khác — đây là hạn chế về độ tin cậy thống kê do cỡ mẫu nhỏ
  theo subject, chưa hẳn là mô hình yếu.
- *"Gán nhãn thế nào?"* → Theo ngưỡng sinh lý %MVC > 60 (mục 2.4, khớp tiêu chí
  ICACE 2019): số %MVC trong tên điều kiện > 60 → Fatigue, ≤ 60 → Normal. Cụ thể:
  `10/20/40/60` → Normal; `90` và `fatigue_70` (70% MVC) → **Fatigue**;
  `10_ap_fatigue` → Normal (số %MVC của nó là 10, đo sau khi nghỉ ở cường độ rất
  thấp — không phải phép đo trong lúc mỏi).
- *"Đã lọc/khử nhiễu tín hiệu chưa?"* → Không — demo **giữ đúng code MATLAB**
  (`Feature_Extraction.m` đọc thẳng CSV, không wavelet denoising); dữ liệu đã lọc
  phần cứng 20–400 Hz. Bài báo có mô tả bước lọc này nhưng code gốc không làm; đây
  là hướng có thể bổ sung.
- *"Vì sao không còn Decision Tree như trước?"* → Pipeline hiện tại (`src/loso.py`)
  chỉ giữ 4 mô hình paper-faithful (KNN 1-NN/5-NN, SVM linear, LDA) để số liệu so
  sánh trực tiếp được với bài báo (bài chỉ dùng SVM/LDA/KNN). Bản Decision Tree +
  SMOTE + GridSearchCV vẫn còn trong code (`src/models.py`) nhưng không còn được
  `app.py` gọi tới.
- *"Top-3 đặc trưng có khớp bài báo không?"* → Bài báo là MAV/Skewness/Mean, demo
  hiện tại là MAV/Spectral_Max/Kurtosis — khớp 1/3 (MAV). Khác phần còn lại do
  cách chia train/test và cài đặt mRMR khác (mục 7.3). Bài báo cũng nói đặc trưng
  mRMR "chỉ mang tính chỉ báo", và hiện tại top-3 không giới hạn đặc trưng của
  model nào (mọi model dùng cả 14).
- *"Số liệu có khớp bài báo không?"* → Fold thuận lợi nhất (subject 6) khá gần
  con số bài báo; trung bình LOSO thật (0.84) thấp hơn bài báo (0.9541). Bảng 2
  của bài báo cũng chỉ có 1 số/model, không có SD/per-subject — khả năng cao đó
  cũng là 1 fold thuận lợi, không phải trung bình (tự suy luận từ cách trình bày
  bảng, không phải bài báo tự thừa nhận).
- *"Mở rộng thế nào cho tin cậy hơn?"* → Xem mục 7.6.
