# 07 — Diễn giải kết quả demo (trình bày với mentor)

Tài liệu này tổng hợp cách **kể câu chuyện kết quả** một cách trung thực, kèm điểm
mạnh, hạn chế và các câu hỏi có thể gặp.

## 7.1. Thông điệp chính (một câu)

> "Pipeline hoạt động đúng và chạy được đầu-cuối trên dữ liệu thật: với đặc trưng
> phổ + biên độ và một bộ phân loại tuyến tính đơn giản (LDA/SVM), ta phân biệt
> được kênh EMG mỏi vs bình thường trên một đối tượng **chưa từng thấy** với
> F1 ≈ 0.92–0.93 và AUC ≈ 0.98–0.99."

## 7.2. Bảng kết quả và cách đọc

| Mô hình | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| SVM (top-3) | 0.951 | 0.853 | 1.000 | 0.921 | 0.977 |
| KNN (14) | 0.933 | 0.938 | 0.820 | 0.875 | 0.899 |
| **LDA (14)** | **0.957** | 0.876 | 0.992 | **0.930** | **0.996** |
| Decision Tree (14) | 0.805 | 0.652 | 0.688 | 0.669 | 0.770 |

Những điểm nên nói:

- **LDA và SVM tốt nhất.** LDA đạt F1 0.930, AUC 0.996; SVM đạt Recall **1.000**
  (không bỏ sót ca mỏi nào) với chỉ **3 đặc trưng**. Điều này cho thấy hai lớp
  **tách tuyến tính khá tốt** trong không gian đặc trưng đã chọn.
- **SVM: Recall 1.0 nhưng Precision 0.85** → mô hình "thà báo nhầm còn hơn bỏ sót".
  Trong bối cảnh cảnh báo mỏi cơ (bỏ sót nguy hiểm hơn báo nhầm), đây là đánh đổi
  hợp lý.
- **Decision Tree kém nhất (F1 0.67)** dù CV-Acc trên train tới 0.98 → ví dụ điển
  hình của **overfit**: cây học thuộc đặc điểm nhóm train nhưng không tổng quát
  sang subject 9. Đây là lý do một cây đơn thường thua các mô hình tuyến tính/ensemble.
- **So với bài báo BME 2024** (KNN F1 ≈ 0.95, AUC ≈ 0.95): xu hướng tương đồng,
  xác nhận pipeline chuyển đổi đúng. Con số tuyệt đối khác vì demo chỉ dùng 5/10
  đối tượng.

## 7.3. Vì sao top-3 mRMR khác bài báo?

Bài báo BME 2024 báo cáo top-3 mRMR là **MAV, Skewness, Mean**. Demo lại chọn
**Spectral_Entropy, Spectral_STD, Max**. Giải thích trung thực:

- Demo chỉ dùng **5/10 đối tượng**, tập con và cách chia khác, và cài đặt mRMR
  (MRMR-FCQ tự viết) khác `fscmrmr` của MATLAB → thứ hạng đặc trưng đổi là bình thường.
- Chính bài báo cũng lưu ý mRMR *"chỉ mang tính chỉ báo"* (indicative): đặc trưng
  mRMR chọn không đảm bảo là tối ưu cho mọi mô hình. Đó là lý do KNN/LDA vẫn dùng
  cả 14 đặc trưng.
- MDF/MNF (chỉ dấu mỏi kinh điển) vẫn nằm trong 14 đặc trưng và vẫn được các mô
  hình dùng 14 đặc trưng khai thác — chúng chỉ không lọt top-3 của mRMR trên tập
  dữ liệu nhỏ này.

## 7.4. Điểm mạnh của bản demo

1. **Convert trung thành code MATLAB gốc:** giữ đúng 14 đặc trưng, thứ tự đặc trưng,
   mRMR, cả 4 mô hình (SVM/KNN/LDA/DecisionTree) và siêu tham số như 7 file `.m`.
2. **Sửa các lỗi rõ ràng của code MATLAB** (theo `PROJECT_SPEC.md`): không gán cứng
   chỉ số TP/FP mà tính từ confusion matrix thật; sửa lỗi luỹ thừa PSD `^4`→`^2`;
   bỏ dead code / lỗi cú pháp.
3. **Chạy được đầu-cuối trên dữ liệu thật**, có cache đặc trưng và giao diện trực quan.
4. **Đánh giá nghiêm túc:** leave-one-subject-out (test trên người chưa thấy),
   không rò rỉ dữ liệu; chọn đặc trưng chỉ fit trên train.

## 7.5. Hạn chế (nên chủ động nêu trước khi mentor hỏi)

1. **Dữ liệu nhỏ:** 5 đối tượng, 20 file, 1270 mẫu kênh. Kết quả mang tính minh hoạ.
2. **Không có bước lọc nhiễu (wavelet denoising):** demo **giữ đúng như code MATLAB
   `Feature_Extraction.m`** — đọc thẳng CSV rồi tính đặc trưng, không lọc lại (dữ
   liệu đã lọc phần cứng 20–400 Hz). Sơ đồ trong bài báo *có* mô tả wavelet denoising
   + bandpass + notch nhưng code gốc không triển khai; nếu muốn bám bài báo hơn thì
   đây là bước có thể bổ sung.
3. **Gán nhãn từ tên file:** khôi phục phân chia Normal/Fatigue của code MATLAB
   (thư mục Normal/Fatigue) bằng quy tắc tên file chứa `fatigue`. Vì cả hai lớp đều
   trải nhiều mức lực (Normal có cả 90% MVC, Fatigue có cả 10% sau mỏi) nên mô hình
   **không thể "ăn gian" chỉ bằng biên độ/lực**; song ranh giới lực–mỏi chưa được cô
   lập hoàn toàn. (Bài báo ICACE nêu tiêu chí ">60% MVC" chỉ mang tính tham khảo —
   xem [tài liệu 02, mục 2.4](02-du-lieu-va-nhan.md).)
4. **Chỉ test 1 đối tượng:** nên làm leave-one-subject-out **luân phiên toàn bộ**
   đối tượng rồi lấy trung bình để đánh giá vững hơn.
5. **Mẫu = toàn bộ tín hiệu một kênh:** chưa dùng cửa sổ trượt theo thời gian, nên
   chưa theo dõi được **diễn tiến mỏi** trong một lần đo.

## 7.6. Hướng phát triển tiếp

- Đánh giá **leave-one-subject-out luân phiên** trên cả 5 đối tượng, báo cáo
  trung bình ± độ lệch.
- Thêm **cửa sổ trượt** (ví dụ 1–2 giây) để quan sát MDF/MNF giảm dần theo thời gian.
- Kiểm soát yếu tố lực: so sánh trong cùng mức %MVC để cô lập ảnh hưởng của mỏi.
- Bổ sung **Random Forest thật** (ensemble) để so với cây đơn.
- Thu thập thêm đối tượng để có kết luận thống kê vững hơn.

## 7.7. Kịch bản demo gợi ý (5–7 phút)

1. **Bài toán & dữ liệu** (tab Overview): giải thích mỏi cơ, quy mô, cân bằng lớp.
2. **Tín hiệu thật** (tab Signal & PSD): mở 1 kênh mỏi vs 1 kênh bình thường, chỉ
   ra khác biệt biên độ/phổ.
3. **Đặc trưng phân biệt** (tab Features): mở boxplot vài đặc trưng, chỉ hộp ít
   chồng lấn.
4. **Chọn đặc trưng** (tab Feature Selection): giải thích mRMR và top-3.
5. **Kết quả** (tab Classification): bảng chỉ số + confusion matrix + ROC; nhấn
   mạnh LDA/SVM, nêu overfit của Decision Tree.
6. **Chốt:** điểm mạnh, hạn chế (mục 7.5) và hướng phát triển (mục 7.6).

## 7.8. Câu hỏi mentor có thể hỏi & gợi ý trả lời

- *"Vì sao không trộn ngẫu nhiên train/test?"* → Sẽ rò rỉ: các kênh của cùng một
  người rất giống nhau, lọt vào cả train lẫn test làm điểm ảo cao. Leave-one-
  subject-out đo đúng khả năng dùng cho **người mới**.
- *"Accuracy 95% có phải quá tốt/đáng ngờ?"* → Nhìn kèm cỡ dữ liệu nhỏ và việc cả
  hai lớp trải nhiều mức lực (mục 7.5). Vì test trên **người chưa thấy** nên con số
  không phải do rò rỉ; vẫn nên báo cáo F1/AUC và nêu rõ hạn chế.
- *"Gán nhãn thế nào?"* → Code MATLAB tách lớp bằng thư mục Normal/Fatigue; dữ liệu
  demo phẳng nên ta khôi phục bằng tên file (chứa `fatigue` → Fatigue). `90%` là
  baseline nên Normal, `10_ap_fatigue` đo sau mỏi nên Fatigue (mục 2.4). (Bài ICACE
  có nêu ngưỡng ">60% MVC" nhưng chỉ là tham khảo.)
- *"Đã lọc/khử nhiễu tín hiệu chưa?"* → Không — demo **giữ đúng code MATLAB**
  (`Feature_Extraction.m` đọc thẳng CSV, không wavelet denoising); dữ liệu đã lọc
  phần cứng 20–400 Hz. Bài báo có mô tả bước lọc này nhưng code gốc không làm; đây
  là hướng có thể bổ sung.
- *"Vì sao Decision Tree kém?"* → Overfit: CV-Acc train 0.98 nhưng test 0.81; một
  cây đơn dễ học thuộc nhóm train. Decision Tree cũng **nằm ngoài phạm vi bài báo**
  (bài chỉ dùng SVM/LDA/KNN).
- *"Top-3 đặc trưng có khớp bài báo không?"* → Bài báo là MAV/Skewness/Mean, demo
  là Spectral_Entropy/Spectral_STD/Max; khác do dữ liệu nhỏ và cài đặt mRMR khác
  (mục 7.3). Bài báo cũng nói đặc trưng mRMR "chỉ mang tính chỉ báo".
- *"Số liệu có khớp bài báo không?"* → Cùng xu hướng (KNN F1 ~0.95, AUC ~0.95);
  khác tuyệt đối do chỉ dùng 5/10 đối tượng và bỏ bước tiền xử lý.
- *"Mở rộng thế nào cho tin cậy hơn?"* → Xem mục 7.6.
