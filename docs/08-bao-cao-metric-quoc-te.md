# 08 — Báo Cáo Phân Tích Chỉ Số Đánh Giá (Evaluation Metrics Report)
**Dự án:** Nhận diện và Phân loại Trạng thái Mỏi Cơ qua Tín hiệu Điện Cơ Bề Mặt (sEMG)
**Phiên bản Tài liệu:** 1.0

> Báo cáo này trình bày chi tiết về khung đánh giá tiêu chuẩn quốc tế được áp dụng trong dự án phân loại tín hiệu sEMG, phục vụ cho việc thẩm định tính hiệu quả và độ tin cậy của các mô hình học máy.

## 1. Khung Đánh Giá Cốt Lõi (Core Evaluation Framework)

Toàn bộ hệ thống đánh giá dựa trên **Ma trận Nhầm lẫn (Confusion Matrix)** được tính toán trên tập dữ liệu kiểm thử (Test Set) độc lập. Trong ngữ cảnh y sinh và sinh lý học của dự án:
- **Lớp Dương (Positive - Class 1):** Trạng thái mỏi cơ (Fatigue).
- **Lớp Âm (Negative - Class 0):** Trạng thái cơ bình thường (Normal).

Các thành phần cơ bản của Ma trận Nhầm lẫn:
- **True Positive (TP):** Số mẫu cơ mỏi được mô hình nhận diện chính xác là mỏi.
- **True Negative (TN):** Số mẫu cơ bình thường được nhận diện chính xác là bình thường.
- **False Positive (FP - Lỗi Loại I):** Cơ bình thường nhưng mô hình cảnh báo mỏi (Báo động giả / False Alarm).
- **False Negative (FN - Lỗi Loại II):** Cơ đang mỏi nhưng mô hình không phát hiện ra (Bỏ sót).

> **Lưu ý Quan trọng:**
> Trong bối cảnh phân tích sEMG, **Lỗi Loại II (FN)** mang rủi ro cao nhất vì việc không phát hiện ra trạng thái mỏi có thể dẫn đến chấn thương thể thao hoặc sai lệch trong chẩn đoán y khoa.

## 2. Các Chỉ Số Hiệu Năng Dẫn Xuất (Derived Performance Metrics)

Từ các thành phần cốt lõi, hệ thống tính toán các chỉ số chuẩn quốc tế để đánh giá hiệu năng mô hình:

### 2.1. Độ Chính Xác Tổng Thể (Accuracy)
- **Định nghĩa:** Tỷ lệ phần trăm các dự đoán đúng (bao gồm cả trạng thái mỏi và không mỏi) trên tổng số dự đoán.
- **Công thức:** `Accuracy = (TP + TN) / (TP + TN + FP + FN)`
- **Lưu ý:** Chỉ số này không được ưu tiên tối thượng do tính chất mất cân bằng tự nhiên của dữ liệu (Imbalanced Data) – số lượng mẫu đo ở trạng thái Normal thường nhiều hơn Fatigue.

### 2.2. Độ Chuẩn Xác (Precision / Positive Predictive Value)
- **Định nghĩa:** Tỷ lệ dự đoán đúng trong số tất cả các ca mà mô hình cảnh báo là "Mỏi cơ".
- **Công thức:** `Precision = TP / (TP + FP)`
- **Ý nghĩa:** Chỉ số này quan trọng để đánh giá độ tin cậy của hệ thống cảnh báo, giúp giảm thiểu sự phiền toái của các báo động giả (FP).

### 2.3. Độ Nhạy (Recall / Sensitivity / True Positive Rate)
- **Định nghĩa:** Tỷ lệ phát hiện thành công trong số tất cả các ca "Mỏi cơ" thực tế.
- **Công thức:** `Recall = TP / (TP + FN)`
- **Ý nghĩa Lâm sàng:** Đóng vai trò tối quan trọng. Recall cao đảm bảo hệ thống không bỏ lỡ các thời điểm cơ bắp bắt đầu đi vào trạng thái suy kiệt.

### 2.4. Điểm F1 (F1-Score)
- **Định nghĩa:** Trung bình điều hòa (Harmonic Mean) giữa Precision và Recall.
- **Công thức:** `F1-Score = 2 * (Precision * Recall) / (Precision + Recall)`
- **Vai trò trong hệ thống:** Được sử dụng làm **chỉ số đánh giá chính (Primary Metric)**. Do sự chênh lệch phân phối mẫu giữa hai lớp (Normal: 320 mẫu vs Fatigue: 128 mẫu trên tập Test hiện tại, subject 6), F1-Score cung cấp một cái nhìn khách quan và toàn diện hơn về chất lượng phân lớp so với Accuracy.

## 3. Đánh Giá Không Phụ Thuộc Ngưỡng (Threshold-Independent Evaluation)

Để so sánh công bằng giữa các thuật toán lõi (KNN 1-NN, KNN 5-NN, SVM linear, LDA) trước khi thiết lập ngưỡng quyết định (decision threshold) cố định:

### 3.1. Đường Cong ROC (Receiver Operating Characteristic)
- Đồ thị biểu diễn sự tương quan và đánh đổi giữa **True Positive Rate (Recall)** và **False Positive Rate (FPR = FP / (FP + TN))** qua nhiều ngưỡng cắt (cut-off) khác nhau.

### 3.2. Diện Tích Dưới Đường Cong (AUC - Area Under Curve)
- **Định nghĩa:** Giá trị tích phân biểu thị toàn bộ diện tích nằm dưới đường cong ROC.
- **Thang đo chuẩn:**
  - `0.9 - 1.0`: Hiệu năng xuất sắc (Outstanding).
  - `0.8 - 0.9`: Hiệu năng tốt (Excellent).
  - `0.7 - 0.8`: Hiệu năng chấp nhận được (Acceptable).
  - `0.5`: Đoán ngẫu nhiên (Random guess).
- **Ý nghĩa:** AUC đánh giá khả năng **xếp hạng (ranking)** của mô hình. Một mô hình có AUC cao sẽ phân tách rõ rệt phổ xác suất của lớp Fatigue và lớp Normal.

## 4. Phương Pháp Kiểm Định & Tổng Quát Hóa (Validation Methodology)

Hệ thống tuân thủ nghiêm ngặt các quy chuẩn kiểm định chéo để ngăn chặn hiện tượng học vẹt (Overfitting):

### 4.1. Leave-One-Subject-Out (LOSO) Cross-Validation
- Dữ liệu sEMG mang tính cá thể hóa rất cao (Subject-specific). Nếu trộn dữ liệu của cùng một người vào cả tập Train và Test, rò rỉ dữ liệu (Data Leakage) sẽ xảy ra, dẫn đến kết quả ảo.
- **Giải pháp:** Hệ thống tách bạch hoàn toàn tập dữ liệu theo đối tượng. (Cấu hình hiện tại: Train trên Subject 5, 7, 8, 9, 10, 11, 12, 13, 14 và Test độc lập trên Subject 6). Điều này phản ánh chính xác **khả năng tổng quát hóa** của AI khi áp dụng lên một người dùng mới. Ngoài cấu hình cố định này, hệ thống còn chạy **leave-one-subject-out đầy đủ qua cả 10 subject** (`scripts/run_loso.py`) để báo cáo trung bình ± độ lệch chuẩn, tránh kết luận chỉ dựa trên 1 fold thuận lợi.

### 4.2. Stratified K-Fold Cross-Validation (CV-Acc)
- Bên trong nội bộ tập Train, thuật toán `Stratified K-Fold` được sử dụng để duy trì tỷ lệ mất cân bằng giữa hai lớp qua các fold.
- **Mục đích:** CV-Acc so sánh với Accuracy trên tập Test giúp các kỹ sư nhận diện sớm sự chênh lệch hiệu năng, đảm bảo tính ổn định và mạnh mẽ (robustness) của kiến trúc mô hình.
