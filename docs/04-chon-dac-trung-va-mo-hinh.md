# 04 — Chọn đặc trưng (mRMR) và các mô hình

## 4.1. Vì sao cần chọn đặc trưng?

Có 14 đặc trưng, nhưng không phải cái nào cũng hữu ích và nhiều cái **trùng lặp
thông tin** (ví dụ RMS, MAV, STD đều đo biên độ). Nếu đưa cả 14 vào một số mô hình
nhạy cảm (như SVM), các đặc trưng dư thừa/nhiễu có thể làm giảm hiệu năng. Ta chọn
một **tập nhỏ đặc trưng vừa liên quan mạnh tới nhãn, vừa ít trùng lặp nhau**.

## 4.2. Thuật toán mRMR (MRMR-FCQ)

**mRMR = minimum Redundancy, Maximum Relevance** (tối thiểu dư thừa, tối đa liên
quan). Cài đặt ở `src/feature_selection.py`. Nguyên tắc chọn tham lam từng bước:

- **Relevance (độ liên quan):** đo bằng **mutual information** giữa mỗi đặc trưng
  và nhãn (dùng `mutual_info_classif`). Càng cao → đặc trưng càng phân biệt được
  Normal/Fatigue.
- **Redundancy (độ dư thừa):** đo bằng **tương quan tuyệt đối trung bình** với các
  đặc trưng đã chọn trước đó. Càng cao → càng trùng lặp thông tin.
- Mỗi bước chọn đặc trưng tối đa hoá `relevance − redundancy`.

Kết quả là một **thứ hạng đầy đủ** 14 đặc trưng (tốt → kém), ta lấy **top-3** đầu.

> Quan trọng: mRMR chỉ được tính trên **tập train** để tránh rò rỉ thông tin từ
> tập test.

### Kết quả xếp hạng trong demo hiện tại

Thứ hạng mRMR (từ tốt nhất):

`Spectral_Entropy → Spectral_STD → Max → Kurtosis → Mean → RMS → Spectral_Max →
Skewness → Spectral_Min → STD → MDF → MAV → MNF → Min`

→ **Top-3 được chọn: Spectral_Entropy, Spectral_STD, Max.**

**Đối chiếu với bài báo:** BME 2024 báo cáo top-3 mRMR là **MAV, Skewness, Mean**
(*"the MAV, Skewness, and Mean features were appropriate for classifying muscle
fatigue"*). Top-3 của demo khác vì (1) chỉ dùng **5/10 đối tượng**, (2) tập con
dữ liệu và cách chia khác, (3) cài đặt mRMR (MRMR-FCQ) khác với `fscmrmr` của MATLAB.
Đây là điều **bình thường** — mRMR nhạy với dữ liệu; chính bài báo cũng lưu ý rằng
đặc trưng do mRMR chọn "chỉ mang tính chỉ báo" (indicative). Xem tab **Feature
Selection** để thấy biểu đồ xếp hạng.

> Lưu ý trung thực: bản thân bài báo BME cũng nêu ở phần Hạn chế rằng mRMR *"khi
> đưa vào mô hình phân loại thì hiệu năng không tối ưu, các đặc trưng này chỉ mang
> tính chỉ báo"*. Vì vậy KNN/LDA trong bài (và trong demo) vẫn dùng **cả 14 đặc
> trưng**, chỉ riêng SVM dùng top-3.

## 4.3. Bốn mô hình phân loại (theo code MATLAB)

Code MATLAB có **4 file classifier** (`SVMClassification.m`, `KNNClassification.m`,
`LDAClassification.m`, `RFClassification.m`), nên demo convert đủ **4 mô hình**.
Định nghĩa ở `src/models.py`, giữ đúng siêu tham số của code MATLAB:

| Mô hình | File MATLAB | Ý tưởng | Cấu hình chính | Đặc trưng dùng |
|---|---|---|---|---|
| **SVM** | `SVMClassification.m` | Tìm siêu phẳng tách hai lớp với lề lớn nhất. | Kernel tuyến tính, `C=1`, có **chuẩn hoá** (StandardScaler). | **Top-3** mRMR |
| **KNN** | `KNNClassification.m` | Gán nhãn theo hàng xóm gần nhất. | `k=1`, khoảng cách Euclidean, có chuẩn hoá. | Cả 14 |
| **LDA** | `LDAClassification.m` | Tìm hướng chiếu tuyến tính tách hai lớp tốt nhất. | Solver `svd`. | Cả 14 |
| **Decision Tree** | `RFClassification.m` | Cây quyết định chia dữ liệu theo ngưỡng từng đặc trưng. | Tiêu chí Gini, tối đa 21 lá (≈ 20 lần chia). | Cả 14 |

> **Vì sao gọi là Decision Tree chứ không phải Random Forest?** File `RFClassification.m`
> tuy đặt tên "RF" nhưng thực chất chỉ tạo **một cây quyết định đơn** (`fitctree`),
> không có ensemble/bagging. Nên ở đây đặt đúng tên là Decision Tree.
>
> *Tham khảo:* bài báo BME 2024 chỉ báo cáo 3 mô hình (SVM/LDA/KNN); Decision Tree
> là phần code MATLAB có thêm ngoài bài báo.

**Chuẩn hoá (Standardize):** SVM và KNN dựa trên khoảng cách, nên các đặc trưng
có thang đo rất khác nhau (ví dụ tần số hàng trăm Hz vs biên độ nhỏ) cần được đưa
về cùng thang bằng StandardScaler (trừ trung bình, chia độ lệch chuẩn). LDA và
Decision Tree không cần bước này.

## 4.4. Huấn luyện và kiểm thử

- **Cross-validation trên tập train:** dùng **StratifiedKFold** (giữ tỉ lệ lớp),
  mặc định 9 fold, để ước lượng độ ổn định của mô hình khi học (cột *CV-Acc*).
- **Kiểm thử thật:** huấn luyện trên toàn bộ tập train rồi dự đoán trên **subject
  9** (đối tượng giữ riêng). Mọi chỉ số báo cáo đều tính trên tập test này.

Chi tiết các chỉ số ở [tài liệu 05](05-chi-so-danh-gia.md).
