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

Kết quả là một **thứ hạng đầy đủ** 14 đặc trưng (tốt → kém), lấy ra **top-3** để
báo cáo (tab Feature Selection) — nhưng lưu ý (mục 4.3): trong pipeline hiện tại
top-3 này **chỉ mang tính minh hoạ/báo cáo**, không còn được dùng để giới hạn
đặc trưng của bất kỳ model nào (mọi model đều train trên cả 14 đặc trưng).

> Quan trọng: mRMR chỉ được tính trên **tập train** để tránh rò rỉ thông tin từ
> tập test.

### Kết quả xếp hạng trong demo hiện tại (dataset 10 subject, cập nhật 2026-07-14)

Thứ hạng mRMR (từ tốt nhất, tính trên tập train = subject {5,7,8,9,10,11,12,13,14}):

`MAV → Spectral_Max → Kurtosis → Mean → Min → STD → MDF → Spectral_Min →
Spectral_STD → RMS → Skewness → Max → MNF → Spectral_Entropy`

→ **Top-3 được chọn: MAV, Spectral_Max, Kurtosis.**

> Con số này thay đổi so với lần chạy trước (`Spectral_Entropy, Spectral_STD,
> Max`) vì dataset đã tăng từ 5 lên **10 đối tượng** (xem
> [tài liệu 02](02-du-lieu-va-nhan.md)) và tập train hiện là subject
> {5,7,8,9,10,11,12,13,14} (test = subject 6) thay vì {5,7,8,11} (test = subject 9).

**Đối chiếu với bài báo:** BME 2024 báo cáo top-3 mRMR là **MAV, Skewness, Mean**
(*"the MAV, Skewness, and Mean features were appropriate for classifying muscle
fatigue"*). Demo hiện đã khớp **1/3 đặc trưng (MAV)** với bài báo — gần hơn so với
lần chạy 5-subject trước đây (0/3 trùng). Phần khác biệt còn lại (Spectral_Max,
Kurtosis vs Skewness, Mean) đến từ cài đặt mRMR (MRMR-FCQ tự viết) khác với
`fscmrmr` của MATLAB, và cách chia train/test khác. Xem tab **Feature Selection**
để thấy biểu đồ xếp hạng.

> Lưu ý trung thực: bản thân bài báo BME cũng nêu ở phần Hạn chế rằng mRMR *"khi
> đưa vào mô hình phân loại thì hiệu năng không tối ưu, các đặc trưng này chỉ mang
> tính chỉ báo"*. Trong demo hiện tại, **tất cả 4 model (KNN 1NN/5NN, SVM, LDA)
> đều dùng cả 14 đặc trưng** — top-3 mRMR chỉ hiển thị để minh hoạ, không còn
> giới hạn đặc trưng của SVM như phiên bản trước.

## 4.3. Bốn mô hình phân loại đang chạy (paper-faithful, `src/loso.py`)

Pipeline hiện tại (`build_loso_models()` trong `src/loso.py`, dùng chung cho CLI
`python -m src.pipeline` và Streamlit) giữ **4 mô hình**, tất cả **chuẩn hoá
StandardScaler + cả 14 đặc trưng**, không SMOTE, không tinh chỉnh ngưỡng
(ngưỡng quyết định mặc định 0.5):

| Mô hình | Gốc MATLAB | Ý tưởng | Cấu hình chính | Đặc trưng dùng |
|---|---|---|---|---|
| **KNN(1-NN)** | `KNNClassification.m` | Gán nhãn theo hàng xóm gần nhất. | `k=1`, khoảng cách Euclidean — đúng bản gốc MATLAB. | Cả 14 |
| **KNN(5-NN)** | *(biến thể thêm)* | Như trên nhưng `k=5` — ổn định hơn 1-NN. | `k=5`, Euclidean. | Cả 14 |
| **SVM(linear)** | `SVMClassification.m` | Tìm siêu phẳng tách hai lớp với lề lớn nhất. | Kernel tuyến tính. | Cả 14 (không còn giới hạn top-3) |
| **LDA** | `LDAClassification.m` | Tìm hướng chiếu tuyến tính tách hai lớp tốt nhất. | Solver mặc định `sklearn`. | Cả 14 |

> **Về Decision Tree/`RFClassification.m`:** bản Decision Tree (thực chất là
> `fitctree` — một cây đơn, không phải Random Forest thật) từng có trong
> `src/models.py`, nhưng **không còn được `app.py` gọi tới** ở pipeline hiện
> tại — chỉ còn dùng trong `tests/test_evaluate.py`. Bộ `src/models.py` +
> `src/evaluate.py` (SVM top-3 + SMOTE, KNN/LDA/DecisionTree + SMOTE +
> GridSearchCV tinh chỉnh ngưỡng) vẫn tồn tại trong code nhưng là phiên bản cũ
> hơn, tách biệt khỏi luồng chạy chính (`src/loso.py` → `src/pipeline.py`).
>
> *Tham khảo:* bài báo BME 2024 báo cáo 3 mô hình (SVM/LDA/KNN); demo thêm
> **KNN(5-NN)** làm biến thể ổn định hơn để so sánh với 1-NN gốc.

**Chuẩn hoá (Standardize):** SVM và KNN dựa trên khoảng cách, nên các đặc trưng
có thang đo rất khác nhau (ví dụ tần số hàng trăm Hz vs biên độ nhỏ) cần được đưa
về cùng thang bằng StandardScaler. LDA cũng được chuẩn hoá trong pipeline hiện
tại (đồng nhất bước tiền xử lý giữa các model).

## 4.4. Huấn luyện và kiểm thử

- **Cross-validation trên tập train:** dùng **StratifiedKFold** (giữ tỉ lệ lớp),
  mặc định 9 fold, để ước lượng độ ổn định của mô hình khi học (cột *CV-Acc*).
- **Kiểm thử thật (pipeline single-split, `config.TEST_SUBJECT`):** huấn luyện
  trên toàn bộ tập train (`config.TRAIN_SUBJECTS = [5,7,8,9,10,11,12,13,14]`)
  rồi dự đoán trên **subject 6** (đối tượng giữ riêng, cấu hình hiện tại). Mọi
  chỉ số báo cáo trong tab Classification đều tính trên tập test này.
- **Đánh giá vững hơn (LOSO đầy đủ, `scripts/run_loso.py`):** lặp lại phép test
  trên **cả 10 subject** (mỗi lần giữ 1 subject làm test), rồi báo cáo cả trung
  bình ± độ lệch chuẩn qua từng fold **và** chỉ số pooled (gộp toàn bộ dự đoán
  lại tính một lần) — xem `results/loso_results.txt`, `results/new_results.txt`
  và [tài liệu 05](05-chi-so-danh-gia.md) mục 5.6.

Chi tiết các chỉ số ở [tài liệu 05](05-chi-so-danh-gia.md).
