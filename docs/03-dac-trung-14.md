# 03 — 14 đặc trưng (features)

Với mỗi kênh EMG (một chuỗi tín hiệu `x` theo thời gian), ta tính **14 con số**
mô tả tín hiệu đó. 8 con số đầu mô tả tín hiệu ở **miền thời gian**, 6 con số sau
mô tả ở **miền tần số**. Toàn bộ cài đặt ở `src/feature_extraction.py`.

Thứ tự 14 đặc trưng là cố định (khai báo trong `config.FEATURE_NAMES`).

## 3.1. Nhóm miền thời gian (8 đặc trưng)

| # | Tên | Công thức / ý nghĩa | Liên hệ với mỏi cơ |
|---|---|---|---|
| 1 | **RMS** (Root Mean Square) | √(trung bình của x²). Đo **năng lượng/biên độ hiệu dụng** của tín hiệu. | Biến đổi khi mỏi (trong co cơ dưới mức tối đa kéo dài thường **tăng**). |
| 2 | **MAV** (Mean Absolute Value) | Trung bình của \|x\|. Cũng đo biên độ, ít nhạy nhiễu đột biến hơn RMS. | Tương tự RMS. Là 1 trong top-3 mRMR của **bài báo BME**. |
| 3 | **Skewness** (độ lệch) | Độ bất đối xứng của phân phối biên độ (âm/dương). | Phản ánh thay đổi hình dạng tín hiệu. |
| 4 | **Kurtosis** (độ nhọn) | Độ "nhọn"/đuôi dày của phân phối (chuẩn = 3). | Thay đổi khi dạng sóng thay đổi. |
| 5 | **Max** | Giá trị lớn nhất của tín hiệu. | Liên quan biên độ đỉnh, mức lực. |
| 6 | **Min** | Giá trị nhỏ nhất. | Tương tự Max. |
| 7 | **STD** (độ lệch chuẩn) | Mức dao động quanh giá trị trung bình. | Gần với RMS về ý nghĩa biên độ. |
| 8 | **Mean** (trung bình) | Giá trị trung bình của tín hiệu (thường gần 0 sau khi khử offset). | Kiểm tra độ lệch nền. |

> Ghi chú cài đặt: `Skewness` và `Kurtosis` dùng công thức chuẩn hoá theo N
> (bias=True), `Kurtosis` là loại **non-Fisher** (giá trị của phân phối chuẩn = 3),
> `STD` chuẩn hoá theo N−1 — khớp quy ước của MATLAB gốc.

## 3.2. Nhóm miền tần số (6 đặc trưng)

Trước hết cần khái niệm **PSD (Power Spectral Density — mật độ phổ công suất)**:
biến đổi Fourier cho biết tín hiệu chứa bao nhiêu **năng lượng ở mỗi tần số**.
Ở dự án dùng hai cách tính PSD tuỳ mục đích (đều trong `feature_extraction.py`):

- Cho 3 đặc trưng Spectral Min/Max/STD: `PSD = |FFT(x, nfft)|² / nfft`, với
  `nfft = 2048`.
- Cho MDF/MNF/Spectral Entropy: dùng **periodogram** một phía trên dải `[0, Fs/2]`.

| # | Tên | Ý nghĩa | Liên hệ với mỏi cơ |
|---|---|---|---|
| 9 | **Spectral_Min** | Giá trị nhỏ nhất của PSD. | Mô tả nền phổ. |
| 10 | **Spectral_Max** | Đỉnh phổ công suất (tần số mạnh nhất). | Liên quan năng lượng đỉnh. |
| 11 | **Spectral_STD** | Độ phân tán của PSD — phổ "trải rộng" hay "tập trung". | Nhạy với thay đổi hình dạng phổ khi mỏi. |
| 12 | **MDF** (Median Frequency) | Tần số **chia đôi tổng công suất** phổ (nửa năng lượng nằm dưới MDF). | **Giảm** khi mỏi — chỉ dấu kinh điển. |
| 13 | **MNF** (Mean Frequency) | Tần số **trung bình có trọng số** theo công suất: Σ(f·PSD)/Σ(PSD). | **Giảm** khi mỏi — chỉ dấu kinh điển. |
| 14 | **Spectral_Entropy** | Entropy Shannon của phổ đã chuẩn hoá thành phân phối xác suất P(f)=S(f)/ΣS. Đo **độ "đều/hỗn loạn"** của phổ. | Thay đổi khi năng lượng dồn về vùng tần số thấp. |

### Vì sao MDF và MNF là chỉ dấu mỏi cơ quan trọng?

Khi cơ mỏi, tốc độ dẫn truyền điện thế dọc sợi cơ chậm lại → các thành phần tần
số cao suy yếu → toàn bộ phổ "nén" về phía tần số thấp. Cả **MDF** (điểm giữa của
phổ) và **MNF** (trọng tâm của phổ) vì thế đều **dịch xuống thấp**. Theo dõi MDF/MNF
giảm dần theo thời gian là phương pháp truyền thống để phát hiện mỏi cơ — được **cả
hai bài báo nhấn mạnh** (*"mean and median frequency show a linear or curvilinear
decrease in time"*).

> **Chú thích cài đặt (bám code MATLAB, có đối chiếu bài báo):**
> - Các hàm khớp đúng quy ước MATLAB: `rms`, `mean(abs())`, `skewness` (bias N),
>   `kurtosis` (bias N, non-Fisher), `std` (N−1), `medfreq`, `meanfreq`, `se`,
>   `fft(x, nfft)` với `nfft = 2^nextpow2(2000) = 2048`. Các công thức này cũng
>   tương ứng với công thức (1)–(6) trong bài báo BME (đối chiếu tham khảo).
> - *Lưu ý tham khảo:* trong bài báo BME, **nhãn của hai công thức (7)/(8) bị in
>   hoán đổi** — công thức đề "Median Frequency" thực chất là MNF (trung bình có
>   trọng số Σf·P/ΣP), còn "Mean Frequency" lại mô tả MDF (chia đôi tổng công suất).
>   Code dùng **đúng định nghĩa chuẩn** như hàm MATLAB `medfreq`/`meanfreq`, nên
>   không bị ảnh hưởng bởi lỗi in này.

## 3.3. Cách xem đặc trưng trong demo

- Tab **Signal & PSD**: chọn một file + một kênh, xem dạng sóng theo thời gian và
  đường PSD, kèm bảng 14 giá trị đặc trưng của chính kênh đó.
- Tab **Features**: vẽ **boxplot** so sánh phân phối từng đặc trưng giữa hai lớp
  Normal và Fatigue — nhìn nhanh xem đặc trưng nào tách hai lớp tốt (xem
  [tài liệu 06](06-huong-dan-doc-streamlit.md)).
