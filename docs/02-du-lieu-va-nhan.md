# 02 — Dữ liệu và cách gán nhãn

## 2.1. Giao thức đo (theo hai bài báo)

Cả hai bài báo dùng chung một bộ dữ liệu:

- **10 đối tượng** khỏe mạnh (3 nữ, 7 nam; tuổi TB 24, SD 1.5).
- Đo sEMG **cơ nhị đầu tay phải (biceps)** bằng ma trận điện cực **64 kênh**, tần
  số lấy mẫu **2 kHz (2000 Hz)**; đã lọc phần cứng 20–400 Hz.
- **Xác định MVC** = trung bình 3 lần đo lực tối đa (cách nhau 5 phút).
- **Trình tự đo:** tập làm quen ở 10% MVC → đo ngẫu nhiên 20%, 40%, 60%, 90% MVC →
  **bài kiểm tra gây mỏi ở 70% MVC** → đo lại 10% MVC (sau mỏi). Nghỉ 5 phút giữa
  các lần đo.

Trong bộ demo có **5 đối tượng** (subject 5, 7, 8, 9, 11).

## 2.2. Định dạng file

Dữ liệu thô nằm trong `dataset/`, mỗi lần đo là một file CSV:

- **Tên file:** `Sujet_{id}_{điều_kiện}_emg.csv`, ví dụ `Sujet_9_fatigue_70_emg.csv`.
- **Ngăn cách cột:** dấu chấm phẩy `;`; số thập phân dùng dấu `.`; **không có dòng tiêu đề**.
- **Cột 0:** chỉ số thời gian. **Cột 1–64:** 64 kênh EMG.
- **Số dòng lớn:** ví dụ `Sujet_9_10_emg.csv` ≈ 66 000 mẫu ≈ 33 giây ở 2000 Hz.

Xử lý trong `data_loader.load_channels()`: đọc bằng `pandas.read_csv(sep=';')`,
bỏ cột 0, giữ ma trận `(số_mẫu_thời_gian, 64)`.

## 2.3. Các điều kiện đo và ý nghĩa

Bài báo BME 2024 phân biệt rõ:
- "10/20/40/60/90% MVC" → **trạng thái trước khi gây mỏi** (pre-fatigue) nếu không
  ghi chú gì thêm.
- "**70% MVC trong lúc gây mỏi**" → được **chú thích rõ ràng** là trạng thái mỏi cơ.
- "**10% MVC sau khi gây mỏi**" (`10_ap_fatigue`) là một phép đo **hồi phục ở mức
  co cơ rất thấp (10% MVC)**, không phải phép đo trong lúc cơ đang mỏi — về mặt
  tín hiệu không thể hiện các dấu hiệu mỏi kinh điển (xem 2.4).

| Điều kiện trong tên file | Ý nghĩa | Trạng thái |
|---|---|---|
| `10`, `20`, `40`, `60` | Co cơ 10–60% MVC (đo trước khi gây mỏi) | Trước mỏi |
| `90` | Co cơ 90% MVC | Trước mỏi theo giao thức đo, nhưng > ngưỡng 60% MVC |
| `fatigue_70` | Bài tập gây mỏi ở 70% MVC | Trong lúc mỏi |
| `10_ap_fatigue` | Đo lại ở 10% MVC **sau** (après) khi đã gây mỏi | Sau mỏi, cường độ thấp |

## 2.4. Cách gán nhãn (ngưỡng %MVC > 60, cập nhật 2026-07-09)

**Trong code MATLAB gốc**, nhãn đến từ **cấu trúc thư mục**: `Feature_Extraction.m`
đọc riêng hai thư mục `Train_Data/Normal` và `Train_Data/Fatigue` (và tương tự cho
`Test_Data`), gán nhãn 0 cho file trong `Normal`, nhãn 1 cho file trong `Fatigue`.
Nội dung các thư mục gốc đó không nằm trong bộ demo, nên nhãn được suy lại từ tên
file — nhưng thay vì dò từ khoá `fatigue` trong tên (cách cũ, hai lần bị chỉnh vì
gán sai `10_ap_fatigue`), nhãn giờ dựa trực tiếp vào **số %MVC trong tên điều
kiện** so với ngưỡng sinh lý, khớp tiêu chí ICACE 2019 (cơ mỏi khi lực co vượt
60% MVC):

- **Fatigue (1):** số %MVC trong tên điều kiện **> 60** (`config.FATIGUE_MVC_THRESHOLD`).
- **Normal (0):** số %MVC **≤ 60**.

Áp dụng cho từng điều kiện trong bộ demo (`data_loader.parse_filename()` trích số
đầu tiên tìm thấy trong tên điều kiện bằng regex `\d+`):

| Điều kiện | Số %MVC trích ra | So với ngưỡng 60 | Nhãn |
|---|---|---|---|
| `10`, `20`, `40`, `60` | 10 / 20 / 40 / 60 | ≤ 60 | Normal (0) |
| `90` | 90 | > 60 | **Fatigue (1)** |
| `fatigue_70` | 70 | > 60 | **Fatigue (1)** |
| `10_ap_fatigue` | 10 (số đầu tiên trong tên) | ≤ 60 | Normal (0) |

So với cách gán trước đó (dò từ khoá `fatigue`), thay đổi chính là **file `90`
chuyển từ Normal → Fatigue**; `10_ap_fatigue` vẫn là Normal như lần sửa trước
(số %MVC của nó là 10, không đổi theo ngưỡng mới). Cách này khớp đúng phần "Ghi
chú tham khảo" mà tài liệu này từng nêu (ICACE 2019, ">60% MVC") — trước đây bị
gạt sang một bên vì ưu tiên bám theo cấu trúc thư mục MATLAB, nay được dùng trực
tiếp theo yêu cầu người dùng.

## 2.5. Đơn vị mẫu và kênh lỗi

- **Một "mẫu" (sample) = một kênh EMG của một file.** Với mỗi kênh, tính một vector
  14 đặc trưng trên toàn bộ chiều dài tín hiệu. Một file 64 kênh → tối đa 64 mẫu.
- **Kênh lỗi (toàn giá trị 0):** bị loại trước khi tính đặc trưng. Đây **không phải
  quy ước tự đặt** — ICACE 2019 nêu rõ: *"the input signal contained many erroneous
  data points, represented by the zero-value in the data table, due to the
  acquisition phase mistaken"* (lỗi ở khâu thu tín hiệu). Mask tính riêng cho từng
  file trong `data_loader.valid_channel_mask()`.

## 2.6. Chia train / test — Leave-One-Subject-Out

Code MATLAB gốc chia train/test bằng **thư mục** (`Train_Data` vs `Test_Data`) và
`Import.m` gom dữ liệu **theo từng subject** (chuẩn bị cho leave-one-subject-out).
Bộ dữ liệu demo phẳng nên ta **tái tạo lại cách chia theo đối tượng**:

- **Train:** subject **{5, 7, 8, 11}**.
- **Test:** subject **9** (giữ riêng hoàn toàn — mô hình chưa từng thấy).

Cách này đo đúng khả năng **tổng quát hoá cho người mới**, thay vì trộn ngẫu nhiên
các kênh của cùng một người vào cả train lẫn test (sẽ rò rỉ và cho điểm ảo cao).

> Ghi chú tham khảo: bài BME 2024 cũng dùng leave-one-subject-out; ICACE 2019 thì
> chia 80/20 ngẫu nhiên. Demo theo cách chia theo subject như trên.

## 2.7. Thống kê tập dữ liệu (demo hiện tại)

| Chỉ số | Giá trị |
|---|---|
| Tổng số mẫu (kênh × file hợp lệ) | **1270** |
| Trong đó Normal / Fatigue | 950 / 320 |
| Số file | 20 |
| Tập train (subject 5,7,8,11) | 824 mẫu (Normal 632, Fatigue 192) |
| Tập test (subject 9) | 446 mẫu (Normal 318, Fatigue 128) |

> Số liệu sau khi đổi sang gán nhãn theo ngưỡng %MVC > 60 (2026-07-09, mục 2.4):
> file `90` chuyển sang Fatigue nên lớp Fatigue tăng trở lại so với lần sửa
> `10_ap_fatigue` trước đó, tỉ lệ mất cân bằng cũng dịu hơn (train ≈ 3.3:1 thay vì
> 5.4:1).

Các con số này in ra khi chạy `python -m src.pipeline` và hiển thị ở tab
**Overview** của Streamlit.
