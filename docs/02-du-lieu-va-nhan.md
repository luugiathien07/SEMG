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
- "**70% MVC trong lúc gây mỏi**" và "**10% MVC sau khi gây mỏi**" → được **chú
  thích rõ ràng** là trạng thái liên quan mỏi cơ.

| Điều kiện trong tên file | Ý nghĩa | Trạng thái |
|---|---|---|
| `10`, `20`, `40`, `60`, `90` | Co cơ 10–90% MVC (đo trước khi gây mỏi) | Trước mỏi |
| `fatigue_70` | Bài tập gây mỏi ở 70% MVC | Trong lúc mỏi |
| `10_ap_fatigue` | Đo lại ở 10% MVC **sau** (après) khi đã gây mỏi | Sau mỏi |

## 2.4. Cách gán nhãn (khôi phục lại phân chia Normal/Fatigue của code MATLAB)

**Trong code MATLAB gốc**, nhãn đến từ **cấu trúc thư mục**: `Feature_Extraction.m`
đọc riêng hai thư mục `Train_Data/Normal` và `Train_Data/Fatigue` (và tương tự cho
`Test_Data`), gán nhãn 0 cho file trong `Normal`, nhãn 1 cho file trong `Fatigue`.

Bộ dữ liệu demo là **phẳng** (không có sẵn thư mục Normal/Fatigue), nên ta **khôi
phục lại phân chia đó từ tên file**: tên chứa chữ `fatigue` → **Fatigue (1)**; còn
lại → **Normal (0)** (`data_loader.parse_filename()`, hằng `config.FATIGUE_KEYWORD`).

- **Fatigue:** `fatigue_70` (đo trong lúc gây mỏi) và `10_ap_fatigue` (đo sau khi
  gây mỏi) — hai điều kiện mà bài báo BME "chú thích rõ ràng" là trạng thái mỏi.
- **Normal:** `10/20/40/60/90` — các mức co cơ đo trước khi gây mỏi.

Cách này bám theo chính nhãn `fatigue` mã hoá trong tên file, phản ánh đúng cách
code MATLAB tách hai lớp.

> **Ghi chú tham khảo (bài báo).** ICACE 2019 phát biểu tiêu chí sinh lý ">60% MVC"
> (cơ mỏi khi lực vượt 60% MVC). Nếu áp cứng ngưỡng này làm nhãn nhị phân thì file
> `90` sẽ thành Fatigue và `10_ap_fatigue` thành Normal — **khác** với cách trên và
> mâu thuẫn với việc `10_ap_fatigue` đo *sau* khi gây mỏi. Vì dự án ưu tiên bám
> **code MATLAB** (tách theo Normal/Fatigue, tức theo pha giao thức) nên ta dùng
> cách gán theo tên file; ">60% MVC" chỉ là bối cảnh sinh lý tham khảo.

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
| Trong đó Normal / Fatigue | 824 / 446 |
| Số file | 20 |
| Tập train (subject 5,7,8,11) | 824 mẫu (Normal 506, Fatigue 318) |
| Tập test (subject 9) | 446 mẫu (Normal 318, Fatigue 128) |

Các con số này in ra khi chạy `python -m src.pipeline` và hiển thị ở tab
**Overview** của Streamlit.
