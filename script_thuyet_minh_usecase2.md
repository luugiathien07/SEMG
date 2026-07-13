# 🎙️ SCRIPT THUYẾT MINH — USE CASE 2: Giám sát Phục hồi Cơ

> **Mục đích:** Đọc / nói từng câu theo thao tác trên màn hình khi quay demo.
> **Thời lượng:** ≤ 5 phút
> **Format:** `[🖱️ Thao tác]` → Lời nói. Dấu `⏸️` = dừng im 1-2 giây.

---

## PHẦN 1 — GIỚI THIỆU VÀ CHUYỂN TAB (0:00 – 0:30)

`[🖱️ Màn hình đang ở tab "Giám sát Mỏi cơ (Real-time)" — tab của Use Case 1]`

> Use Case 1 vừa trình bày cách phát hiện mỏi cơ **trong một buổi tập**. Bây giờ chúng ta chuyển sang Use Case 2: **Giám sát Phục hồi Cơ** — theo dõi quá trình phục hồi **qua nhiều buổi trị liệu**.

`[🖱️ Click vào tab "Giám sát phục hồi cơ"]`

> ⏸️

> Câu hỏi cốt lõi của Use Case 2 là: **"Bệnh nhân tập nhiều buổi rồi, cơ có thực sự khoẻ hơn không?"** — và nếu có, thì khoẻ hơn **bao nhiêu**, đo bằng **con số** nào.

---

## PHẦN 2 — DASHBOARD TỔNG QUAN (0:30 – 1:10)

### 2a. Metric cards

`[🖱️ Di chuột qua 4 ô metric cards ở đầu trang, từ trái sang phải]`

> Đầu tiên là **bảng tổng quan** gồm 4 chỉ số chính.

`[🖱️ Chỉ vào card "Số buổi tập"]`

> Card đầu tiên: **Số buổi tập** — ở đây là 8 buổi. Hệ thống mô phỏng một bệnh nhân trải qua 8 buổi trị liệu vật lý.

`[🖱️ Chỉ vào card "Sức bền ban đầu"]`

> Card thứ hai: **Sức bền ban đầu** — buổi đầu tiên, cơ chỉ chịu được khoảng **5 giây** trước khi bắt đầu mỏi.

`[🖱️ Chỉ vào card "Sức bền gần nhất" — có mũi tên xanh tăng]`

> Card thứ ba: **Sức bền gần nhất** — sau 8 buổi trị liệu, sức bền tăng lên **23 giây**, tương đương tăng hơn **400%**. Con số delta xanh cho thấy xu hướng cải thiện rõ rệt.

`[🖱️ Chỉ vào card "F1 (KNN phân loại mỏi)"]`

> Card cuối: **F1 score** — chỉ số đánh giá độ chính xác của mô hình phân loại mỏi. Ở đây F1 = **0.928** — nghĩa là mô hình phân biệt đúng "mỏi" và "chưa mỏi" rất tốt.

### 2b. Narrative và giải thích P(mỏi)

`[🖱️ Chỉ vào ô thông tin xanh dương (info box) phía dưới cards]`

> Hệ thống tự động tóm tắt: *Sau 8 buổi trị liệu, sức bền cơ tăng từ 5 giây lên 23 giây — cơ mỏi muộn hơn, phục hồi tốt hơn.*

`[🖱️ Click vào expander "💡 P(mỏi) là gì?"]`

> Mở phần giải thích **P(mỏi)** — đây là khái niệm quan trọng nhất của Use Case 2.

`[🖱️ Để expander mở, di chuột chậm qua từng dòng giải thích]`

> Cách tính: cắt tín hiệu mỗi buổi thành các **đoạn 1 giây**. Mỗi đoạn trích **7 đặc trưng** — gồm RMS, MAV, Waveform Length, Zero Crossing, Slope Sign Change, MDF và MNF — tạo thành **1 điểm** trong không gian 7 chiều.
>
> Với mỗi đoạn mới, mô hình KNN tìm **7 đoạn GIỐNG NHẤT** trong dữ liệu huấn luyện, rồi đếm xem bao nhiêu đoạn trong đó đã được gán nhãn "mỏi". **P(mỏi) = số hàng xóm mỏi chia 7.** Ví dụ 5/7 hàng xóm là mỏi thì P = 0,71.
>
> **Trực giác:** Đầu buổi cơ khoẻ — RMS thấp, MDF cao — giống nhóm "chưa mỏi" nên P thấp. Cuối buổi cơ mỏi — RMS tăng, MDF giảm — giống nhóm "mỏi" nên P cao. Giây P vượt **0,5** = lúc bắt đầu mỏi.

---

## PHẦN 3 — BIỂU ĐỒ XÁC SUẤT MỎI TRONG BUỔI TẬP (1:10 – 2:10)

### 3a. Giải thích biểu đồ

`[🖱️ Cuộn xuống phần "📈 Xác suất mỏi (KNN) trong bài co cơ duy trì"]`

> Đây là biểu đồ thứ nhất: **Xác suất mỏi theo thời gian** trong một buổi tập.

`[🖱️ Chỉ vào trục Y]`

> **Trục dọc** là P(mỏi) — giá trị từ 0 đến 1. Càng gần 1 nghĩa là cơ càng mỏi.

`[🖱️ Chỉ vào trục X]`

> **Trục ngang** là thời gian trong buổi tập, tính bằng giây.

`[🖱️ Chỉ vào đường nét đứt ngang ở giữa]`

> **Đường nét đứt ngang** ở mức 0,5 là **ngưỡng mỏi** — khi P vượt qua đường này, hệ thống đánh giá cơ đã bắt đầu mỏi.

### 3b. So sánh 2 buổi

`[🖱️ Chỉ vào đường đỏ cam (Buổi 1)]`

> **Đường đỏ cam** là Buổi 1 — giai đoạn đầu trị liệu. Đường P leo nhanh và cắt ngưỡng 0,5 ngay ở **khoảng 4,5 giây** — cơ mỏi rất sớm.

`[🖱️ Chỉ vào đường dọc đứt nét đỏ cam ghi "4.5s"]`

> Đường dọc đánh dấu thời điểm khởi phát mỏi: chỉ **4,5 giây**.

`[🖱️ Chỉ vào đường xanh teal (Buổi 8)]`

> **Đường xanh teal** là Buổi 8 — buổi gần nhất. Đường P leo rất chậm, mãi đến **23 giây** mới vượt ngưỡng.

`[🖱️ Chỉ vào đường dọc đứt nét xanh ghi "23.0s"]`

> Khởi phát mỏi dời ra tận **23 giây** — gấp hơn 5 lần buổi đầu.

### 3c. Thử đổi buổi so sánh

`[🖱️ Click vào dropdown "Buổi A" → chọn Buổi 3, rồi "Buổi B" → chọn Buổi 6]`

> Hệ thống cho phép chọn **bất kỳ 2 buổi** để so sánh trực tiếp. Ví dụ Buổi 3 với Buổi 6. Có thể thấy xu hướng cải thiện dần dần.

`[🖱️ Đổi lại về Buổi 1 vs Buổi 8 để screenshot đẹp]`

> Tôi chuyển lại Buổi 1 và Buổi 8 để thấy rõ nhất sự khác biệt.

---

## PHẦN 4 — BIỂU ĐỒ SỨC BỀN QUA CÁC BUỔI (2:10 – 2:50)

`[🖱️ Cuộn xuống phần "🏋️ Điểm sức bền cơ (thời điểm khởi phát mỏi) qua các buổi"]`

> Đây là biểu đồ **quan trọng nhất** của Use Case 2: **Điểm sức bền cơ qua các buổi trị liệu**.

`[🖱️ Chỉ vào trục X]`

> Trục ngang là **số buổi trị liệu** — từ buổi 1 đến buổi 8.

`[🖱️ Chỉ vào trục Y]`

> Trục dọc là **thời gian cho tới khi mỏi**, tính bằng giây. Giá trị càng cao nghĩa là cơ chịu được càng lâu — sức bền tốt hơn.

`[🖱️ Di chuột dọc theo đường teal từ trái sang phải — đang tăng dần]`

> Đường xu hướng **tăng rõ rệt** từ trái sang phải: buổi đầu cơ chịu được khoảng **5 giây**, đến buổi 8 tăng lên **23 giây**. Mỗi buổi đều được ghi nhãn giá trị cụ thể ngay trên biểu đồ.

`[🖱️ Chỉ vào vùng tô xanh nhạt bên dưới đường — area fill]`

> Vùng tô nhạt bên dưới giúp nhìn thấy rõ **xu hướng tăng trưởng tổng thể** — cơ bệnh nhân đang phục hồi theo đúng kỳ vọng trị liệu.

`[🖱️ Di chuột xuống bảng số liệu phía dưới biểu đồ]`

> Bên dưới là bảng chi tiết: **Buổi**, **Điểm sức bền (giây)**, và **% cửa sổ chưa mỏi** — tức tỷ lệ thời gian trong buổi mà cơ còn khoẻ. Buổi đầu chỉ khoảng 15%, buổi cuối lên 70-80%.

---

## PHẦN 5 — CHỈ SỐ PHỤC HỒI (3 chỉ số sinh lý) (2:50 – 3:50)

`[🖱️ Cuộn xuống phần "📉 Chỉ số phục hồi (dữ liệu PhysioMio)"]`

> Phần tiếp theo sử dụng **dữ liệu thật từ bộ dữ liệu PhysioMio** — HD-sEMG 64 kênh, 2000 Hz — ghi nhận từ bệnh nhân đột quỵ trong chương trình phục hồi chức năng.

### 5a. Chọn bệnh nhân

`[🖱️ Chỉ vào dropdown "Bệnh nhân" → hiện patient1, patient2]`

> Hệ thống có dữ liệu của 2 bệnh nhân. Mỗi bệnh nhân có **tay liệt** (impaired arm) và **tay lành** (healthy arm) — tay lành dùng làm chuẩn đối chiếu.

`[🖱️ Chọn patient1]`

> Chọn bệnh nhân 1 — có 6 buổi trị liệu tay liệt.

### 5b. Chỉ số phục hồi đối xứng (Symmetry Index)

`[🖱️ Chỉ vào biểu đồ bên trái — "Chỉ số phục hồi (đối xứng RMS lành-liệt, %)"]`

> Biểu đồ bên trái là **Symmetry Index** — chỉ số đối xứng lành-liệt. Công thức: **RMS tay liệt chia RMS tay lành, nhân 100%**.

`[🖱️ Chỉ vào đường nét đứt 100% ở trên cùng]`

> Đường nét đứt ở mức **100%** là mục tiêu — nghĩa là tay liệt đã khoẻ **ngang bằng** tay lành. Càng gần 100%, cơ phục hồi càng tốt.

`[🖱️ Di chuột theo đường teal — đang tăng dần về phía 100%]`

> Đường xu hướng cho thấy Symmetry tăng dần qua 6 buổi — từ khoảng **74%** lên **95%** — cơ tay liệt đang dần bắt kịp tay lành.

### 5c. MDF — Tần số trung vị

`[🖱️ Chỉ vào biểu đồ bên phải — "MDF (Hz) — dịch lên khi phục hồi"]`

> Biểu đồ bên phải là **MDF** — Median Frequency — tần số trung vị phổ công suất.

`[🖱️ Chỉ vào đường nét đứt xanh lá "baseline tay lành"]`

> Đường nét đứt xanh lá là **baseline** — MDF trung bình của tay lành. Khi cơ tay liệt phục hồi, MDF sẽ **dịch lên** về phía baseline này — cho thấy tốc độ dẫn truyền sợi cơ đang cải thiện.

### 5d. Bảng chi tiết

`[🖱️ Cuộn xuống bảng số liệu — 4 cột: Buổi, RMS, MDF, Symmetry (%)]`

> Bảng chi tiết hiển thị số liệu thô cho từng buổi: giá trị **RMS**, **MDF** và **Symmetry Index**. Chuyên viên phục hồi chức năng có thể dùng bảng này để theo dõi tiến triển từng buổi.

### 5e. Đổi bệnh nhân

`[🖱️ Click dropdown → chọn patient2]`

> Chuyển sang bệnh nhân 2 — biểu đồ cập nhật tự động. Mỗi bệnh nhân có mẫu phục hồi riêng, chuyên viên có thể so sánh tiến triển giữa các bệnh nhân.

---

## PHẦN 6 — KHÁM PHÁ TÍN HIỆU (3:50 – 4:30)

`[🖱️ Cuộn xuống phần "🔬 Khám phá tín hiệu 1 buổi / 1 kênh"]`

> Phần cuối cho phép **khám phá tín hiệu gốc** — zoom vào một kênh cụ thể của một buổi tập bất kỳ.

`[🖱️ Chỉ vào 2 dropdown: Bệnh nhân + Tay]`

> Chọn **bệnh nhân** và **tay** cần xem — tay liệt hoặc tay lành.

`[🖱️ Chọn patient1, impaired_arm, Buổi 1]`

> Chọn bệnh nhân 1, tay liệt, buổi 1.

`[🖱️ Chỉ vào dropdown "Cử chỉ" → chọn 1 cử chỉ (ví dụ: Grasp)]`

> Chọn **loại cử chỉ** — ví dụ Grasp (nắm tay). Đây là một trong các bài tập phục hồi tiêu chuẩn.

`[🖱️ Chỉ vào dropdown "Kênh" → chọn channel_01]`

> Chọn **kênh điện cực** — ví dụ kênh 01.

`[🖱️ Chỉ vào biểu đồ tín hiệu — dạng sóng EMG]`

> Biểu đồ hiển thị **tín hiệu EMG gốc** của kênh đó: trục ngang là thời gian (giây), trục dọc là biên độ (µV). Chuyên viên có thể quan sát trực tiếp chất lượng tín hiệu, phát hiện nhiễu hoặc bất thường.

`[🖱️ Đổi sang Buổi 6, cùng cử chỉ, cùng kênh — so sánh biên độ bằng mắt]`

> Đổi sang buổi 6 — cùng cử chỉ, cùng kênh. So sánh bằng mắt: biên độ tín hiệu buổi 6 thường lớn hơn buổi 1 — cho thấy cơ huy động tốt hơn sau trị liệu.

---

## PHẦN 7 — KẾT LUẬN (4:30 – 5:00)

`[🖱️ Cuộn lên đầu trang — cho thấy lại Dashboard tổng quan với 4 metric cards]`

> Tổng kết Use Case 2:
>
> Hệ thống theo dõi phục hồi cơ qua **3 góc nhìn bổ trợ lẫn nhau**:
>
> **Thứ nhất — Sức bền cơ**: Đo thời điểm khởi phát mỏi bằng mô hình KNN, cho thấy bệnh nhân chịu được **lâu hơn gấp 5 lần** sau 8 buổi.
>
> **Thứ hai — Chỉ số đối xứng**: So sánh RMS tay liệt với tay lành, theo dõi mức độ bắt kịp qua từng buổi.
>
> **Thứ ba — Xác suất mỏi trong buổi**: Cho phép so sánh chi tiết đường P(mỏi) giữa bất kỳ 2 buổi nào — trực quan hoá sự cải thiện.

> Toàn bộ dựa trên tín hiệu **HD-sEMG 64 kênh**, xử lý tự động, kết quả trình bày trên **một giao diện duy nhất** — phù hợp để chuyên viên phục hồi chức năng theo dõi tiến triển bệnh nhân qua từng buổi trị liệu.

`[🖱️ Để màn hình ở dashboard — đẹp và ấn tượng]`

> Cảm ơn quý thầy cô đã theo dõi.

---

## 📐 SƠ ĐỒ LAYOUT THAM KHẢO

```
┌─────────────────────────────────────────────────────────────────────┐
│  Giám sát Mỏi cơ                                                   │
│  [Tab: Giám sát Real-time] [Tab: ★ Phục hồi cơ ★]                  │
├══════════════════════════════════════════════════════════════════════╡
│ ① DASHBOARD TỔNG QUAN                                               │
│ ┌──────────┐ ┌──────────────┐ ┌──────────────────┐ ┌─────────────┐ │
│ │ Số buổi  │ │ Sức bền      │ │ Sức bền gần nhất │ │ F1 (KNN)    │ │
│ │ tập: 8   │ │ ban đầu: 5s  │ │ 23s  ▲+411%      │ │ 0.928       │ │
│ └──────────┘ └──────────────┘ └──────────────────┘ └─────────────┘ │
│                                                                      │
│ ℹ️ Sau 8 buổi trị liệu, sức bền cơ tăng từ 5s → 23s (+411%)...    │
│                                                                      │
│ ▸ 💡 P(mỏi) là gì? — Giải thích cho người không chuyên   [mở/đóng] │
├══════════════════════════════════════════════════════════════════════╡
│ ② BIỂU ĐỒ XÁC SUẤT MỎI (KNN) TRONG BUỔI TẬP                       │
│ ┌─────────────────────────────┬─────────────────────────────┐       │
│ │ Buổi A (so sánh) ▼  Buổi 1 │ Buổi B (so sánh) ▼  Buổi 8 │       │
│ └─────────────────────────────┴─────────────────────────────┘       │
│ ┌──────────────────────────────────────────────────────────────┐     │
│ │ 1.0 ──  đỏ cam: Buổi 1 (mỏi sớm)                           │     │
│ │     │   xanh teal: Buổi 8 (mỏi muộn)                        │     │
│ │ P   │   --- ngang: ngưỡng P=0.5                              │     │
│ │ (   │   ⋮ đứt dọc: 4.5s (đỏ) | 23.0s (teal)                │     │
│ │ m   │                                                         │     │
│ │ ỏ   │  ▅▅▅▅▅▅ (đỏ lên sớm)                                  │     │
│ │ i   │                   ▃▃▃▃▃▃▃▃ (teal lên muộn)             │     │
│ │ )   │──────────────── 0.5 ──── ──── ──── ──── ────           │     │
│ │ 0.0 └─────────────────────────────────────────────           │     │
│ │      0s        10s        20s       30s                       │     │
│ └──────────────────────────────────────────────────────────────┘     │
│ KNN F1 = 0.928 (mô phỏng). Điểm sức bền = thời điểm P(mỏi)>0.5.  │
├══════════════════════════════════════════════════════════════════════╡
│ ③ BIỂU ĐỒ SỨC BỀN CƠ QUA CÁC BUỔI (★ chart quan trọng nhất)      │
│ ┌──────────────────────────────────────────────────────────────┐     │
│ │ 25s │                                           ● 23s       │     │
│ │     │                                  ● 19s                 │     │
│ │ 20s │                          ● 17s                         │     │
│ │     │                   ● 14s                                │     │
│ │ 15s │            ● 12s                                       │     │
│ │     │      ● 9s                                   ░░░░ area  │     │
│ │ 10s │   ● 7s                                      ░░░░ fill  │     │
│ │     │ ● 5s                                                   │     │
│ │  0s └─────────────────────────────────────────────           │     │
│ │      1    2    3    4    5    6    7    8                      │     │
│ │                  Buổi trị liệu                                │     │
│ └──────────────────────────────────────────────────────────────┘     │
│ ┌──────────┬──────────────────┬──────────────────┐                  │
│ │ Buổi     │ Điểm sức bền (s) │ % cửa sổ chưa mỏi│                  │
│ │ 1        │ 4.5              │ 13.8              │                  │
│ │ ...      │ ...              │ ...               │                  │
│ │ 8        │ 23.0             │ 77.6              │                  │
│ └──────────┴──────────────────┴──────────────────┘                  │
├══════════════════════════════════════════════════════════════════════╡
│ ④ CHỈ SỐ PHỤC HỒI (dữ liệu PhysioMio)                              │
│ Bệnh nhân ▼ [patient1]                                               │
│ ┌─────────────────────────────┬─────────────────────────────┐       │
│ │ Symmetry Index (%)          │ MDF (Hz)                     │       │
│ │ ┌─────────────────────┐     │ ┌─────────────────────────┐ │       │
│ │ │ --- 100% (tay lành)  │     │ │ --- baseline tay lành   │ │       │
│ │ │ ── teal: symmetry   │     │ │ ── coral: MDF           │ │       │
│ │ │    ░░ area fill     │     │ │    ■ markers vuông      │ │       │
│ │ └─────────────────────┘     │ └─────────────────────────┘ │       │
│ └─────────────────────────────┴─────────────────────────────┘       │
│ ┌──────┬──────┬──────┬──────────────┐                                │
│ │ Buổi │ RMS  │ MDF  │ Symmetry (%) │                                │
│ │ 1    │ 0.09 │ 87.6 │ 74.0         │                                │
│ │ ...  │ ...  │ ...  │ ...          │                                │
│ └──────┴──────┴──────┴──────────────┘                                │
├══════════════════════════════════════════════════════════════════════╡
│ ⑤ KHÁM PHÁ TÍN HIỆU 1 BUỔI / 1 KÊNH                                │
│ ┌───────────────┬───────────────┐                                    │
│ │ Bệnh nhân ▼   │ Tay ▼         │                                    │
│ ├───────────────┤               │                                    │
│ │ Buổi ▼        │ Cử chỉ ▼     │  Kênh ▼                            │
│ └───────────────┴───────────────┘                                    │
│ ┌──────────────────────────────────────────────────────────────┐     │
│ │  Biểu đồ tín hiệu EMG gốc (1 kênh, 1 cử chỉ)             │     │
│ │  Trục X: thời gian (s)    Trục Y: biên độ (µV)             │     │
│ └──────────────────────────────────────────────────────────────┘     │
└══════════════════════════════════════════════════════════════════════╘
```

---

## ⏱ TIMELINE TÓM TẮT

| Thời gian | Phần | Đang giải thích gì |
|---|---|---|
| 0:00 – 0:30 | Phần 1 | Giới thiệu mục đích UC2, chuyển tab |
| 0:30 – 1:10 | Phần 2 | ① Dashboard: 4 metric cards + narrative + giải thích P(mỏi) |
| 1:10 – 2:10 | Phần 3 | ② Biểu đồ P(mỏi) in-session: 2 đường, ngưỡng, onset, đổi buổi |
| 2:10 – 2:50 | Phần 4 | ③ Biểu đồ sức bền qua 8 buổi (chart quan trọng nhất) + bảng |
| 2:50 – 3:50 | Phần 5 | ④ Chỉ số phục hồi PhysioMio: Symmetry + MDF + đổi bệnh nhân |
| 3:50 – 4:30 | Phần 6 | ⑤ Khám phá tín hiệu gốc: chọn buổi/cử chỉ/kênh |
| 4:30 – 5:00 | Phần 7 | Kết luận: 3 góc nhìn bổ trợ, ý nghĩa lâm sàng |
