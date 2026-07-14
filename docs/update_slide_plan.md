# Kế hoạch cập nhật `sEMG_slide_trinh_bay(4).pptx` theo Feedback deck sEMG

> **Mục tiêu tổng thể:** Chuyển deck từ "công nghệ này là gì" sang **"AI hỗ trợ được quyết định lâm sàng nào, và rủi ro đã xử lý tới đâu"** — để Vinmec nhìn thấy hiệu quả.
> **Nguyên tắc bất biến:** Giữ **nguyên** template, font (Arial), bảng màu, header/footer, phong cách card/table/badge của deck hiện tại. Chỉ thay đổi **thứ tự slide, nội dung, và thêm slide mới** đúng theo phong cách đã có.
> **Quy mô:** 10 slide → **15 slide** (tái cấu trúc đầy đủ theo feedback).

---

## 1. Bốn thay đổi cốt lõi (theo feedback)

| # | Hiện tại | Chuyển thành | Slide xử lý |
|---|----------|--------------|-------------|
| 1 | Mở bằng định nghĩa MFCV | Mở bằng **khoảng trống lâm sàng** Motion Lab chưa nhìn thấy | Slide 2 (mới) |
| 2 | Rủi ro Noraxon để ngỏ ở slide cuối | **Đưa lên sớm, nói trực diện, kèm Plan B** | Slide 7 (dời lên) |
| 3 | Lợi ích liệt kê chung | Tách rõ **đã có bằng chứng** vs **giả thuyết cần thí điểm** | Slide 8 |
| 4 | Demo ở cuối | **Demo đan xen** trong nội dung | Slide 9 (Demo 1), Slide 12 (Demo 2) |

---

## 2. Hệ thống thiết kế cần GIỮ NGUYÊN (design system)

Trích xuất trực tiếp từ file hiện tại — mọi slide mới phải tuân theo.

### 2.1 Khổ & font
- **Khổ slide:** 16:9 — `13.333 in × 7.5 in` (12192000 × 6858000 EMU).
- **Font:** `Arial` cho toàn bộ text (không đổi).

### 2.2 Bảng màu (hex — vai trò)
| Hex | Vai trò |
|-----|---------|
| `#00A896` | Teal — **kicker label** đầu mỗi content slide (chữ hoa, đậm, 12pt) |
| `#028090` | Teal đậm — inline nhấn mạnh (bold), vòng trang trí slide title |
| `#9BE7DC` | Mint nhạt — kicker/badge **trên nền tối** (title, CTA) |
| `#1E293B` | Slate đậm — **tiêu đề slide (24pt)** & body chính |
| `#64748B` | Slate xám — text phụ, caption, **footer** |
| `#FFFFFF` | Trắng — text trên nền tối / trên chip teal |
| `#F2F7F8` | Nền **card** (xám-teal rất nhạt) |
| `#0B2A4A` | Navy đậm — **hộp nhấn mạnh / badge số liệu** (chữ trắng) |
| `#E08A1E` | Cam hổ phách — hộp **"quyết định cần chốt"** (chỉ dùng ở CTA) |

### 2.3 Mẫu bố cục content slide (header)
- Icon shape `0.62×0.62 in` tại `(0.50, 0.45)`, có ảnh icon `0.30×0.30` bên trong.
- **Kicker** tại `(1.32, 0.43)` — 12pt, bold, `#00A896`, CHỮ HOA.
- **Tiêu đề** tại `(1.32, 0.69)` — 24pt, bold, `#1E293B`.

### 2.4 Footer (mọi content slide)
- Trái `(0.50, 7.08)`: `CHƯƠNG 2 · sEMG — ĐÁNH GIÁ MỎI CƠ` — 9pt, `#64748B`.
- Phải `(11.63, 7.08)`: **số trang** — 9pt, `#64748B`.

### 2.5 Các thành phần lặp lại
- **Card:** rounded-rect nền `#F2F7F8`; trong card: icon tròn + tiêu đề card (bold, `#1E293B`, 14–16pt) + body (11.5–12.5pt, `#1E293B` hoặc `#64748B`).
- **Chip nhóm đối tượng:** rounded-rect nền `#028090`, chữ trắng 11.5pt.
- **Hộp nhấn mạnh:** rounded-rect nền `#0B2A4A`, chữ trắng 13pt.
- **Badge số liệu:** nền `#0B2A4A`, số lớn 24pt `#9BE7DC`, chú thích 10pt trắng.
- **Bảng:** hàng tiêu đề CHỮ HOA; ô teal/xám nhạt xen kẽ (theo slide 7 hiện tại).
- **Slide nền tối (title/CTA):** giữ nguyên nền hiện có bằng cách **nhân bản slide gốc** (không dựng lại nền).

---

## 3. Cấu trúc deck MỚI — ánh xạ cũ → mới

| Mới | Slide | Nguồn | Trạng thái |
|-----|-------|-------|-----------|
| 1 | Trang bìa | Slide 1 cũ | Giữ (chỉnh phụ đề nhẹ) |
| 2 | **Khoảng trống lâm sàng** | — | **MỚI** |
| 3 | **Ca minh họa (giả định)** | — | **MỚI** |
| 4 | Phạm vi & giới hạn | Slide 2 cũ | Giữ gần nguyên |
| 5 | Khoa đối tác & đối tượng | Slide 3 cũ | Giữ + 1 dòng "người dùng hằng ngày" |
| 6 | Bối cảnh & khác biệt | Slide 4 cũ | **Gọn thành bảng so sánh** |
| 7 | Tích hợp phần cứng + Plan B | Slide 6 cũ | **Dời lên sớm** + sửa claim/ước lượng |
| 8 | Lợi ích: bằng chứng vs giả thuyết | Slide 5 cũ | **Tách 2 cột** + chú thích F1 |
| 9 | **Demo 1 — Mỏi cơ trong buổi tập** | — | **MỚI** (cue + ảnh) |
| 10 | Hạ tầng vận hành (OH/eForm/EMR) | Slide 7 cũ | Giữ nguyên |
| 11 | Quy trình 7 bước | Slide 8 cũ | + dòng logistics + **nâng abstention gate** |
| 12 | **Demo 2 — Tiến độ phục hồi** | — | **MỚI** (cue + ảnh) |
| 13 | **Thí điểm 1 trang** | Slide 9 cũ (một phần) | **Gộp thành bảng thí điểm** |
| 14 | Bước tiếp theo & Quyết định cần chốt | Slide 9 cũ | CTA đóng (nền tối) + cảm ơn |
| 15 | Tài liệu tham khảo | Slide 10 cũ | Giữ nguyên (phụ lục) |

**Logic dòng chảy:** Vấn đề (2–3) → Ranh giới an toàn (4–5) → Khác biệt & tích hợp trung thực + Plan B (6–7) → Giá trị có bằng chứng vs giả thuyết + Demo cảm nhận (8–9) → Cách vận hành thật + Demo dọc thời gian (10–12) → Lời đề nghị cụ thể (13–14) → Phụ lục (15).

---

## 4. Chi tiết từng slide

> Ký hiệu: **[GIỮ]** không đổi · **[SỬA]** chỉnh trên slide cũ · **[MỚI]** dựng mới (nhân bản slide cùng phong cách rồi thay nội dung).

### Slide 1 — Trang bìa **[GIỮ, chỉnh phụ đề]**
- Nền tối hiện có, tiêu đề `sEMG: Đánh giá Mỏi cơ & Chức năng cơ bằng AI` (40pt trắng) — **giữ**.
- **Chỉnh:** phụ đề dòng dưới (16pt `#00A896`) đổi từ mô tả chương → câu định hướng câu chuyện, ví dụ:
  *"Chỉ dấu sinh lý báo mỏi sớm — hỗ trợ quyết định tăng/giữ tải trong phục hồi"*.
- Giữ dòng người trình bày (12pt `#64748B`).

### Slide 2 — Khoảng trống lâm sàng **[MỚI]**
- **Mục đích:** Mở bằng vấn đề, không bằng định nghĩa. Nêu điều Motion Lab **chưa** nhìn thấy.
- **Kicker:** `KHOẢNG TRỐNG LÂM SÀNG` (12pt bold `#00A896`).
- **Tiêu đề:** `Motion Lab đã đo được nhiều thứ — trừ dấu hiệu mỏi sớm` (24pt bold `#1E293B`).
- **Bố cục — hàng 3 card "đã đo được"** (nền `#F2F7F8`, ngang ~3.9in mỗi card, y≈1.8):
  | Card | Tiêu đề (bold 13pt) | Phụ (10pt `#64748B`) |
  |------|--------------------|-----------------------|
  | 1 | Sức cơ | Compass 600 / David Health — đo trực tiếp |
  | 2 | Động học | VICON Valkyrie — chuyển động 3D |
  | 3 | Kích hoạt cơ | Noraxon Ultium — biên độ/vị trí |
- **Hộp "khoảng trống"** (rounded-rect nền `#0B2A4A`, chữ trắng, y≈3.3, rộng full):
  `Còn thiếu: chỉ dấu sinh lý báo mỏi TRƯỚC KHI sức cơ tụt rõ.` (14pt, chữ "TRƯỚC KHI" bold `#9BE7DC`).
- **Câu hỏi lâm sàng** (dạng trích dẫn nhấn mạnh, y≈4.6): icon + text 18pt bold `#1E293B`:
  *"Hôm nay có nên tăng tải cho bệnh nhân này không?"* — và dòng phụ 12pt `#64748B`:
  `Quyết định lặp lại mỗi buổi PHCN — hiện dựa nhiều vào cảm nhận chủ quan.`
- Footer chuẩn, số trang `2`.

### Slide 3 — Ca minh họa (giả định) **[MỚI]**
- **Mục đích:** Cho thấy **hình dạng của giá trị** qua một ca cụ thể before/after. Ghi rõ là **giả định**.
- **Kicker:** `CA MINH HỌA — GIẢ ĐỊNH` (12pt bold `#00A896`).
- **Tiêu đề:** `Cùng một ca ACL, hai quyết định khác nhau` (24pt bold `#1E293B`).
- **Bố cục 2 cột** (mỗi cột 1 card `#F2F7F8`, ~5.97in):
  - **Cột trái — "Phác đồ hiện tại"** (tiêu đề card 15pt bold `#1E293B`):
    `Tuần 6 sau tái tạo ACL. Sức cơ đạt 85% chân lành → phác đồ hiện tại: tăng tải.` (12pt `#1E293B`).
  - **Cột phải — "Với MFCV"** (tiêu đề card 15pt bold `#028090`):
    `Phát hiện mỏi sớm khu trú ở vastus medialis ở mức tải thấp hơn dự kiến → giữ tải, tránh quá tải điểm yếu.` (12pt `#1E293B`).
- **Băng disclaimer** (nền `#0B2A4A` hoặc viền, full width, y≈5.6, 11pt):
  `Ca giả định để minh họa cơ chế giá trị — chưa phải dữ liệu bệnh nhân Vinmec. Kiểm chứng chính là mục tiêu của thí điểm.` (chữ "Ca giả định" bold `#9BE7DC`).
- Footer chuẩn, số trang `3`.

### Slide 4 — Phạm vi & giới hạn **[GIỮ gần nguyên]** *(là slide 2 cũ — điểm mạnh nhất, feedback nói giữ)*
- Giữ toàn bộ: định nghĩa MFCV + 2 ranh giới ("không chẩn đoán bệnh lý TK ngoại biên", "human-in-the-loop").
- **Chỉnh duy nhất:** cập nhật số trang → `4`; kicker giữ `PHẠM VI & GIỚI HẠN`.

### Slide 5 — Khoa đối tác & đối tượng **[SỬA nhẹ]** *(slide 3 cũ)*
- Giữ 2 card đơn vị + 3 chip nhóm đối tượng.
- **Thêm 1 dòng** (feedback): làm rõ **ai là người dùng thật hằng ngày**. Đặt dưới cụm "đơn vị triển khai" hoặc thành dải nhỏ 11pt `#64748B`:
  `Người dùng hằng ngày: KTV Phục hồi chức năng vận hành đo; Bác sĩ chỉ định & đọc/ký kết quả.`
- Cập nhật số trang `5`.

### Slide 6 — Bối cảnh & khác biệt (bảng so sánh) **[SỬA — chuyển thành bảng]** *(slide 4 cũ)*
- **Mục đích:** Feedback: "gọn lại thành bảng so sánh thuần túy."
- **Kicker:** `BỐI CẢNH & KHÁC BIỆT`; **Tiêu đề:** giữ `Motion Lab đã có công cụ đo sức cơ — MFCV bổ sung, không trùng lặp`.
- **Thay** cụm 3 card thiết bị + 2 card đối lập bằng **1 bảng 3 cột** (phong cách bảng slide 7):
  | CÔNG CỤ HIỆN CÓ | ĐO GÌ | MFCV BỔ SUNG ĐIỀU GÌ |
  |-----------------|-------|----------------------|
  | Compass 600 / David Health | Sức mạnh, sức bền cơ — trực tiếp, dễ diễn giải | — |
  | VICON Valkyrie | Động học / động lực học vận động | — |
  | Noraxon + myoRESEARCH | Biên độ / kích hoạt cơ theo vị trí | Chạy trên cùng dữ liệu này |
  | **MFCV (lớp AI)** | — | **Chỉ dấu sinh lý màng sợi cơ — tiềm năng phát hiện mỏi sớm hơn thời điểm sức cơ giảm rõ** |
  - Hàng tiêu đề CHỮ HOA nền teal, chữ trắng; các hàng nội dung xen kẽ nền `#F2F7F8`.
- **Giữ** dòng chú thích David Health cuối slide (10.5pt `#64748B`) — vẫn cần thiết để chính xác.
- Cập nhật số trang `6`.

### Slide 7 — Tích hợp phần cứng + Plan B **[SỬA — dời lên + sửa claim]** *(slide 6 cũ)*
- **Mục đích (feedback #2 + Slide 6):** đưa rủi ro Noraxon **lên sớm, trực diện, kèm Plan B**; đây là slide "cần sửa nhất về kỹ thuật" — **mọi claim ước lượng phải ghi rõ là ước lượng**.
- Giữ bố cục 2 cột: **"Cấu hình đã sẵn sàng"** vs **"Cần bổ sung nhẹ (Plan B)"** + hộp đề xuất "01 buổi làm việc kỹ thuật".
- **Sửa nội dung để trung thực:**
  - Đổi tiêu đề cột phải rõ ràng thành `Cần bổ sung — Plan B`.
  - Thêm nhãn **(ước lượng — cần xác nhận với Motion Lab)** vào mọi câu điều kiện ("nếu Noraxon đã hỗ trợ dãy điện cực tuyến tính và sampling đủ cao…", "gần như không phát sinh đầu tư…").
  - Câu mở đầu nêu thẳng: `Điều kiện kỹ thuật quyết định: cấu hình Noraxon có ước lượng được MFCV không (điện cực tuyến tính, sampling ≥ 1000 Hz). Cả hai kịch bản đều có đường đi — không phải rủi ro để ngỏ.` (bold cụm "Điều kiện kỹ thuật quyết định" `#028090`).
- Giữ hộp đề xuất nền `#0B2A4A` chữ trắng.
- Cập nhật số trang `7`.

### Slide 8 — Lợi ích: bằng chứng vs giả thuyết **[SỬA — tách 2 cột]** *(slide 5 cũ)*
- **Mục đích (feedback #3 + Slide 5):** tách rõ **đã có bằng chứng y khoa** vs **giả thuyết cần thí điểm**; hạ nhiệt con số F1.
- **Bố cục 2 cột** (2 card `#F2F7F8`):
  - **Cột trái — "Đã có bằng chứng y khoa"** (tiêu đề 16pt bold `#028090`):
    - `MFCV giảm theo tiến triển mỏi cơ — cơ chế sinh lý đã được y văn xác lập.` (dẫn Farina 2004, Bigland-Ritchie 1981 ở slide tham khảo).
    - `Chỉ số phổ sEMG (median frequency, RMS) là chỉ báo mỏi kinh điển.`
  - **Cột phải — "Giả thuyết cần thí điểm"** (tiêu đề 16pt bold `#64748B` hoặc cam nhạt để phân biệt):
    - `Giá trị đổi thành quyết định lâm sàng cụ thể (giữ/tăng tải) — cần thí điểm xác nhận.`
    - `Phát hiện mỏi sớm hơn thời điểm sức cơ giảm rõ trên bệnh nhân Vinmec.`
    - `Cá thể hóa cường độ & phòng ngừa tái chấn thương — mục tiêu đo lường trong thí điểm.`
- **Badge F1 — sửa quan trọng:** giữ badge `F1 = 95,12%` (24pt `#9BE7DC` trên `#0B2A4A`) **nhưng thêm 2 dòng điều kiện** (10pt):
  `Mô hình KNN · dữ liệu Springer 2025` **và** `Chưa validate trên BN Vinmec — đây chính là mục tiêu thí điểm.`
  → chuyển F1 từ "bằng chứng lâm sàng" sang "kết quả thuật toán có điều kiện".
- Cập nhật số trang `8`.

### Slide 9 — Demo 1: Mỏi cơ trong buổi tập **[MỚI — cue + ảnh]**
- **Vị trí (feedback):** ngay **sau slide Lợi ích**.
- **Kicker:** `DEMO TRỰC TIẾP` (12pt bold `#00A896`).
- **Tiêu đề:** `Phát hiện mỏi cơ theo thời gian thực trong buổi tập` (24pt bold `#1E293B`).
- **1 dòng giá trị** (14pt `#1E293B`): `Đường cong mỏi cơ + cảnh báo thời điểm bắt đầu mỏi — trên tín hiệu sEMG thật của một buổi tập.`
- **Khung ảnh chụp** (rounded-rect nền `#F2F7F8`, ~11.3×3.6in, giữa slide): **placeholder** để dán ảnh chụp màn hình Use Case 1 (team cung cấp).
- **Caption** (10pt `#64748B`): `Use Case 1 · real-time · ma trận 64 kênh, 2000 Hz.`
- Footer chuẩn, số trang `9`.

### Slide 10 — Hạ tầng vận hành (OH/eForm/EMR) **[GIỮ nguyên]** *(slide 7 cũ)*
- Giữ bảng 3 cột + dòng kết luận "phần lớn hạ tầng có sẵn / dùng chung".
- Cập nhật số trang `10`.

### Slide 11 — Quy trình 7 bước **[SỬA — logistics + nâng abstention]** *(slide 8 cũ)*
- Giữ sơ đồ 7 bước ngang.
- **Thêm dòng logistics dưới sơ đồ** (feedback — trả lời câu hỏi vận hành, 12pt `#64748B`):
  `Mỗi buổi ~[X] phút, lồng vào buổi PHCN sẵn có · KTV vận hành đo, Bác sĩ đọc & ký kết quả · 0 lượt hẹn Motion Lab phát sinh.` *(điền [X] khi có số)*.
- **Nâng abstention gate (Bước 3)** — điểm khác biệt an toàn mạnh nhất, hiện đang ở dòng chú thích nhỏ:
  - Tô Bước 3 khác biệt về màu (badge số `#028090` hoặc viền nhấn) để nổi so với 6 bước còn lại.
  - Đổi dòng chú thích nhỏ thành **hộp nhấn mạnh** (nền `#0B2A4A` chữ trắng, 12–13pt):
    `Cổng abstention (Bước 3): phân biệt "chưa phát hiện mỏi" với "dữ liệu chưa đủ điều kiện phân tích" — tránh kết luận sai khi tín hiệu kém. Đây là lớp an toàn khác biệt của đề xuất.`
- Cập nhật số trang `11`.

### Slide 12 — Demo 2: Tiến độ phục hồi qua các buổi **[MỚI — cue + ảnh]**
- **Vị trí (feedback):** **sau quy trình 7 bước**, gắn Bước 7 (Longitudinal).
- **Kicker:** `DEMO TRỰC TIẾP` (12pt bold `#00A896`).
- **Tiêu đề:** `Theo dõi tiến độ phục hồi cơ qua nhiều buổi` (24pt bold `#1E293B`).
- **1 dòng giá trị** (14pt): `Dashboard tổng hợp: sức bền tăng qua từng buổi, đối chiếu đường cong mỏi — trả lời "cơ có thực sự khỏe hơn không, bao nhiêu".`
- **Khung ảnh chụp** (placeholder `#F2F7F8`): ảnh chụp Dashboard Use Case 2 (team cung cấp).
- **Caption** (10pt `#64748B`): `Use Case 2 · longitudinal · gắn Bước 7 của quy trình.`
- Footer chuẩn, số trang `12`.

### Slide 13 — Thí điểm 1 trang **[MỚI/GỘP — bảng]** *(gộp từ slide 9 cũ)*
- **Mục đích (feedback Slide 9):** gộp "Bước tiếp theo" thành **bảng thí điểm 1 trang** cụ thể.
- **Kicker:** `ĐỀ XUẤT THÍ ĐIỂM`; **Tiêu đề:** `Thí điểm 1 trang — sẵn sàng triển khai` (24pt bold `#1E293B`).
- **Bảng 2 cột** (phong cách bảng slide 7; hàng tiêu đề `HẠNG MỤC | ĐỀ XUẤT`):
  | Hạng mục | Đề xuất *(số cần team điền lại)* |
  |----------|-----------------------------------|
  | Đối tượng | N ≈ 30 BN sau tái tạo ACL, tuần 4–12 |
  | Cơ mục tiêu | Vastus medialis, vastus lateralis, hamstring |
  | Thời gian | 12 tuần thu thập + 4 tuần phân tích |
  | Tần suất | 1 buổi/tuần, **lồng vào buổi PHCN sẵn có — 0 lượt hẹn Motion Lab phát sinh** |
  | Chi phí | Phần cứng (module array nếu cần) + nhân lực |
- **Nhãn** dưới bảng (10pt `#64748B`): `Các con số là đề xuất khởi điểm — team điều chỉnh theo nguồn lực thực tế.`
- Cập nhật số trang `13`.

### Slide 14 — Bước tiếp theo & Quyết định cần chốt **[SỬA — CTA đóng]** *(slide 9 cũ, phần còn lại)*
- Giữ slide **nền tối** hiện có (nhân bản slide 9 cũ để giữ nền).
- Giữ danh sách hành động (4 mục) + hộp cam `#E08A1E` "Quyết định cần chốt" + dòng cảm ơn.
- **Chỉnh khung câu chuyện:** vì rủi ro Noraxon đã được xử lý ở Slide 7, hộp "Quyết định cần chốt" ở đây là **xác nhận/chốt**, không phải lần đầu nêu rủi ro. Có thể đổi tiêu đề hộp thành `XÁC NHẬN ĐỂ KHỞI ĐỘNG` với nội dung: `Cấu hình Noraxon hỗ trợ ước lượng MFCV? (điện cực tuyến tính, sampling ≥ 1000 Hz)`.
- Không đánh số trang (giữ như hiện tại) hoặc `14` tùy nhất quán — **giữ nguyên cách hiện tại** (slide này không có footer chương).

### Slide 15 — Tài liệu tham khảo **[GIỮ nguyên]** *(slide 10 cũ)*
- Giữ toàn bộ 2 cột (khoa học & phương pháp / nguồn hạ tầng Vinmec).
- Footer phụ lục giữ `Phụ lục`.

---

## 5. Ma trận truy vết Feedback → Hành động

| Điểm feedback | Xử lý ở |
|---------------|---------|
| Mở bằng khoảng trống lâm sàng | Slide 2 (mới) |
| Ca minh họa before/after (giả định, đánh dấu rõ) | Slide 3 (mới) |
| S3 = giữ slide phạm vi & giới hạn | Slide 4 |
| Slide 2 (phạm vi) giữ nguyên | Slide 4 |
| Slide 3: thêm ai là người dùng hằng ngày (BS/KTV) | Slide 5 |
| Slide 4: gọn thành bảng so sánh | Slide 6 |
| Slide 5: tách bằng chứng vs giả thuyết | Slide 8 |
| Slide 5: F1 ghi rõ điều kiện + "chưa validate trên BN Vinmec" | Slide 8 (badge) |
| Slide 6: sửa claim, ghi rõ ước lượng | Slide 7 |
| Rủi ro Noraxon lên sớm + Plan B | Slide 7 (dời lên trước Lợi ích) |
| Slide 7 (OH/eForm/EMR) giữ nguyên | Slide 10 |
| Slide 8: buổi đo mấy phút / ai đọc-ký / có nghẽn Motion Lab? | Slide 11 (dòng logistics) |
| Nâng abstention gate (Bước 3) | Slide 11 (hộp nhấn mạnh + tô màu) |
| Slide 9: gộp "Thí điểm 1 trang" (bảng 5 hàng) | Slide 13 |
| Demo 1 sau slide Lợi ích | Slide 9 |
| Demo 2 sau quy trình 7 bước (gắn Bước 7) | Slide 12 |

✅ **Mọi điểm feedback đều có slide xử lý.**

---

## 6. Kế hoạch triển khai kỹ thuật

### 6.1 Công cụ & an toàn
- Dùng **`python-pptx` 1.0.2** (đã có trong `.venv`).
- **Thao tác trên bản sao**: copy `sEMG_slide_trinh_bay(4).pptx` → `sEMG_slide_trinh_bay(5).pptx` (hoặc backup) trước khi sửa, giữ bản gốc nguyên vẹn cho tới khi user duyệt.
- **Không tạo template mới**: mọi slide mới **nhân bản** một slide cùng phong cách trong file (deep-copy XML `<p:sld>` + đăng ký vào `sldIdLst`), rồi thay text/shape. Cách này giữ nguyên theme màu, font, nền tối.

### 6.2 Kỹ thuật cần dùng (python-pptx không hỗ trợ sẵn)
- **Nhân bản slide:** copy phần tử XML của slide nguồn + copy quan hệ (rels) ảnh/hình; thêm `sldId` mới vào `presentation.xml`.
- **Sắp xếp lại thứ tự slide:** hoán đổi thứ tự các `<p:sldId>` trong `sldIdLst` để đạt trật tự 1–15 ở mục 3.
- **Chỉnh text an toàn:** sửa ở mức `run` để giữ định dạng; nếu thay cả đoạn thì set lại font/size/color/bold theo bảng màu mục 2.

### 6.3 Trình tự thực hiện
1. Backup file gốc.
2. Sửa nội dung các slide **[GIỮ/SỬA]** tại chỗ (4, 5, 6, 7, 8, 10, 11, 13→từ 9, 14, 15) — vì phần lớn chỉ đổi text/table.
3. Dựng các slide **[MỚI]** (2, 3, 9, 12) bằng nhân bản slide phù hợp:
   - Slide 2, 3: nhân bản một content slide có card (ví dụ slide 3/5 cũ).
   - Slide 9, 12 (demo): nhân bản content slide, thay thân bằng khung ảnh placeholder.
4. **Sắp xếp lại** toàn bộ theo trật tự mục 3.
5. **Cập nhật số trang** footer cho khớp thứ tự mới (2…13; slide 14 CTA & 15 phụ lục theo quy ước hiện tại).
6. Mở lại bằng python-pptx để **verify**: đủ 15 slide, đúng thứ tự, không lỗi shape; xuất log text mỗi slide để đối chiếu.
7. (Tùy chọn) Kết xuất ảnh/PDF để mắt thường kiểm tra bố cục nếu có LibreOffice.

### 6.4 Hạng mục cần TEAM cung cấp (điền sau, đã có placeholder)
| Hạng mục | Slide | Mặc định nếu chưa có |
|----------|-------|----------------------|
| Ảnh chụp màn hình Demo 1 (Use Case 1) | 9 | Khung placeholder + caption |
| Ảnh chụp màn hình Demo 2 (Dashboard UC2) | 12 | Khung placeholder + caption |
| Số phút mỗi buổi đo `[X]` | 11 | Để "~[X] phút" chờ điền |
| Xác nhận claim cấu hình Noraxon | 7 | Ghi "(ước lượng — cần xác nhận)" |
| Số thí điểm (N, thời gian, chi phí) | 13 | Dùng số đề xuất trong feedback |
| Số liệu ca giả định (nếu muốn cụ thể hơn) | 3 | Dùng ca giả định 85%/tuần-6 như feedback |

### 6.5 Rủi ro & lưu ý
- Nhân bản slide qua XML dễ sai quan hệ ảnh (rels) → cần copy đúng phần ảnh icon đi kèm; ưu tiên nhân bản slide **ít ảnh** rồi thêm nội dung.
- Sau khi reorder, **kiểm tra lại footer/số trang** vì chúng là text tĩnh, không tự cập nhật.
- Giữ đúng hex màu ở mục 2 khi set run mới — không để python-pptx rơi về màu theme mặc định.
- Không đụng tới slide master/layout để tránh lệch toàn bộ template.

---

## 7. Tiêu chí hoàn thành (Definition of Done)
- [ ] Đủ **15 slide** đúng trật tự mục 3, mở được không lỗi.
- [ ] 4 slide mới (2, 3, 9, 12) đúng phong cách (màu, font, header, footer) như deck cũ.
- [ ] Mọi điểm trong ma trận mục 5 đã phản ánh trên slide.
- [ ] F1 có đủ 2 dòng điều kiện; badge không còn đứng như "bằng chứng lâm sàng".
- [ ] Abstention gate được nâng thành hộp nhấn mạnh + Bước 3 nổi bật.
- [ ] Rủi ro Noraxon xuất hiện ở Slide 7 (sớm) kèm Plan B, không còn để ngỏ ở cuối.
- [ ] Footer & số trang khớp thứ tự mới.
- [ ] Bản gốc được backup; file mới không phá template.
