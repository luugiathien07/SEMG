# Kịch bản thuyết minh — Demo Usecase 1: Giám sát Mỏi cơ Real-time

Tổng thời lượng gợi ý: **~2 phút slide + ~5 phút demo trực tiếp**.
Slide: `slides/10-slide-usecase1-demo.html`. App: `streamlit run app_realtime_session.py`, tab
"Giám sát Mỏi cơ (Real-time)".

Mở sẵn cả hai trước khi bắt đầu (slide toàn màn hình, app đã load xong trong tab trình
duyệt khác) để không mất thời gian chờ giữa buổi.

---

## Phần 1 — Giới thiệu qua slide (~2 phút)

### Slide 1/3 — Bối cảnh (~40s)

> "Đây là bài toán giám sát mỏi cơ real-time trong một buổi tập phục hồi chức năng.
> Kỹ thuật viên gắn một cảm biến sEMG 64 điện cực lên cơ đang tập — ví dụ cơ nhị đầu
> tay — và hệ thống theo dõi, phân loại Mỏi / Không mỏi liên tục trong suốt buổi tập."

Chỉ lần lượt vào 3 card:
- **Mục tiêu**: phát hiện đúng thời điểm mỏi, tránh tập sai tư thế hoặc chấn thương.
- **Chức năng**: giám sát 64 điện cực theo thời gian thực, phát hiện mỏi theo từng vùng.
- **Output**: trạng thái real-time + sơ đồ điện cực đổi màu + khuyến nghị hành động.

### Slide 2/3 — Thuật ngữ (~50s)

> "Trước khi vào demo, có vài thuật ngữ sẽ xuất hiện liên tục trên màn hình, tôi nói
> nhanh qua để lát nữa không bị rối."

- **RMS** — biên độ tín hiệu, "cơ đang gồng mạnh cỡ nào", tăng dần khi mỏi.
- **MDF** — tần số trung vị, "tín hiệu dao động nhanh hay chậm", giảm dần khi mỏi.
- **MFCV** — tốc độ dẫn truyền dọc sợi cơ, cũng giảm dần khi mỏi — *"phần này tôi sẽ
  quay lại kỹ hơn ở slide sau vì nó gắn với điểm khác biệt an toàn của hệ thống."*
- **%MVC** — mức cường độ co cơ so với lực tối đa từng đo được, dùng để chuẩn hoá giữa
  các bệnh nhân.
- **Sơ đồ 64 điện cực** và **trạng thái Mỏi/Không mỏi** — mỗi điểm trên sơ đồ là một
  kênh đo độc lập, tự tô màu riêng theo model học máy tổng hợp từ 14 đặc trưng tín hiệu.

### Slide 3/3 — MFCV & cổng abstention (~40s)

> "Đây là phần tôi muốn nhấn mạnh trước khi vào demo, vì nó là điểm khác biệt an toàn
> so với một hệ thống chỉ luôn luôn 'phán' ra một kết quả."

Chỉ vào hình biểu đồ thật (`docs/MFCV.png`):

> "Đây là dữ liệu thật — một buổi tập đến mỏi. Đường RMS màu cam đi lên, đường MFCV màu
> xanh đi xuống — đúng chu kỳ sinh lý của mỏi cơ. Nhưng để ý: có 18 trên 160 cửa sổ bị
> đánh dấu xám — hệ thống **từ chối** ước lượng CV ở những đoạn đó, chủ yếu lúc cơ chưa
> co hoặc vừa buông lực."

> "Lý do là hệ thống chỉ chấp nhận kết quả khi đạt đủ 4 điều kiện: tương quan liên kênh
> đủ cao, CV nằm trong dải sinh lý, cơ đang thực sự co — không phải lúc nghỉ, và không
> dính vùng innervation zone. Thiếu một trong 4, hệ thống **im lặng** thay vì đoán bừa
> một con số. Cái đứt quãng trên đường CV lát nữa các anh/chị sẽ thấy trong demo không
> phải lỗi — đó là tính năng."

**Chuyển sang demo:**

> "Giờ mình xem trực tiếp trên hệ thống."

---

## Phần 2 — Demo trực tiếp (~5 phút)

| Thời điểm | Việc làm | Lời thuyết minh |
|---|---|---|
| 0:00–0:30 | Trỏ vào dropdown "Nhóm cơ" (đã chọn sẵn Cơ nhị đầu tay), chưa bấm gì | "Hệ thống hiện hỗ trợ đầy đủ cơ nhị đầu tay — hai nhóm cơ còn lại đang trong lộ trình mở rộng. Tôi bấm 'Bắt đầu mô phỏng buổi tập'." |
| 0:30–1:00 | Bấm **Bắt đầu mô phỏng buổi tập**, chờ load | "Hệ thống đang ghép nhiều file ghi thật của cùng một bệnh nhân — từ mức %MVC thấp đến cao — thành một buổi tập liên tục, chạy 14 đặc trưng tín hiệu qua các model đã huấn luyện trước cho từng đoạn." |
| 1:00–1:45 | Bấm **Phát**, tăng tốc **5×**, trỏ vào waveform (thô/đã xử lý) và banner **Trạng thái** | "Đây là tín hiệu sEMG thô và tín hiệu đã lọc — thông dải 20–400Hz, notch 50Hz. Banner bên phải là kết luận real-time: Không mỏi / Mỏi, cùng mức %MVC hiện tại. Khi hệ thống báo Mỏi, khuyến nghị lâm sàng xuất hiện ngay bên dưới — ví dụ giảm cường độ hoặc cho nghỉ giữa hiệp." |
| 1:45–2:30 | Trỏ vào **sơ đồ 64 điện cực** | "Mỗi ô là một điện cực độc lập trên cơ — tự đổi màu xanh/đỏ theo từng vùng, không phải một kết luận chung cho cả cơ. Điện cực xám là kênh không hợp lệ trên bệnh nhân này — hệ thống không cố ép ra kết quả từ kênh đã biết là hỏng." |
| 2:30–3:30 | Cuộn xuống biểu đồ **"Xu hướng RMS, MDF & MFCV"**, tạm dừng phát để trỏ rõ | "RMS tăng dần, MDF giảm dần — đúng như slide vừa nói. Và đây, đường MFCV — để ý những **đoạn đứt quãng** này: đó chính là cổng abstention tôi vừa giải thích, từ chối ước lượng khi tín hiệu chưa đủ tin cậy, thay vì nội suy liều lĩnh qua đó." |
| 3:30–4:15 | Mở mục **"3. Chi tiết dự đoán theo model"** | "Đây là bảng dự đoán của từng model riêng lẻ tại đúng thời điểm hiện tại — cho thấy các model đồng thuận với nhau chứ không phải một hộp đen duy nhất quyết định." |
| 4:15–5:00 | Mở mục **"4. Kết quả phân loại của các mô hình"**, trỏ bảng Accuracy/F1 + confusion matrix | "Đây là hiệu năng của các model trên tập kiểm định — đo trên bệnh nhân hoàn toàn chưa từng dùng để huấn luyện, phản ánh đúng khả năng dùng cho bệnh nhân mới. Đó là toàn bộ luồng: từ tín hiệu thô, qua cổng chất lượng, đến kết luận và khuyến nghị." |

**Câu chốt:**

> "Điểm tôi muốn các anh/chị nhớ nhất: hệ thống không chỉ phân loại Mỏi/Không mỏi — nó
> còn biết khi nào **không đủ tin cậy để kết luận**, và nói rõ điều đó thay vì im lặng
> đoán bừa. Đó là yêu cầu bắt buộc cho một công cụ hỗ trợ lâm sàng."

---

## Ghi chú vận hành

- Nếu cần demo lại từ đầu: bấm **Đặt lại**, không cần tải lại trang.
- Tốc độ phát 1×/2×/5×/10× — dùng 5× hoặc 10× khi cần rút ngắn thời gian chờ hết một
  đoạn %MVC, quay về 1× khi đang giải thích một chi tiết cụ thể.
- Toggle **Envelope** tắt/bật lớp tín hiệu đã xử lý nếu muốn chỉ riêng tín hiệu thô lúc
  giải thích bộ lọc.
