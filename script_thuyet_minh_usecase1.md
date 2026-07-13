# 🎙️ SCRIPT THUYẾT MINH — USE CASE 1: Giám sát Mỏi cơ (Real-time)

> **Mục đích:** Đọc / nói từng câu theo thao tác trên màn hình khi quay demo.
> **Thời lượng:** ≤ 5 phút
> **Format:** `[🖱️ Thao tác]` → Lời nói. Dấu `⏸️` = dừng im 1-2 giây.

---

## PHẦN 1 — MÀN HÌNH CHỜ (0:00 – 0:50)

`[🖱️ Màn hình đang ở trạng thái chờ, chưa bấm gì. Chuột đặt ở giữa]`

> Đây là màn hình **Giám sát Mỏi cơ thời gian thực** — Use Case 1 trong hệ thống phân tích tín hiệu điện cơ bề mặt, viết tắt là sEMG.

`[🖱️ Di chuột lên tiêu đề "Giám sát Mỏi cơ", rồi di qua hai tab]`

> Giao diện có hai tab: tab đầu tiên là **Giám sát Mỏi cơ Real-time** — là phần chúng ta sẽ demo hôm nay. Tab thứ hai là Giám sát phục hồi cơ, thuộc Use Case 2.

`[🖱️ Di chuột xuống ô dropdown "Nhóm cơ"]`

> Ở đây có ô chọn **Nhóm cơ** cần phân tích.

`[🖱️ Click mở dropdown — hiện ra 3 tùy chọn]`

> Hiện tại hệ thống hỗ trợ phân tích **cơ nhị đầu tay**, tức Biceps brachii. Hai nhóm cơ khác — cơ tứ đầu đùi và cơ delta vai — đang được phát triển, đánh dấu "Sắp có".

`[🖱️ Chọn lại "Cơ nhị đầu tay (Biceps brachii)" — dropdown đóng]`

> Chúng ta chọn cơ nhị đầu tay. Dữ liệu được thu thập bằng **ma trận điện cực mật độ cao gồm 64 kênh**, tần số lấy mẫu **2000 Hz**.

`[🖱️ Di chuột sang nút xanh "Bắt đầu mô phỏng buổi tập"]`

> Bây giờ tôi sẽ bấm nút này để khởi chạy mô phỏng một buổi tập luyện hoàn chỉnh. Hệ thống sẽ ghép các bản ghi sEMG của cùng một bệnh nhân — sắp xếp theo cường độ co cơ tăng dần — thành một phiên tập liên tục.

---

## PHẦN 2 — KHỞI CHẠY VÀ LOADING (0:50 – 1:15)

`[🖱️ Bấm nút "Bắt đầu mô phỏng buổi tập"]`

`[⏳ Spinner hiện ra: "Đang huấn luyện / tải mô hình…"]`

> Hệ thống đang thực hiện toàn bộ pipeline Machine Learning phía sau:

`[⏳ Spinner: "Đang ghép dữ liệu buổi tập…"]`

> Bước 1 — trích xuất **14 đặc trưng** từ mỗi kênh tín hiệu. Gồm hai nhóm:
>
> **8 đặc trưng miền thời gian** — phân tích trực tiếp dạng sóng:
> - **RMS** — Root Mean Square — biên độ trung bình bình phương, phản ánh cường độ co cơ tổng thể.
> - **MAV** — Mean Absolute Value — giá trị tuyệt đối trung bình, tương tự RMS nhưng ít nhạy với nhiễu.
> - **Skewness** — độ lệch — cho biết phân bố biên độ nghiêng về phía dương hay âm.
> - **Kurtosis** — độ nhọn — phân bố biên độ nhọn hay tròn, phát hiện tín hiệu bất thường.
> - **Max, Min** — giá trị cực đại và cực tiểu, phản ánh biên độ đỉnh.
> - **STD** — độ lệch chuẩn — mức dao động của tín hiệu quanh giá trị trung bình.
> - **Mean** — giá trị trung bình, phát hiện lệch đường cơ sở.
>
> **6 đặc trưng miền tần số** — phân tích phổ công suất:
> - **MDF** — Median Frequency — tần số chia đôi tổng năng lượng phổ. Khi mỏi, MDF giảm — đây là chỉ báo mỏi cơ kinh điển nhất.
> - **MNF** — Mean Frequency — tần số trung bình có trọng số, xu hướng tương tự MDF.
> - **Spectral Entropy** — entropy phổ — đo mức hỗn loạn của phổ, tăng khi cơ mỏi.
> - **Spectral Max, Spectral Min** — đỉnh và đáy phổ công suất.
> - **Spectral STD** — độ phân tán phổ.

`[⏳ Spinner: "Đang đánh giá mỏi cho từng kênh (64 kênh)…"]`

> Bước 2 — chọn đặc trưng tối ưu bằng thuật toán **mRMR** — viết tắt của Minimum Redundancy, Maximum Relevance. Thuật toán này chọn ra những đặc trưng **liên quan nhất** tới việc phân biệt mỏi hay không mỏi, đồng thời **loại bỏ các đặc trưng dư thừa** — tức những đặc trưng mang thông tin trùng lặp với nhau. Kết quả: chọn ra **top 3 đặc trưng** hiệu quả nhất.
>
> Bước 3 — huấn luyện **6 mô hình** Machine Learning song song:
> - **SVM** — Support Vector Machine — tìm siêu phẳng phân tách tối ưu, dùng top 3 đặc trưng.
> - **KNN** — K-Nearest Neighbors — phân loại dựa trên k điểm láng giềng gần nhất.
> - **LDA** — Linear Discriminant Analysis — phân tích biệt thức tuyến tính, tìm hướng chiếu tách biệt hai lớp tốt nhất.
> - **Decision Tree** — Cây quyết định — chia nhánh theo ngưỡng từng đặc trưng.
> - **Random Forest** — Rừng ngẫu nhiên — tập hợp nhiều cây quyết định, bỏ phiếu lấy kết quả.
> - **Logistic Regression** — Hồi quy logistic — ước lượng xác suất mỏi bằng hàm sigmoid.
>
> Tất cả đều được đánh giá mỏi trên **toàn bộ 64 kênh** của bệnh nhân.

`[✅ Animation tự động bắt đầu chạy]`

---

## PHẦN 3 — GIẢI THÍCH BIỂU ĐỒ SÓNG EMG (1:15 – 2:15)

### 3a. Biểu đồ tín hiệu EMG thời gian thực (panel lớn bên trái)

`[🖱️ Di chuột vào vùng biểu đồ sóng bên trái — đang chạy animation]`

> Đây là khu vực quan trọng nhất: **biểu đồ tín hiệu EMG thời gian thực**.

`[🖱️ Bấm "Tạm dừng" để freeze hình — dễ giải thích]`

> Tôi tạm dừng để giải thích chi tiết.

`[🖱️ Di chuột dọc theo trục Y bên trái]`

> **Trục dọc** là biên độ tín hiệu, đơn vị đã chuẩn hóa. Giá trị càng lớn nghĩa là cơ co càng mạnh.

`[🖱️ Di chuột dọc theo trục X phía dưới]`

> **Trục ngang** là thời gian tính bằng giây. Biểu đồ hiển thị một **cửa sổ trượt 3 giây** — tức là tại mỗi thời điểm, ta luôn nhìn thấy 3 giây tín hiệu gần nhất, giống như màn hình monitor bệnh nhân trong bệnh viện.

`[🖱️ Chỉ vào đường sóng xanh dương, rồi nhìn xuống legend "Tín hiệu thô"]`

> **Đường xanh dương** là tín hiệu EMG thô — tín hiệu điện gốc thu được từ điện cực, phản ánh hoạt động co cơ trực tiếp. Ta thấy sóng dao động lên xuống quanh trục 0.

`[🖱️ Chỉ vào đường sóng vàng cam, rồi nhìn xuống legend "Rectified envelope"]`

> **Đường vàng cam** là **Rectified Envelope** — tức tín hiệu đã chỉnh lưu toàn sóng rồi lấy đường bao biên độ. Nôm na là: lấy giá trị tuyệt đối rồi làm trơn. Đường này giúp nhìn rõ hơn xu hướng biên độ cơ đang tăng hay giảm, không bị nhiễu bởi dao động dương-âm.

`[🖱️ Chỉ vào nút "Envelope" (đang sáng vàng) trên thanh điều khiển]`

> Nút **Envelope** này cho phép bật/tắt đường bao. Khi tắt, chỉ còn tín hiệu thô.

`[🖱️ Bấm nút Envelope để tắt → chỉ còn đường xanh → bấm lại để bật]`

> ⏸️

`[🖱️ Nếu có đường đứt gãy dọc trong biểu đồ — di chuột vào đó]`

> Các **đường đứt gãy dọc** trên biểu đồ đánh dấu ranh giới giữa các giai đoạn tập luyện. Bên cạnh ghi nhãn mức phần trăm MVC — tức phần trăm lực co cơ tối đa tự nguyện. Ví dụ: "10%" nghĩa là bệnh nhân đang co cơ ở mức 10% sức tối đa.

---

### 3b. Sơ đồ 64 điện cực (panel nhỏ bên phải)

`[🖱️ Di chuột sang khung bên phải, có tiêu đề "Sơ đồ 64 điện cực — Cơ nhị đầu tay"]`

> Bên phải là **Sơ đồ vị trí 64 điện cực** được đặt trên cơ nhị đầu tay.

`[🖱️ Di chuột chậm qua lưới các chấm tròn, từ trên xuống dưới]`

> Mỗi **chấm tròn** đại diện cho một kênh điện cực. Bố trí theo lưới **13 hàng × 5 cột**, khoảng cách 8mm giữa các điện cực — đúng theo cấu hình ma trận HD-sEMG thực tế.

`[🖱️ Chỉ vào chấm xanh lá]`

> **Chấm xanh lá** = kênh được mô hình phân loại là **Không mỏi** — cơ ở kênh đó vẫn hoạt động bình thường.

`[🖱️ Chỉ vào chấm xám (nếu có)]`

> **Chấm xám** = kênh **không hợp lệ** — tín hiệu bị nhiễu hoặc điện cực tiếp xúc kém, hệ thống tự động loại bỏ để không ảnh hưởng kết quả.

`[🖱️ Chỉ xuống legend: "Không mỏi | Mỏi | Kênh không hợp lệ"]`

> Khi bệnh nhân bắt đầu mỏi, các chấm sẽ chuyển sang **đỏ** — cho thấy rõ **vùng nào trên cơ** đang mỏi nhiều nhất. Chúng ta sẽ thấy rõ điều này khi tua nhanh.

---

### 3c. Thanh điều khiển phát lại

`[🖱️ Di chuột xuống thanh điều khiển bên dưới biểu đồ sóng]`

> Phía dưới là thanh điều khiển phát lại. Từ trái qua phải:

`[🖱️ Chỉ vào nút "Phát" / "Tạm dừng"]`

> Nút **Phát / Tạm dừng** — kiểm soát animation.

`[🖱️ Chỉ vào nút "Envelope"]`

> Nút **Envelope** — bật tắt đường bao biên độ mà tôi vừa giải thích.

`[🖱️ Di chuột dọc theo thanh tiến trình]`

> **Thanh tiến trình** — có thể click vào bất kỳ vị trí nào để nhảy tới thời điểm tương ứng trong buổi tập, giống thanh seek trên YouTube.

`[🖱️ Chỉ vào đồng hồ thời gian bên phải]`

> Bên phải hiện **thời gian hiện tại / tổng thời gian** buổi tập.

`[🖱️ Chỉ vào các nút 1× 2× 5× 10×]`

> Cuối cùng là các nút **tốc độ phát lại**: 1× là thời gian thực, 10× là gấp 10 lần. Bây giờ tôi sẽ tua nhanh để xem điều gì xảy ra khi bệnh nhân mệt.

---

## PHẦN 4 — TUA NHANH VÀ QUAN SÁT CHUYỂN TRẠNG THÁI (2:15 – 3:15)

### 4a. Banner trạng thái — Lúc KHÔNG MỎI

`[🖱️ Di chuột xuống khu vực Banner trạng thái (nền xanh lá nhạt)]`

> Trước khi tua nhanh, hãy nhìn khu vực **Banner Trạng thái** bên dưới.

`[🖱️ Chỉ vào dòng nhỏ "Trạng thái (LogisticRegression)"]`

> Dòng nhỏ ghi **"Trạng thái"** kèm tên mô hình đang được dùng — ở đây là **Logistic Regression**, mô hình được chọn vì có kết quả ổn định nhất khi kiểm tra trên nhiều bệnh nhân khác nhau.

`[🖱️ Chỉ vào chữ lớn "KHÔNG MỎI" (màu xanh)]`

> Chữ lớn hiện **"KHÔNG MỎI"** — tức mô hình nhận định cơ đang ở trạng thái bình thường. Banner nền xanh, viền xanh.

`[🖱️ Di chuột sang Card "Mức độ MVC" bên phải]`

> Bên cạnh là **Card Mức độ MVC** — hiển thị giai đoạn tập hiện tại là bao nhiêu phần trăm sức co cơ tối đa. Ví dụ lúc này đang ở mức thấp.

### 4b. Tua nhanh — Xem chuyển đổi

`[🖱️ Bấm nút "5×" hoặc "10×"]`

> Tôi tăng tốc lên **5 lần** để mô phỏng buổi tập kéo dài. Bây giờ quan sát:

`[🖱️ Bấm "Phát" — animation chạy nhanh. Di chuột lên biểu đồ sóng]`

> Biên độ tín hiệu EMG đang **tăng dần** khi cường độ co cơ tăng — sóng dao động rộng hơn rõ rệt.

`[👀 Chờ đến lúc banner chuyển từ xanh sang đỏ — ĐÂY LÀ KHOẢNH KHẮC QUAN TRỌNG]`

`[🖱️ Bấm "Tạm dừng" ngay khi banner chuyển đỏ]`

> ⏸️ *(dừng nói 2 giây để người xem thấy rõ)*

### 4c. Banner trạng thái — Lúc MỎI

`[🖱️ Chỉ vào banner đỏ "MỎI"]`

> Hệ thống đã chuyển sang trạng thái **MỎI**. Banner chuyển **nền đỏ, viền đỏ**, chữ lớn hiện **"MỎI"** — cảnh báo rõ ràng cho chuyên viên phục hồi chức năng.

`[🖱️ Chỉ vào dòng khuyến nghị bên dưới banner (nền đỏ nhạt)]`

> Đồng thời xuất hiện **khuyến nghị lâm sàng tự động**: *"Giảm cường độ hoặc cho bệnh nhân nghỉ giữa hiệp"*. Khuyến nghị này chỉ hiện khi phát hiện mỏi — giúp chuyên viên ra quyết định can thiệp kịp thời.

`[🖱️ Di chuột sang card MVC — lúc này hiện mức cao, ví dụ "90%" hoặc "10% (sau mỏi)"]`

> Card MVC cho thấy giai đoạn tập hiện tại — ở đây là sau giai đoạn co cơ mạnh hoặc lần đo lại sau mỏi.

### 4d. Sơ đồ điện cực lúc mỏi

`[🖱️ Di chuột lên sơ đồ 64 điện cực — lúc này nhiều chấm đỏ]`

> Nhìn lại sơ đồ 64 điện cực: rất nhiều chấm đã chuyển sang **đỏ**. So với lúc đầu toàn xanh, bây giờ ta thấy rõ **vùng cơ bị ảnh hưởng bởi mỏi** tập trung ở đâu. Thông tin này giúp chuyên viên xác định chính xác khu vực cần can thiệp.

---

## PHẦN 5 — GIẢI THÍCH BIỂU ĐỒ XU HƯỚNG RMS & MDF (3:15 – 3:50)

`[🖱️ Di chuột xuống biểu đồ "2. Xu hướng RMS & MDF"]`

> Tiếp theo là biểu đồ **Xu hướng RMS & MDF** — biểu đồ cốt lõi thể hiện cơ chế sinh lý của mỏi cơ.

`[🖱️ Chỉ vào đường đỏ, rồi nhìn lên nhãn "RMS" ở trục trái]`

> **Đường đỏ là RMS** — Root Mean Square — đo biên độ trung bình của tín hiệu. Trục Y bên trái. Khi cơ mỏi, cơ phải huy động thêm đơn vị vận động để duy trì lực, nên **RMS có xu hướng tăng**.

`[🖱️ Chỉ vào đường xanh dương, rồi nhìn lên nhãn "MDF (Hz)" ở trục phải]`

> **Đường xanh là MDF** — Median Frequency — tần số trung vị của phổ công suất, đơn vị Hertz. Trục Y bên phải. Khi cơ mỏi, tốc độ dẫn truyền sợi cơ giảm, phổ công suất dịch về vùng tần số thấp, nên **MDF giảm dần**. Đây là dấu hiệu kinh điển và bền vững nhất của mỏi cơ trên sEMG.

`[🖱️ Chỉ vào vùng nền xanh lá nhạt (Normal) và vùng nền đỏ nhạt (Fatigue)]`

> **Nền xanh lá nhạt** là các giai đoạn mô hình nhận định cơ **bình thường**. **Nền đỏ nhạt** là giai đoạn **mỏi**. Ranh giới giữa các giai đoạn được đánh dấu bằng đường đứt nét kèm nhãn mức MVC.

`[🖱️ Chỉ vào đường dọc đang di chuyển (playback cursor)]`

> **Đường dọc di chuyển** là vị trí phát lại hiện tại — cho thấy ta đang xem tới đâu trong buổi tập.

`[🖱️ Di chuột qua trục X phía dưới]`

> Trục X là toàn bộ thời gian buổi tập từ đầu tới cuối. Khác với biểu đồ sóng ở trên chỉ xem 3 giây, biểu đồ này cho thấy **toàn cảnh xu hướng** — như nhìn bức tranh lớn thay vì chỉ một khung hình.

---

## PHẦN 6 — CHI TIẾT DỰ ĐOÁN VÀ METRICS (3:50 – 4:35)

### 6a. Bảng chi tiết dự đoán theo model

`[🖱️ Click vào mục "3. Chi tiết dự đoán theo model (giai đoạn hiện tại)" — mở rộng ra]`

> Mở phần **Chi tiết dự đoán** — đây là bảng so sánh kết quả của **tất cả 6 mô hình** tại giai đoạn tập hiện tại.

`[🖱️ Di chuột dọc theo bảng — 3 cột: Model, Dự đoán, P(Fatigue)]`

> Cột **Model**: tên 6 mô hình — SVM, KNN, LDA, Decision Tree, Random Forest, Logistic Regression.
>
> Cột **Dự đoán**: kết quả phân loại — "Normal" hoặc "Fatigue".
>
> Cột **P(Fatigue)**: xác suất mỏi — ví dụ 85.3% nghĩa là mô hình tin rằng 85% khả năng cơ đang mỏi.

`[🖱️ Chỉ vào dòng mà hầu hết model đều nói Fatigue]`

> Ở giai đoạn này, ta thấy đa số mô hình đều nhận định **Fatigue** — kết quả nhất quán giữa các mô hình khác nhau, tăng thêm độ tin cậy cho chẩn đoán.

### 6b. Bảng kết quả phân loại tổng thể

`[🖱️ Click vào mục "4. Kết quả phân loại của các mô hình (Classification results)" — mở rộng]`

> Phần cuối là **Bảng kết quả phân loại tổng thể** — đánh giá chất lượng của từng mô hình trên **dữ liệu test**.

`[🖱️ Di chuột qua bảng metrics — hàng tiêu đề]`

> Đây là bảng metrics tiêu chuẩn trong Machine Learning. Mỗi dòng là một model:

`[🖱️ Chỉ vào từng cột header khi nói tên]`

> **Accuracy** — tỷ lệ dự đoán đúng tổng thể.
> **Precision** — trong các trường hợp hệ thống nói "mỏi", bao nhiêu phần trăm thực sự mỏi.
> **Recall** — trong tất cả ca thực sự mỏi, hệ thống phát hiện được bao nhiêu phần trăm.
> **F1** — trung bình điều hòa của Precision và Recall — chỉ số cân bằng nhất.
> **CV-Acc** — Accuracy đánh giá chéo trên tập huấn luyện.
> **AUC** — diện tích dưới đường ROC, đo khả năng phân biệt giữa hai lớp.

`[🖱️ Nếu cần nhấn mạnh: chỉ vào dòng LogisticRegression]`

> Điểm quan trọng: dữ liệu test là một bệnh nhân **hoàn toàn tách biệt** — Subject 9 — chưa từng tham gia huấn luyện. Phương pháp này gọi là **Leave-One-Subject-Out**, đảm bảo đánh giá khả năng tổng quát hóa cho bệnh nhân mới.

### 6c. Confusion Matrix

`[🖱️ Cuộn xuống vùng Confusion Matrix — các khung vuông có số lớn]`

> Bên dưới là **Confusion Matrix** — ma trận nhầm lẫn — cho từng mô hình.

`[🖱️ Chỉ vào 1 confusion matrix, ví dụ LogisticRegression]`

> Cách đọc: hàng là **nhãn thực tế**, cột là **dự đoán của mô hình**.
>
> Ô **xanh trên trái** — True Negative — cơ thực sự bình thường và mô hình nói đúng là bình thường.
> Ô **xanh dưới phải** — True Positive — cơ thực sự mỏi và mô hình phát hiện đúng.
> Ô **đỏ trên phải** — False Positive — báo nhầm mỏi khi thực tế không mỏi.
> Ô **đỏ dưới trái** — False Negative — bỏ sót mỏi khi thực tế đang mỏi — đây là lỗi nguy hiểm nhất trong lâm sàng.

`[🖱️ Dừng chỉ, để tay nghỉ]`

> Nhìn tổng thể: các ô xanh có số lớn, các ô đỏ có số nhỏ — nghĩa là mô hình hoạt động tốt.

---

## PHẦN 7 — KẾT LUẬN (4:35 – 5:00)

`[🖱️ Cuộn lên trên cùng — cho thấy lại tổng thể screen. Bấm "Phát" để animation chạy vài giây ở 1×]`

> Tổng kết Use Case 1:
>
> Hệ thống nhận tín hiệu điện cơ bề mặt **64 kênh** ở **2000 Hz** — trích xuất **14 đặc trưng** miền thời gian và tần số — chọn đặc trưng tối ưu bằng **mRMR** — huấn luyện **6 mô hình** Machine Learning — và trình bày kết quả trên giao diện **thời gian thực** kèm **khuyến nghị lâm sàng tự động**.

> Toàn bộ animation chạy ở **60 khung hình/giây** trực tiếp trên trình duyệt, không cần server xử lý từng frame.

> Mô hình được đánh giá bằng phương pháp **Leave-One-Subject-Out** — đảm bảo kết quả phản ánh đúng khả năng áp dụng cho bệnh nhân mới.

`[🖱️ Dừng animation, để screen ở trạng thái đẹp]`

> Cảm ơn quý thầy cô đã theo dõi.

---

## 📐 SƠ ĐỒ LAYOUT THAM KHẢO

```
┌─────────────────────────────────────────────────────────────────────┐
│  Giám sát Mỏi cơ                                                   │
│  [Tab: Giám sát Real-time] [Tab: Phục hồi cơ]                      │
├───────────────┬──────────────────┬──────────────┐                   │
│ Nhóm cơ ▼    │ [Bắt đầu mô     │ [Đặt lại]    │                   │
│ Biceps       │  phỏng buổi tập] │              │                   │
├══════════════════════════════════════════════════╤══════════════════╡
│ ① BIỂU ĐỒ SÓNG EMG THỜI GIAN THỰC              │ ② SƠ ĐỒ 64      │
│ ┌──────────────────────────────────────────┐     │    ĐIỆN CỰC      │
│ │  Trục Y: biên độ          3s window      │     │ ┌──────────────┐ │
│ │  ─── xanh: tín hiệu thô                 │     │ │ ●● ●● ●●    │ │
│ │  ─── vàng: rectified envelope            │     │ │ ●● ●● ●●    │ │
│ │  --- đứt: ranh giới segment + nhãn %MVC  │     │ │ 13×5 grid    │ │
│ │  Trục X: thời gian (giây)                │     │ │ xanh/đỏ/xám │ │
│ └──────────────────────────────────────────┘     │ └──────────────┘ │
│  Legend: [Tín hiệu thô] [Rectified envelope]    │ [Ko mỏi][Mỏi]   │
│                                                  │ [Kênh ko hợp lệ]│
├════════════════════════════════════════════════════╧═════════════════╡
│ ③ THANH ĐIỀU KHIỂN                                                   │
│ [Phát/Dừng] [Envelope] [════════ thanh seek ════════] 12.5s/45.0s   │
│                                                    [1×][2×][5×][10×] │
├════════════════════════════════════════╤═════════════════════════════╡
│ ④ BANNER TRẠNG THÁI                   │ ⑤ CARD MVC                  │
│ ┌────────────────────────────────┐     │ ┌─────────────────────────┐ │
│ │ Trạng thái (LogisticRegression)│     │ │ Mức độ MVC              │ │
│ │ KHÔNG MỎI / MỎI               │     │ │ 10% / 90% (sau mỏi)    │ │
│ │ (nền xanh hoặc đỏ)            │     │ └─────────────────────────┘ │
│ └────────────────────────────────┘     │                             │
├════════════════════════════════════════╧═════════════════════════════╡
│ ⑥ KHUYẾN NGHỊ (chỉ hiện khi MỎI)                                    │
│ [❗ Khuyến nghị: giảm cường độ hoặc cho bệnh nhân nghỉ giữa hiệp]   │
├══════════════════════════════════════════════════════════════════════╡
│ ⑦ BIỂU ĐỒ XU HƯỚNG RMS & MDF                                       │
│ ┌──────────────────────────────────────────────────────────────────┐ │
│ │ RMS (trục trái)        toàn timeline buổi tập      MDF Hz (phải)│ │
│ │ ─── đỏ: RMS             nền xanh = Normal                      │ │
│ │ ─── xanh: MDF            nền đỏ  = Fatigue                     │ │
│ │ --- đứt: ranh giới segment     | cursor dọc                    │ │
│ │ Trục X: thời gian toàn buổi tập                                │ │
│ └──────────────────────────────────────────────────────────────────┘ │
│ Legend: [RMS] [MDF (Hz)] [Normal] [Fatigue]                          │
├══════════════════════════════════════════════════════════════════════╡
│ ⑧ ▸ 3. Chi tiết dự đoán theo model (giai đoạn hiện tại)    [mở/đóng]│
│   ┌──────────────┬──────────────┬──────────────┐                     │
│   │ Model        │ Dự đoán      │ P(Fatigue)   │                     │
│   │ SVM          │ Fatigue      │ 85.3%        │                     │
│   │ KNN          │ Fatigue      │ 78.1%        │                     │
│   │ ...          │ ...          │ ...          │                     │
│   └──────────────┴──────────────┴──────────────┘                     │
├══════════════════════════════════════════════════════════════════════╡
│ ⑨ ▸ 4. Kết quả phân loại (Classification results)          [mở/đóng]│
│   ┌──────┬────────┬─────────┬────────┬──────┬──────┬──────┐         │
│   │Model │Accuracy│Precision│ Recall │  F1  │CV-Acc│ AUC  │         │
│   │SVM   │ 95.1%  │ 85.3%   │100.0%  │92.1% │...   │97.7% │         │
│   └──────┴────────┴─────────┴────────┴──────┴──────┴──────┘         │
│                                                                      │
│   ⑩ CONFUSION MATRIX (cho từng model)                                │
│   ┌────────────────────┐  ┌────────────────────┐                     │
│   │    SVM              │  │    KNN              │                     │
│   │      Pred:N  Pred:F │  │      Pred:N  Pred:F │                     │
│   │ Act:N [TN]   [FP]  │  │ Act:N [TN]   [FP]  │                     │
│   │ Act:F [FN]   [TP]  │  │ Act:F [FN]   [TP]  │                     │
│   └────────────────────┘  └────────────────────┘                     │
└══════════════════════════════════════════════════════════════════════╘
```

---

## ⏱ TIMELINE TÓM TẮT

| Thời gian | Phần | Đang giải thích gì |
|---|---|---|
| 0:00 – 0:50 | Phần 1 | Màn hình chờ, dropdown nhóm cơ, mục đích hệ thống |
| 0:50 – 1:15 | Phần 2 | Loading, pipeline ML đang chạy |
| 1:15 – 2:15 | Phần 3 | ① Biểu đồ sóng EMG (2 đường, trục, ranh giới) → ② Sơ đồ 64 điện cực → ③ Thanh điều khiển |
| 2:15 – 3:15 | Phần 4 | ④ Banner KHÔNG MỎI → tua nhanh → ④ Banner MỎI → ⑥ Khuyến nghị → ⑤ MVC → ② Sơ đồ lúc mỏi |
| 3:15 – 3:50 | Phần 5 | ⑦ Biểu đồ RMS & MDF (2 đường, 2 trục, nền màu, cursor) |
| 3:50 – 4:35 | Phần 6 | ⑧ Bảng dự đoán 6 model → ⑨ Metrics → ⑩ Confusion Matrix |
| 4:35 – 5:00 | Phần 7 | Kết luận tổng kết |
