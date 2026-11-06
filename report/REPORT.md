# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** CPU

**Python / PyTorch / Ultralytics:** 3.13.18 / 2.11.0+cpu / 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt` (cả ba PASS SHA-256 trong ô setup)

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (không đổi)

**Thay đổi so với notebook nguồn:** Không thay đổi code, checkpoint hay threshold. Chỉ thêm hai ô phụ ở cuối: `!zip -r day1_lab_outputs.zip day1_lab_outputs` để tải kết quả về máy trước khi viết báo cáo, và một ô in phiên bản Python/PyTorch/device. Hai ô này không thay đổi output.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. **Record hạng 1:** `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K  
   (Top-5 còn lại: minibus 0.164, police_van 0.086, recreational_vehicle 0.054, streetcar 0.048.)

2. **Cả bức ảnh.** Record không có `bbox_xyxy` hay polygon, chỉ có một `class_name` + `score` gắn với `sample_id=traffic` (kích thước 640×428). Năm lớp top-5 là năm phương án loại trừ nhau cho *toàn ảnh*, tổng score ≈ 0.86. Thực tế ảnh có hàng chục xe buýt, ô tô, người đi bộ, nhưng model chỉ trả **một** nhãn “cab” cho cả cảnh — nó không chỉ ra chiếc taxi nào.

3. **Đi kèm taxonomy/checkpoint.** `yolo11n-cls.pt` được huấn luyện trên ImageNet-1K nên chỉ chọn được trong 1000 lớp cố định đó (`result.names`). Model không tự nghĩ ra lớp mới. Vì vậy ảnh `traffic` không có lớp “traffic scene / đường phố”, và ảnh `kitchen` ra top-1 là “gong” (0.42) — một lớp nghe vô lý nhưng là lớp gần nhất trong taxonomy.

4. `class_name` có thể trùng/không rõ nghĩa giữa các taxonomy (vd. `dining_table` có trong cả ImageNet-1K là id 532 và COCO-80 là id 60). `class_id` chỉ có nghĩa khi biết nó thuộc taxonomy nào. Ghi đủ ba trường giúp: tra cứu lại chính xác lớp, map/convert giữa các taxonomy, và phát hiện khi ai đó trộn nhãn từ hai bộ lớp khác nhau.

5. Guideline cần nói rõ: (a) nhãn mô tả **chủ thể chính / chủ đề chiếm ưu thế** của ảnh hay vật thể nổi bật nhất; (b) tiêu chí chọn khi có nhiều vật (vd. vật chiếm diện tích lớn nhất, hoặc theo thứ tự ưu tiên lớp đã định sẵn); (c) có cho phép lớp “khác/không xác định” không; (d) khi nào phải escalation. Với ảnh `traffic`: nếu taxonomy chỉ có xe cụ thể thì chọn “bus” (chiếm nhiều diện tích nhất) hay “cab”? Guideline phải trả lời trước, không để mỗi annotator chọn một kiểu.

6. `score` chỉ là mức tin của model, có thể **tự tin mà sai**: ảnh `kitchen` được gán “gong” với score 0.42, cao gấp ~6 lần lớp thứ 2, nhưng trong ảnh không có cái chiêng nào (có thể là chảo đồng treo trên tường bị nhầm). Ground truth phải do người gán theo guideline, nếu không thì lỗi của model sẽ bị đưa ngược vào dữ liệu huấn luyện.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`.

1. **Record:** `class_name`=person ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68

2. Gốc (0, 0) là góc trên-trái ảnh (ảnh 640×427). Góc trên-trái của box ở **x=385, y=69** (cách mép trái 385 px, cách mép trên 69 px — ngang đầu người đầu bếp). Góc dưới-phải ở **x=499, y=349** (ngang chân). Box rộng ≈ **114 px**, cao ≈ **280 px** — hình chữ nhật đứng, đúng dáng một người đứng quay lưng.

3. Ảnh `kitchen`:
   - Ngưỡng **0.20 → 17** prediction (thêm spoon ×3, potted plant, dining table, bottle so với 0.35)
   - Ngưỡng 0.35 → 11 prediction (số dùng trong JSON)
   - Ngưỡng **0.60 → 6** prediction (person ×2, bowl ×2, oven ×2)

4. Khi hạ ngưỡng, reviewer phải: kiểm tra từng box thêm vào xem là vật thật hay false positive (vd. “spoon” ở 0.20 thực ra là muôi/vá treo tường — có đúng lớp không?); xử lý box trùng/chồng lấn cùng một vật; kiểm tra lớp có đúng không (“oven 0.69” bên trái thực chất là bếp lò + bàn — box có bao đúng không?). Đổi lại, ngưỡng thấp ít bỏ sót vật hơn (vd. “dining table” chỉ xuất hiện ở 0.20). Khối lượng review tăng ~55% (11 → 17 box).

5. **Quy tắc box:** “Vẽ hộp chữ nhật **sát nhất** bao trọn **phần nhìn thấy được** của vật thể (cạnh hộp chạm điểm ngoài cùng của vật, không chừa nền thừa > vài pixel). Mỗi vật thể riêng biệt có một hộp riêng, kể cả khi hai vật cùng lớp đứng sát nhau. Không đoán phần bị che.”

6. **Vật bị che/cắt mép:** “person” 0.61 ở mép trái, `bbox_xyxy`=[0.12, 263.18, 61.55, 310.78] — thực ra chỉ là **một cánh tay/bàn tay** thò vào khung hình từ mép trái, phần còn lại của người nằm ngoài ảnh. → **Viết thành quy tắc guideline**, vì tình huống “vật bị cắt mép ảnh” lặp lại rất thường xuyên trong mọi dataset, cần một câu trả lời nhất quán: *“Vẫn gán nhãn person nếu nhận ra được là bộ phận cơ thể người; box chỉ bao phần nhìn thấy; đánh dấu thuộc tính truncated=true.”* Chỉ escalation nếu không chắc phần nhìn thấy có đủ để nhận ra lớp hay không.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`.

1. **Record:** `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899318 ; `polygon_point_count`=348 ; 5 điểm đầu của `polygon_xy`=[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]

2. Box chỉ cho 4 số (hình chữ nhật), luôn chứa cả nền thừa — với `kitchen-001` box chứa cả tường, móc treo, khăn ở hai bên người. Polygon 348 điểm bám theo **đường viền thật** (đầu, vai, khuỷu tay chống hông, hai chân), cho biết hình dạng, tư thế, diện tích pixel thật của vật và tách được vật khỏi nền/vật khác nằm trong cùng box.

3. `instance_id` chỉ để **phân biệt từng đối tượng riêng trong output này**: ảnh `kitchen` có 4 bowl (kitchen-002, -003, -006, -010) — cùng `class_id`=45 nhưng 4 `instance_id` khác nhau, mỗi cái một polygon riêng. Nó không phải class ID vì không nói vật thuộc lớp gì (lớp nằm ở `class_id`/`class_name`). Nó không phải tracking ID vì chỉ là số thứ tự trong một lần predict trên một ảnh tĩnh — chạy lại hay sang ảnh khác thì số này không gắn với cùng một vật thật.

4. **Quy tắc vẽ biên mask:** “Vẽ polygon theo **mép ngoài phần nhìn thấy** của vật, sai lệch tối đa ~2 px. Không bao gồm vật khác che phía trước (phần bị che thì cắt bỏ khỏi mask, không đoán). Vật có lỗ thủng (quai cốc, khoảng giữa chân và tay) thì không tô phần nền lọt qua. Hai instance chạm nhau thì mask không được chồng lấn.”

5. **Hai vật chạm nhau/che nhau:** mask `dining table` (kitchen-005, 0.58, 598 điểm) tô **luôn cả các bát, cuộn bột, đĩa bánh** đặt trên mặt bàn và tràn xuống gầm bàn; đồng thời mask “bowl” 0.70/0.74 chồng lên chính vùng bàn đó → một pixel thuộc hai instance. Ngoài ra “oven” 0.49 tô cả vùng bếp tối lẫn mép bàn. → **Viết thành guideline**, vì “vật đặt trên bàn” là tình huống chung: *“Pixel thuộc vật nằm trên/trước thì gán cho vật đó; mask bàn phải khoét bỏ các vật đặt trên nó.”* Riêng việc “oven” bên trái có thực sự là lò nướng hay là bếp lò củi cổ (lớp không rõ trong COCO-80) thì **escalation** cho Lab Coach, vì đó là câu hỏi taxonomy chứ không phải câu hỏi vẽ biên.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | **1 nhãn cho cả ảnh**: `class_id` + `class_name` + `taxonomy_name` | `kitchen` top-1 = “gong” 0.42 (không có chiêng trong ảnh); `traffic` top-1 = “cab” 0.51 dù xe buýt chiếm nhiều diện tích hơn | Xem toàn ảnh, chọn 1 lớp theo quy tắc “chủ thể chính” trong guideline; không copy top-1 của model | Nhãn có đúng taxonomy không; các annotator có chọn nhất quán cho ảnh nhiều chủ đề không; tỉ lệ đồng thuận |
| Phát hiện vật thể | **1 box xyxy (pixel) + 1 lớp cho mỗi vật** | Cánh tay ở mép trái bị gán “person” 0.61 [0.12, 263.18, 61.55, 310.78]; nhiều bowl/cup nhỏ chồng box nhau; số box đổi 17 → 11 → 6 theo ngưỡng 0.20/0.35/0.60 | Vẽ box sát phần nhìn thấy cho **mọi** vật thuộc taxonomy, kể cả vật model bỏ sót; đánh dấu truncated/occluded | Box có sát không; có vật bị bỏ sót (nồi, chảo treo tường không được gán) hay box thừa; đúng lớp không |
| Instance segmentation | **1 polygon + 1 lớp + 1 instance_id cho mỗi vật** | Mask `dining table` (kitchen-005) tô lẫn bát/bột trên bàn và chồng với mask bowl; “oven” 0.49 lẫn vùng tối | Vẽ polygon theo mép phần nhìn thấy, khoét vật nằm trên, mỗi instance một mask riêng không chồng lấn | Biên có bám sát mép không; có pixel thuộc 2 mask không; 2 vật cùng lớp có bị gộp thành 1 instance không |

## 5. An toàn dữ liệu

1. Chỉ dùng 3 ảnh COCO công khai do notebook tải và kiểm SHA-256; không tải ảnh cá nhân/nội bộ lên Colab và không ghi họ tên, MSSV hay thông tin cá nhân vào báo cáo hoặc output.
2. Dừng ngay, không gán nhãn/không đưa lên repo, và báo cho **Lab Coach / giảng viên phụ trách**.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
