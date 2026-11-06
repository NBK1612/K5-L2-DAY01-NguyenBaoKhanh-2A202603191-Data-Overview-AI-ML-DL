# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** CPU

**Python / PyTorch / Ultralytics:** 3.13.18 / 2.11.0+cpu / 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (không đổi)

**Thay đổi so với notebook nguồn:** Không thay đổi code, checkpoint hay threshold. Chỉ thêm hai ô phụ ở cuối notebook: một ô nén `day1_lab_outputs` thành zip để tải về máy, một ô in phiên bản Python/PyTorch/device. Hai ô này không thay đổi output.

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

1. `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K

2. **Cả bức ảnh.** Đây là tác vụ `image_classification`: model trả một nhãn "cab" cho toàn ảnh `traffic`. Record không có tọa độ (không có `bbox_xyxy` hay `polygon_xy` như ở mục 2, 3) nên không chỉ vào vật thể cụ thể nào.

3. **Đi kèm taxonomy/checkpoint**, model không tự nghĩ ra. `yolo11n-cls.pt` được huấn luyện trên ImageNet-1K nên chỉ chọn trong 1000 lớp cố định của taxonomy này.

4. Vì cùng một tên/số có thể mang nghĩa khác nhau ở taxonomy khác nhau. Ví dụ `dining_table` là id 532 trong ImageNet-1K nhưng `dining table` là id 60 trong COCO-80. Ghi đủ ba trường thì mới tra lại đúng lớp và không trộn nhãn giữa hai bộ lớp.

5. Guideline phải nói rõ **tiêu chí chọn chủ thể chính**, ví dụ: chọn vật chiếm diện tích lớn nhất hoặc chủ đề nổi bật nhất của ảnh, và nếu không xác định được thì escalation. Như ảnh `traffic` có cả xe buýt, ô tô, người đi bộ — không có quy tắc thì mỗi annotator sẽ chọn một lớp khác nhau.

6. Vì score chỉ là mức tin của model, không phải độ đúng — model có thể tự tin mà sai. Ví dụ ảnh `kitchen` có top-1 là "gong" (score 0.42) dù trong ảnh không có cái chiêng nào. Lấy prediction làm ground truth sẽ đưa lỗi của model vào dữ liệu huấn luyện.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. `class_name`=person ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68

2. Gốc (0, 0) ở góc trên-trái ảnh. Góc trên-trái của box ở **x = 385, y = 69** pixel (ngang đầu người đầu bếp), góc dưới-phải ở **x = 499, y = 349** (ngang chân). Box rộng khoảng **114 px**, cao khoảng **280 px**.

3. - Ngưỡng **0.20**: **17** prediction  
   - Ngưỡng **0.60**: **6** prediction

4. Reviewer phải kiểm tra thêm nhiều box có score thấp: box đó có đúng là vật thật không (loại bỏ false positive), có đúng lớp không, có box nào trùng lên cùng một vật không. Ở ảnh `kitchen`, hạ ngưỡng từ 0.60 xuống 0.20 làm số box cần xem tăng từ 6 lên 17.

5. *"Vẽ hộp chữ nhật sát nhất bao trọn phần nhìn thấy được của vật thể; mỗi vật thể một hộp riêng, kể cả khi các vật cùng lớp đứng sát nhau; không vẽ đoán phần bị che hoặc nằm ngoài ảnh."*

6. Box "person" score 0.61, `bbox_xyxy`=[0.12, 263.18, 61.55, 310.78] ở mép trái ảnh: chỉ thấy một cánh tay, phần còn lại của người bị cắt ra ngoài khung hình.  
   → **Viết thành quy tắc guideline**, vì vật bị cắt mép ảnh là tình huống gặp thường xuyên, cần một cách xử lý thống nhất cho mọi annotator (vd. *"vẫn gán nhãn nếu nhận ra được lớp; box chỉ bao phần nhìn thấy"*).

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899318 ; `polygon_point_count`=348 ; 5 điểm đầu của `polygon_xy`=[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]

2. Box chỉ là hình chữ nhật 4 số nên chứa cả phần nền xung quanh. Polygon (348 điểm với `kitchen-001`) bám theo **đường viền thật** của vật, cho biết **hình dạng** chính xác và những pixel nào thuộc vật, tách được vật khỏi nền.

3. `instance_id` dùng để **phân biệt từng đối tượng riêng** trong output. Ví dụ ảnh `kitchen` có 4 bowl cùng `class_id`=45 nhưng có 4 `instance_id` khác nhau (kitchen-002, -003, -006, -010).  
   - Không phải class ID: nó không cho biết vật thuộc lớp gì (lớp nằm ở `class_id`).  
   - Không phải tracking ID: nó chỉ là số thứ tự trong một lần chạy trên một ảnh, không dùng để theo dõi cùng một vật qua nhiều ảnh/khung hình.

4. *"Vẽ polygon bám sát mép ngoài phần nhìn thấy của vật; không tô lên vật khác hoặc phần nền; hai vật chạm nhau thì mask không được chồng lên nhau."*

5. Mask `dining table` (kitchen-005, score 0.58) tô luôn cả các bát và bột đặt trên mặt bàn, chồng lên mask của "bowl" → hai vật chạm nhau bị lẫn mask.  
   → **Viết thành quy tắc guideline**, vì "vật đặt trên bàn" là tình huống chung, lặp lại nhiều: *"pixel thuộc vật nằm trên/phía trước thì gán cho vật đó; mask bàn phải bỏ phần các vật đặt trên nó."*

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | 1 nhãn lớp cho cả ảnh (`class_id` + `class_name` + `taxonomy_name`) | `kitchen` top-1 = "gong" (0.42) nhưng ảnh không có chiêng; `traffic` có nhiều loại xe nhưng chỉ được chọn 1 lớp | Xem cả ảnh, chọn 1 lớp theo guideline, không chép kết quả model | Nhãn có đúng lớp và đúng taxonomy không; các annotator có chọn thống nhất không |
| Phát hiện vật thể | Mỗi vật 1 box `xyxy` (pixel) + 1 lớp | "person" 0.61 ở mép trái chỉ là cánh tay bị cắt mép; số box đổi từ 17 → 6 khi tăng ngưỡng 0.20 → 0.60 | Vẽ box sát phần nhìn thấy cho mọi vật thuộc taxonomy, kể cả vật model bỏ sót | Box có sát vật không; có vật bị bỏ sót hoặc box thừa không; đúng lớp không |
| Instance segmentation | Mỗi vật 1 polygon + 1 lớp + 1 `instance_id` | Mask `dining table` (kitchen-005) tô lẫn cả bát trên bàn, chồng với mask bowl | Vẽ polygon theo mép vật, mỗi vật một mask riêng, không chồng nhau | Biên có bám sát mép không; có pixel thuộc 2 mask không; 2 vật có bị gộp thành 1 không |

## 5. An toàn dữ liệu

1. Chỉ dùng 3 ảnh COCO công khai do notebook tải về, không đưa ảnh cá nhân hay dữ liệu nội bộ vào Colab, và không ghi họ tên, MSSV hay thông tin cá nhân vào báo cáo/output.

2. Dừng ngay, không gán nhãn và không đưa lên repo, rồi báo cho **Lab Coach**.

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
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
