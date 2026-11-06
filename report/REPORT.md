
# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

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

1. Chép record hạng 1: `class_id`=… ; `class_name`=… ; `rank`=1 ; `score`=… ; `taxonomy_name`=…
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
6. Vì sao không được lấy `score` của model làm ground truth?

trả lời:

{
    "sample_id": "traffic",
    "coco_image_id": 210273,
    "image_width": 640,
    "image_height": 428,
    "task": "image_classification",
    "taxonomy_name": "ImageNet-1K",
    "model_file": "yolo11n-cls.pt",
    "model_sha256": "c62d41bf9625777760018bf914d2e6cd472420ccd01706d97a61cb6c82502bd7",
    "ultralytics_version": "8.4.145",
    "rank": 1,
    "class_id": 468,
    "class_name": "cab",
    "score": 0.510915
  },

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=… ; `score`=… ; `bbox_xyxy`=[…, …, …, …] ; `bbox_width`=… ; `bbox_height`=…
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?

trả lời:

1. Chép một record
- `class_name = person`
- `score = 0.912625`
- `bbox_xyxy = [385.33, 69.24, 498.92, 348.92]`
- `bbox_width = 113.58`
- `bbox_height = 279.68`

2. Giải thích box bằng lời thường
`bbox_xyxy = [385.33, 69.24, 498.92, 348.92]` nghĩa là:
- Góc trên-trái của bounding box ở tọa độ `(385.33, 69.24)`.
- Góc dưới-phải ở tọa độ `(498.92, 348.92)`.
- Chiều rộng của box là `113.58 pixel`.
- Chiều cao của box là `279.68 pixel`.
Bounding box này bao quanh một đối tượng được model dự đoán là `person` trong ảnh `kitchen`.

3. So sánh hai ngưỡng
- Ngưỡng `0.20`: phát hiện `17` vật thể.
- Ngưỡng `0.60`: phát hiện `6` vật thể.
Ngoài ra, ở ngưỡng mặc định `0.35`, model phát hiện `11` vật thể.
Khi tăng ngưỡng từ `0.20` lên `0.60`, số prediction giảm từ 17 xuống còn 6.

4. Khi hạ ngưỡng, reviewer phải làm thêm việc gì?
Khi hạ ngưỡng, model giữ lại nhiều prediction có score thấp hơn nên số lượng vật thể cần kiểm tra tăng lên. Reviewer phải kiểm tra kỹ hơn các prediction để phát hiện false positive, box chưa chính xác hoặc đối tượng bị nhận nhầm. Vì vậy, ngưỡng thấp làm tăng khối lượng QC của reviewer.

5. Một quy tắc box cho người gán nhãn
Bounding box phải bao quanh sát phần nhìn thấy của vật thể, không cắt mất phần vật thể cần gán nhãn và không chứa quá nhiều vùng nền không liên quan. Quy tắc này phải được áp dụng nhất quán cho tất cả các ảnh.

6. Vật bị che khuất hoặc cắt mép trong ảnh `kitchen`
Trong ảnh `kitchen` có một đối tượng `person` ở sát mép trái ảnh, với box bắt đầu gần `x = 0`, cho thấy đối tượng bị cắt bởi mép ảnh.
Tôi sẽ **đưa trường hợp này lên hỏi Lab Coach (escalation)** nếu guideline hiện tại chưa quy định rõ cách gán nhãn cho vật thể bị cắt mép. Người gán nhãn không nên tự đặt ra quy tắc mới. Sau khi Lab Coach thống nhất cách xử lý, quy tắc đó có thể được bổ sung vào guideline để áp dụng nhất quán cho các trường hợp tương tự.


## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=… ; `class_name`=… ; `score`=… ; số điểm polygon=`polygon_point_count`=… ; 3–5 điểm đầu của `polygon_xy`=…
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?

trả lời:

1. Chép một record của ảnh `kitchen`

Tôi chọn record có `instance_id = kitchen-001`.

- `instance_id = kitchen-001`
- `class_id = 0`
- `class_name = person`
- `score = 0.899318`
- `bbox_xyxy = [385.45, 66.44, 498.02, 348.58]`
- `polygon_point_count = 348`

Năm điểm đầu tiên của `polygon_xy` là:

- `[446.0, 70.0]`
- `[445.0, 71.0]`
- `[444.0, 71.0]`
- `[443.0, 72.0]`
- `[442.0, 72.0]`

Record này cho biết model dự đoán một đối tượng thuộc lớp `person` với score `0.899318`. Đối tượng này không chỉ có bounding box mà còn có một polygon gồm 348 điểm tọa độ để mô tả đường biên của đối tượng.

2. So với bounding box, polygon cung cấp thêm thông tin gì?

Bounding box chỉ biểu diễn vật thể bằng một hình chữ nhật bao quanh đối tượng. Vì vậy, bounding box cho biết vị trí tổng quát của vật thể nhưng không thể hiện chính xác hình dạng thật của vật thể.

Trong khi đó, polygon gồm nhiều điểm tọa độ nối lại với nhau để mô tả đường biên của đối tượng. Nhờ đó, polygon có thể bám sát hình dạng thực tế của vật thể hơn. Ví dụ, đối với một người, polygon có thể đi theo đường viền đầu, vai, cánh tay, thân và chân thay vì chỉ dùng một hình chữ nhật lớn bao quanh toàn bộ người.

Trong record `kitchen-001`, đối tượng `person` có tới 348 điểm polygon. Điều này cho thấy segmentation cung cấp thông tin chi tiết hơn detection về vùng pixel nào thuộc về vật thể và vùng pixel nào thuộc về nền.

Vì vậy, có thể hiểu ngắn gọn:

- **Bounding box** trả lời: “Vật thể nằm ở khu vực nào?”
- **Polygon/mask** trả lời chi tiết hơn: “Chính xác phần nào của ảnh thuộc về vật thể?”

3. `instance_id` dùng để làm gì? Vì sao nó không phải class ID và cũng không phải tracking ID?

`instance_id` dùng để phân biệt **từng đối tượng riêng biệt** trong cùng một ảnh.

Ví dụ, trong ảnh `kitchen` có nhiều đối tượng thuộc cùng lớp `bowl`. Một đối tượng có:

- `instance_id = kitchen-002`
- `class_id = 45`
- `class_name = bowl`

và một đối tượng khác có:

- `instance_id = kitchen-003`
- `class_id = 45`
- `class_name = bowl`

Hai đối tượng này đều có cùng `class_id = 45` vì chúng đều thuộc lớp `bowl`, nhưng chúng có `instance_id` khác nhau vì chúng là **hai cái bát riêng biệt trong ảnh**. 

Do đó:

- `class_id` trả lời câu hỏi: **“Đối tượng thuộc loại gì?”**
- `instance_id` trả lời câu hỏi: **“Đây là đối tượng cụ thể nào trong ảnh?”**

`instance_id` cũng không phải tracking ID. Tracking ID thường được dùng trong video để theo dõi cùng một vật thể qua nhiều frame. Ví dụ, một người xuất hiện ở frame 1, frame 2 và frame 3 có thể được giữ cùng một tracking ID để hệ thống biết đó vẫn là cùng một người.

Trong bài thực hành này, dữ liệu là các ảnh tĩnh riêng biệt và mục tiêu là instance segmentation, không phải object tracking. Vì vậy, `instance_id` chỉ dùng để phân biệt từng instance trong ảnh/output hiện tại, không có ý nghĩa theo dõi một đối tượng xuyên suốt nhiều frame.

4. Quy tắc về biên mask cho người gán nhãn

Một quy tắc có thể áp dụng thống nhất cho nhiều ảnh là:

> Biên mask phải bám sát phần nhìn thấy thực tế của vật thể. Người gán nhãn không được cố ý lấy thêm nhiều vùng nền không thuộc vật thể và cũng không được bỏ sót những phần rõ ràng thuộc về vật thể. Khi hai vật thể chạm hoặc nằm sát nhau, mỗi instance phải có mask riêng và đường biên cần được xác định theo phần nhìn thấy của từng vật thể. Nếu vật thể bị che khuất, chỉ gán phần nhìn thấy được theo guideline; không tự suy đoán phần bị che nếu guideline không cho phép.

Quy tắc này quan trọng vì segmentation cần mô tả chính xác hình dạng của từng vật thể, chứ không chỉ cần một hình chữ nhật bao quanh như detection.

Nếu annotator tự suy đoán khác nhau ở các trường hợp biên mờ hoặc vật thể bị che, ground truth sẽ thiếu nhất quán. Vì vậy guideline cần quy định rõ cách xử lý các trường hợp như:

- vật thể bị cắt bởi mép ảnh;
- vật thể bị che một phần;
- hai vật thể cùng lớp chạm nhau;
- biên vật thể khó phân biệt với nền;
- vật thể rất nhỏ hoặc chỉ nhìn thấy một phần.

5. Một trường hợp khó trong ảnh `kitchen`: dùng guideline hay escalation?

Trong dữ liệu của ảnh `kitchen` có một đối tượng được model dự đoán là `potted plant` với bounding box bắt đầu tại khoảng `[0.05, 1.4, 63.45, 148.89]`. Vì tọa độ x và y bắt đầu rất gần 0, đối tượng này nằm sát góc trên-trái của ảnh và có thể bị cắt bởi mép ảnh.

Đây là một trường hợp cần chú ý khi gán mask vì annotator phải quyết định nên vẽ biên mask như thế nào đối với phần vật thể không còn xuất hiện trong ảnh.

Cách xử lý của tôi là:

- Nếu guideline đã có quy định rõ về vật thể bị cắt mép, tôi sẽ làm đúng theo guideline.
- Ví dụ, nếu guideline quy định “vật thể bị cắt mép nhưng vẫn nhận diện được thì chỉ gán mask cho phần nhìn thấy trong ảnh”, tôi sẽ áp dụng đúng quy tắc đó.
- Nếu guideline chưa nói rõ trường hợp này, hoặc tôi không chắc phần nào thuộc vật thể, tôi sẽ **escalation lên Lab Coach/Reviewer** thay vì tự đặt ra một quy tắc mới.

Lý do là ground truth phải được tạo nhất quán giữa tất cả annotator. Nếu mỗi người tự suy đoán cách xử lý vật thể bị cắt mép thì cùng một loại trường hợp có thể được gán nhãn theo nhiều cách khác nhau, làm giảm chất lượng dữ liệu huấn luyện.

Kết luận
Qua phần instance segmentation, tôi hiểu rằng prediction của model không chỉ có class và bounding box mà còn có `instance_id` và polygon/mask để mô tả từng đối tượng riêng biệt. `class_id` dùng để xác định loại vật thể, còn `instance_id` dùng để phân biệt từng cá thể trong cùng ảnh. Polygon cung cấp thông tin hình dạng chi tiết hơn bounding box vì nó mô tả đường biên của vật thể. Khi tạo ground truth, annotator phải tuân thủ guideline để giữ biên mask nhất quán; nếu gặp trường hợp không được guideline quy định rõ thì phải escalation thay vì tự suy đoán.



## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh |[SUY LUẬN TỪ YÊU CẦU BÀI] Một ảnh có một nhãn lớp cho toàn bộ ảnh, vì phần này được mô tả là “phân loại ảnh – một lớp cho toàn bộ ảnh”.  |Ảnh traffic có nhiều nội dung/vật thể cùng xuất hiện nên việc chọn một lớp duy nhất có thể là trường hợp cần bám sát guideline.  |Theo phần thuật ngữ của bài: Annotator là người gán nhãn. Vì vậy annotator phải gán nhãn theo guideline, không lấy prediction làm ground truth.  |Theo phần thuật ngữ của bài: Reviewer là người kiểm tra chất lượng. Vì vậy reviewer kiểm tra nhãn đã gán có phù hợp với guideline hay không.  |
| Phát hiện vật thể |[SUY LUẬN TỪ JSON] Mỗi vật thể cần một lớp và một bounding box. Prediction trong JSON có class_name, class_id và bbox_xyxy.| Trong ảnh kitchen có một prediction person có bbox_xyxy = [0.12, 263.18, 61.55, 310.78]. Vì tọa độ x bắt đầu rất gần 0 nên đây là trường hợp vật thể nằm sát hoặc bị cắt ở mép trái ảnh.|  Annotator gán nhãn vật thể theo guideline. Nếu gặp trường hợp khó mà guideline chưa nói rõ thì theo quy tắc bài lab phải escalation, không tự bịa nhãn.|  Reviewer kiểm tra chất lượng nhãn. Với trường hợp khó như vật thể sát/cắt mép, reviewer đối chiếu với guideline để xem cách gán nhãn có nhất quán hay không |
| Instance segmentation |[SUY LUẬN TỪ JSON] Mỗi đối tượng có class, instance_id và polygon/mask riêng. Ví dụ kitchen-001 là person và có polygon_point_count = 348.|  Trong ảnh kitchen, potted plant có box bắt đầu gần (0,0), nên đây là một trường hợp vật thể nằm sát/cắt mép ảnh và có thể khó xác định biên mask.|  Annotator gán mask/polygon cho từng đối tượng theo guideline. Nếu biên không rõ hoặc guideline chưa quy định thì escalation hỏi Lab Coach, không tự quyết định.|  Reviewer kiểm tra chất lượng mask/nhãn theo guideline và chú ý các trường hợp biên khó xác định hoặc vật thể bị cắt mép. |


## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?

trả lời:

1. **Quy tắc bảo vệ dữ liệu:**  
Chỉ sử dụng ba ảnh COCO công khai có sẵn trong bài lab; không tải ảnh cá nhân, dữ liệu nội bộ hoặc dữ liệu ngoài phạm vi bài thực hành.

2. **Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi:**  
Tôi sẽ dừng xử lý và báo cho Lab Coach hoặc giảng viên để được hướng dẫn, không tự ý sử dụng dữ liệu đó.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
