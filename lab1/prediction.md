## Part C — Prediction Before Experiment

### Prediction 1 — Vocabulary
Nếu corpus có 30K documents, vocabulary sẽ có khoảng bao nhiêu unique terms?
    30 000 * 15 / 3 = 450 000 / 3 = 150 000
    (Do đang ước lượng trung bình một document có khoảng 15 từ và giả định unique terms chiếm 1/3.)

### Prediction 2 — Sparsity
TF-IDF matrix sẽ dense hay sparse? 
    sparse 

Tỷ lệ zero entries có thể lớn đến mức nào? 
    Khoảng >90% 

### Prediction 3 — Search
Với một query bất kỳ:
Các documents đứng đầu kết quả tìm kiếm có nhất thiết là documents gần nghĩa nhất
không?
    Không nhất thiết. Với các bài đang xét chủ yếu đề cập đến sự trùng khớp giữa các term chứ không phải 
    về mặt ngữ nghĩa.
    Ví dụ: "Bạn Liên làm bài" với "Bạn Liên không làm bài" trùng khớp đến 3 terms nhưng trái nghĩa nhau hoàn toàn.

