## Part K - Reflection

### 1. Prediction nào của em sai?

Prediction về vocabulary size tương đối gần với kết quả thực tế, tuy nhiên cách ước lượng ban đầu còn đơn giản. Prediction về sparsity và việc TF-IDF chủ yếu dựa trên sự trùng khớp term phù hợp với kết quả thực nghiệm.

### 2. Kết quả nào bất ngờ nhất?

Kết quả bất ngờ nhất là một số documents có similarity cao nhưng không thực sự liên quan đến query. Ví dụ, query `transformer language model` trả về document nói về `transformer/circuit board` ở vị trí đầu tiên.

### 3. Experiment nào cung cấp evidence mạnh nhất?

Phần **Error Analysis** cung cấp evidence rõ nhất vì có thể quan sát trực tiếp các trường hợp TF-IDF trả về document có lexical overlap nhưng khác về ngữ nghĩa.

### 4. Failure case quan trọng nhất là gì?

Failure case quan trọng nhất là `transformer language model`, trong đó từ `transformer` xuất hiện trong cả query và document nhưng mang hai nghĩa khác nhau. Điều này cho thấy hạn chế của lexical matching.

### 5. Nếu được xây lại search engine, em sẽ thay đổi điều gì?

Em sẽ giữ TF-IDF làm baseline và thử bổ sung **n-gram hoặc embedding-based representation** để cải thiện khả năng tìm kiếm theo ngữ nghĩa.

### 6. AI đã được sử dụng ở những phần nào và đóng góp cụ thể là gì?
- Giải thích và kiểm tra các hàm trong phần implementation/ experiments/...
- Hỗ trợ xây dựng và kiểm tra unit test.
- Giải thích các pipeline A, B, C (đặc biệt là C).
- Hỗ trợ kiểm tra lỗi code và điều chỉnh chương trình trong quá trình thực hiện.