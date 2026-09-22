# Part H: Evaluation
- `1`: Relevant 
- `0`: Not Relevant

### Query 1: medical image classification

| Rank | Document ID | Similarity | Relevant |
|---:|---:|---:|---:|
| 1 | D18971 | 0.4000 | 0 |
| 2 | D8527 | 0.3500 | 0 |
| 3 | D19908 | 0.2534 | 0 |
| 4 | D17794 | 0.2405 | 0 |
| 5 | D12658 | 0.2331 | 0 |

Nhận xét:

Không có document nào trong Top-5 thực sự nói về medical image classification. D12658 có nội dung liên quan đến medical devices nhưng không nói về phân loại ảnh y tế. D17794 liên quan đến image metadata nhưng cũng không phải medical image classification.

---

### Query 2: transformer language model

| Rank | Document ID | Similarity | Relevant |
|---:|---:|---:|---:|
| 1 | D27936 | 0.4718 | 0 |
| 2 | D25428 | 0.2791 | 0 |
| 3 | D4075 | 0.2227 | 0 |
| 4 | D701 | 0.2161 | 0 |
| 5 | D13690 | 0.2136 | 0 |

Nhận xét:

Không có document nào trong Top-5 thực sự nói về transformer language model.

D27936 đứng đầu với similarity cao nhất nhưng nội dung nói về transformer/circuit board, không phải transformer trong NLP.

D25428 có nội dung về language trên iPhone nhưng không nói về transformer language model.

---

### Query 3: deep learning healthcare

| Rank | Document ID | Similarity | Relevant |
|---:|---:|---:|---:|
| 1 | D9252 | 0.2782 | 0 |
| 2 | D6123 | 0.2753 | 0 |
| 3 | D11119 | 0.2747 | 0 |
| 4 | D11979 | 0.2730 | 0 |
| 5 | D7564 | 0.2405 | 0 |

Nhận xét:

Các document đứng đầu chủ yếu nói về healthcare, nhưng không tập trung vào deep learning.

D9252 nói về healthcare policy và healthcare costs. D6123 nói về các chương trình đào tạo trong lĩnh vực healthcare. D11119 nói về Big Data trong healthcare. D11979 nói về đào tạo quản lý healthcare. D7564 lại chủ yếu nói về cách sử dụng từ "learning".

Do đó, similarity chủ yếu đến từ sự xuất hiện của các từ liên quan đến "healthcare" hoặc "learning", chứ chưa phản ánh đầy đủ ý nghĩa của cụm "deep learning healthcare".

---

### Query 4: natural language processing

| Rank | Document ID | Similarity | Relevant |
|---:|---:|---:|---:|
| 1 | D25428 | 0.3623 | 0 |
| 2 | D8705 | 0.3418 | 0 |
| 3 | D4075 | 0.2890 | 0 |
| 4 | D5699 | 0.2829 | 0 |
| 5 | D701 | 0.2805 | 0 |

Nhận xét:

Không có document nào trong Top-5 thực sự tập trung vào natural language processing.

D25428 nói về ngôn ngữ trên iPhone. D4075 và D701 nói về việc giảng dạy ngoại ngữ. D5699 có cụm "natural language" nhưng nội dung chính là về truth-value gaps trong ngôn ngữ tự nhiên. D8705 nói về food processing chứ không phải language processing.

---

### Query 5: machine learning classification

| Rank | Document ID | Similarity | Relevant |
|---:|---:|---:|---:|
| 1 | D18971 | 0.3938 | 0 |
| 2 | D8527 | 0.3446 | 0 |
| 3 | D2501 | 0.2528 | 1 |
| 4 | D21822 | 0.2415 | 0 |
| 5 | D19280 | 0.2374 | 0 |

Nhận xét:

D2501 được đánh dấu relevant vì document nói rõ về việc sử dụng machine learning để dự đoán nguy cơ PTSD và hỗ trợ chẩn đoán. Đây là document gần với nội dung machine learning và classification/prediction hơn các document còn lại.

D18971 và D8527 có similarity cao nhưng chủ yếu liên quan đến từ "classification", trong khi không nói về machine learning.

---

### Query 6: image processing

| Rank | Document ID | Similarity | Relevant |
|---:|---:|---:|---:|
| 1 | D8705 | 0.4068 | 0 |
| 2 | D19908 | 0.3408 | 0 |
| 3 | D17794 | 0.3234 | 0 |
| 4 | D15682 | 0.3133 | 0 |
| 5 | D8320 | 0.3027 | 1 |

Nhận xét:

D8320 được đánh dấu relevant vì document nói trực tiếp về Image processing technology, bao gồm các ứng dụng như identifying, measuring objects và telemedicine.

D8705 đứng đầu nhưng nội dung lại nói về food processing, không phải image processing. Đây là một ví dụ rõ ràng về việc một term chung như "processing" có thể tạo similarity cao dù document không liên quan đến chủ đề cần tìm.

---

## Tổng hợp relevance labels

| Query | Relevant documents trong Top-5 |
|---|---|
| medical image classification | Không có |
| transformer language model | Không có |
| deep learning healthcare | Không có |
| natural language processing | Không có |
| machine learning classification | D2501 |
| image processing | D8320 |

---

## Precision@5

Công thức:

\[
P@5 = \frac{\#\text{relevant documents retrieved in Top-5}}{5}
\]

Kết quả:

| Query | Relevant trong Top-5 | P@5 |
|---|---:|---:|
| medical image classification | 0 | 0.00 |
| transformer language model | 0 | 0.00 |
| deep learning healthcare | 0 | 0.00 |
| natural language processing | 0 | 0.00 |
| machine learning classification | 1 | 0.20 |
| image processing | 1 | 0.20 |

---

## Recall@5

Công thức:

\[
R@5 =
\frac{\#\text{relevant documents retrieved in Top-5}}
{\#\text{relevant documents}}
\]

Lưu ý: Đây chỉ là cho top 5 chứ không phải toàn bộ 30K docs.



# Part I: Error Analysis

## I.1. Query tốt 1: `machine learning classification`

| Rank | Document | Similarity | Relevant |
|---:|---:|---:|:---:|
| 1 | D18971 | 0.3938 | No |
| 2 | D8527 | 0.3446 | No |
| 3 | D2501 | 0.2528 | Yes |
| 4 | D21822 | 0.2415 | No |
| 5 | D19280 | 0.2374 | No |

- **Vì sao D18971 đứng đầu?** Có nhiều từ liên quan đến `classification`, trùng với query.
- **Từ đóng góp:** `classification`.
- **Lexical overlap:** Có, đặc biệt là `classification`.
- **Relevant bị xếp thấp:** D2501 relevant nhưng chỉ ở rank 3.
- **Nguyên nhân:** TF-IDF dựa nhiều vào lexical matching, chưa hiểu đầy đủ ngữ nghĩa của `machine learning classification` và `environmental classification`.

---

## I.2. Query tốt 2: `image processing`

| Rank | Document | Similarity | Relevant |
|---:|---:|---:|:---:|
| 1 | D8705 | 0.4068 | No |
| 2 | D19908 | 0.3408 | No |
| 3 | D17794 | 0.3234 | No |
| 4 | D15682 | 0.3133 | No |
| 5 | D8320 | 0.3027 | Yes |

- **Vì sao D8705 đứng đầu?** Document chứa nhiều từ liên quan đến `processing`.
- **Từ đóng góp:** `processing`.
- **Lexical overlap:** Có.
- **Relevant bị xếp thấp:** D8320 relevant nhưng ở rank 5.
- **Nguyên nhân:** TF-IDF không phân biệt được `image processing` với `food processing`.

---

## I.3. Query kém 1: `transformer language model`

| Rank | Document | Similarity | Relevant |
|---:|---:|---:|:---:|
| 1 | D27936 | 0.4718 | No |
| 2 | D25428 | 0.2791 | No |
| 3 | D4075 | 0.2227 | No |
| 4 | D701 | 0.2161 | No |
| 5 | D13690 | 0.2136 | No |

- **Vì sao D27936 đứng đầu?** Có từ `transformer` trùng trực tiếp với query.
- **Từ đóng góp:** `transformer`, một phần `model`.
- **Lexical overlap:** Có.
- **Relevant bị bỏ sót:** Không có relevant document trong Top-5.
- **Nguyên nhân:** `transformer` trong document nói về thiết bị điện, trong khi query nói về Transformer trong NLP. TF-IDF không hiểu khác biệt về ngữ nghĩa.

---

## I.4. Query kém 2: `natural language processing`

| Rank | Document | Similarity | Relevant |
|---:|---:|---:|:---:|
| 1 | D25428 | 0.3623 | No |
| 2 | D8705 | 0.3418 | No |
| 3 | D4075 | 0.2890 | No |
| 4 | D5699 | 0.2829 | No |
| 5 | D701 | 0.2805 | No |

- **Vì sao D25428 đứng đầu?** Có nhiều từ liên quan đến `language`.
- **Từ đóng góp:** `language`, `processing`.
- **Lexical overlap:** Có.
- **Relevant bị bỏ sót:** Không có relevant document trong Top-5.
- **Nguyên nhân:** Các từ `language` và `processing` xuất hiện trong nhiều ngữ cảnh khác nhau, nhưng TF-IDF không hiểu toàn bộ cụm `natural language processing`.

---

## I.5. Tổng hợp lỗi

Các lỗi chính đều liên quan đến **lexical matching**:

| Query | Failure chính |
|---|---|
| `machine learning classification` | `classification` |
| `image processing` | `processing` |
| `transformer language model` | `transformer` khác nghĩa |
| `natural language processing` | `language`, `processing` khác ngữ cảnh |

TF-IDF có thể cho similarity cao khi query và document có các từ giống nhau, nhưng không hiểu đầy đủ ý nghĩa và ngữ cảnh của các từ đó.

---

## I.6. Failure case quan trọng nhất

### Query: `transformer language model`

D27936 đứng **rank 1** với similarity **0.4718**, nhưng document nói về `transformer/circuit board`, không phải Transformer trong NLP.

Đây là một **lexical matching failure**:

```text
Query:    transformer language model
Document: transformer / circuit board
```


# Part J
Có thể sử dụng word embeddings/contextual embeddings, trong đó các từ xuất hiện trong những ngữ cảnh tương tự được biểu diễn bằng các vector gần nhau. Nhờ đó, hệ thống có thể đo similarity về nghĩa thay vì chỉ dựa vào sự trùng khớp từ.

# Learning Check
Question 1: Vì mỗi document chỉ chứa một phần nhỏ vocabulary nên phần lớn giá trị TF-IDF bằng 0, tạo ra vector thưa.

Question 2: Vì term xuất hiện trong nhiều documents nên ít có khả năng phân biệt document này với document khác nên IDF thấp.

Question 3: IDF chỉ thể hiện độ hiếm của term. Nếu term đó không xuất hiện hoặc có TF thấp trong document thì TF-IDF vẫn thấp.

Question 4: Vì cosine similarity đo góc giữa các vector, ít bị ảnh hưởng bởi độ dài document và phù hợp để so sánh các vector TF-IDF.

Question 5: Vì preprocessing quyết định các term được giữ lại và cách chúng được biểu diễn, từ đó làm thay đổi vocabulary, TF-IDF và similarity.

Question 6: Query transformer language model trả về document nói về transformer/circuit board ở rank 1 do có từ transformer trùng với query, mặc dù khác nghĩa.

Question 7: Điều này gợi ý cần representation có khả năng biểu diễn ngữ nghĩa, chẳng hạn word embeddings hoặc contextual embeddings.

