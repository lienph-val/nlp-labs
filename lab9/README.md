Các file .py chạy trên local 
python baseline_ner.py                                   
python baseline_ner.py --max_train_samples 2000 --epochs 1   
python baseline_ner.py --strategy B                     

python information_extraction.py 

Các file .ipynb chạy trên Kaggle. Cần import dữ liệu vào trong input dạng file parquet với 3 file data: train/validation/test
Khi chạy hết file ner_finetuning.ipynb ra được folder ner_model thì cập nhật về local để chạy file information_extraction.py

```
AI-assisted code:
What was generated: Boilerplate code, visualization snippets, and code examples for tokenizer documentation.
What was modified: AI-suggested code was reviewed and adapted to fit the project requirements. Debugging suggestions and runtime error explanations were used to fix implementation issues.
What was verified: Code execution, tokenizer behavior, and visualization outputs were tested and checked manually.
```