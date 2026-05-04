# Kết Quả Thực Nghiệm Pipeline IT CV-JD Matching

> Tổng hợp toàn bộ kết quả đo lường của 2 thành phần cốt lõi trong pipeline:
> **(1) Mô hình NER** và **(2) Bước Mapping O\*NET.**
>
> Dữ liệu: 19.853 tài liệu (CV + JD) từ Hugging Face `lang-uk/recruitment-dataset-*`.
> Mô hình: `roberta-base`, Kiến trúc: Span-based NER.

---

## Phần 1: Kết Quả Mô Hình NER (Span-based RoBERTa)

Thực nghiệm so sánh 2 phiên bản dữ liệu huấn luyện để cải thiện Precision của mô hình.

| Chỉ số                     | v5 (Baseline)                         | v6 (Improved)                         | Chênh lệch         |
| :---                       | :---:                                 | :---:                                 | :---:              |
| **Phiên bản dữ liệu**      | `cleaned_v5` (negative_multiplier=10) | `cleaned_v6` (negative_multiplier=5)  |                    |
| **Cải tiến so với v5**     | -                                     | Trim noisy prefixes, giảm negative sampling | -           |
| **Best Epoch**             | 7 / 8                                 | 7 / 8                                 | -                  |
| **Dev Precision**          | 0.4934 (49.3%)                        | **0.4984 (49.8%)**                    | +0.5%              |
| **Dev Recall**             | 0.8695 (87.0%)                        | 0.8669 (86.7%)                        | -0.3%              |
| **Dev F1 Score (Best)**    | 0.6295 (62.9%)                        | **0.6330 (63.3%)**                    | **+0.35%**         |
| **Test Precision**         | 0.4864 (48.6%)                        | **0.4869 (48.7%)**                    | +0.05%             |
| **Test Recall**            | 0.8741 (87.4%)                        | 0.8728 (87.3%)                        | -0.1%              |
| **Test F1 Score (Final)**  | 0.6250 (62.5%)                        | **0.6251 (62.5%)**                    | +0.01%             |
| **Checkpoint path**        | `artifacts/span_ner_roberta_base_openai_cleaned_v5_hardneg/span_ner.pt` | `artifacts/span_ner_roberta_base_openai_cleaned_v6/span_ner.pt` | |

### Ghi chú NER

- Việc làm sạch các **tiền tố nhiễu** (`strong`, `experience with`, `knowledge of`, v.v.) và **giảm Negative Sampling Multiplier** xuống còn 5 (từ 10) giúp Precision nhích lên nhẹ mà không đánh đổi Recall đáng kể.
- Cả hai phiên bản đều cho thấy mô hình có khả năng **Recall rất cao (~87%)** — phù hợp cho bài toán trích xuất (không muốn bỏ sót thực thể), nhưng **Precision còn thấp (~49%)** do nhiễu từ nhãn LLM (Noisy Weak Labels). Đây là hướng cải thiện tiếp theo.

---

## Phần 2: Kết Quả Mapping O*NET

Thực nghiệm so sánh 3 phiên bản của bước Mapping để cải thiện tỉ lệ thực thể được ánh xạ sang O*NET.

### Tổng quan theo phiên bản

| Chỉ số                         | v5-Lexical (Baseline) | v5-Semantic              | v6-Semantic (Best)       |
| :---                           | :---:                 | :---:                    | :---:                    |
| **Dữ liệu đầu vào**            | `cleaned_v5`          | `cleaned_v5`             | `cleaned_v6`             |
| **Phương pháp Matching**       | Lexical only          | Lexical + Semantic (MiniLM-L6-v2) | Lexical + Semantic (MiniLM-L6-v2) |
| **Tổng thực thể được hỗ trợ**  | 221.928               | 221.928                  | 221.859                  |
| **Tổng thực thể được map**     | 148.509               | 164.264                  | **164.507**              |
| **Mapped Rate (Tổng)**         | 66.92%                | 74.02%                   | **74.15%**               |
| **File kết quả**               | `stage2_onet_mapped_cleaned_v5.jsonl` | `stage2_onet_mapped_semantic_full.jsonl` | `stage2_onet_mapped_semantic_v6_full.jsonl` |

### Chi tiết theo nhãn (v5-Lexical vs v6-Semantic)

| Nhãn (Label)      | v5-Lexical (mapped/total) | v6-Semantic (mapped/total) |
| :---              | :---:                     | :---:                      |
| **JOB_ROLE**      | 23.524 / 24.438           | 24.106 / 24.438            |
| **TECHNOLOGY**    | 115.763 / 154.354         | 120.519 / 154.354          |
| **WORK_ACTIVITY** | 2.585 / 21.476            | 10.811 / 21.420            |
| **SKILL**         | 4.413 / 17.890            | 6.237 / 17.878             |
| **PROJECT_TYPE**  | 2.224 / 3.770             | 2.834 / 3.769              |
| **INDUSTRY**      | 0 / 5.434                 | 0 / 5.433                  |
| **DEGREE**        | 0 / 1.172                 | 0 / 1.157                  |
| **CERTIFICATION** | 0 / 629                   | 0 / 626                    |

### Ghi chú Mapping

- **Cải thiện lớn nhất:** Nhờ Semantic Embedding, nhãn `WORK_ACTIVITY` tăng từ **12% lên 50.5%** (tăng hơn **4 lần**). Đây là nhãn khó nhất vì LLM thường trích xuất các cụm từ dài, mang ngữ nghĩa phức tạp mà lexical matching không bắt được.
- **Nhãn SKILL:** Tăng từ 24.7% lên 34.9%, vẫn còn tiềm năng cải thiện thêm nếu tích hợp thêm bộ chuẩn **ESCO**.
- **Thông số tốt nhất:** `min_score=0.35`, `top_k=5`, `embedding_model=sentence-transformers/all-MiniLM-L6-v2`, `device=cuda`.

---

## Hướng cải thiện tiếp theo (Future Work)

1. **Nâng cấp Backbone NER:** Thay `roberta-base` bằng `microsoft/deberta-v3-base` hoặc `jjzha/jobbert-base-cased` để tăng Precision lên đáng kể mà không cần thêm dữ liệu.
2. **Tích hợp bộ chuẩn ESCO:** Bổ sung kho từ điển kỹ năng của ESCO (EU) vào bước Mapping để tăng tỉ lệ map của nhãn `SKILL` và `TECHNOLOGY`.
3. **Fine-tune Embedding Model:** Dùng Contrastive Learning trên tập dữ liệu cặp (CV term ↔ O\*NET descriptor) để mô hình Embedding hiểu sâu hơn về ngôn ngữ IT tuyển dụng.
4. **Hard-Constraint Scoring (Stage 3):** Viết thuật toán chấm điểm kết hợp:
   - Soft-matching bằng Cosine Similarity (thay vì exact-matching).
   - Trọng số kỹ năng theo tần suất xuất hiện trong JD (TF-weighting).
   - Suy luận kỹ năng ngầm từ O\*NET (Implicit Skill Inference).
