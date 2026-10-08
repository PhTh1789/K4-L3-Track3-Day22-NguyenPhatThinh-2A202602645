---
language:
- vi
license: apache-2.0
base_model: unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit
library_name: peft
tags:
- dpo
- alignment
- rlhf
- lora
- trl
- unsloth
- vietnamese
datasets:
- saillab/alpaca-vietnamese-cleaned
- sailor2/sea-ultrafeedback-onpolicy
metrics:
- accuracy
- win_rate
pipeline_tag: text-generation
model_name: Qwen3-4B-Instruct-Vietnamese-SFT-DPO
---

# Qwen3-4B-Instruct (Vietnamese SFT + DPO Alignment) — Model Card

Thẻ mô tả mô hình (Model Card) cho mô hình căn chỉnh DPO tiếng Việt trong khuôn khổ Lab 22 (Track 3).

- **Học viên:** Nguyễn Phát Thịnh (MSSV: `2A202602645`)
- **Lớp / Khóa:** A20-K4 (Track 3), VinUniversity
- **Kiến trúc:** Qwen3-4B với LoRA rank 16, alpha 32
- **Thư viện:** Unsloth + TRL (DPOTrainer)

---

## 1. Thông tin mô hình & Dữ liệu

| Thuộc tính | Giá trị |
|---|---|
| **Mô hình nền (Base model)** | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` |
| **Dữ liệu SFT** | `saillab/alpaca-vietnamese-cleaned` (1.000 mẫu tiếng Việt, 1 epoch) |
| **Mô hình tham chiếu ($\pi_{ref}$)** | `models/sft-merged` (Mô hình SFT gộp trọng số) |
| **Dữ liệu sở thích (Preference)** | `sailor2/sea-ultrafeedback-onpolicy` (800 train / 100 held-out) |
| **Siêu tham số DPO** | $\beta = 0.1$, learning rate $5\times 10^{-6}$, cosine scheduler, 8-bit AdamW |

---

## 2. Kết quả thực nghiệm

### A. Động học Reward (NB3)
- **Loss khởi đầu:** `0.6911` (khớp sát lý thuyết $\ln 2 \approx 0.6931$).
- **Reward margin cuối trên train:** `+0.0899` (Chosen: `+0.3859`, Rejected: `+0.2960`).
- **Reward margin trên held-out:** `+0.0867` (Chosen: `+0.4020`, Rejected: `+0.3153`).
- **Độ chính xác reward trên held-out:** `66.0%`.
- **Chẩn đoán:** `INTENDED` (không bị Likelihood Displacement hay quá khớp).

### B. Đánh giá tự động qua Hội đồng Giám khảo (NB4)
- **Giám khảo:** Hội đồng RM (`Skywork-Reward-V2-Qwen3-4B` + `Skywork-Reward-V2-Llama-3.2-3B`).
- **Sanity Accuracy (Độ chính xác tiếng Việt):** `91.7%` (vượt ngưỡng 80%).
- **Win rate trên held-out:** `76.0%` (Khoảng tin cậy 95%: [66.0%, 86.0%]).
- **Win rate các cặp dài tương đồng:** `74.0%` (loại trừ hiện tượng hack độ dài).
- **Độ dài trung bình:** SFT `635` ký tự $\rightarrow$ DPO `627` ký tự (không lạm dụng độ dài).

---

## 3. Triển khai & Sử dụng (Inference)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Nạp SFT base model và adapter DPO
base_path = "models/sft-merged"
tokenizer = AutoTokenizer.from_pretrained(base_path)
model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, "adapters/dpo")

# Sinh phản hồi
messages = [{"role": "user", "content": "Tóm tắt 3 lợi ích chính của thuật toán DPO so với RLHF truyền thống."}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7)
print(tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```
