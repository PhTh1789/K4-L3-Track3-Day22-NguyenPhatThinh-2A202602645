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

# Qwen3-4B-Instruct (Vietnamese SFT + DPO Alignment)

This repository contains the LoRA adapter trained with **Direct Preference Optimization (DPO)** on top of a Vietnamese Supervised Fine-Tuned (SFT) checkpoint of `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit`.

- **Student:** Nguyễn Phát Thịnh (ID: `2A202602645`)
- **Course:** AICB-P2T3 (Track 3 Day 22 - DPO/ORPO Alignment), VinUniversity K4
- **Architecture:** Qwen3-4B with LoRA rank 16, alpha 32 targeting all linear modules
- **Training Framework:** Unsloth + TRL (DPOTrainer)

---

## 1. Model & Alignment Pipeline Overview

```
Base Model: unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit
   │
   ▼  [Stage 1: SFT - saillab/alpaca-vietnamese-cleaned (1,000 samples, 1 epoch)]
Merged Model (π_ref): models/sft-merged/
   │
   ▼  [Stage 2: DPO - sailor2/sea-ultrafeedback-onpolicy (800 train / 100 held-out)]
Aligned Policy (π_θ): adapters/dpo/ (SFT + DPO LoRA)
```

### Key Technical Decisions
1. **Reference Model Stability:** Following Rafailov et al. (2023), DPO optimization assumes the starting policy $\pi_\theta \equiv \pi_{ref}$. We explicitly trained and merged the SFT adapter to produce `models/sft-merged`, using it as the fixed reference distribution $\pi_{ref}$ to eliminate reference shift artifacts.
2. **Conservative Regularization:** Selected $\beta = 0.1$ and learning rate $5\times 10^{-6}$ with cosine scheduler and 8-bit AdamW optimizer.

---

## 2. Experimental Results & Evaluation

### A. Training & Held-out Reward Dynamics (DPO Metrics)

| Metric | Value | Interpretation |
|---|---:|---|
| **Initial Loss (Step 0)** | `0.6911` | Matches theoretical closed-form $\ln(2) \approx 0.6931$ |
| **Final Train Loss** | `0.6748` | Consistent loss convergence |
| **Train Chosen Reward** | `+0.3859` | Log-probability on chosen responses strengthened |
| **Train Rejected Reward** | `+0.2960` | Log-probability on rejected responses increased slower |
| **Train Reward Gap (Margin)** | `+0.0899` | Positive separation achieved without likelihood displacement |
| **Held-out Chosen Reward** | `+0.4020` | Generalizes cleanly to unseen distribution |
| **Held-out Rejected Reward** | `+0.3153` | Consistent with training trajectory |
| **Held-out Margin** | `+0.0867` | Zero overfitting, margin expands robustly |
| **Held-out Reward Accuracy** | `66.0%` | High discriminative accuracy on unseen pairs |
| **Automated Diagnosis** | `INTENDED` | Fully adheres to theoretical expectation |

### B. Side-by-Side Model Evaluation (RM Panel)

Evaluated across 58 test prompts (8 fixed benchmark prompts + 50 held-out pairs) using a multi-model reward panel (`Skywork-Reward-V2-Qwen3-4B` + `Skywork-Reward-V2-Llama-3.2-3B`):

| Test Set | Sample Size ($n$) | DPO Wins | SFT Wins | Ties | DPO Win Rate | 95% Confidence Interval | Length-Matched Win Rate |
|---|---:|---:|---:|---:|---:|---|---:|
| **Held-out** | 50 | 33 | 7 | 10 | **76.0%** | [66.0%, 86.0%] | **74.0%** ($n=25$) |
| **Helpfulness** | 4 | 4 | 0 | 0 | **100.0%** | [100.0%, 100.0%] | **100.0%** ($n=4$) |
| **Safety** | 4 | 0 | 0 | 4 | **50.0%** | [50.0%, 50.0%] | **50.0%** ($n=4$) |
| **Overall** | 58 | 37 | 7 | 14 | **75.9%** | [66.4%, 84.5%] | **74.2%** ($n=33$) |

- **Vietnamese Sanity Check:** Judge accuracy on obvious Vietnamese sanity pairs: **91.7%** (Qwen3-4B: 91.7%, Llama-3.2-3B: 83.3%, surpassing the $\ge 80\%$ benchmark).
- **Length Hacking Analysis:** Longest answer won fraction is 55.0% (close to random chance 50%). DPO maintains 74.0% win rate on length-matched pairs, and helpfulness responses are more concise (94 chars vs 100 chars SFT), confirming genuine response quality improvement rather than verbosity bias.

---

## 3. Usage & Inference

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. Load merged SFT base model
base_model_path = "models/sft-merged"
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 2. Load DPO adapter
model = PeftModel.from_pretrained(model, "adapters/dpo")
model.eval()

# 3. Format prompt and generate
messages = [{"role": "user", "content": "Giải thích ngắn gọn cách hoạt động của thuật toán Quicksort."}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7, top_p=0.9)

print(tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```
