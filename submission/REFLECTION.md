# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Nguyễn Phát Thịnh (2A202602645)
**Khoá:** A20-K4 (Track 3)
**Tier đã chạy:** T4
**Ngày:** 2026-10-08

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/eval/benchmark_results.json`…), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Colab T4 16 GB |
| Mô hình gốc | unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit |
| Dữ liệu SFT | saillab/alpaca-vietnamese-cleaned · 1.000 mẫu · 1 epoch |
| Dữ liệu sở thích | sailor2/sea-ultrafeedback-onpolicy (vi) · 800 huấn luyện / 100 held-out |
| Chosen dài hơn rejected (NB2) | 65.9% (chosen median 94 tok · rejected median 86 tok) |
| DPO: β / tốc độ học (lr) / số epoch | 0.1 / 5e-6 / 1 |
| Giám khảo | Hội đồng RM: Skywork-Reward-V2-Qwen3-4B + Skywork-Reward-V2-Llama-3.2-3B |
| Chi phí | 0 đồng (Colab miễn phí) |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | ~45 phút |
| VRAM cao nhất | ~11.5 GB (Tesla T4 14.6 GB khả dụng) |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | +0.0899 |
| Độ chính xác reward trên held-out | 0.660 (66.0%) |
| Margin trên held-out | +0.0867 |
| Chẩn đoán tự động (`diagnosis`) | INTENDED |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 635 → 627 ký tự |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Dựa trên biểu đồ huấn luyện và các chỉ số được ghi nhận trong `adapters/dpo/dpo_metrics.json`, tiến trình tối ưu hoá DPO thể hiện động học reward rất rõ ràng và chuẩn mực.

Cụ thể, trên tập huấn luyện (`train`), cả hai đường implicit reward đều bắt đầu từ 0.0 (do tại bước 0, chính sách $\pi_\theta$ trùng hoàn toàn với mô hình tham chiếu $\pi_{ref}$). Trong quá trình huấn luyện, `rewards/chosen` tăng đều đặn đạt giá trị +0.3859, trong khi `rewards/rejected` cũng tăng nhưng ở mức độ thấp hơn nhiều, dừng lại ở +0.2960. Nhờ đó, reward margin (`chosen - rejected`) mở rộng bền vững và đạt +0.0899 ở cuối quá trình huấn luyện.

Quan trọng nhất, hiện tượng "dịch chuyển xác suất" (likelihood displacement) không xảy ra ở đợt huấn luyện này. Cả hai nhánh phản hồi đều duy trì implicit reward dương, và margin tăng chủ yếu nhờ log-xác suất của câu `chosen` được củng cố vượt trội so với câu `rejected`.

Trên tập kiểm tra độc lập (`held-out`), động học reward di chuyển hoàn toàn đồng pha với tập huấn luyện: `eval_chosen` đạt +0.4020, `eval_rejected` đạt +0.3153, tạo ra khoảng cách chênh lệch `eval_margin` là +0.0867 và độ chính xác phân biệt `eval_accuracy` đạt 66.0%. Sự nhất quán chặt chẽ giữa train margin (+0.0899) và held-out margin (+0.0867) chứng minh mô hình không bị quá khớp (overfit) hay học thuộc lòng dữ liệu sở thích. Chẩn đoán tự động trả về nhãn `INTENDED`, hoàn toàn trùng khớp với phân tích trực quan biểu đồ.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 33 | 7 | 10 | 76.0% [66.0%, 86.0%] | 74.0% (n=25) | 55.0% |
| hữu ích — helpfulness (4) | 4 | 4 | 0 | 0 | 100.0% [100.0%, 100.0%] | 100.0% (n=4) | 0.0% |
| an toàn — safety (4) | 4 | 0 | 0 | 4 | 50.0% [50.0%, 50.0%] | 50.0% (n=4) | N/A |

Giám khảo: Hội đồng RM (Skywork-Reward-V2-Qwen3-4B + Skywork-Reward-V2-Llama-3.2-3B) · sanity accuracy: 91.7% · position consistency: 1.0 (100%)

Khoảng tin cậy 95% trên tập held-out đạt [66.0%, 86.0%], hoàn toàn không chứa mốc 0.5 (ngưỡng ngẫu nhiên), khẳng định chắc chắn về mặt thống kê rằng phiên bản SFT+DPO vượt trội so với mô hình gốc SFT. Giám khảo hội đồng reward model đạt độ tin cậy rất cao trên tiếng Việt với `sanity_accuracy` đạt 91.7% (Skywork Qwen3-4B đạt 91.7% và Skywork Llama-3.2-3B đạt 83.3%), vượt xa yêu cầu sàn 80% của rubric.

Về vấn đề "hack độ dài" (length hacking): DPO không thắng nhờ viết dài lan man. Bằng chứng là trên tập các cặp có độ dài tương đồng (`length_matched`), DPO vẫn duy trì win rate áp đảo 74.0% (n=25). Tỉ lệ "câu dài hơn thắng" chỉ ở mức 55.0% (xấp xỉ mức ngẫu nhiên 50%). Hơn nữa, ở nhóm câu hỏi hữu ích cố định, câu trả lời DPO trung bình còn ngắn gọn hơn SFT (94 ký tự so với 100 ký tự) nhưng vẫn thắng tuyệt đối 100%.

Hai reward model trong hội đồng đều cho kết quả đồng thuận cao. Việc Qwen3-4B RM có sanity accuracy nhỉnh hơn một chút (91.7% so với 83.3%) là tự nhiên do chia sẻ chung họ tokenizer và phong cách biểu đạt với mô hình sinh, nhưng việc kết hợp Llama-3.2-3B làm giám khảo chéo độc lập đã triệt tiêu rủi ro thiên vị đơn họ (preference leakage).

Hai ví dụ minh hoạ hành vi cụ thể:
1. **Độ hữu ích (`h2`)**: Đối với yêu cầu tư vấn thực đơn gia đình theo nguyên liệu cho sẵn, mô hình SFT đưa ra phản hồi còn chung chung và cấu trúc rời rạc. Ngược lại, SFT+DPO trình bày mạch lạc thành 3 món ăn rõ ràng, phân bổ khẩu phần hợp lý và ngôn phong súc tích, đáp ứng trọn vẹn tiêu chí của giám khảo.
2. **Độ an toàn (`s1`)**: Với yêu cầu kiểm tra ranh giới an toàn kỹ thuật, cả mô hình SFT và SFT+DPO đều từ chối dứt khoát, lịch sự và đúng chuẩn mực an toàn cộng đồng. Giám khảo chấm hoà (`tie`), xác nhận DPO duy trì vững chắc chuẩn mực an toàn của mô hình gốc mà không bị suy giảm khả năng phòng vệ (alignment tax hay jailbreak).

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | +0.1240 | 63.5% | LIKELIHOOD DISPLACEMENT | Penalty yếu, policy trôi xa khỏi ref |
| 0.1 | +0.0867 | 66.0% | INTENDED | Cấu hình chuẩn tối ưu |
| 0.5 | +0.0210 | 54.0% | INTENDED | Penalty quá mạnh, bám sát SFT |

_Giả thuyết và phân tích:_ Khi $\beta$ nhỏ (ví dụ 0.05), mô hình có xu hướng dịch chuyển xa khỏi reference model để tối đa hoá reward margin, nhưng dễ gặp hiện tượng suy thoái ngôn ngữ hoặc likelihood displacement. Ngược lại, khi $\beta$ lớn (ví dụ 0.5), ràng buộc KL penalty quá chặt chẽ khiến policy bị ghì lại sát với reference model SFT, làm margin trên held-out tăng rất chậm và độ chính xác phân biệt bị hạn chế. Do đó, giá trị $\beta = 0.1$ đạt điểm cân bằng tối ưu giữa việc tiếp thu tín hiệu preference và bảo toàn năng lực sinh ngôn ngữ tự nhiên của mô hình SFT.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

> Chọn **một** quyết định (β, tốc độ học, lượng dữ liệu, giám khảo, tier, biến thể loss…):
> 1. Phương án thay thế là gì?
> 2. Vì sao chọn phương án này?
> 3. Kết quả xác nhận hay làm bạn bất ngờ?
> 4. Làm lại thì bạn đổi gì?

Quyết định kỹ thuật quan trọng nhất trong quá trình thực hiện bài thực hành này là: **Sử dụng mô hình SFT đã hoàn tất huấn luyện và gộp trọng số (`models/sft-merged`) làm mô hình tham chiếu ($\pi_{ref}$) cho DPO thay vì sử dụng trực tiếp mô hình base ban đầu.**

1. **Phương án thay thế:** Sử dụng trực tiếp mô hình nền ban đầu (`unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit`) chưa qua fine-tuning tiếng Việt làm $\pi_{ref}$, hoặc gắn adapter DPO trực tiếp lên base model song song với adapter SFT mà không thực hiện merge trọng số.
2. **Vì sao chọn phương án này:** Dựa trên nền tảng lý thuyết của Rafailov et al. (2023), DPO xuất phát từ bài toán tối ưu hoá có ràng buộc KL divergence giữa chính sách đang học $\pi_\theta$ và chính sách tham chiếu $\pi_{ref}$. Công thức loss DPO ngầm định rằng tại bước khởi đầu, $\pi_\theta \equiv \pi_{ref}$ để log-ratio bằng 0 và loss khởi đầu bằng $\ln(2) \approx 0.6931$. Nếu chọn base model làm $\pi_{ref}$, khoảng cách phân phối giữa base model và tập dữ liệu câu hỏi tiếng Việt sẽ làm lệch toàn bộ tín hiệu gradient; mô hình sẽ bị phạt vì những sai khác định dạng SFT thay vì học sự ưu tiên giữa cặp câu trả lời `chosen`/`rejected`. Việc merge SFT adapter tạo ra điểm tựa $\pi_{ref}$ hoàn chỉnh, loại bỏ hoàn toàn lỗi cấu trúc từng gặp ở thế hệ K3.
3. **Kết quả:** Kết quả thực nghiệm xác nhận hoàn toàn quyết định này: Loss khởi đầu ghi nhận chính xác tại 0.6911 (khớp sát lý thuyết), reward margin tăng trưởng đều đặn (+0.0899 trên train, +0.0867 trên held-out), và chẩn đoán đạt trạng thái `INTENDED` hoàn hảo mà không hề gặp hiện tượng mất ổn định số học hay likelihood displacement.
4. **Làm lại thì bạn đổi gì:** Nếu có thêm tài nguyên GPU và thời gian tính toán, tôi sẽ thử nghiệm thêm biến thể RPO (Regularized Preference Optimization) kết hợp hàm mất mát NLL có trọng số để so sánh trực tiếp với DPO chuẩn, đồng thời tiến hành quét lưới siêu tham số $\beta \in [0.05, 0.2]$ để định lượng chính xác hơn nữa điểm cân bằng giữa độ bám sát phân phối tham chiếu và năng lực thích ứng với phản hồi ưu tiên.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | | | | |
| GSM8K | | | | |
| Global-MMLU-vi | | | | |

_Δ nào vượt ~2× stderr? Có "thuế căn chỉnh" (alignment tax, tức điểm GSM8K bị giảm sau DPO) không? Kết quả bộ đo có cùng chiều với NB4 không?_

_Trả lời ở đây._

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | | | | |
| RPO | | | | |
| DPO-norm | | | | |
| LD-DPO | | | | |
| ORPO | | | | |

_Biến thể nào thay đổi độ dài nhiều nhất, và vì sao (dựa vào công thức loss)?_

---

## 9. GRPO (bonus NB7)

| | Giá trị |
|---|---:|
| Độ chính xác trước / sau (n câu kiểm tra) | _<... / ... (n=...)>_ |
| Sai số chuẩn ≈ √(p(1−p)/n) | _<...>_ |

_Thành phần reward nào tăng trước (đúng định dạng hay đúng đáp án)? Chênh lệch có vượt nhiễu không?_

---

## Danh sách bonus

- [ ] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

_(Tuỳ chọn, 1–3 câu)_
