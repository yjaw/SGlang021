# SGLang 推論 30 天學習計畫

Oct 11, 2026 · @j

## 目標與前置需求

30 天後，你能用 SGLang 部署並調校一個 LLM 推論服務，說清楚它為什麼快，並交出 3 個專案、3 個 PR、1 篇技術文章，直接用來申請 RadixArk 推論相關職缺。每天約 3–4 小時：1–1.5 小時學習，2–2.5 小時實作。每個階段最後半天留作緩衝，用來補進度或複習。

**前置能力**

- Python 熟練，看得懂 PyTorch 基本程式
- 了解 Transformer 結構（attention、KV cache 的概念即可）
- 會用 Linux、Docker、Git

**硬體**

- 最低：1 張 24GB GPU（RTX 3090/4090、L4、A10G），可跑 7B–8B 模型
- 第三階段的多卡實驗（TP/DP）需 2 張 GPU，可用雲端按小時租（RunPod、Lambda、Vast.ai），預算約 US$30–60
- 沒有 GPU：前 5 天可用 Colab；之後強烈建議租卡

**每天的固定產出**：在 GitHub repo 的 `notes/dayXX.md` 寫學習筆記，數據放 `benchmarks/`。這個 repo 本身就是履歷上的證據。

## 總覽

四個階段由「會用」到「懂原理」到「會調校」，最後產出可展示的專案。

| 階段 | 天數 | 主題 | 階段產出 |
| --- | --- | --- | --- |
| 1 | Day 1–6 | 推論基礎、從原始碼建置 SGLang | 專案 A：基準測試報告 + 第 1 個 PR（文件或測試） |
| 2 | Day 7–14 | 讀原始碼、RadixAttention、排程、記憶體管理、除錯 | 專案 B：重現並修復一個真實 issue（第 2 個 PR） |
| 3 | Day 15–23 | 推論效能：kernel、profiling、量化、推測解碼、平行、PD 分離 | Triton kernel、profiling 報告、各優化手段的前後對比數據 |
| 4 | Day 24–30 | 深度貢獻、分享與投遞 | 專案 C：有分量的 PR + deep-dive 文章 + 履歷 |

## 職缺要求對照（RadixArk AI Infra Residency）

這個職缺由 SGLang 和 Miles 的開發團隊招募，所以**合併進 SGLang 的 PR 是最強的證據**：面試官就是 reviewer。計畫因此把開源貢獻從最後一天提前到每個階段。

| 職缺要求 | 怎麼證明 | 對應天數 |
| --- | --- | --- |
| 系統程式、ML systems 基礎 | 讀懂 SGLang 架構並寫成 deep-dive | Day 7–12、27 |
| 快速學習、除錯複雜問題 | 重現並修復真實 issue，寫除錯過程 | Day 13–14、24–26 |
| CUDA / 分散式（加分） | Triton kernel、TP/DP 實驗 | Day 15–16、21 |
| 跨層除錯：model → runtime → kernel | Triton kernel + profiling + 推論優化對比數據 | Day 15–23 |
| 貢獻 SGLang 開源 | 3 個 PR（至少 1 個合併） | Day 6、14、24–26 |
| 撰寫文件與指南、技術分享 | 技術文章 + 10 分鐘簡報 | Day 27–28 |

學歷要求（CS 相關學士或碩士）無法靠 30 天補；如果你不是相關科系，PR 和文章的分量就更重要。

## 第一階段 Day 1–6：推論基礎與開發環境

這一階段結束時，你能從原始碼建置 SGLang、量出 TTFT 與吞吐量，並送出第一個 PR。

- [ ] **Day 1｜LLM 推論基礎**
  - 學習：prefill vs decode、KV cache、memory-bound vs compute-bound、TTFT / TPOT / throughput 的定義
  - 實作：用 HuggingFace `generate()` 跑 Qwen2.5-7B 或 Llama-3.1-8B，記錄延遲與顯存，當作 baseline
- [ ] **Day 2｜安裝與第一個服務**
  - 學習：讀 SGLang 官方文件 Quick Start、SGLang 論文摘要與架構圖
  - 實作：用 Docker 映像 `lmsysorg/sglang` 啟動伺服器（預設 port 30000），用 OpenAI SDK 呼叫 `/v1/chat/completions`，試串流輸出
- [ ] **Day 3｜從原始碼建置 + 加入社群**
  - 學習：讀 SGLang 的 contribution guide、CI 流程；了解 `--mem-fraction-static`、`--chunked-prefill-size` 等關鍵參數
  - 實作：fork repo、`pip install -e "python"` 開發模式安裝、跑單元測試；加入 SGLang Slack，瀏覽 `good first issue` 選定一個
- [ ] **Day 4｜基準測試工具**
  - 學習：`sglang.bench_serving` 與 `bench_one_batch` 的用法、request rate 與並發的關係
  - 實作：在不同 request rate 下量 TTFT、TPOT、throughput，畫出延遲–吞吐曲線
- [ ] **Day 5｜對照 vLLM，完成專案 A**
  - 學習：Continuous batching、PagedAttention 的概念，以及它和 RadixAttention 的取捨
  - 實作：同模型、同 GPU、同資料集下比較 SGLang 與 vLLM，聚焦多輪對話與共享前綴情境
  - **專案 A 產出**：README + 圖表 + 可重現腳本，結論寫「在哪種負載下誰較好、為什麼」
- [ ] **Day 6｜第 1 個 PR + 緩衝**
  - 實作：送出第 1 個 PR，修文件、補測試或補範例都可以，目標是熟悉 CI 與 review 流程
  - 緩衝：補完前 5 天沒做完的部分，整理 `notes/`

## 第二階段 Day 7–14：讀原始碼與除錯

這一階段的重點是讀原始碼，搞懂 SGLang 為什麼在多輪對話與 Agent 場景特別快。

- [ ] **Day 7｜架構總覽**
  - 學習：TokenizerManager → Scheduler → ModelRunner → DetokenizerManager 的多進程架構與 ZMQ 通訊
  - 實作：讀 `python/sglang/srt/` 的目錄結構，列出每個模組的職責，寫成筆記
- [ ] **Day 8｜追蹤一個請求的一生**
  - 學習：HTTP server 進入點、tokenize、排程、forward、detokenize 的完整路徑
  - 實作：用 debugger 或加 log 追一個請求，畫成流程圖（Day 27 文章的素材）
- [ ] **Day 9｜RadixAttention**
  - 學習：Radix tree 前綴快取、LRU 淘汰；讀 `radix_cache.py`
  - 實作：設計共享長 system prompt 的負載，開關 `--disable-radix-cache` 比較快取命中率與 TTFT
- [ ] **Day 10｜排程器**
  - 學習：zero-overhead batch scheduler、chunked prefill、`--schedule-policy`（lpm / fcfs）
  - 實作：比較不同排程策略在混合長短請求下的 P99 延遲
- [ ] **Day 11｜KV cache 記憶體管理**
  - 學習：KV cache 記憶體池、`--mem-fraction-static`、請求被搶佔（retract）的條件
  - 實作：刻意製造記憶體壓力，在 log 中找出 retract 事件，量測對延遲的影響
- [ ] **Day 12｜結構化輸出與前端語言**
  - 學習：JSON schema / regex 受限解碼（xgrammar 後端）、frontend 的 `gen`、`select`、`fork`
  - 實作：量測受限解碼的額外開銷，做一個結構化輸出的小 Agent
- [ ] **Day 13–14｜專案 B：修一個真實 bug**
  - 從 SGLang issues 中挑一個標為 bug、尚無人處理的問題，先留言認領
  - 重現 → 最小化測試案例 → 定位原因 → 修復 → 加測試，送出第 2 個 PR
  - **專案 B 產出**：一篇除錯紀錄（症狀、假設、驗證、修法），對應職缺的「debug complex technical problems」

## 第三階段 Day 15–23：推論效能深化

這一階段只聚焦 LLM 推論，從 kernel 一路往上到多機部署，每天都要有前後對比數據。

- [ ] **Day 15｜GPU 基礎與 Triton 入門**
  - 學習：GPU 記憶體階層、roofline model、memory-bound vs compute-bound kernel、Triton 程式模型
  - 實作：跑通 Triton 官方教學的 vector add 與 softmax，理解 block 與 program id
- [ ] **Day 16｜寫一個推論用的 kernel**
  - 學習：瀏覽 `sgl-kernel` 目錄，看 SGLang 用了哪些自訂 kernel
  - 實作：用 Triton 寫 fused RMSNorm 或 SiLU-and-mul，對照 PyTorch 原生實作量測頻寬與加速比
- [ ] **Day 17｜Profiling：model → runtime → kernel**
  - 學習：Nsight Systems、PyTorch profiler 的用法與讀圖方式
  - 實作：profile 一次 prefill 與一次 decode step，找出最耗時的 kernel，寫成報告
- [ ] **Day 18｜CUDA Graph 與 attention 後端**
  - 學習：CUDA Graph 為何能降低 decode 的 CPU 開銷、`--enable-torch-compile`、FlashInfer 等 attention backend
  - 實作：比較 CUDA Graph 開關與不同 attention backend 在小 batch 下的延遲差異
- [ ] **Day 19｜量化**
  - 學習：FP8、AWQ、GPTQ 權重量化；FP8 KV cache（`--kv-cache-dtype`）
  - 實作：比較 BF16 / FP8 / AWQ 的吞吐、顯存與準確率（GSM8K 抽 200 題）
- [ ] **Day 20｜推測解碼**
  - 學習：speculative decoding 原理、EAGLE 系列、Spec V2
  - 實作：量測 EAGLE 在不同 batch size 下的接受率與加速比，解釋大 batch 時加速變小的原因
- [ ] **Day 21｜平行化**
  - 學習：Tensor Parallel、Data Parallel、MoE 的 Expert Parallel；TP 的 all-reduce 通訊開銷
  - 實作：租 2 張 GPU 比較 TP=2 與 DP=2，用 profiler 看通訊佔比
- [ ] **Day 22｜PD 分離與 HiCache**
  - 學習：Prefill/Decode 分離的動機與 KV 傳輸、HiCache 分層 KV 快取（GPU → CPU → 儲存）
  - 實作：讀 PD 分離的文件與範例，在 2 張 GPU 上跑通最小設定，記錄 TTFT 與 TPOT 的變化
- [ ] **Day 23｜Model Gateway 路由 + 緩衝**
  - 學習：快取感知路由（cache-aware routing）的原理
  - 實作：用 Gateway 在 2 個 worker 前比較 cache-aware 與 round-robin 的命中率與 TTFT
  - 緩衝：補進度，把 Day 15–22 的數據整理成一份效能報告

## 第四階段 Day 24–30：深度貢獻、分享與投遞

最後 7 天做一個有分量的貢獻，並把學到的東西寫出來、講出來：職缺明確要求文件撰寫與技術分享。

- [ ] **Day 24–26｜專案 C：有分量的 PR**
  - 從前面挖到的問題中挑一個：效能改善、補一個缺少的功能、補 benchmark 或 kernel 測試
  - 先在 issue 或 Slack 和 maintainer 討論做法，再實作；附上前後數據送出第 3 個 PR，並積極回應 review 意見
- [ ] **Day 27｜技術 deep-dive 文章**
  - 主題例如「一個請求在 SGLang 裡的一生：從 TokenizerManager 到 RadixAttention」
  - 用 Day 8 的流程圖、Day 9–11 與第三階段的實驗數據；發在個人部落格或 Medium，並分享到 SGLang Slack
- [ ] **Day 28｜10 分鐘技術簡報**
  - 把文章濃縮成投影片並錄影上傳，證明「present technical deep-dives」與英文溝通能力
- [ ] **Day 29｜作品集與履歷**
  - 3 個專案 README 統一格式（問題 → 方法 → 數據 → 結論），更新履歷與 LinkedIn
  - 模擬面試：用「面試準備」那段的問題練習，能用英文講清楚每個專案
- [ ] **Day 30｜投遞 + 緩衝**
  - 投遞時附上 PR 連結、文章與影片；在 Slack 互動過的 maintainer 可以禮貌地告知你已申請
  - 回頭跟進還沒合併的 PR

## 履歷寫法

每一條都用「動作 + 技術 + 量化結果」的格式；下方 X、Y 換成你自己量到的真實數字。履歷寫英文時可直接使用。

**Open-Source Contributions & LLM Inference Projects** | Python, SGLang, Triton, CUDA, Nsight

- Contributed X PRs to SGLang (Y merged), including a fix for \[issue summary\] (PR #XXXX) and \[performance/feature summary\] (PR #XXXX).
- Reproduced and debugged a \[scheduler / cache / runtime\] bug end to end, from a minimal repro to a tested fix.
- Wrote a fused Triton kernel for \[RMSNorm / SiLU-and-mul\], reaching X% of peak memory bandwidth and Y× speedup over PyTorch.
- Profiled SGLang decode steps with Nsight Systems; benchmarked RadixAttention prefix caching (X% TTFT reduction), FP8 quantization and EAGLE speculative decoding.
- Benchmarked TP vs. DP serving on 2 GPUs and cache-aware routing with SGLang Model Gateway (X% higher cache hit rate than round-robin).
- Published a technical deep-dive on SGLang internals (\[link\]) and a 10-minute recorded talk (\[link\]).

**面試準備**：每條都要能解釋「為什麼」，例如 RadixAttention 為何比逐請求快取好、推測解碼為何在大 batch 下效果變差。

## 學習資源

- [SGLang GitHub](https://github.com/sgl-project/sglang)：原始碼、範例、`good first issue`
- [SGLang 2026 Roadmap](https://github.com/sgl-project/sglang/issues/12780)：了解 PD 分離、HiCache、Model Gateway 的發展方向
- SGLang 論文：*SGLang: Efficient Execution of Structured Language Model Programs*（RadixAttention 原理）
- LMSYS 部落格的 SGLang 版本發表文章（v0.2–v0.4、大規模 Expert Parallel）
- vLLM 論文 *Efficient Memory Management for LLM Serving with PagedAttention*（Day 5 對照用）
- [SGLang vs vLLM 比較（2026 年 9 月）](https://www.morphllm.com/comparisons/sglang-vs-vllm)：目前最新為 SGLang v0.5.20

版本更新很快，實作前請以官方文件的參數名稱為準。
