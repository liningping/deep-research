# 📊 Benchmarking Guide

This guide demonstrates how to evaluate the Enterprise Deep Research Agent using various benchmarks and evaluation frameworks.

## 🚀 Quick Start

### Prerequisites

Complete the [main installation setup](../README.md) first, then configure your environment for benchmarking.

### 🔧 Recommended Configuration

```bash
# .env file settings for optimal benchmarking
LLM_PROVIDER=google
LLM_MODEL=gemini-2.5-pro
GOOGLE_CLOUD_PROJECT=your-project-id
TAVILY_API_KEY=your-tavily-key
MAX_WEB_RESEARCH_LOOPS=5

# Optional: LangSmith for tracing (not required)
# LANGCHAIN_API_KEY=your-key
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_PROJECT=your-project
```

---

## 📋 Evaluation Modes

### 🔄 Sequential Processing
Process queries one at a time using `run_research.py`:

```bash
python run_research.py "Your research query" \
  --provider google \
  --model gemini-2.5-pro \
  --max-loops 2 \
  --output result.json
```

### ⚡ Concurrent Processing
Process multiple queries in parallel using `run_research_concurrent.py`:

```bash
python run_research_concurrent.py \
  --benchmark drb \
  --max_concurrent 4 \
  --provider google \
  --model gemini-2.5-pro \
  --max_loops 5
```

**Optional: Enable trajectory collection for detailed execution traces**
```bash
python run_research_concurrent.py \
  --benchmark drb \
  --max_concurrent 4 \
  --collect-traj  # Saves detailed trajectory data for analysis
```

---

## 🎯 Supported Benchmarks

> 💡 **Default Paths**: The scripts automatically use default input/output paths for each benchmark. You can override with `--input` and `--output_dir` flags.

### 1. DeepResearchBench (DRB)

Comprehensive research evaluation with 100 PhD-curated diverse queries.

**Setup:**
```bash
cd benchmarks
git clone https://github.com/Ayanami0730/deep_research_bench.git
```

To run DeepResearchBench evaluation:

**Step 1: Generate responses for all 100 queries**
```bash
python run_research_concurrent.py \
  --benchmark drb \
  --max_concurrent 4 \
  --provider google \
  --model gemini-2.5-pro \
  --max_loops 5
```

> 💡 **Tip**: Add `--collect-traj` to save detailed execution traces for debugging or analysis.

> ✅ **Step 2 is built into Step 1**: After a DRB batch finishes, `run_research_concurrent.py` automatically runs `process_drb.py`, writing `deep_research_bench/data/test_data/raw_data/<name>.jsonl`. Default `<name>` is `edr_<sanitized --model>` (e.g. `gemini-2.5-pro` → `edr_gemini-2.5-pro`). Override with `--drb-jsonl-name my_run`. Skip with `--no-process-drb`.

**Step 2 (optional): Convert manually** — only if you used `--no-process-drb`, an old output folder, or need a custom JSONL name:
```bash
python process_drb.py \
  --input-dir deep_research_bench/results/edr_reports_qwen3-max \
  --model-name edr_qwen3-max
```

> 📝 **Note**: Run from `benchmarks/`. Add the same `<model-name>` (JSONL stem) to `TARGET_MODELS` in `deep_research_bench/run_benchmark.sh` when using their shell driver.

**Step 3: Run DeepResearchBench evaluation**
```bash
cd deep_research_bench
# Jina（网页抓取，清洗引用等）
export JINA_API_KEY="your_jina_api_key_here"
bash run_benchmark.sh
```

**RACE / `deepresearch_bench_race.py` 打分模型（`utils/api.py`）** 有两种方式：

| 方式 | 环境变量 | 说明 |
|------|----------|------|
| **OpenAI 兼容网关**（默认） | `OPENAI_API_KEY`、`OPENAI_BASE_URL`（可选） | 走 `chat.completions`，模型名由网关决定（如 `gemini-2.5-pro-preview-06-05`）。 |
| **Google 官方 GenAI** | `DRB_EVAL_BACKEND=google_genai` + `GOOGLE_API_KEY`（或 `GEMINI_API_KEY`） | 使用 `pip install google-genai`，直连 [Google AI Studio](https://aistudio.google.com/apikey) 官方 API。可选：`DRB_EVAL_MODEL`（默认 `gemini-2.5-pro`）、`DRB_FACT_MODEL`（默认 `gemini-2.5-flash`）。 |

示例（官方 GenAI）：
```bash
cd benchmarks/deep_research_bench
export DRB_EVAL_BACKEND=google_genai
export GOOGLE_API_KEY="your_key_from_ai_studio"
export JINA_API_KEY="..."
python deepresearch_bench_race.py your_model_jsonl_stem --output_dir results/race/your_run
```

> 🎉 **Results**: The evaluation results will be written to `deep_research_bench/results/`

---

### 2. DeepConsult

Multi-perspective research evaluation with diverse query types.

**Setup:**

Clone the DeepConsult repo and follow the [installation steps](https://github.com/Su-Sea/ydc-deep-research-evals?tab=readme-ov-file#installation):
```bash
git clone https://github.com/Su-Sea/ydc-deep-research-evals.git
```

To run DeepConsult evaluation:

**Step 1: Process DeepConsult CSV queries**
```bash
python run_research_concurrent.py \
  --benchmark deepconsult \
  --max_concurrent 4 \
  --max_loops 10 \
  --provider google \
  --model gemini-2.5-pro
```

**Step 2: Create responses CSV for evaluation**
```bash
python process_deepconsult.py \
  --queries-file /path/to/queries.csv \
  --baseline-file /path/to/baseline_responses.csv \
  --reports-dir /path/to/generated_reports \
  --output-file /path/to/custom_output.csv
```

> 📋 **This script combines**:
> - Questions from the original `queries.csv`
> - Baseline answers from existing responses  
> - Your generated candidate answers from the JSON files
> - **Output**: `responses_EDR_vs_ARI_YYYY-MM-DD.csv`

**Step 3: Run pairwise evaluation**
```bash
cd benchmarks/ydc-deep-research-evals/evals
export OPENAI_API_KEY="your_openai_key_here"
python deep_research_pairwise_evals.py \
    --input-data /path/to/csv/previous/step \
    --output-dir results \
    --model gpt-4.1-2025-04-14 \
    --num-workers 4 \
    --metric-num-workers 3 \
    --metric-num-trials 3
```

---

### 3. LiveResearchBench (LRB)

A user-centric benchmark with 100 expert‑curated tasks, evaluating AI agents on their ability to retrieve, synthesize, and reason over real-world information.

**Setup:**
Automatically loads data from [Huggingface](https://huggingface.co/datasets/Salesforce/LiveResearchBench)

To run LiveResearchBench evaluation:

**Step 1: Produce markdown reports for the queries**
```bash
python run_research_concurrent.py \
  --benchmark lrb \
  --max_concurrent 3 \
  --provider google \
  --save_md \
  --model gemini-2.5-pro
```

**Step 2: Use DeepEval for evaluation**
This produces the required markdown file format of the reports which can be further evaluated using the [DeepEval](https://github.com/SalesforceAIResearch/LiveResearchBench?tab=readme-ov-file#basic-usage-to-evaluate-long-form-reports) pipeline.

---

## 📈 Monitoring and Debugging

### 🔍 Real-time Progress Monitoring

The concurrent script provides **detailed progress tracking**:

- ⏱️ **Live progress updates** every 10 seconds showing completion rate and ETA
- 📊 **Individual task logging** with timing and performance metrics  
- 📋 **Comprehensive summary** with success/failure statistics

#### 💻 Example Output
```bash
🚀 Starting concurrent processing of 100 tasks
📊 Max workers: 4
⏱️  Rate limit delay: 1.0s
🤖 Using google/gemini-2.5-pro

📈 Progress: 45/100 completed, 2 failed, 8 in progress, ETA: 12.3min
[Task 23] ✅ SUCCESS - Completed in 45.67s
[Task 23] 📊 Metrics: 3 loops, 12 sources, 8,234 chars
[Task 23] 📈 Throughput: 180 chars/second
```

---

## 🐛 Troubleshooting

### ⚠️ Common Issues

**Rate Limiting:**
```bash
# Increase rate limit delay
--rate-limit 2.0

# Reduce concurrent workers
--max-workers 2
```

**API Errors:**
- ✅ Verify all API keys are correctly set
- ✅ Check API quotas and billing
- ✅ Ensure proper network connectivity
