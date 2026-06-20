# gitm-signal-scan

## Description

Signal discovery skill for Git.M's GTM agent stack. Crawls GitHub, LinkedIn, and X for public GPU-pain signals matching Git.M's ICP. Produces ranked signal records written to Airtable `signals` table. Output schema aligns with Danny's scorer input contract (input_contract.md).

---

## Part 1: Trigger Keyword Library

These are the exact phrases and terms to search across all sources. Organized by pain category.

### Capacity / Utilization Pain
- `"GPU utilization"`
- `"GPUs sitting idle"`
- `"idle GPUs"`
- `"underutilizing GPU"`
- `"GPU bubbles"`
- `"improve GPU utilization"`
- `"can't fit more workloads"`
- `"cluster utilization"`

### Throughput / Latency Pain
- `"throughput-limited"`
- `"latency-limited"`
- `"p99 latency"`
- `"unpredictable latency"`
- `"throughput regression"`
- `"latency regression"`
- `"tokens per second"`
- `"requests per second"`

### Capacity Ceiling Pain
- `"hitting our allocation cap"`
- `"allocation cap"`
- `"had to add more GPUs"`
- `"need more GPUs"`
- `"scaling GPU capacity"`
- `"out of GPU memory"`
- `"OOM"`

### Performance Regression Pain
- `"performance regression"`
- `"chasing a regression"`
- `"training slowdown"`
- `"inference slowdown"`
- `"kernel regression"`
- `"CUDA regression"`

### Tooling / Observability Pain
- `"need to build performance tooling"`
- `"GPU profiling"`
- `"runtime inefficiency"`
- `"wasted compute"`
- `"compute overhead"`
- `"GPU bottleneck"`
- `"memory bottleneck"`
- `"communication bottleneck"`
- `"NCCL bottleneck"`
- `"NVLink bottleneck"`

### ICP-Specific Terms (Phase 2 verticals)
- `"molecular dynamics GPU"`
- `"protein folding compute"`
- `"CFD GPU"`
- `"robotics simulation GPU"`
- `"perception training"`
- `"scientific compute GPU"`
- `"HPC GPU"`
- `"drug discovery compute"`

---

## Part 2: Crawl Source List

### Source 1: GitHub (priority — cleanest API, run first)

Target repos and orgs:

| Repo / Org | Why |
|---|---|
| `pytorch/pytorch` | Core ML framework — issues surface runtime pain |
| `openai/triton` | GPU kernel language — perf issues are explicit |
| `vllm-project/vllm` | LLM inference — throughput and latency pain |
| `NVIDIA/nccl` | Collective comms — NCCL bottleneck signals |
| `NVIDIA/apex` | Mixed precision training — perf regression signals |
| `microsoft/DeepSpeed` | Distributed training — memory and throughput pain |
| `huggingface/transformers` | Model training — GPU utilization signals |
| `huggingface/accelerate` | Multi-GPU training — scheduling and sync pain |
| `facebookresearch/xformers` | Attention kernels — memory and throughput |
| `NVIDIA/cuda-samples` | CUDA — low-level runtime pain |

Search targets within each repo:
- Issues (open and closed, last 90 days)
- PRs (title and body, last 90 days)
- Discussion threads where available

### Source 2: X (Twitter)

Search via X API v2 recent search endpoint.

Query structure:
```
({keyword}) (lang:en) -is:retweet
```

Target accounts to monitor (in addition to keyword search):
- ML infrastructure leads at GPU-heavy companies
- HPC and scientific compute practitioners
- Anyone with `ML infra`, `GPU`, `HPC`, `compute`, `ML platform` in bio

### Source 3: LinkedIn

Search public posts via LinkedIn search (manual or via scraping where permitted).

Target queries:
- Keyword search in posts
- Posts from people with titles matching buyer persona: VP Infra, Head of ML Platform, Director of ML Engineering, CTO, PI, Computing Director

---

## Part 3: Output Schema

### Airtable Table: `signals`

One row per signal found. Aligned with Danny's scorer input contract.

| Field | Type | Description | Scorer field |
|---|---|---|---|
| `signal_id` | Single line text (primary) | e.g. `sig_001` | — |
| `prospect_name` | Single line text | Full name of person who posted | — |
| `prospect_company` | Single line text | Their current company | — |
| `prospect_title` | Single line text | Their current title | — |
| `prospect_linkedin_url` | URL | LinkedIn profile if found | — |
| `source` | Single select | `github`, `x`, `linkedin` | — |
| `source_url` | URL | Direct link to the post, issue, or PR | `pain_signal_url` |
| `signal_date` | Date | Date of the post or issue | `pain_signal_date` |
| `days_since_signal` | Number | Days between signal_date and today | Used for `signal_recency` |
| `signal_recency` | Number | `exp(-days_since_signal / 30)` | `signal_recency` |
| `raw_quote` | Long text | Exact quote from the post or issue | `pain_signal_summary` |
| `pain_summary` | Long text | One-sentence summary of the pain | `pain_signal_summary` |
| `pain_category` | Single select | `utilization`, `throughput`, `latency`, `regression`, `tooling`, `capacity`, `vertical_specific` | — |
| `keyword_matched` | Single line text | The trigger keyword that matched | — |
| `pain_acknowledged` | Number | 1 (all signals in this table are pain-confirmed) | `pain_acknowledged` |
| `icp_fit` | Single select | `phase_1`, `phase_2`, `unknown` | — |
| `vertical` | Single select | `biotech`, `robotics`, `hpc`, `managed_gpu`, `sovereign_compute`, `other` | — |
| `rank_score` | Number | Composite rank: signal_recency × icp_weight | — |
| `status` | Single select | `new`, `enriched`, `scored`, `outreached`, `discarded` | — |
| `notes` | Long text | Manual caveats | — |

### Ranking Formula

```
rank_score = signal_recency × icp_weight

icp_weight:
  phase_1 company = 1.0
  phase_2 company = 0.8
  unknown          = 0.4

signal_recency = exp(-days_since_signal / 30)
```

Higher rank_score = higher priority for outreach queue.

---

## Part 4: GitHub Scan — Step by Step

This is the first source to run. GitHub has the cleanest API and requires no scraping.

### Auth
```
GITHUB_TOKEN    # already in ~/.hermes/.env
GITHUB_ORG      # GitM-Labs
```

### Search query format

GitHub code search API:
```
https://api.github.com/search/issues?q={keyword}+repo:{owner}/{repo}&sort=created&order=desc&per_page=20
```

### Steps

1. For each repo in the target repo list:
   - For each keyword in the trigger keyword library:
     - Query GitHub issues and PRs created in the last 90 days
     - Filter results where the keyword appears in title OR body
     - Extract: issue/PR URL, title, body excerpt (first 500 chars), author login, created_at date
     - Look up author profile: name, company field, location
     - Skip bots (author login contains `[bot]`)

2. For each result:
   - Compute `days_since_signal` = today - created_at
   - Compute `signal_recency` = exp(-days_since_signal / 30)
   - Extract the matching keyword and a 1-2 sentence `pain_summary`
   - Classify `pain_category` from keyword matched
   - Set `pain_acknowledged = 1`
   - Set `icp_fit` based on company field (phase_1 if orchestrator/managed GPU, phase_2 if biotech/robotics/HPC, else unknown)
   - Compute `rank_score`

3. Deduplicate: if same author appears more than once, keep the highest rank_score row only.

4. Write top 20 results sorted by `rank_score` descending to Airtable `signals` table.

5. Log run to Airtable `status_loop_runs`: `{ date, mode: "signal_scan_github", signals_found, signals_written: 20 }`.

### Target: first 20 signals

Pull until 20 unique non-bot authors with valid signal records are written to Airtable.

---

## Cron Schedule

Add to Hermes cron after first manual run passes:

```
hermes cron create --skill gitm-signal-scan --prompt "run github signal scan" --schedule "0 */6 * * *" --name "gitm-signal-scan-github"
```

Runs every 6 hours. Expand to X and LinkedIn once GitHub scan is stable.

---

## Seeding the signals table in Airtable

Create table `signals` with all fields listed in Part 3 before running the scan.

Single select options for `source`: `github`, `x`, `linkedin`
Single select options for `pain_category`: `utilization`, `throughput`, `latency`, `regression`, `tooling`, `capacity`, `vertical_specific`
Single select options for `icp_fit`: `phase_1`, `phase_2`, `unknown`
Single select options for `vertical`: `biotech`, `robotics`, `hpc`, `managed_gpu`, `sovereign_compute`, `other`
Single select options for `status`: `new`, `enriched`, `scored`, `outreached`, `discarded`

---

## Running the scan manually

Install the skill on the VM, then open a Hermes chat session:

```bash
hermes chat --skill gitm-signal-scan
```

Type inside the session:
```
Run the GitHub signal scan. For each repo in the target list, search GitHub issues and PRs from the last 90 days using the trigger keyword library. Extract prospect name, company, title, source URL, signal date, raw quote, and pain summary for each match. Compute signal_recency and rank_score. Deduplicate by author. Write the top 20 results sorted by rank_score to the Airtable signals table using AIRTABLE_API_KEY and AIRTABLE_BASE_ID from the environment. Log the run to status_loop_runs.
```

---

## Install

Upload to GitM-Labs/runtime, then on the VM:

```bash
curl -o ~/.hermes/skills/gitm-signal-scan.md \
  https://raw.githubusercontent.com/GitM-Labs/runtime/main/skills/gitm-signal-scan.md

hermes skills list | grep gitm-signal-scan
```
