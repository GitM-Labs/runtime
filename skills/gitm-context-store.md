# gitm-context-store

## Description

Context store skill for Git.M's GTM agent stack. Provides structured retrieval of company voice rules, sender personas, copy variants, product state, and scoring context. Used by drafting and scoring agents to answer queries like "what's Git.M's voice for cold LinkedIn from Asmar as sender?"

Storage: Pinecone (semantic search) + Airtable (structured records). Airtable is the source of truth. Pinecone is the query layer.

---

## Query Interface

When invoked, the agent accepts a natural language query and returns the most relevant context. Example queries:

- "what's Git.M's voice for cold LinkedIn from Asmar as sender?"
- "what copy variant should I use for a biotech VP of Infra?"
- "what are the objection responses for vLLM comparison?"
- "what is Git.M's ICP Phase 1?"
- "what is Jalon's sender persona?"

The agent queries Pinecone first for semantic matches, then hydrates with structured fields from Airtable.

---

## Pinecone Setup

Index name: `gitm-context-store`
Dimensions: 1536 (OpenAI text-embedding-3-small)
Metric: cosine
Environment: us-east-1

Each vector has metadata:
- `record_type`: one of `voice_rule`, `sender_persona`, `copy_variant`, `product_state`, `objection_response`, `customer_win`, `icp_profile`, `scoring_context`
- `record_id`: matching Airtable row ID
- `label`: human-readable label
- `sender`: sender name if sender-specific, else `global`
- `vertical`: vertical if vertical-specific, else `global`

### Environment variables required

```
PINECONE_API_KEY
PINECONE_INDEX_NAME=gitm-context-store
OPENROUTER_API_KEY          # already active in Hermes env
OPENROUTER_EMBEDDING_MODEL=openai/text-embedding-3-small
```

Note: OpenRouter is used instead of OpenAI directly. The embedding model `openai/text-embedding-3-small` is available via OpenRouter's OpenAI-compatible endpoint at `https://openrouter.ai/api/v1/embeddings`. Use `OPENROUTER_API_KEY` as the bearer token.

---

## Airtable Schema

### Table: `context_voice_rules`

Company-wide voice rules and brand guidelines.

| Field | Type | Description |
|---|---|---|
| `rule_id` | Single line text (primary) | e.g. `voice_001` |
| `category` | Single select | `tone`, `structure`, `language`, `anti_pattern` |
| `rule` | Long text | The rule in plain English |
| `example_good` | Long text | Example of the rule applied correctly |
| `example_bad` | Long text | Example of what to avoid |
| `applies_to` | Multi-select | `linkedin`, `email`, `all` |

### Table: `context_sender_personas`

Per-sender profile for message personalization.

| Field | Type | Description |
|---|---|---|
| `sender_id` | Single line text (primary) | `jalon`, `adit`, `jane`, `asmar`, `giancarlos`, `evan` |
| `full_name` | Single line text | Full name |
| `role` | Single line text | e.g. `Founder`, `ML Intern`, `GTM Intern` |
| `school` | Single line text | e.g. `UC Berkeley` |
| `background` | Long text | 2-3 sentences on their background relevant to outreach |
| `voice_register` | Single select | `founder`, `peer-curious`, `technical-peer` |
| `credibility_anchor` | Long text | What makes them credible to the prospect |
| `tone_notes` | Long text | Specific tone guidance for this sender |
| `linkedin_url` | URL | Sender's LinkedIn |
| `active` | Checkbox | Whether this sender is active in current sprint |

### Table: `context_copy_variants`

Approved copy variants with taxonomy tags.

| Field | Type | Description |
|---|---|---|
| `variant_id` | Single line text (primary) | e.g. `v0_pick_your_brain` |
| `variant_name` | Single line text | Human label e.g. `Pick your brain` |
| `opener_style` | Single select | `student-credibility` |
| `hook_type` | Single select | `curiosity-flattery`, `research-framing`, `hypothesis-validation` |
| `cta` | Single select | `15-min-soft`, `15-min-or-reply` |
| `voice_register` | Single select | `peer-curious` |
| `signal_reference_style` | Single select | `none`, `hypothesis-direct` |
| `body_template` | Long text | Full message template with `{placeholders}` |
| `sender_id` | Single line text | Sender this is written for |
| `vertical` | Single select | `biotech`, `robotics`, `hpc`, `global` |
| `status` | Single select | `active`, `paused`, `testing` |
| `reply_rate` | Number | Tracked reply rate as data comes in |
| `approved_by` | Single line text | e.g. `Jalon` |
| `approved_date` | Date | When approved |

### Table: `context_product_state`

Current Git.M product facts for grounding agent outputs.

| Field | Type | Description |
|---|---|---|
| `field_id` | Single line text (primary) | e.g. `product_001` |
| `category` | Single select | `positioning`, `icp`, `differentiator`, `not_this`, `proof_point` |
| `label` | Single line text | Short label e.g. `ICP Phase 1` |
| `content` | Long text | The fact, verbatim from GTM brief or arch doc |
| `source` | Single select | `gtm_brief`, `arch_doc`, `founder_input`, `sales_call` |
| `last_updated` | Date | |

### Table: `context_objection_responses`

Approved responses to common objections.

| Field | Type | Description |
|---|---|---|
| `objection_id` | Single line text (primary) | e.g. `obj_001` |
| `objection` | Single line text | The objection as a prospect would say it |
| `category` | Single select | `competitor`, `timing`, `trust`, `budget`, `scope` |
| `response` | Long text | Approved response |
| `compare_to` | Single line text | Competitor name if applicable e.g. `vLLM`, `Nsight` |
| `approved_by` | Single line text | |

### Table: `context_customer_wins`

Reference wins for social proof in messaging.

| Field | Type | Description |
|---|---|---|
| `win_id` | Single line text (primary) | e.g. `win_001` |
| `company_name` | Single line text | Customer or prospect name |
| `vertical` | Single select | `biotech`, `robotics`, `hpc`, `sovereign_compute`, `managed_gpu` |
| `outcome` | Long text | What Git.M recovered or improved |
| `metric` | Single line text | Quantified result if available |
| `usable_in_copy` | Checkbox | Whether this can be referenced in outbound |
| `approved_by` | Single line text | |

---

## Seed Data

### Voice Rules (context_voice_rules)

| rule_id | category | rule | example_good | example_bad |
|---|---|---|---|---|
| `voice_001` | tone | Direct and technical. No filler. Buyers read roofline charts — don't dumb it down. | "We model what the workload should achieve, detect where it falls short, and apply fixes." | "We help you optimize your GPU performance using cutting-edge AI technology." |
| `voice_002` | tone | Lead with output, not cost. Buyers want more capacity, not a lower bill. | "Run more workloads on the hardware you have." | "Reduce your GPU spend by up to 30%." |
| `voice_003` | language | Never call Git.M a profiler, scheduler, or monitoring tool. Break the pattern match immediately. | "Unlike profilers that describe what happened, we prescribe and apply fixes." | "Git.M is a GPU optimization and monitoring platform." |
| `voice_004` | structure | One ask per message. No feature lists. No bullet points in cold outreach. | Single clear CTA at the end. | Three bullet points listing product features mid-message. |
| `voice_005` | anti_pattern | Never use "utilize", "leverage", "synergy", "streamline", or "innovative". | "We find where your workload loses capacity and fix it." | "We leverage innovative techniques to streamline GPU utilization." |
| `voice_006` | tone | Peer-curious register for intern senders. Humble, direct, no sales language. | "I'm a Berkeley student studying GPU runtime inefficiency — your work at [company] is exactly the kind of system I'm trying to understand." | "Hi, I wanted to reach out about an exciting opportunity to optimize your GPU infrastructure." |
| `voice_007` | language | Capacity, not cost. Recovered capacity is the value. Cost reduction is downstream. | "More of your GPU ceiling doing useful work." | "Save money on your GPU bill." |
| `voice_008` | anti_pattern | No generic openers. Never start with "Hope this finds you well" or "I came across your profile." | Start with a specific observation or credibility anchor. | "Hope this finds you well! I came across your profile and was impressed by your work." |

### Sender Personas (context_sender_personas)

| sender_id | full_name | role | voice_register | credibility_anchor | tone_notes |
|---|---|---|---|---|---|
| `asmar` | Asmar | GTM Intern | `peer-curious` | UC Berkeley student studying GPU runtime systems. Direct access to Git.M's engineering team and early findings. | Humble and curious. Reaches out to learn, not to sell. Never pitches. |
| `jane` | Jane | GTM Intern | `peer-curious` | UC Berkeley student. Research framing — studying how GPU-intensive teams handle performance gaps. | Slightly more research-oriented. Can lead with a study framing. |
| `giancarlos` | Giancarlos | GTM Intern | `peer-curious` | UC Berkeley student with interest in ML systems and HPC workloads. | Can lean more technical with HPC and scientific compute prospects. |
| `jalon` | Jalon | Founder | `founder` | Founded Git.M. Built the runtime optimization layer. Deep technical credibility across GPU workloads. | Direct, peer-to-peer, no fluff. Founder-to-founder or founder-to-CTO. Short messages. |
| `adit` | Adit | Founder | `founder` | Co-founder. Technical and commercial credibility. | Similar to Jalon. Peer register with senior buyers. |
| `evan` | Evan | Advisor/Team | `technical-peer` | Deep systems background. | Technical peer register. Use for highly technical prospects. |

### Copy Variants (context_copy_variants)

| variant_id | variant_name | hook_type | cta | signal_reference_style | body_template |
|---|---|---|---|---|---|
| `v0_pick_your_brain` | Pick your brain | `curiosity-flattery` | `15-min-soft` | `none` | "Hi {first_name}, I'm a student at UC Berkeley studying how {vertical} teams manage GPU runtime inefficiency. Your work at {company} is exactly the kind of system I'm trying to understand better. Would you be open to 15 minutes? I'd love to hear how your team thinks about this." |
| `v0_industry_study` | Industry study | `research-framing` | `15-min-soft` | `none` | "Hi {first_name}, I'm running a small study at Berkeley on how {vertical} teams handle GPU performance gaps. Happy to share aggregated findings afterward. Would you be open to 15 minutes to share your experience?" |
| `v0_sanity_check` | Sanity-check my read | `hypothesis-validation` | `15-min-or-reply` | `hypothesis-direct` | "Hi {first_name}, I'm a Berkeley student looking at GPU runtime inefficiency in {vertical} workloads. My read is that {failure_mode} is one of the harder gaps to close with existing tools — curious if that matches what you see. Worth a 15-minute call, or happy to hear your take in a reply." |

### Product State (context_product_state)

| field_id | category | label | content | source |
|---|---|---|---|---|
| `product_001` | positioning | What Git.M is | Runtime software that models how a GPU workload should execute, detects where the real runtime path loses productive capacity, proves which fixes recover it, and safely applies those fixes over time. | `arch_doc` |
| `product_002` | positioning | What Git.M is NOT | Not a profiler. Not a scheduler. Not a kernel optimizer. Not an inference framework. Not a monitoring tool. | `gtm_brief` |
| `product_003` | icp | ICP Phase 1 | Multi-tenant GPU platforms, HPC platforms, managed GPU clouds, sovereign compute providers. Route: SDK embed inside their offering. | `gtm_brief` |
| `product_004` | icp | ICP Phase 2 | Mid-market companies running non-text GPU workloads: biotech, robotics, autonomous systems, HPC, defense, research labs. | `gtm_brief` |
| `product_005` | differentiator | vs profilers | Profilers describe what happened. Git.M prescribes and applies fixes. | `gtm_brief` |
| `product_006` | differentiator | vs inference frameworks | vLLM and TensorRT are LLM-specific. Git.M works across all GPU workloads. | `gtm_brief` |
| `product_007` | differentiator | vs orchestrators | Kubernetes and Slurm decide where workloads run. Git.M optimizes execution inside the cluster. Phase 1 partners, not competitors. | `gtm_brief` |
| `product_008` | not_this | Why not pure LLM serving yet | Mature tooling (vLLM, TensorRT, SGLang) and established benchmarks. Need proof before entering. | `gtm_brief` |
| `product_009` | positioning | Value statement | Capacity, not cost. More of the GPU ceiling doing useful work. | `gtm_brief` |
| `product_010` | icp | Buyer persona | VP/Head of Infrastructure, VP/Head of ML Platform, Director of ML Engineering, CTO at smaller companies, PIs and lab leadership at research orgs. | `gtm_brief` |

### Objection Responses (context_objection_responses)

| objection_id | objection | category | response | compare_to |
|---|---|---|---|---|
| `obj_001` | "We already use Nsight / PyTorch Profiler." | `competitor` | Profilers describe what happened. Git.M diagnoses why and applies the fix. Profilers require a human to interpret results and implement changes. Git.M closes the loop automatically. | `Nsight` |
| `obj_002` | "We use vLLM / TensorRT for inference." | `competitor` | Git.M is not an inference framework. We work at the runtime layer underneath — across training, inference, and scientific compute. vLLM and Git.M are not in conflict. | `vLLM` |
| `obj_003` | "We use Datadog / Grafana for GPU monitoring." | `competitor` | Monitoring surfaces symptoms. Git.M acts on causes. We don't replace your observability stack — we sit underneath it and close the gaps it can't act on. | `Datadog` |
| `obj_004` | "We don't have GPU performance problems." | `trust` | Every GPU workload operates below its attainable ceiling. The gap is usually invisible to existing tools because they don't model what the workload should achieve. We find what's invisible. | `` |
| `obj_005` | "We have an internal perf team." | `scope` | Internal perf teams focus on kernel-level optimization and profiling. The runtime path — scheduling, memory movement, contention, sync — is where most of the remaining gap lives, and it's where existing tooling doesn't reach. | `` |

---

## Seeding Skill

To seed the context store, run inside a Hermes chat session:

```
Seed the gitm-context-store. For each table in context_voice_rules, context_sender_personas, context_copy_variants, context_product_state, and context_objection_responses: read the seed data from the gitm-context-store skill, write each row to Airtable, generate an embedding for each record using OpenRouter with model openai/text-embedding-3-small (endpoint: https://openrouter.ai/api/v1/embeddings, bearer token: OPENROUTER_API_KEY), and upsert into the Pinecone index gitm-context-store with the correct metadata fields (record_type, record_id, label, sender, vertical).
```

---

## Query Skill

To query the context store, run inside a Hermes chat session:

```
Query the gitm-context-store: {your question here}
```

Example:
```
Query the gitm-context-store: what's Git.M's voice for cold LinkedIn from Asmar as sender?
```

Expected response: voice rules for LinkedIn + Asmar's sender persona + active copy variants for Asmar.

---

## Install

```bash
curl -o ~/.hermes/skills/gitm-context-store.md \
  https://raw.githubusercontent.com/GitM-Labs/runtime/main/skills/gitm-context-store.md

hermes skills list | grep gitm-context-store
```
