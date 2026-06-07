# gitm-internal-status-loop

## Description

Internal meta-loop skill for Git.M's GTM sprint. Runs two output modes: a daily team standup loop (Mode 1) and a founder-approval DM loop (Mode 2). No founder interruption on Mode 1. Mode 2 fires only on explicit approval triggers.

---

## Mode 1: Team standup (auto, 9:00 AM daily)

### Trigger
Scheduled. Runs at 9:00 AM every weekday. No manual trigger required.

### Inputs
- GitHub: commits from the last 24h across the gtm-related repos
- Slack: `#gtm-standup` — any messages posted since yesterday 9:00 AM
- Airtable: sprint tracker — task status changes, new rows, status field updates

### Steps

1. For each intern (Asmar, Jane, Giancarlos, Danny, Khoa, Arshad):
   - Pull their GitHub commits from the last 24h (filter by author)
   - Pull their Airtable task rows where `owner` = their name and `updated_at` >= yesterday 9:00 AM
   - Pull any Slack messages they posted in `#gtm-standup` since yesterday
   - Synthesize into three fields:
     - **Yesterday:** what they shipped or progressed (from commits + Airtable updates)
     - **Today:** next open tasks from Airtable (status = In Progress or Up Next)
     - **Blockers:** any blocker flags in Airtable OR explicit blocker language in their Slack messages
   - DM each intern their own summary for review. Message format:

```
Hey {name} — here's your standup for today:

Yesterday: {yesterday}
Today: {today}
Blockers: {blockers_or_none}

Reply to correct anything before I post to #gtm-standup.
```

2. Wait 15 minutes for replies. Apply any corrections.

3. Compile all six summaries into a single standup digest. Post to `#gtm-standup`:

```
*GTM standup — {date}*

*Asmar:* Yesterday: … | Today: … | Blockers: …
*Jane:* Yesterday: … | Today: … | Blockers: …
*Giancarlos:* Yesterday: … | Today: … | Blockers: …
*Danny:* Yesterday: … | Today: … | Blockers: …
*Khoa:* Yesterday: … | Today: … | Blockers: …
*Arshad:* Yesterday: … | Today: … | Blockers: …
```

4. Log run to Airtable `status_loop_runs` table: `{ date, mode: "standup", status: "ok", interns_DMed: 6 }`.

### Rules
- Do not DM Jalon or any founder.
- Do not post to `#gtm-founder-approvals`.
- If an intern has no activity (no commits, no Airtable updates, no Slack messages), mark Yesterday as `no activity logged` — do not omit them.
- If Slack, GitHub, or Airtable is unreachable, log the error and skip that source. Do not fail the whole run.

---

## Mode 2: Founder approval (DM Jalon, event-driven)

### Trigger
Event-driven. Fires when any of the following conditions are detected in Airtable or Slack:

| Trigger type | Detection method |
|---|---|
| Tool account approval needed | Airtable `approval_queue` table — new row with `type = tool_account` and `status = pending` |
| Copy variant sign-off needed | Airtable `approval_queue` table — new row with `type = copy_variant` and `status = pending` |
| Cross-team blocker | Airtable `blockers` table — new row with `escalation = founder` OR Slack message in any `#gtm-*` channel containing trigger phrases (see below) |
| VM / infra decision | Airtable `approval_queue` table — new row with `type = infra` and `status = pending` |

Trigger phrases for Slack scan: `"need sign-off"`, `"founder decision"`, `"blocked on Jalon"`, `"needs approval"`, `"escalate"`.

### Steps

1. Detect trigger (polling interval: every 10 minutes on Airtable `approval_queue` and `blockers`; real-time on Slack if webhook available, else 10-minute poll).

2. Fetch full context for the item:
   - For copy variants: pull the variant text, vertical, role, sender from Airtable
   - For tool accounts: pull the tool name, requested by, and purpose
   - For blockers: pull the blocker description, who raised it, what is blocked
   - For infra decisions: pull the options, recommendation, and who needs to act

3. DM Jalon on Slack with a structured approval request:

```
*Approval needed: {trigger_type}*

{context_summary}

Options:
{options_if_applicable}

Recommendation: {recommendation_if_applicable}

Reply with *approve*, *reject*, or *hold* — or ask a question and I'll loop in the right person.
```

4. On reply:
   - `approve` → update Airtable row `status = approved`, notify the requesting intern via DM
   - `reject` → update Airtable row `status = rejected`, DM requesting intern with Jalon's reason
   - `hold` → update Airtable row `status = on_hold`, DM requesting intern
   - Any other reply → DM Jalon asking to clarify with approve / reject / hold

5. Post a summary to `#gtm-founder-approvals`:

```
*{trigger_type} — {approved/rejected/on_hold}*
Requested by: {intern}
Decision: {decision}
Time to decision: {elapsed}
```

6. Log to Airtable `status_loop_runs`: `{ date, mode: "founder_approval", trigger_type, decision, elapsed_minutes }`.

### Rules
- Only DM Jalon. Do not DM other founders unless explicitly configured.
- Do not post approval requests to `#gtm-standup`.
- Batch multiple pending items into a single DM if more than one trigger fires within a 5-minute window.
- Do not re-trigger on items already in `approved`, `rejected`, or `on_hold` status.

---

## Airtable schema dependencies

| Table | Fields used |
|---|---|
| `sprint_tracker` | `owner`, `status`, `updated_at`, `task_name`, `blockers` |
| `approval_queue` | `type`, `status`, `requested_by`, `context`, `created_at` |
| `blockers` | `description`, `raised_by`, `escalation`, `created_at` |
| `status_loop_runs` | `date`, `mode`, `status`, `interns_DMed`, `trigger_type`, `decision`, `elapsed_minutes` |

---

## Environment variables required

```
SLACK_BOT_TOKEN         # needs chat:write, im:write, channels:read, channels:history
SLACK_STANDUP_CHANNEL   # #gtm-standup channel ID
SLACK_APPROVALS_CHANNEL # #gtm-founder-approvals channel ID
SLACK_JALON_USER_ID     # Jalon's Slack user ID for DMs
SLACK_INTERN_IDS        # JSON map: { "Asmar": "U...", "Jane": "U...", ... }
AIRTABLE_API_KEY
AIRTABLE_BASE_ID
GITHUB_TOKEN            # read:org, repo scope
GITHUB_ORG              # git.m org name
```

---

## Error handling

- Source unreachable (GitHub / Airtable / Slack): log warning, skip source, continue run. Do not fail silently — append `[source unavailable]` to affected fields.
- Slack DM delivery failure: retry once after 60s, then log to Airtable and continue.
- No activity detected for an intern: do not skip — report `no activity logged`.
- Duplicate trigger detected (same `approval_queue` row): no-op, already handled.

---

## Install

Place this file at:

```
~/.hermes/skills/gitm-internal-status-loop.md
```

Then verify:

```bash
hermes skills list | grep gitm-internal-status-loop
```
