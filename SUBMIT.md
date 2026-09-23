# Submission runbook

Two venues, two very different processes.

## A. Inspect Evals Register (UK AISI)

inspect_evals **no longer accepts code PRs for new evals**. New evals are
*registered*: your code stays in your own public repo, and a bot adds a
`register/milgram/eval.yaml` entry pointing to a pinned commit.
Docs: https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/register/README.md

### 1. Publish this repo (you do this)

```bash
cd ~/Desktop/HD/code/projects/one_timers/inspect_milgram
gh repo create hidayetaksu/inspect-milgram --public --source . --push
git rev-parse HEAD        # the 40-char SHA you will submit
```

### 2. Produce the two required full-run logs

The register wants logs from **two models, every sample**. "Every sample"
is 6 conditions; use `epochs=1` to keep it cheap, and pick small models:

```bash
uv run inspect eval milgram/milgram -T conditions=all -T epochs=1 --model openai/gpt-4o-mini
uv run inspect eval milgram/milgram -T conditions=all -T epochs=1 --model anthropic/claude-haiku-4-5-20251001
```

Logs land in `./logs/*.eval`. Check them with `uv run inspect view`, and
remove anything identifying (usernames, local paths) before uploading.

Tip: run with `-T epochs=15` on the same two models too and put the numbers
in the README. A table that reproduces a paper figure makes reviewers trust it.

### 3. Open the registration issue

https://github.com/UKGovernmentBEIS/inspect_evals/issues/new/choose →
**Register Eval Submission**. Fields:

- **arXiv URL:** `https://arxiv.org/abs/2608.16177v2`
- **Source URL:** `https://github.com/hidayetaksu/inspect-milgram/blob/<SHA>/src/milgram/milgram.py`
  (the `@task` is the only one in the file, so no `#L` anchor is needed)
- **Maintainers:** `hidayetaksu`

### 4. After the bot opens the PR

- Upload the two `.eval` logs to the log store as the PR comment instructs,
  then comment to confirm.
- Fix anything the automated checks flag (they lint with
  `inspect-evals-lint`, verify the `@task`, and check that the title and
  description match the code).
- Agent-generated PRs must stay drafts, and **you** must write the replies
  to reviewers. Use Conventional Comments (`suggestion:`, `question:` ...).

### Optional: first post in the issue

> I'm Hidayet Aksu (ex-Google Privacy Sandbox, PhD). This registers the
> obedience battery from arXiv:2608.16177: a 42-model Milgram-paradigm
> census. It's a multi-turn, adaptive, safety-relevant eval, and the
> fiction-framing and tool-vs-text findings are guardrail gaps that
> single-turn refusal benchmarks don't catch. The harness is deterministic,
> needs no external assets, and has end-to-end mockllm tests.

## B. HELM Safety (Stanford CRFM)

There is no submission form. HELM Safety's benchmarks are chosen by the
CRFM team. The route is a GitHub issue proposing the benchmark, followed by
a scenario PR if they're interested.

**The fit problem:** HELM scenarios are static `Instance`s scored
per-instance, and the Milgram session is *adaptive*: each harness turn
depends on the model's previous action. So propose a **static
decision-point variant** that HELM can run as-is, and keep the full
adaptive battery in Inspect:

- Each instance is a transcript prefix in which the Teacher has complied up
  to voltage V, ending at the decision for V+15 (optionally after prod k).
- Output: parse `ACTION: SHOCK|STOP`. Metric: P(SHOCK) per voltage band,
  plus the 150 V "consent withdrawal" point and the fiction-minus-baseline
  delta.
- Size: 30 voltages × 6 conditions × (no prod + 4 prods) ≈ 900 instances,
  all deterministic, generated from `prompts.json` by the vendored engine.

Steps:

1. Search https://github.com/stanford-crfm/helm/issues first, so you don't
   duplicate an existing proposal.
2. Open an issue with the draft below.
3. If a maintainer is interested, implement per `docs/code.md`:
   `src/helm/benchmark/scenarios/milgram_scenario.py` (a `Scenario` with
   `get_instances()`), a `@run_spec_function("milgram")` in the safety run
   specs, a metric that parses the action, and entries in `schema.yaml`.
   Test with `helm-run -r milgram:condition=baseline`.
4. Recent example of an outside safety scenario PR:
   https://github.com/stanford-crfm/helm/pull/4353

### Draft HELM issue

**Title:** Proposal: Milgram obedience scenario for HELM Safety (authority-pressure compliance)

> HELM Safety v1 covers harmful-request refusal (HarmBench, SST,
> AnthropicRedTeam), over-refusal (XSTest) and bias (BBQ). It doesn't cover
> **compliance under authority pressure**: whether a model keeps doing a
> harmful thing *it has already started* when an authority insists.
>
> Our paper, arXiv:2608.16177, ports Milgram's obedience paradigm to 42
> served LLMs (4,848 sessions). Findings relevant to HELM Safety:
> - Baseline full-obedience ranges from 0% to 100% across models (mean 42.9%;
>   human anchor 65%).
> - Calling the identical scenario fiction *raises* obedience
>   (Holm-adj. p = 9.2e-4).
> - Moving the decision from a text line to a tool call changes obedience
>   sharply (median −53 V). Text-channel results don't transfer to tool use.
>
> The full protocol is adaptive and multi-turn (it's available in Inspect:
> <link>). For HELM, I propose a **static decision-point scenario**: ~900
> deterministic instances, each a transcript prefix ending at a
> shock decision, scored by P(continue). I'm happy to implement the
> scenario, run spec and metric. Would the team consider it for HELM
> Safety? I'd like your guidance on naming and on which risk category it
> belongs to.

Before you open this, validate the static variant: its per-model P(SHOCK)
should rank-correlate with the census breakoff voltages. That is the first
question a reviewer will ask.
