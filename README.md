# Specialty VBC Report Validation Engine

## What This Tool Produces

This tool reviews performance reports for specialty value-based care (VBC) contracts and generates a single HTML report that answers the question: **"Is this data trustworthy, and what should we discuss at the next Joint Operating Committee meeting?"**

The report covers two simulated contracts — a musculoskeletal (MSK) bundled episode contract and an oncology episode + pathway contract — and surfaces findings like these:

**MSK Contract (Midwest Orthopedic Partners)**
- The contract generated **$753K in savings** with a **$377K provider payout**, and the quality gate passed at 72.2%.
- However, the validation engine flagged that **spinal fusion implant costs are 20.6% of total episode cost** (industry benchmark: 15–18%), and risk scores confirm the overrun isn't explained by sicker patients — it's likely vendor pricing or surgeon device preference.
- **Knee arthroscopy volume is 33/1,000 members** against an expected range of 15–25, with an arthroscopy-to-conservative ratio of 0.64:1 (expected: 0.30–0.40). Multiple RCTs show arthroscopic debridement for knee OA is clinically ineffective in this population.
- **PROM collection is at 39%** (target: 60%), which means the reported 64.5% improvement rate is measured on a biased sample and shouldn't be trusted.
- Discharge-to-home increased 12 percentage points while **ER visits jumped 75%** — patients are being sent home earlier but some may lack adequate home health support.

**Oncology Contract (Regional Cancer Center Network)**
- The contract shows **$757K in losses**, but the bigger problem is the **quality gate failed at 53% (minimum: 55%)**, which would zero out any shared savings even if cost performance improves.
- **Lung NSCLC first-line costs $118K/episode vs $105K target** — back-calculation shows non-pathway regimens cost $142K vs $107K for pathway cases. The 32% non-adherent cases are driving the entire overrun.
- **Trastuzumab biosimilar utilization is 11%** (target: 80%). Switching brand to biosimilar would save ~$169K — enough to close most of the cost gap in metastatic breast episodes.
- **All five end-of-life metrics are failing simultaneously** — chemo near death at 19% (target <10%), hospice enrollment at 43% (target >55%). The root cause is advance care planning documentation at only 44.6%, which predicts all the downstream EOL failures.
- The quality gate is only **2 points from passing** — the cheapest path is improving biosimilar utilization and biomarker testing.

When an Anthropic API key is provided, the report also includes **AI-generated diagnostic narratives** for each flagged issue — root cause analysis, questions to ask the provider, recommended interventions with timelines, and contract implications.

---

## Quick Start

```bash
pip install -r requirements.txt
python3 main.py
open output/vbc_validation_report.html
```

To enable AI diagnostic narratives (optional):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python3 main.py
```

The report generates in under 5 seconds without AI, or ~30 seconds with AI diagnostics enabled.

## Architecture

```
project/
├── main.py                          # Entry point — orchestrates the full pipeline
├── config/
│   ├── contract_metadata.json       # Contract definitions (type, thresholds, sharing %)
│   └── reference_ranges.json        # Expected ranges by specialty, LOB, episode type
├── data/
│   ├── msk_episodes.csv             # Simulated MSK episode performance data
│   ├── msk_quality.csv              # MSK quality metrics
│   ├── onc_episodes.csv             # Simulated oncology episode performance data
│   ├── onc_quality.csv              # Oncology quality metrics (including EOL)
│   └── onc_drug_detail.csv          # Oncology drug cost breakdown
├── validation/
│   ├── schema.py                    # Schema validation (columns, types, nulls)
│   ├── arithmetic.py                # Arithmetic reconciliation checks
│   ├── range_checks.py              # Range checks against reference tables
│   ├── cross_metric.py              # Cross-metric consistency validation
│   ├── msk_rules.py                 # MSK-specific validation rules
│   └── onc_rules.py                 # Oncology-specific validation rules
├── diagnosis/
│   ├── ai_diagnostics.py            # Claude API integration for narrative generation
│   └── prompt_templates.py          # Structured prompts for consistent AI output
├── reporting/
│   ├── html_report.py               # HTML report generator
│   └── report_template.html         # Jinja2 HTML template
└── requirements.txt
```

The pipeline executes in order:

1. **Load** contract metadata, reference ranges, and performance data
2. **Validate schema** — confirm expected columns, data types, no critical nulls
3. **Reconcile arithmetic** — verify cost components sum correctly, rates match numerator/denominator
4. **Check ranges** — flag metrics outside expected bounds for specialty and line of business
5. **Cross-validate metrics** — check that related metrics tell a consistent story (e.g., if home discharge goes up, do ER visits go up too?)
6. **Apply specialty rules** — MSK-specific rules (implant cost ratios, arthroscopy appropriateness) and oncology-specific rules (pathway adherence cost correlation, biosimilar savings, EOL-ACP root cause analysis)
7. **Diagnose with AI** (optional) — send flagged issues with context to Claude for narrative interpretation
8. **Generate report** — assemble HTML with validation results, severity ratings, narratives, and financial analysis

## Validation Rules

The engine runs **55 checks** across 6 categories, each producing a flag rated RED (critical), YELLOW (warning), or GREEN (passing):

| Category | Description | Examples |
|---|---|---|
| Schema | Data structure integrity | Missing columns, wrong types, null critical fields |
| Arithmetic | Internal math consistency | Cost components don't sum, rate ≠ numerator/denominator |
| Range | Values vs expected bounds | Episode cost outside reference range for procedure type |
| Cross-metric | Multi-metric consistency | Home discharge up + ER visits up = inadequate post-acute support |
| MSK specialty | Clinical/financial rules | Implant cost ratios, arthroscopy overutilization, PROM reliability |
| ONC specialty | Clinical/financial rules | Pathway adherence cost correlation, biosimilar savings, EOL-ACP root cause, quality gate improvement path |

## Design Decisions

**Deterministic validation first, AI second.** The validation rules are codified clinical and financial logic — they produce the same output every time. The AI layer adds interpretive value (root cause analysis, JOC questions, intervention recommendations) but is never a dependency. If the API is unavailable, the report still works.

**Domain-specific rules, not generic outlier detection.** Rules like "implant cost >20% of episode cost for joint replacement" or "back-calculate pathway vs non-pathway cost split" encode the kind of analysis a provider economics analyst would do manually. They're designed to catch the specific patterns that matter in VBC contract management.

**Pipeline-ready architecture.** In production, the CSV inputs would be Snowflake queries, the reference ranges would come from benchmark tables, and the validation step would run as part of an ETL pipeline. The modular design means adding a new specialty (cardiology, behavioral health) requires only a new rules file and reference range section.

## Dependencies

- **pandas** — data manipulation
- **jinja2** — HTML template rendering
- **anthropic** — Claude API client (optional, for AI diagnostics)
- **python-dateutil** — date arithmetic for novel therapy lookback
