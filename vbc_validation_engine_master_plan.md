# Specialty VBC Report Validation Engine — Master Implementation Plan

## Context & Purpose

This document is a complete implementation specification for building a **Specialty Value-Based Care (VBC) Report Validation Engine**. The tool is designed as an interview demonstration piece for a Business Information Consultant position on the Provider Economics team at Carelon (Elevance Health). The interviewer is a Principal Data Analyst and VBC Subject Matter Expert whose background includes designing attribution/affinity algorithms in PySpark, building ETL pipelines on AWS Glue and Databricks, implementing risk adjustment models, and creating Tableau dashboards for VBC programs across Commercial, Medicare, and Medicaid lines of business. He has specific oncology analytics experience (IBM Watson-based Care Insights) and implemented TennCare's Medicaid VBC programs including episodes of care with risk adjustment using linear regression.

The tool must demonstrate:
1. **Data engineering sensibility** — structured validation pipeline with modular, reusable rules, clear separation of concerns, and pipeline-ready architecture
2. **Deep VBC domain knowledge** — realistic data, clinically meaningful validation rules, and accurate financial logic for specialty episode-based and shared savings contracts
3. **Practical AI integration** — Claude API used for narrative diagnosis of flagged issues, clearly separated from deterministic validation logic, demonstrating "stable and trustworthy" AI augmentation
4. **Communication capability** — final output is a clean, professional HTML report suitable for a Joint Operating Committee (JOC) meeting

---

## Architecture Overview

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
│   ├── __init__.py
│   ├── schema.py                    # Schema validation (columns, types, nulls)
│   ├── arithmetic.py                # Arithmetic reconciliation checks
│   ├── range_checks.py              # Range checks against reference tables
│   ├── cross_metric.py              # Cross-metric consistency validation
│   ├── trend.py                     # Year-over-year trend deviation checks
│   ├── msk_rules.py                 # MSK-specific validation rules
│   └── onc_rules.py                 # Oncology-specific validation rules
├── diagnosis/
│   ├── __init__.py
│   ├── ai_diagnostics.py            # Claude API integration for narrative generation
│   └── prompt_templates.py          # Structured prompts for consistent AI output
├── reporting/
│   ├── __init__.py
│   ├── html_report.py               # HTML report generator
│   └── report_template.html         # Jinja2 HTML template
├── requirements.txt
└── README.md
```

The pipeline executes in this order:
1. **Load** contract metadata, reference ranges, and performance data
2. **Validate schema** — confirm expected columns, data types, no critical nulls
3. **Reconcile arithmetic** — verify PMPM × member months = totals, episode cost × volume = total spend
4. **Check ranges** — flag metrics outside expected bounds for specialty/LOB
5. **Cross-validate metrics** — check that related metrics tell a consistent story
6. **Apply specialty rules** — MSK-specific or oncology-specific clinical/financial logic
7. **Diagnose with AI** — send flagged issues with context to Claude API for narrative interpretation
8. **Generate report** — assemble HTML report with validation results, severity ratings, narratives, and executive summary

---

## Detailed Data Specifications

### Contract Metadata (`config/contract_metadata.json`)

```json
{
  "contracts": [
    {
      "contract_id": "MSK-2024-001",
      "contract_name": "Midwest Orthopedic Partners — MA Bundled Episodes",
      "specialty": "MSK",
      "lob": "Medicare Advantage",
      "contract_type": "episode_bundled",
      "risk_arrangement": "two_sided",
      "sharing_rate_savings": 0.50,
      "sharing_rate_losses": 0.30,
      "performance_period": "CY 2024",
      "attribution_method": "episode_trigger",
      "episode_window_days": 90,
      "attributed_members": 6200,
      "member_months": 74400,
      "quality_gate_minimum": 60,
      "stop_loss_threshold_per_episode": 150000,
      "benchmark_trend_factor": 1.032,
      "risk_adjustment_model": "CMS-HCC V28 + surgical complexity",
      "minimum_episode_volume": 20,
      "data_as_of": "2025-01-15",
      "claims_runout_days": 120,
      "notes": "Second performance year. Provider completed year 1 with 3.2% savings."
    },
    {
      "contract_id": "ONC-2024-001",
      "contract_name": "Regional Cancer Center Network — MA Episode + Pathway",
      "specialty": "Oncology",
      "lob": "Medicare Advantage",
      "contract_type": "episode_pathway",
      "risk_arrangement": "two_sided",
      "sharing_rate_savings": 0.50,
      "sharing_rate_losses": 0.25,
      "performance_period": "CY 2024",
      "attribution_method": "chemo_trigger",
      "episode_window_days": 180,
      "attributed_members": 4800,
      "member_months": 57600,
      "quality_gate_minimum": 55,
      "pathway_adherence_target": 0.80,
      "pathway_bonus_pmpm": 75.00,
      "stop_loss_threshold_per_episode": 500000,
      "benchmark_trend_factor": 1.045,
      "risk_adjustment_model": "OCM-derived (cancer type, stage, comorbidity)",
      "novel_therapy_carveout": true,
      "novel_therapy_lookback_months": 18,
      "data_as_of": "2025-01-15",
      "claims_runout_days": 120,
      "notes": "First performance year. Network includes 3 practice sites."
    }
  ]
}
```

### Reference Ranges (`config/reference_ranges.json`)

```json
{
  "msk": {
    "episode_cost_ranges": {
      "TKR": {"min": 18000, "max": 45000, "expected": 28000, "description": "Total Knee Replacement, 90-day all-in"},
      "THR": {"min": 17000, "max": 42000, "expected": 27500, "description": "Total Hip Replacement, 90-day all-in"},
      "spinal_fusion_1_2": {"min": 25000, "max": 65000, "expected": 38000, "description": "Spinal Fusion 1-2 levels"},
      "spinal_fusion_3_plus": {"min": 45000, "max": 120000, "expected": 72000, "description": "Spinal Fusion 3+ levels"},
      "knee_arthroscopy": {"min": 4000, "max": 15000, "expected": 9200, "description": "Knee Arthroscopy"},
      "rotator_cuff": {"min": 12000, "max": 30000, "expected": 18000, "description": "Rotator Cuff Repair"},
      "conservative_lbp": {"min": 1500, "max": 8000, "expected": 4800, "description": "Conservative Low Back Pain, 12-month"},
      "conservative_joint": {"min": 1200, "max": 6000, "expected": 3500, "description": "Conservative Joint Pain, 12-month"}
    },
    "utilization_ranges_ma": {
      "surgical_conversion_rate_lbp": {"min": 0.05, "max": 0.25, "expected": 0.12, "unit": "proportion", "description": "% LBP patients converting to surgery within 12mo"},
      "surgical_conversion_rate_joint": {"min": 0.10, "max": 0.35, "expected": 0.20, "unit": "proportion"},
      "discharge_to_home_tkr": {"min": 0.50, "max": 0.95, "expected": 0.72, "unit": "proportion"},
      "discharge_to_home_thr": {"min": 0.45, "max": 0.90, "expected": 0.68, "unit": "proportion"},
      "discharge_to_snf_tkr": {"min": 0.05, "max": 0.40, "expected": 0.18, "unit": "proportion"},
      "readmission_rate_joint": {"min": 0.02, "max": 0.10, "expected": 0.05, "unit": "proportion"},
      "readmission_rate_spine": {"min": 0.03, "max": 0.12, "expected": 0.07, "unit": "proportion"},
      "implant_cost_tkr": {"min": 2000, "max": 12000, "expected": 5500, "unit": "dollars"},
      "implant_cost_thr": {"min": 2500, "max": 14000, "expected": 6000, "unit": "dollars"},
      "opioid_mme_discharge_avg": {"min": 0, "max": 90, "expected": 40, "unit": "MME"},
      "prom_collection_rate": {"min": 0.30, "max": 0.95, "expected": 0.65, "unit": "proportion"}
    },
    "quality_targets": {
      "ssi_rate": {"target": 0.02, "max_acceptable": 0.05, "description": "Surgical site infection rate"},
      "readmit_90day": {"target": 0.05, "max_acceptable": 0.10},
      "er_visit_90day": {"target": 0.08, "max_acceptable": 0.15},
      "revision_12mo": {"target": 0.02, "max_acceptable": 0.05},
      "prom_improvement": {"target": 0.70, "description": "% patients showing clinically meaningful improvement on HOOS/KOOS/ODI"}
    }
  },
  "oncology": {
    "episode_cost_ranges": {
      "breast_early": {"min": 20000, "max": 65000, "expected": 42000, "description": "Breast, early stage, 6-month episode"},
      "breast_metastatic": {"min": 50000, "max": 180000, "expected": 95000, "description": "Breast, metastatic, 6-month episode"},
      "lung_nsclc_1L": {"min": 60000, "max": 200000, "expected": 105000, "description": "NSCLC first-line, 6-month episode"},
      "lung_nsclc_2L_plus": {"min": 40000, "max": 160000, "expected": 85000, "description": "NSCLC second-line+, 6-month episode"},
      "colorectal_adjuvant": {"min": 15000, "max": 55000, "expected": 35000, "description": "Colorectal adjuvant, 6-month episode"},
      "colorectal_metastatic": {"min": 40000, "max": 150000, "expected": 82000, "description": "Colorectal metastatic, 6-month episode"},
      "prostate_early": {"min": 8000, "max": 35000, "expected": 18000, "description": "Prostate, early stage, 6-month episode"},
      "prostate_advanced": {"min": 35000, "max": 130000, "expected": 68000, "description": "Prostate, advanced, 6-month episode"}
    },
    "pathway_adherence_benchmarks": {
      "breast_early": {"min_acceptable": 0.75, "expected": 0.85},
      "breast_metastatic": {"min_acceptable": 0.60, "expected": 0.75},
      "lung_nsclc_1L": {"min_acceptable": 0.60, "expected": 0.72},
      "lung_nsclc_2L_plus": {"min_acceptable": 0.50, "expected": 0.65},
      "colorectal_adjuvant": {"min_acceptable": 0.80, "expected": 0.90},
      "colorectal_metastatic": {"min_acceptable": 0.55, "expected": 0.68}
    },
    "eol_benchmarks": {
      "chemo_14d_death": {"target": 0.08, "max_acceptable": 0.15, "description": "Chemotherapy within 14 days of death"},
      "hospice_enrollment": {"target": 0.60, "min_acceptable": 0.40, "description": "Hospice enrollment rate among decedents"},
      "hospice_7d_before_death": {"target": 0.45, "min_acceptable": 0.30, "description": "Hospice enrolled >7 days before death"},
      "icu_30d_death": {"target": 0.12, "max_acceptable": 0.20, "description": "ICU admission within 30 days of death"},
      "er_30d_death": {"target": 0.25, "max_acceptable": 0.40, "description": "ER visit within 30 days of death"}
    },
    "drug_cost_benchmarks": {
      "biosimilar_utilization_trastuzumab": {"target": 0.80, "min_acceptable": 0.50},
      "biosimilar_utilization_bevacizumab": {"target": 0.75, "min_acceptable": 0.45},
      "biosimilar_utilization_rituximab": {"target": 0.70, "min_acceptable": 0.40},
      "office_vs_hopd_ratio": {"target": 0.65, "min_acceptable": 0.40, "description": "% infusions in physician office vs HOPD"},
      "generic_utilization_supportive": {"target": 0.85, "min_acceptable": 0.65, "description": "Generic rate for supportive care drugs"}
    },
    "incidence_rates_ma_per_1000": {
      "breast": {"min": 2.0, "max": 4.5, "expected": 3.2},
      "lung": {"min": 1.5, "max": 3.5, "expected": 2.4},
      "colorectal": {"min": 1.0, "max": 2.5, "expected": 1.6},
      "prostate": {"min": 1.5, "max": 3.0, "expected": 2.1}
    }
  }
}
```

### MSK Episode Performance Data (`data/msk_episodes.csv`)

Generate a CSV with the following columns and characteristics. **Plant the specific errors and anomalies described in the Planted Issues section below.**

**Columns:**
```
episode_type,episode_count,avg_episode_cost,target_price,total_cost,total_target,variance_pct,
implant_cost_avg,facility_cost_avg,professional_cost_avg,post_acute_cost_avg,readmission_cost_avg,
discharge_home_pct,discharge_snf_pct,discharge_irf_pct,discharge_other_pct,
readmission_rate,er_visit_rate_90d,ssi_rate,revision_rate_12mo,
avg_los_days,avg_opioid_mme_discharge,
prom_collection_rate,prom_improvement_rate,
prior_year_episode_count,prior_year_avg_cost,
risk_score_actual,risk_score_expected
```

**Episode Types and Realistic Baseline Values (before planting errors):**

| Episode Type | Count | Avg Cost | Target | Implant Avg | Facility Avg | Professional Avg | Post-Acute Avg | Readmit Cost Avg | Home% | SNF% | IRF% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TKR | 142 | $26,800 | $28,000 | $5,200 | $12,400 | $4,100 | $4,200 | $900 | 74% | 18% | 5% |
| THR | 95 | $25,400 | $27,500 | $5,800 | $11,800 | $3,900 | $3,100 | $800 | 71% | 20% | 6% |
| Spinal Fusion 1-2 | 58 | $41,200 | $38,000 | $8,500 | $18,200 | $6,800 | $6,400 | $1,300 | 58% | 28% | 10% |
| Spinal Fusion 3+ | 18 | $78,500 | $72,000 | $16,200 | $32,000 | $12,800 | $14,500 | $3,000 | 32% | 38% | 22% |
| Knee Arthroscopy | 205 | $8,400 | $9,200 | $350 | $4,800 | $2,200 | $800 | $250 | 95% | 2% | 0% |
| Rotator Cuff | 72 | $16,800 | $18,000 | $2,100 | $7,500 | $3,600 | $2,800 | $800 | 82% | 8% | 4% |
| Conservative LBP | 485 | $4,100 | $4,800 | N/A | $1,200 | $1,800 | $1,100 | N/A | N/A | N/A | N/A |
| Conservative Joint | 320 | $3,200 | $3,500 | N/A | $900 | $1,500 | $800 | N/A | N/A | N/A | N/A |

### MSK Quality Metrics (`data/msk_quality.csv`)

**Columns:**
```
measure_name,measure_id,numerator,denominator,rate,target,max_points,points_earned,
prior_year_rate,benchmark_50th,benchmark_90th
```

**Rows:**
| Measure | Numerator | Denominator | Rate | Target | Points |
|---|---|---|---|---|---|
| Surgical Site Infection (TKR/THR) | 5 | 237 | 2.1% | <2.0% | 8/10 |
| 90-Day Readmission (Surgical) | 18 | 390 | 4.6% | <5.0% | 10/10 |
| 90-Day ER Visit (Surgical) | 35 | 390 | 9.0% | <8.0% | 7/10 |
| 12-Month Revision Rate | 4 | 237 | 1.7% | <2.0% | 10/10 |
| PROM Collection Rate | 152 | 390 | 39.0% | >60% | 2/10 |
| PROM Clinically Meaningful Improvement | 98 | 152 | 64.5% | >70% | 6/10 |
| Opioid MME Compliance | 285 | 390 | 73.1% | >80% | 6/10 |
| Conservative Tx Attempted Pre-Surgery | 340 | 390 | 87.2% | >85% | 9/10 |
| Patient Satisfaction (Top Box) | 312 | 390 | 80.0% | >82% | 7/10 |
| Composite Quality Score | — | — | — | Min 60 | 65/90 |

### Oncology Episode Performance Data (`data/onc_episodes.csv`)

**Columns:**
```
cancer_type,stage_group,line_of_therapy,episode_count,avg_episode_cost,target_price,
total_cost,total_target,variance_pct,
drug_cost_avg,administration_cost_avg,inpatient_cost_avg,er_cost_avg,
imaging_cost_avg,lab_cost_avg,supportive_care_cost_avg,other_cost_avg,
pathway_adherence_rate,pathway_regimen_pct,non_pathway_regimen_pct,
biosimilar_utilization_rate,office_infusion_pct,hopd_infusion_pct,
hospitalization_rate,er_visit_rate,
prior_year_episode_count,prior_year_avg_cost,
risk_score_actual,risk_score_expected
```

**Episode Types and Baseline Values:**

| Cancer/Stage/Line | Count | Avg Cost | Target | Drug Avg | Admin Avg | IP Avg | ER Avg | Path Adh |
|---|---|---|---|---|---|---|---|---|
| Breast Early / 1L | 78 | $39,200 | $42,000 | $18,500 | $4,200 | $6,800 | $1,200 | 88% |
| Breast Metastatic / 1L | 32 | $102,400 | $95,000 | $62,000 | $8,500 | $14,200 | $3,800 | 72% |
| Breast Metastatic / 2L+ | 18 | $88,500 | $90,000 | $48,000 | $7,200 | $16,800 | $4,500 | 65% |
| Lung NSCLC / 1L | 42 | $118,300 | $105,000 | $72,000 | $9,800 | $18,500 | $4,200 | 68% |
| Lung NSCLC / 2L+ | 24 | $82,600 | $85,000 | $45,000 | $7,500 | $15,200 | $3,800 | 62% |
| Colorectal Adjuvant | 36 | $33,500 | $35,000 | $14,200 | $5,800 | $5,500 | $1,800 | 91% |
| Colorectal Metastatic / 1L | 28 | $92,800 | $82,000 | $52,000 | $8,200 | $15,600 | $4,200 | 64% |
| Prostate Early | 45 | $16,200 | $18,000 | $6,800 | $2,100 | $3,200 | $800 | 85% |
| Prostate Advanced / 1L | 22 | $72,400 | $68,000 | $42,000 | $6,500 | $11,200 | $2,800 | 74% |

### Oncology Quality & EOL Metrics (`data/onc_quality.csv`)

**Columns:**
```
measure_name,measure_id,numerator,denominator,rate,target,max_points,points_earned,
prior_year_rate,national_benchmark
```

**Rows:**
| Measure | Numerator | Denominator | Rate | Target | Points |
|---|---|---|---|---|---|
| Pathway Adherence (Overall) | 238 | 325 | 73.2% | >80% | 12/20 |
| Chemo Within 14 Days of Death | 11 | 58 | 19.0% | <10% | 3/10 |
| Hospice Enrollment (Decedents) | 25 | 58 | 43.1% | >55% | 5/10 |
| Hospice >7 Days Before Death | 16 | 58 | 27.6% | >40% | 3/10 |
| ICU Within 30 Days of Death | 14 | 58 | 24.1% | <15% | 2/10 |
| ER Within 30 Days of Death | 19 | 58 | 32.8% | <25% | 5/10 |
| Biomarker Testing (NSCLC) | 55 | 66 | 83.3% | >90% | 7/10 |
| Biosimilar Utilization | 42 | 85 | 49.4% | >70% | 4/10 |
| Advance Care Planning Documented | 145 | 325 | 44.6% | >65% | 4/10 |
| Patient Experience (Top Box) | 268 | 325 | 82.5% | >80% | 8/10 |
| Composite Quality Score | — | — | — | Min 55 | 53/100 |

### Oncology Drug Detail (`data/onc_drug_detail.csv`)

**Columns:**
```
drug_category,drug_name,is_biosimilar,is_pathway,cancer_types_used,
total_claims,total_cost,avg_cost_per_claim,
site_of_service_office_pct,site_of_service_hopd_pct,site_of_service_home_pct,
prior_year_total_cost,prior_year_claims,
fda_approval_date,is_novel_therapy
```

Include 15-20 drugs covering: pembrolizumab, nivolumab, trastuzumab (brand + biosimilar), bevacizumab (brand + biosimilar), rituximab (brand + biosimilar), FOLFOX components, CDK4/6 inhibitors (palbociclib, ribociclib), enzalutamide, docetaxel, carboplatin, paclitaxel, and 1-2 recently approved agents.

---

## Planted Issues (Critical — These Drive the Demo)

The simulated data must contain these specific, deliberate anomalies. Each is designed to demonstrate a different type of validation failure and diagnostic reasoning.

### MSK Contract — Planted Issues

**Issue 1: Spinal Fusion Cost Overrun with Implant Cost Anomaly**
- Spinal Fusion 1-2 level: avg cost $41,200 vs target $38,000 (+8.4%)
- Implant cost avg: $8,500 — this is at the high end but not flagged by range alone
- **However:** Set `risk_score_actual` to 1.08 and `risk_score_expected` to 1.05 — nearly identical. This means the overrun is NOT explained by case complexity
- **The real issue:** The implant cost for spinal fusion is 20.6% of total episode cost, compared to industry benchmark of 15-18%. Implant vendor pricing or surgeon preference for premium devices is the driver
- **Validation rule that catches this:** Cross-check implant cost as % of total episode cost against expected ratio by procedure type

**Issue 2: Knee Arthroscopy Volume Anomaly**
- 205 knee arthroscopy episodes for 6,200 attributed members = 33.1 per 1,000
- Expected MA rate: 15-25 per 1,000
- **The real issue:** Potential overutilization. Evidence shows arthroscopic debridement for knee OA (the most common MA-age indication) is clinically ineffective per multiple RCTs. High volume in an MA population likely indicates unnecessary procedures
- **Cross-validation:** Check arthroscopy volume vs conservative joint episode volume. Ratio of 205:320 (0.64:1) is high — expected is 0.3-0.4:1
- **Validation rule that catches this:** Episode volume per 1,000 attributed members vs expected range, plus arthroscopy-to-conservative ratio

**Issue 3: PROM Collection Rate Failure Threatening Quality Gate**
- PROM collection rate: 39% (target >60%)
- This earned only 2/10 quality points
- Composite quality score: 65/90 = 72.2% — passes the 60% gate
- **However:** If PROM collection were at 0% (total failure), composite drops to 63/90 = 70% — still passes. But this masks a real problem: with only 39% collection, the PROM improvement rate (64.5%) is measured on a biased sample. Compliant patients who return PROMs likely have better outcomes
- **The real issue:** The provider doesn't have a systematic PROM collection workflow. This isn't a care quality problem — it's an operational/data capture problem identical in nature to the MSSP quality failure pattern
- **Validation rule that catches this:** Flag any quality measure where collection/reporting rate is below 50%, and note that associated outcome measures are unreliable

**Issue 4: Discharge Disposition Shift — SNF to Home, but Readmission Rate Unchanged**
- TKR discharge to home: 74% (up from prior year 62%)
- TKR discharge to SNF: 18% (down from prior year 30%)
- TKR readmission rate: 4.6% (prior year: 4.8%)
- **This looks like a success story** — home discharge up, SNF down, readmissions stable
- **BUT:** Set `er_visit_rate_90d` for TKR to 14% (prior year: 8%)
- **The real issue:** Patients are being sent home earlier (good for cost), but some who previously would have gone to SNF are now showing up in the ER (masked by the aggregate readmission rate). The ER visits are not converting to readmissions (yet), but this is an early warning sign of inadequate home health support
- **Validation rule that catches this:** When discharge-to-home increases >10 percentage points AND ER visit rate increases >50%, flag potential inadequate post-acute support

**Issue 5: Conservative LBP Episode — Suspiciously Low Cost**
- Conservative LBP episodes: 485 at $4,100 average
- Prior year: 520 episodes at $4,600 average
- **Surface read:** Good — costs down 10.9%, volumes slightly down
- **BUT:** Set `prior_year_episode_count` for Spinal Fusion 1-2 to 42 (current year: 58)
- Spinal fusion volume increased 38% while conservative LBP decreased 6.7%
- **The real issue:** The conversion rate from conservative to surgical is increasing. Calculate implied conversion: if 35 "missing" conservative patients converted to surgery, that's a 7.2% conversion rate shift. The provider may be fast-tracking patients to surgery
- **Validation rule that catches this:** When conservative episode volumes decrease AND corresponding surgical volumes increase disproportionately, flag potential surgical pipeline acceleration

### Oncology Contract — Planted Issues

**Issue 6: Lung NSCLC 1L Cost Overrun Driven by Pathway Non-Adherence**
- NSCLC 1L: avg cost $118,300 vs target $105,000 (+12.7%)
- Pathway adherence: 68% (target: 72%)
- **Plant in drug detail:** Show that non-pathway NSCLC cases average $142,000 while pathway cases average $107,000
- **The real issue:** The 32% non-pathway cases are using pembrolizumab + chemotherapy + bevacizumab (a triplet regimen) when the pathway specifies pembrolizumab + chemotherapy (a doublet). The added bevacizumab costs ~$6,000-8,000 per cycle with marginal evidence in most NSCLC subtypes
- **Cross-validation:** If you back-calculate: (0.68 × $107,000) + (0.32 × $142,000) = $118,200 ≈ reported $118,300 ✓ — this confirms the non-pathway regimen is the cost driver
- **Validation rule that catches this:** For any episode type where cost exceeds target by >10%, check pathway adherence correlation. If pathway vs non-pathway cost differential is >25%, the pathway non-adherence is the primary cost driver

**Issue 7: Metastatic Breast — Biosimilar Utilization Failure**
- Breast metastatic 1L: avg cost $102,400 vs target $95,000 (+7.8%)
- Pathway adherence: 72% — below target but not terrible
- **Plant in drug detail:** Trastuzumab (brand Herceptin) has 65 claims at avg $4,800/claim. Trastuzumab biosimilar has only 8 claims at avg $2,200/claim
- Biosimilar utilization: 8/(65+8) = 11% — dramatically below the 50% minimum and 80% target
- **The real issue:** If all brand trastuzumab were switched to biosimilar, savings would be: 65 × ($4,800 - $2,200) = $169,000 across ~32 episodes = ~$5,280 per episode saved. This alone would close most of the $7,400/episode gap
- **Validation rule that catches this:** When biosimilar alternatives exist and brand utilization exceeds 50%, flag with calculated savings opportunity

**Issue 8: End-of-Life Metrics — Systemic Palliative Care Failure**
- Chemo within 14 days of death: 19% (target <10%)
- Hospice enrollment: 43.1% (target >55%)
- Hospice >7 days: 27.6% (target >40%)
- ICU within 30 days: 24.1% (target <15%)
- **All EOL metrics failing simultaneously**
- **Plant cross-validation data:** Advance care planning documentation rate: 44.6% (target >65%)
- **The real issue:** This is not random — it's a systemic failure to integrate palliative care. The advance care planning rate predicts the EOL failures. Without goals-of-care conversations, patients default to aggressive treatment. This is both the highest-quality and highest-cost issue in the report
- **Validation rule that catches this:** When ≥3 EOL metrics fail simultaneously AND advance care planning is below 50%, flag as systemic palliative care integration failure (not individual measure failure)

**Issue 9: Colorectal Metastatic — Cost Overrun with Hidden Site-of-Service Problem**
- Colorectal metastatic: avg cost $92,800 vs target $82,000 (+13.2%)
- Pathway adherence: 64% — below target
- **Plant in drug detail:** Bevacizumab (brand Avastin) claims show 85% administered at HOPD, only 15% in physician office
- **The real issue:** Even if pathway adherence improved, the site-of-service mix is adding $3,000-5,000 per infusion episode in facility fees. HOPD infusion of bevacizumab can cost 2-3× physician office administration. Combined with low biosimilar utilization for bevacizumab (plant: 35% biosimilar rate), there are two compounding cost drivers
- **Validation rule that catches this:** For drug categories with >$2,000 average cost per claim, check site-of-service distribution. If HOPD >60% for drugs that can be safely administered in office, flag with estimated savings

**Issue 10: Quality Gate Failure — Composite Just Below Threshold**
- Composite quality score: 53/100 (gate minimum: 55)
- **This means the entire shared savings calculation may be zeroed out**
- Total reported savings across all oncology episodes: approximately $180,000
- Provider share at 50%: $90,000
- **BUT:** Quality gate failure → $0 payout
- **The real issue:** The provider is 2 points away from the gate. The cheapest path to those 2 points is biosimilar utilization (currently 4/10, could reach 6/10 with moderate improvement) or biomarker testing (currently 7/10, could reach 9/10 with process improvement)
- **Validation rule that catches this:** When quality composite is within 5 points of the gate threshold, identify the lowest-effort measures that could close the gap, and calculate the financial impact of quality gate passage vs failure

---

## Validation Module Specifications

### `validation/schema.py`

**Purpose:** Verify data structure integrity before any calculations.

**Checks:**
1. **Column presence:** All expected columns exist in each CSV
2. **Data types:** Numeric columns are numeric (not strings), rate columns are 0-1 or 0-100 (detect and normalize), count columns are integers
3. **Null/missing values:** Flag any null in critical fields (episode_count, avg_episode_cost, target_price). Warn on nulls in non-critical fields
4. **Value constraints:** episode_count >= 0, costs >= 0, rates between 0 and 1 (after normalization), percentages sum to ~100% where applicable (discharge dispositions)

**Output format:**
```python
@dataclass
class ValidationFlag:
    flag_id: str              # e.g., "SCHEMA-001"
    severity: str             # "RED", "YELLOW", "GREEN"
    category: str             # "schema", "arithmetic", "range", "cross_metric", "specialty"
    metric_name: str          # The specific field/metric flagged
    metric_value: Any         # The actual value
    expected_value: Any       # What was expected (range, exact, or description)
    episode_type: str         # Which episode/cancer type this relates to
    contract_id: str          # Which contract
    description: str          # Plain English description of the issue
    detail: str               # More detailed technical explanation
    related_metrics: dict     # Other metrics relevant to this flag
```

### `validation/arithmetic.py`

**Purpose:** Verify internal mathematical consistency.

**Checks:**
1. **Episode cost reconciliation:** `episode_count × avg_episode_cost ≈ total_cost` (within 1% tolerance for rounding)
2. **Target reconciliation:** `episode_count × target_price ≈ total_target`
3. **Variance calculation:** `(avg_episode_cost - target_price) / target_price ≈ variance_pct`
4. **Cost component sum:** `implant_cost_avg + facility_cost_avg + professional_cost_avg + post_acute_cost_avg + readmission_cost_avg ≈ avg_episode_cost` (within 5% — some costs may be uncategorized)
5. **Discharge disposition sum:** `discharge_home_pct + discharge_snf_pct + discharge_irf_pct + discharge_other_pct ≈ 1.0` (within 2%)
6. **Quality score arithmetic:** `points_earned / max_points × 100 ≈ composite score` (verify the composite is correctly calculated from components)
7. **Rate calculation:** `numerator / denominator ≈ rate` for quality measures
8. **Member month check:** For contract-level totals, verify `attributed_members × 12 ≈ member_months` (within 5% for mid-year enrollment changes)

### `validation/range_checks.py`

**Purpose:** Flag values outside expected ranges for the specialty and LOB.

**Implementation:**
- Load reference ranges from `config/reference_ranges.json`
- For each metric in each episode row, compare against the appropriate range
- **Severity logic:**
  - GREEN: Within expected range
  - YELLOW: Outside expected range but within min/max bounds
  - RED: Outside min/max bounds entirely
- **Special handling:** Conservative episodes don't have implant costs, discharge dispositions, or readmission metrics — skip those checks with N/A notation

### `validation/cross_metric.py`

**Purpose:** Check that combinations of metrics tell a consistent story.

**Checks:**
1. **Discharge shift + ER correlation (MSK):** If discharge-to-home increased >10pp YoY AND ER visits increased >50% YoY, flag
2. **Risk score vs benchmark calibration:** If |risk_score_actual - risk_score_expected| / risk_score_expected > 0.10, flag with explanation of benchmark validity concern
3. **Pathway adherence vs cost (Oncology):** For episodes where adherence <75% and cost >target, calculate pathway vs non-pathway cost split to verify adherence is the driver
4. **EOL metric clustering (Oncology):** If ≥3 of 5 EOL metrics fail, flag as systemic issue rather than individual measure failures
5. **Volume vs population (both):** Episode count / (attributed_members / 1000) should be within expected incidence range
6. **Conservative-to-surgical pipeline (MSK):** If conservative volumes decrease AND surgical volumes increase YoY for related procedures, flag potential surgical acceleration
7. **Quality gate proximity:** If composite score is within 5 points of the minimum gate, flag with financial impact calculation
8. **Biosimilar × site-of-service compounding (Oncology):** If a drug has both low biosimilar utilization AND high HOPD percentage, calculate combined savings opportunity

### `validation/msk_rules.py`

**Purpose:** MSK-specific clinical and financial validation logic.

**Rules:**
1. **Implant cost ratio:** For each surgical episode type, calculate implant cost as % of total episode cost. Flag if >20% for joints or >25% for spine (above industry benchmarks)
2. **Arthroscopy volume reasonableness:** Calculate arthroscopies per 1,000 attributed members. Flag if >25 per 1,000 for MA population (potential overutilization, especially given evidence against arthroscopic debridement for OA)
3. **Arthroscopy-to-conservative ratio:** Calculate ratio of arthroscopy episodes to conservative joint episodes. Flag if >0.5:1
4. **Post-acute cost efficiency:** For TKR/THR, calculate post-acute cost as % of total. Compare against benchmark. If >20% of total cost, flag
5. **Opioid prescribing:** Flag if avg MME at discharge exceeds 50 (CDC guideline-informed threshold)
6. **PROM reliability:** If PROM collection rate <50%, flag all PROM-derived quality measures as unreliable with explanation
7. **Spinal fusion level distribution:** If spinal fusion 3+ cases are >30% of total fusions, flag as potential case complexity concern requiring risk adjustment review

### `validation/onc_rules.py`

**Purpose:** Oncology-specific clinical and financial validation logic.

**Rules:**
1. **Pathway adherence cost correlation:** For each cancer type, if pathway adherence is below target AND cost exceeds target, back-calculate the cost split: `(adherence% × pathway_cost) + ((1-adherence%) × non_pathway_cost) = avg_cost`. Solve for implied non-pathway cost and flag the differential
2. **Biosimilar savings calculator:** For each drug with a biosimilar alternative, calculate: `brand_claims × (brand_avg_cost - biosimilar_avg_cost) = potential savings`. Express as per-episode impact
3. **Site-of-service savings calculator:** For drugs with >$2,000/claim avg cost, if HOPD% >60%, calculate: `hopd_claims × (hopd_cost - office_cost) = excess facility costs`. Use industry ratio of HOPD = 2× office for estimation
4. **EOL-ACP correlation:** If advance care planning rate is below 50% AND ≥3 EOL metrics fail, identify as root cause rather than symptom
5. **Novel therapy impact:** If any drug flagged as `is_novel_therapy` AND `fda_approval_date` is within the contract's novel therapy lookback period, note that costs may be carved out per contract terms and recalculate savings/losses excluding those claims
6. **Episode volume vs incidence:** For each cancer type, calculate episodes per 1,000 members and compare against expected incidence. Flag if >2× expected (potential attribution problem) or <0.5× expected (potential underdiagnosis or access problem)
7. **Quality gate financial impact:** When composite is within 5 points of gate, calculate: total potential savings × sharing_rate = amount at risk. Identify which 1-2 measures are cheapest to improve (closest to next point threshold with highest point values)

---

## AI Diagnostic Layer Specifications

### `diagnosis/ai_diagnostics.py`

**Purpose:** Send validation flags with context to Claude API and receive structured narrative diagnostics.

**Implementation:**

```python
import json
import requests  # or anthropic SDK if available

def generate_diagnostics(flags: list[ValidationFlag], contract_metadata: dict, report_data: dict) -> list[DiagnosticNarrative]:
    """
    Groups related flags, constructs prompts, calls Claude API,
    and parses structured responses.
    """
```

**Key Design Decisions:**
1. **Batch related flags:** Group flags by episode type/cancer type before sending to API. Don't send one flag at a time — the diagnostic value comes from seeing patterns across flags
2. **Include context:** Each API call should include the contract metadata, the specific flagged metrics AND their related unflagged metrics (Claude needs the full picture to diagnose)
3. **Structured output:** Prompt Claude to respond in JSON format with specific fields (see below)
4. **Fallback:** If API is unavailable, the validation report still works — just without narrative sections. Print a clear notice that AI diagnostics were skipped

**API Call Structure:**
- Model: `claude-sonnet-4-20250514`
- Max tokens: 1500 per call (sufficient for a focused diagnostic narrative)
- System prompt: Include role context (Carelon provider economics analyst), the specific contract type, and output format requirements
- Temperature: Low/default (we want consistent, reliable analysis)

**Prompt Template (in `prompt_templates.py`):**

```python
DIAGNOSTIC_PROMPT = """
You are a senior analyst on the Provider Economics team at Carelon (Elevance Health),
reviewing a VBC performance report for a {specialty} specialty contract.

Contract: {contract_name}
Type: {contract_type}
LOB: {lob}
Performance Period: {performance_period}
Attribution: {attributed_members} members

The automated validation system has flagged the following issues for {episode_type}:

{formatted_flags}

Additional context — full metrics for this episode type:
{formatted_metrics}

Prior year comparison:
{formatted_prior_year}

Respond in JSON with this exact structure:
{{
  "diagnosis_summary": "2-3 sentence summary of the most likely root cause",
  "probable_causes": [
    {{
      "cause": "description",
      "likelihood": "high/medium/low",
      "evidence": "which specific metrics support this"
    }}
  ],
  "questions_for_provider": [
    "Specific question to ask at JOC meeting"
  ],
  "recommended_interventions": [
    {{
      "intervention": "description",
      "timeframe": "immediate/short-term/contract-renewal",
      "expected_impact": "estimated financial or quality impact"
    }}
  ],
  "contract_implications": "How this affects shared savings/losses and what contract amendments to consider"
}}
"""
```

**Output dataclass:**

```python
@dataclass
class DiagnosticNarrative:
    episode_type: str
    diagnosis_summary: str
    probable_causes: list[dict]
    questions_for_provider: list[str]
    recommended_interventions: list[dict]
    contract_implications: str
    flags_addressed: list[str]  # flag_ids this narrative covers
```

---

## HTML Report Specifications

### Structure

The HTML report should be a single self-contained file (all CSS inline or in `<style>` tags) with the following sections:

**1. Executive Summary Header**
- Contract name, type, LOB, performance period
- Overall financial result (total savings/losses, provider share, quality gate status)
- Headline assessment: 1-2 sentences on the contract's overall status
- Severity summary: count of RED/YELLOW/GREEN flags

**2. Financial Overview**
- Summary table: total episodes, total cost, total target, variance, implied shared savings/losses
- Quality gate status with composite score and threshold
- If quality gate fails: calculate forfeited savings amount prominently

**3. Episode-Level Detail (one section per episode type)**
- Metrics table with current values, targets, prior year, and variance
- Validation flags for this episode type, color-coded by severity
- AI diagnostic narrative (if available)
- Recommended actions

**4. Quality Scorecard**
- Full quality measure table with numerators, denominators, rates, targets, and points
- Flags on failing measures
- Quality gate analysis (if near threshold, show path to passing)

**5. Drug Detail (oncology only)**
- Top drugs by spend
- Biosimilar utilization analysis
- Site-of-service analysis

**6. Cross-Cutting Findings**
- Flags that span multiple episode types or represent systemic issues
- AI narrative on systemic patterns

**7. Appendix — Validation Log**
- Complete list of all flags (RED, YELLOW, GREEN) with technical detail
- Useful for data team review, not for JOC presentation

### Visual Design

- Clean, professional, healthcare-appropriate styling
- Color coding: RED (#DC3545), YELLOW/AMBER (#FFC107), GREEN (#28A745)
- Severity badges as colored pills/labels
- Tables with alternating row shading
- Key metrics highlighted in summary cards at the top
- Expandable/collapsible sections for detail (use `<details>` HTML elements)
- Print-friendly (no dark backgrounds, reasonable font sizes)
- Carelon-inspired color palette for headers/accents: use professional blues (#003366, #0066CC) and grays

---

## Implementation Notes

### Python Dependencies

```
pandas
jinja2
requests (for Claude API calls)
```

No heavy frameworks needed. This is deliberately lightweight — it should run with `python main.py` after installing requirements.

### main.py Flow

```python
def main():
    # 1. Load config
    contract_metadata = load_json("config/contract_metadata.json")
    reference_ranges = load_json("config/reference_ranges.json")

    # 2. Load data
    msk_episodes = pd.read_csv("data/msk_episodes.csv")
    msk_quality = pd.read_csv("data/msk_quality.csv")
    onc_episodes = pd.read_csv("data/onc_episodes.csv")
    onc_quality = pd.read_csv("data/onc_quality.csv")
    onc_drugs = pd.read_csv("data/onc_drug_detail.csv")

    # 3. Run validation pipeline for each contract
    all_flags = []

    # MSK validation
    msk_contract = get_contract(contract_metadata, "MSK-2024-001")
    all_flags += validate_schema(msk_episodes, "msk_episodes", msk_contract)
    all_flags += validate_schema(msk_quality, "msk_quality", msk_contract)
    all_flags += validate_arithmetic(msk_episodes, msk_quality, msk_contract)
    all_flags += validate_ranges(msk_episodes, reference_ranges["msk"], msk_contract)
    all_flags += validate_cross_metrics(msk_episodes, msk_quality, msk_contract)
    all_flags += validate_msk_rules(msk_episodes, msk_quality, reference_ranges["msk"], msk_contract)

    # Oncology validation
    onc_contract = get_contract(contract_metadata, "ONC-2024-001")
    all_flags += validate_schema(onc_episodes, "onc_episodes", onc_contract)
    all_flags += validate_schema(onc_quality, "onc_quality", onc_contract)
    all_flags += validate_arithmetic(onc_episodes, onc_quality, onc_contract)
    all_flags += validate_ranges(onc_episodes, reference_ranges["oncology"], onc_contract)
    all_flags += validate_cross_metrics(onc_episodes, onc_quality, onc_contract)
    all_flags += validate_onc_rules(onc_episodes, onc_quality, onc_drugs, reference_ranges["oncology"], onc_contract)

    # 4. AI diagnostics (optional — graceful failure if API unavailable)
    diagnostics = []
    try:
        diagnostics = generate_all_diagnostics(all_flags, contract_metadata, {
            "msk_episodes": msk_episodes,
            "msk_quality": msk_quality,
            "onc_episodes": onc_episodes,
            "onc_quality": onc_quality,
            "onc_drugs": onc_drugs
        })
    except Exception as e:
        print(f"AI diagnostics unavailable: {e}. Proceeding with validation-only report.")

    # 5. Generate reports
    generate_html_report(
        all_flags, diagnostics, contract_metadata,
        msk_episodes, msk_quality,
        onc_episodes, onc_quality, onc_drugs,
        output_path="output/vbc_validation_report.html"
    )

    print(f"Report generated: output/vbc_validation_report.html")
    print(f"Total flags: {len(all_flags)} (RED: {count_severity(all_flags, 'RED')}, "
          f"YELLOW: {count_severity(all_flags, 'YELLOW')}, GREEN: {count_severity(all_flags, 'GREEN')})")
```

### Error Handling & Robustness

- If any CSV fails to load, skip that contract's validation and note in the report
- If Claude API is unavailable (network disabled, rate limited, etc.), generate report without narrative sections — include a clear notice
- If a validation check encounters unexpected data (e.g., division by zero), catch the exception, log it as a RED flag with the error details, and continue
- All flag IDs should be unique and sequential within categories (e.g., SCHEMA-001, ARITH-001, RANGE-001, CROSS-001, MSK-001, ONC-001)

### Demo Script

Include a `README.md` with:
1. Install instructions (`pip install -r requirements.txt`)
2. How to run (`python main.py`)
3. How to set the Claude API key (`export ANTHROPIC_API_KEY=...`) with note that AI diagnostics are optional
4. What to expect in the output
5. Brief explanation of the architecture for interview discussion

---

## Interview Talking Points (for reference, not part of the code)

When demoing this tool, emphasize:

1. **Pipeline architecture:** "I structured this the way you'd structure a validation step in a production ETL pipeline — modular rules that can be extended, clear separation of data loading, validation, and reporting. In production, the CSV inputs would be Snowflake queries, and the reference ranges would come from Carelon's benchmark tables."

2. **Data quality first:** "The schema and arithmetic checks run before any business logic. If the upstream data is broken, you need to know that before you start interpreting performance. This is the same principle as using Great Expectations in a data pipeline."

3. **Domain-specific rules:** "The MSK rules encode things like implant cost ratios and arthroscopy appropriateness criteria. The oncology rules encode pathway adherence correlations and biosimilar savings calculations. These aren't generic statistical outlier detectors — they're clinical and financial rules that a provider economics analyst would apply manually, now codified."

4. **AI integration philosophy:** "The deterministic layer tells you WHAT is wrong. The AI layer tells you WHY it might be wrong and WHAT to do about it. If the AI is unavailable, you still have a complete validation report. The AI adds interpretive value but isn't a dependency."

5. **Scalability:** "Adding a new specialty — say behavioral health or cardiology — means adding a new rules file and reference range section. The schema, arithmetic, and cross-metric checks are generic. The reporting template adapts automatically."
