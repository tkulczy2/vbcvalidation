"""Schema validation: columns, types, nulls, value constraints."""

import pandas as pd
import numpy as np
from validation import ValidationFlag

# Expected columns per dataset
EXPECTED_COLUMNS = {
    "msk_episodes": [
        "episode_type", "episode_count", "avg_episode_cost", "target_price",
        "total_cost", "total_target", "variance_pct", "implant_cost_avg",
        "facility_cost_avg", "professional_cost_avg", "post_acute_cost_avg",
        "readmission_cost_avg", "discharge_home_pct", "discharge_snf_pct",
        "discharge_irf_pct", "discharge_other_pct", "readmission_rate",
        "er_visit_rate_90d", "ssi_rate", "revision_rate_12mo", "avg_los_days",
        "avg_opioid_mme_discharge", "prom_collection_rate", "prom_improvement_rate",
        "prior_year_episode_count", "prior_year_avg_cost",
        "risk_score_actual", "risk_score_expected",
    ],
    "msk_quality": [
        "measure_name", "measure_id", "numerator", "denominator", "rate",
        "target", "max_points", "points_earned", "prior_year_rate",
        "benchmark_50th", "benchmark_90th",
    ],
    "onc_episodes": [
        "cancer_type", "stage_group", "line_of_therapy", "episode_count",
        "avg_episode_cost", "target_price", "total_cost", "total_target",
        "variance_pct", "drug_cost_avg", "administration_cost_avg",
        "inpatient_cost_avg", "er_cost_avg", "imaging_cost_avg", "lab_cost_avg",
        "supportive_care_cost_avg", "other_cost_avg", "pathway_adherence_rate",
        "pathway_regimen_pct", "non_pathway_regimen_pct",
        "biosimilar_utilization_rate", "office_infusion_pct", "hopd_infusion_pct",
        "hospitalization_rate", "er_visit_rate", "prior_year_episode_count",
        "prior_year_avg_cost", "risk_score_actual", "risk_score_expected",
    ],
    "onc_quality": [
        "measure_name", "measure_id", "numerator", "denominator", "rate",
        "target", "max_points", "points_earned", "prior_year_rate",
        "national_benchmark",
    ],
    "onc_drug_detail": [
        "drug_category", "drug_name", "is_biosimilar", "is_pathway",
        "cancer_types_used", "total_claims", "total_cost", "avg_cost_per_claim",
        "site_of_service_office_pct", "site_of_service_hopd_pct",
        "site_of_service_home_pct", "prior_year_total_cost", "prior_year_claims",
        "fda_approval_date", "is_novel_therapy",
    ],
}

# Critical fields that must not be null
CRITICAL_FIELDS = {
    "msk_episodes": ["episode_type", "episode_count", "avg_episode_cost", "target_price"],
    "msk_quality": ["measure_name", "measure_id", "max_points", "points_earned"],
    "onc_episodes": ["cancer_type", "episode_count", "avg_episode_cost", "target_price"],
    "onc_quality": ["measure_name", "measure_id", "max_points", "points_earned"],
    "onc_drug_detail": ["drug_name", "total_claims", "total_cost"],
}

# Numeric columns
NUMERIC_COLUMNS = {
    "msk_episodes": [
        "episode_count", "avg_episode_cost", "target_price", "total_cost",
        "total_target", "variance_pct", "implant_cost_avg", "facility_cost_avg",
        "professional_cost_avg", "post_acute_cost_avg", "readmission_cost_avg",
        "discharge_home_pct", "discharge_snf_pct", "discharge_irf_pct",
        "discharge_other_pct", "readmission_rate", "er_visit_rate_90d",
        "ssi_rate", "revision_rate_12mo", "avg_los_days",
        "avg_opioid_mme_discharge", "prom_collection_rate",
        "prom_improvement_rate", "prior_year_episode_count",
        "prior_year_avg_cost", "risk_score_actual", "risk_score_expected",
    ],
    "onc_episodes": [
        "episode_count", "avg_episode_cost", "target_price", "total_cost",
        "total_target", "variance_pct", "drug_cost_avg", "administration_cost_avg",
        "inpatient_cost_avg", "er_cost_avg", "imaging_cost_avg", "lab_cost_avg",
        "supportive_care_cost_avg", "other_cost_avg", "pathway_adherence_rate",
        "pathway_regimen_pct", "non_pathway_regimen_pct",
        "biosimilar_utilization_rate", "office_infusion_pct", "hopd_infusion_pct",
        "hospitalization_rate", "er_visit_rate", "prior_year_episode_count",
        "prior_year_avg_cost", "risk_score_actual", "risk_score_expected",
    ],
}

# Rate columns that should be 0-1
RATE_COLUMNS = {
    "msk_episodes": [
        "discharge_home_pct", "discharge_snf_pct", "discharge_irf_pct",
        "discharge_other_pct", "readmission_rate", "er_visit_rate_90d",
        "ssi_rate", "revision_rate_12mo", "prom_collection_rate",
        "prom_improvement_rate",
    ],
    "onc_episodes": [
        "pathway_adherence_rate", "pathway_regimen_pct", "non_pathway_regimen_pct",
        "biosimilar_utilization_rate", "office_infusion_pct", "hopd_infusion_pct",
        "hospitalization_rate", "er_visit_rate",
    ],
}

_schema_counter = 0


def _next_id():
    global _schema_counter
    _schema_counter += 1
    return f"SCHEMA-{_schema_counter:03d}"


def validate_schema(df: pd.DataFrame, dataset_name: str, contract: dict) -> list[ValidationFlag]:
    """Run all schema validation checks on a dataframe."""
    global _schema_counter
    flags = []
    contract_id = contract["contract_id"]

    # 1. Column presence
    expected = EXPECTED_COLUMNS.get(dataset_name, [])
    actual = list(df.columns)
    missing = [c for c in expected if c not in actual]
    extra = [c for c in actual if c not in expected]

    for col in missing:
        flags.append(ValidationFlag(
            flag_id=_next_id(), severity="RED", category="schema",
            metric_name=col, metric_value="MISSING",
            expected_value="Column should exist",
            episode_type="ALL", contract_id=contract_id,
            description=f"Missing expected column '{col}' in {dataset_name}",
            detail=f"The column '{col}' is expected in {dataset_name} but was not found. "
                   f"Available columns: {actual}",
        ))

    for col in extra:
        flags.append(ValidationFlag(
            flag_id=_next_id(), severity="YELLOW", category="schema",
            metric_name=col, metric_value="EXTRA",
            expected_value="Column not expected",
            episode_type="ALL", contract_id=contract_id,
            description=f"Unexpected column '{col}' in {dataset_name}",
            detail=f"The column '{col}' exists in {dataset_name} but is not in the expected schema.",
        ))

    # 2. Data types — check numeric columns are numeric
    numeric_cols = NUMERIC_COLUMNS.get(dataset_name, [])
    for col in numeric_cols:
        if col not in df.columns:
            continue
        non_null = df[col].dropna()
        if len(non_null) == 0:
            continue
        if not pd.api.types.is_numeric_dtype(non_null):
            flags.append(ValidationFlag(
                flag_id=_next_id(), severity="RED", category="schema",
                metric_name=col, metric_value=str(non_null.dtype),
                expected_value="numeric",
                episode_type="ALL", contract_id=contract_id,
                description=f"Column '{col}' in {dataset_name} is not numeric",
                detail=f"Expected numeric type but found {non_null.dtype}. "
                       f"Sample values: {list(non_null.head(3))}",
            ))

    # 3. Null/missing values in critical fields
    critical = CRITICAL_FIELDS.get(dataset_name, [])
    for col in critical:
        if col not in df.columns:
            continue
        null_count = df[col].isna().sum()
        if null_count > 0:
            flags.append(ValidationFlag(
                flag_id=_next_id(), severity="RED", category="schema",
                metric_name=col, metric_value=f"{null_count} nulls",
                expected_value="No nulls in critical field",
                episode_type="ALL", contract_id=contract_id,
                description=f"Critical field '{col}' has {null_count} null value(s) in {dataset_name}",
                detail=f"The field '{col}' is critical for downstream calculations and should not be null.",
            ))

    # 4. Value constraints
    # episode_count >= 0
    if "episode_count" in df.columns:
        neg = df[df["episode_count"] < 0]
        if len(neg) > 0:
            flags.append(ValidationFlag(
                flag_id=_next_id(), severity="RED", category="schema",
                metric_name="episode_count", metric_value=list(neg["episode_count"]),
                expected_value=">= 0",
                episode_type="ALL", contract_id=contract_id,
                description=f"Negative episode counts found in {dataset_name}",
                detail=f"Found {len(neg)} rows with negative episode counts.",
            ))

    # costs >= 0
    cost_cols = [c for c in df.columns if "cost" in c.lower() and c in numeric_cols]
    for col in cost_cols:
        if col not in df.columns:
            continue
        neg = df[df[col].notna() & (df[col] < 0)]
        if len(neg) > 0:
            flags.append(ValidationFlag(
                flag_id=_next_id(), severity="RED", category="schema",
                metric_name=col, metric_value=f"{len(neg)} negative values",
                expected_value=">= 0",
                episode_type="ALL", contract_id=contract_id,
                description=f"Negative cost values in '{col}' in {dataset_name}",
                detail=f"Cost fields should not be negative. Found {len(neg)} negative values.",
            ))

    # Rate columns between 0 and 1
    rate_cols = RATE_COLUMNS.get(dataset_name, [])
    for col in rate_cols:
        if col not in df.columns:
            continue
        non_null = df[col].dropna()
        if len(non_null) == 0:
            continue
        out_of_range = non_null[(non_null < 0) | (non_null > 1)]
        if len(out_of_range) > 0:
            # Check if they might be percentages (0-100 scale)
            if non_null.max() > 1 and non_null.max() <= 100:
                flags.append(ValidationFlag(
                    flag_id=_next_id(), severity="YELLOW", category="schema",
                    metric_name=col, metric_value=f"max={non_null.max()}",
                    expected_value="0-1 proportion scale",
                    episode_type="ALL", contract_id=contract_id,
                    description=f"Rate column '{col}' appears to be on 0-100 scale instead of 0-1",
                    detail=f"Values range from {non_null.min()} to {non_null.max()}. "
                           f"These may need to be divided by 100.",
                ))
            else:
                flags.append(ValidationFlag(
                    flag_id=_next_id(), severity="RED", category="schema",
                    metric_name=col, metric_value=f"{len(out_of_range)} values outside [0,1]",
                    expected_value="Between 0 and 1",
                    episode_type="ALL", contract_id=contract_id,
                    description=f"Rate column '{col}' has values outside valid range in {dataset_name}",
                    detail=f"Found {len(out_of_range)} values outside [0,1] range.",
                ))

    # Discharge dispositions sum to ~100% for surgical episodes
    if dataset_name == "msk_episodes":
        disp_cols = ["discharge_home_pct", "discharge_snf_pct", "discharge_irf_pct", "discharge_other_pct"]
        if all(c in df.columns for c in disp_cols):
            for _, row in df.iterrows():
                ep_type = row.get("episode_type", "unknown")
                if "Conservative" in str(ep_type):
                    continue
                vals = [row[c] for c in disp_cols if pd.notna(row[c])]
                if vals:
                    total = sum(vals)
                    if abs(total - 1.0) > 0.02:
                        flags.append(ValidationFlag(
                            flag_id=_next_id(), severity="YELLOW", category="schema",
                            metric_name="discharge_disposition_sum",
                            metric_value=round(total, 4),
                            expected_value="~1.0 (within 2%)",
                            episode_type=ep_type, contract_id=contract_id,
                            description=f"Discharge dispositions sum to {total:.1%} for {ep_type}",
                            detail=f"Expected discharge percentages to sum to ~100%. "
                                   f"Values: {dict(zip(disp_cols, vals))}",
                        ))

    # All checks passed notification
    if not flags:
        flags.append(ValidationFlag(
            flag_id=_next_id(), severity="GREEN", category="schema",
            metric_name="schema_check", metric_value="PASS",
            expected_value="All checks pass",
            episode_type="ALL", contract_id=contract_id,
            description=f"Schema validation passed for {dataset_name}",
            detail=f"All {len(expected)} expected columns present, types correct, "
                   f"no critical nulls, constraints satisfied.",
        ))

    return flags
