"""Range checks: flag values outside expected ranges for specialty and LOB."""

import pandas as pd
import numpy as np
from validation import ValidationFlag

_range_counter = 0


def _next_id():
    global _range_counter
    _range_counter += 1
    return f"RANGE-{_range_counter:03d}"


# Maps episode_type strings to reference range keys
MSK_EP_TYPE_MAP = {
    "TKR": "TKR",
    "THR": "THR",
    "Spinal Fusion 1-2": "spinal_fusion_1_2",
    "Spinal Fusion 3+": "spinal_fusion_3_plus",
    "Knee Arthroscopy": "knee_arthroscopy",
    "Rotator Cuff": "rotator_cuff",
    "Conservative LBP": "conservative_lbp",
    "Conservative Joint": "conservative_joint",
}

ONC_EP_TYPE_MAP = {
    ("Breast", "Early", "1L"): "breast_early",
    ("Breast", "Metastatic", "1L"): "breast_metastatic",
    ("Breast", "Metastatic", "2L+"): "breast_metastatic",
    ("Lung", "NSCLC", "1L"): "lung_nsclc_1L",
    ("Lung", "NSCLC", "2L+"): "lung_nsclc_2L_plus",
    ("Colorectal", "Adjuvant", "1L"): "colorectal_adjuvant",
    ("Colorectal", "Metastatic", "1L"): "colorectal_metastatic",
    ("Prostate", "Early", "1L"): "prostate_early",
    ("Prostate", "Advanced", "1L"): "prostate_advanced",
}


def _check_range(value, range_def, metric_name, ep_label, contract_id):
    """Check a single value against a range definition. Returns a flag or None."""
    if pd.isna(value):
        return None

    min_val = range_def.get("min", range_def.get("min_acceptable"))
    max_val = range_def.get("max", range_def.get("max_acceptable"))
    expected = range_def.get("expected", range_def.get("target"))
    desc = range_def.get("description", "")

    if min_val is not None and max_val is not None:
        if value < min_val or value > max_val:
            return ValidationFlag(
                flag_id=_next_id(), severity="RED", category="range",
                metric_name=metric_name, metric_value=value,
                expected_value=f"range [{min_val}, {max_val}], expected ~{expected}",
                episode_type=ep_label, contract_id=contract_id,
                description=f"{metric_name} = {value} is outside bounds [{min_val}, {max_val}] "
                            f"for {ep_label}",
                detail=f"{desc}. Value {value} falls outside the acceptable range. "
                       f"Expected approximately {expected}.",
            )
        elif expected is not None:
            # Within bounds but check if it's far from expected
            range_width = max_val - min_val
            if range_width > 0:
                deviation = abs(value - expected) / range_width
                if deviation > 0.4:
                    return ValidationFlag(
                        flag_id=_next_id(), severity="YELLOW", category="range",
                        metric_name=metric_name, metric_value=value,
                        expected_value=f"expected ~{expected} (range [{min_val}, {max_val}])",
                        episode_type=ep_label, contract_id=contract_id,
                        description=f"{metric_name} = {value} is within bounds but "
                                    f"significantly deviates from expected {expected} for {ep_label}",
                        detail=f"{desc}. Value is within [{min_val}, {max_val}] but "
                               f"deviates {deviation:.0%} of range width from expected.",
                    )
    elif max_val is not None:
        # Target-based (like quality targets with max_acceptable)
        target = range_def.get("target")
        if value > max_val:
            return ValidationFlag(
                flag_id=_next_id(), severity="RED", category="range",
                metric_name=metric_name, metric_value=value,
                expected_value=f"target {target}, max acceptable {max_val}",
                episode_type=ep_label, contract_id=contract_id,
                description=f"{metric_name} = {value} exceeds maximum acceptable "
                            f"{max_val} for {ep_label}",
                detail=f"{desc}. Target is {target}, maximum acceptable is {max_val}.",
            )
        elif target is not None and value > target:
            return ValidationFlag(
                flag_id=_next_id(), severity="YELLOW", category="range",
                metric_name=metric_name, metric_value=value,
                expected_value=f"target {target}",
                episode_type=ep_label, contract_id=contract_id,
                description=f"{metric_name} = {value} exceeds target {target} "
                            f"for {ep_label} (but within acceptable range)",
                detail=f"{desc}. Value exceeds target but remains below "
                       f"maximum acceptable threshold of {max_val}.",
            )

    return None


def validate_ranges(episodes_df: pd.DataFrame, reference_ranges: dict,
                    contract: dict) -> list[ValidationFlag]:
    """Check all metrics against reference ranges."""
    flags = []
    contract_id = contract["contract_id"]
    specialty = contract.get("specialty", "")

    cost_ranges = reference_ranges.get("episode_cost_ranges", {})

    if specialty == "MSK":
        for _, row in episodes_df.iterrows():
            ep_type = row.get("episode_type", "")
            ref_key = MSK_EP_TYPE_MAP.get(ep_type)
            if ref_key and ref_key in cost_ranges:
                flag = _check_range(
                    row.get("avg_episode_cost"), cost_ranges[ref_key],
                    "avg_episode_cost", ep_type, contract_id
                )
                if flag:
                    flags.append(flag)

        # Check utilization ranges
        util_ranges = reference_ranges.get("utilization_ranges_ma", {})
        for _, row in episodes_df.iterrows():
            ep_type = row.get("episode_type", "")
            if "Conservative" in str(ep_type):
                continue

            # Opioid MME
            if pd.notna(row.get("avg_opioid_mme_discharge")) and "opioid_mme_discharge_avg" in util_ranges:
                flag = _check_range(
                    row.get("avg_opioid_mme_discharge"),
                    util_ranges["opioid_mme_discharge_avg"],
                    "avg_opioid_mme_discharge", ep_type, contract_id
                )
                if flag:
                    flags.append(flag)

            # PROM collection rate
            if pd.notna(row.get("prom_collection_rate")) and "prom_collection_rate" in util_ranges:
                flag = _check_range(
                    row.get("prom_collection_rate"),
                    util_ranges["prom_collection_rate"],
                    "prom_collection_rate", ep_type, contract_id
                )
                if flag:
                    flags.append(flag)

        # Quality targets
        quality_targets = reference_ranges.get("quality_targets", {})
        for _, row in episodes_df.iterrows():
            ep_type = row.get("episode_type", "")
            if "Conservative" in str(ep_type):
                continue
            for metric, ref_key in [("readmission_rate", "readmit_90day"),
                                     ("er_visit_rate_90d", "er_visit_90day"),
                                     ("ssi_rate", "ssi_rate"),
                                     ("revision_rate_12mo", "revision_12mo")]:
                val = row.get(metric)
                if pd.notna(val) and ref_key in quality_targets:
                    flag = _check_range(val, quality_targets[ref_key],
                                        metric, ep_type, contract_id)
                    if flag:
                        flags.append(flag)

    elif specialty == "Oncology":
        for _, row in episodes_df.iterrows():
            cancer = row.get("cancer_type", "")
            stage = row.get("stage_group", "")
            line = row.get("line_of_therapy", "")
            ep_label = f"{cancer} {stage} {line}".strip()
            ref_key = ONC_EP_TYPE_MAP.get((cancer, stage, line))

            if ref_key and ref_key in cost_ranges:
                flag = _check_range(
                    row.get("avg_episode_cost"), cost_ranges[ref_key],
                    "avg_episode_cost", ep_label, contract_id
                )
                if flag:
                    flags.append(flag)

            # Pathway adherence
            pathway_benchmarks = reference_ranges.get("pathway_adherence_benchmarks", {})
            if ref_key and ref_key in pathway_benchmarks:
                adherence = row.get("pathway_adherence_rate")
                if pd.notna(adherence):
                    benchmark = pathway_benchmarks[ref_key]
                    min_acceptable = benchmark.get("min_acceptable", 0)
                    expected = benchmark.get("expected", 0)
                    if adherence < min_acceptable:
                        flags.append(ValidationFlag(
                            flag_id=_next_id(), severity="RED", category="range",
                            metric_name="pathway_adherence_rate",
                            metric_value=adherence,
                            expected_value=f"min acceptable {min_acceptable}, expected {expected}",
                            episode_type=ep_label, contract_id=contract_id,
                            description=f"Pathway adherence {adherence:.0%} below minimum "
                                        f"acceptable {min_acceptable:.0%} for {ep_label}",
                            detail=f"Expected adherence of {expected:.0%}. Current rate is "
                                   f"significantly below benchmark.",
                        ))
                    elif adherence < expected:
                        flags.append(ValidationFlag(
                            flag_id=_next_id(), severity="YELLOW", category="range",
                            metric_name="pathway_adherence_rate",
                            metric_value=adherence,
                            expected_value=f"expected {expected}",
                            episode_type=ep_label, contract_id=contract_id,
                            description=f"Pathway adherence {adherence:.0%} below expected "
                                        f"{expected:.0%} for {ep_label}",
                            detail=f"Rate is above minimum acceptable ({min_acceptable:.0%}) "
                                   f"but below expected benchmark.",
                        ))

    return flags
