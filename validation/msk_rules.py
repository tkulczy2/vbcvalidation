"""MSK-specific validation rules: clinical and financial logic for musculoskeletal episodes."""

import pandas as pd
import numpy as np
from validation import ValidationFlag

_msk_counter = 0


def _next_id():
    global _msk_counter
    _msk_counter += 1
    return f"MSK-{_msk_counter:03d}"


# Implant cost ratio benchmarks by procedure category
IMPLANT_RATIO_BENCHMARKS = {
    "TKR": {"max_ratio": 0.20, "category": "joint_replacement"},
    "THR": {"max_ratio": 0.20, "category": "joint_replacement"},
    "Spinal Fusion 1-2": {"max_ratio": 0.25, "category": "spinal_fusion"},
    "Spinal Fusion 3+": {"max_ratio": 0.25, "category": "spinal_fusion"},
    "Rotator Cuff": {"max_ratio": 0.20, "category": "joint_replacement"},
    "Knee Arthroscopy": {"max_ratio": 0.10, "category": "arthroscopy"},
}


def validate_msk_rules(episodes_df: pd.DataFrame, quality_df: pd.DataFrame,
                       reference_ranges: dict, contract: dict) -> list[ValidationFlag]:
    """Run MSK-specific validation rules."""
    flags = []
    contract_id = contract["contract_id"]
    attributed_members = contract.get("attributed_members", 0)

    # 1. Implant cost ratio
    for _, row in episodes_df.iterrows():
        ep_type = row.get("episode_type", "")
        if ep_type not in IMPLANT_RATIO_BENCHMARKS:
            continue

        implant_cost = row.get("implant_cost_avg")
        avg_cost = row.get("avg_episode_cost")
        if pd.isna(implant_cost) or pd.isna(avg_cost) or avg_cost == 0:
            continue

        ratio = implant_cost / avg_cost
        benchmark = IMPLANT_RATIO_BENCHMARKS[ep_type]
        max_ratio = benchmark["max_ratio"]

        if ratio > max_ratio:
            flags.append(ValidationFlag(
                flag_id=_next_id(), severity="RED", category="specialty",
                metric_name="implant_cost_ratio",
                metric_value=f"{ratio:.1%}",
                expected_value=f"<{max_ratio:.0%} for {benchmark['category']}",
                episode_type=ep_type, contract_id=contract_id,
                description=f"{ep_type}: Implant cost is {ratio:.1%} of total episode cost "
                            f"(${implant_cost:,.0f}/${avg_cost:,.0f}), exceeding "
                            f"{max_ratio:.0%} benchmark",
                detail=f"Implant cost avg ${implant_cost:,.0f} represents {ratio:.1%} of "
                       f"total episode cost ${avg_cost:,.0f}. Industry benchmark for "
                       f"{benchmark['category']} is <{max_ratio:.0%}. This may indicate "
                       f"premium device selection or unfavorable vendor pricing. "
                       f"Note: risk score actual ({row.get('risk_score_actual', 'N/A')}) vs "
                       f"expected ({row.get('risk_score_expected', 'N/A')}) shows the overrun "
                       f"is NOT explained by case complexity.",
                related_metrics={
                    "implant_cost_avg": implant_cost,
                    "avg_episode_cost": avg_cost,
                    "implant_ratio": round(ratio, 4),
                    "benchmark_max": max_ratio,
                    "risk_score_actual": row.get("risk_score_actual"),
                    "risk_score_expected": row.get("risk_score_expected"),
                },
            ))

    # 2. Arthroscopy volume reasonableness
    arth_row = episodes_df[episodes_df["episode_type"] == "Knee Arthroscopy"]
    if len(arth_row) > 0 and attributed_members > 0:
        arth_count = arth_row.iloc[0].get("episode_count", 0)
        rate_per_1000 = arth_count / (attributed_members / 1000)
        if rate_per_1000 > 25:
            flags.append(ValidationFlag(
                flag_id=_next_id(), severity="RED", category="specialty",
                metric_name="arthroscopy_volume",
                metric_value=f"{rate_per_1000:.1f}/1,000 members",
                expected_value="15-25/1,000 for MA population",
                episode_type="Knee Arthroscopy", contract_id=contract_id,
                description=f"Knee arthroscopy rate of {rate_per_1000:.1f}/1,000 exceeds "
                            f"expected MA range — potential overutilization",
                detail=f"{arth_count} arthroscopy episodes for {attributed_members:,} members "
                       f"= {rate_per_1000:.1f}/1,000. Multiple RCTs show arthroscopic "
                       f"debridement for knee OA (the most common MA-age indication) is "
                       f"clinically ineffective. This rate significantly exceeds the expected "
                       f"MA range of 15-25/1,000.",
                related_metrics={"episode_count": arth_count,
                                 "attributed_members": attributed_members,
                                 "rate_per_1000": round(rate_per_1000, 1)},
            ))

    # 3. Arthroscopy-to-conservative ratio (also checked in cross_metric, but
    #    this provides the specialty-specific clinical context)
    # Already covered in cross_metric.py — skip to avoid duplicate

    # 4. Post-acute cost efficiency
    for _, row in episodes_df.iterrows():
        ep_type = row.get("episode_type", "")
        if ep_type not in ("TKR", "THR"):
            continue

        post_acute = row.get("post_acute_cost_avg")
        avg_cost = row.get("avg_episode_cost")
        if pd.isna(post_acute) or pd.isna(avg_cost) or avg_cost == 0:
            continue

        pa_ratio = post_acute / avg_cost
        if pa_ratio > 0.20:
            flags.append(ValidationFlag(
                flag_id=_next_id(), severity="YELLOW", category="specialty",
                metric_name="post_acute_cost_ratio",
                metric_value=f"{pa_ratio:.1%}",
                expected_value="<20% of total episode cost",
                episode_type=ep_type, contract_id=contract_id,
                description=f"{ep_type}: Post-acute costs are {pa_ratio:.1%} of total "
                            f"episode cost (${post_acute:,.0f}/${avg_cost:,.0f})",
                detail=f"Post-acute spending (SNF, IRF, home health) at {pa_ratio:.1%} of "
                       f"total episode cost exceeds the 20% benchmark. Consider "
                       f"care coordination improvements and discharge planning optimization.",
                related_metrics={"post_acute_cost_avg": post_acute,
                                 "avg_episode_cost": avg_cost,
                                 "discharge_home_pct": row.get("discharge_home_pct"),
                                 "discharge_snf_pct": row.get("discharge_snf_pct")},
            ))

    # 5. Opioid prescribing
    for _, row in episodes_df.iterrows():
        ep_type = row.get("episode_type", "")
        if "Conservative" in str(ep_type):
            continue
        mme = row.get("avg_opioid_mme_discharge")
        if pd.notna(mme) and mme > 50:
            severity = "RED" if mme > 90 else "YELLOW"
            flags.append(ValidationFlag(
                flag_id=_next_id(), severity=severity, category="specialty",
                metric_name="opioid_mme_discharge",
                metric_value=f"{mme} MME",
                expected_value="<50 MME (CDC guideline-informed)",
                episode_type=ep_type, contract_id=contract_id,
                description=f"{ep_type}: Average discharge opioid prescription of {mme} MME "
                            f"exceeds 50 MME threshold",
                detail=f"CDC-informed guidelines suggest discharge opioid prescriptions "
                       f"should average <50 MME. Current average of {mme} MME for {ep_type} "
                       f"may indicate opportunity for enhanced recovery protocols or "
                       f"multimodal pain management.",
                related_metrics={"avg_opioid_mme_discharge": mme},
            ))

    # 6. PROM reliability
    for _, row in episodes_df.iterrows():
        ep_type = row.get("episode_type", "")
        if "Conservative" in str(ep_type):
            continue
        prom_coll = row.get("prom_collection_rate")
        prom_improve = row.get("prom_improvement_rate")
        if pd.notna(prom_coll) and prom_coll < 0.50:
            flags.append(ValidationFlag(
                flag_id=_next_id(), severity="RED", category="specialty",
                metric_name="prom_collection_reliability",
                metric_value=f"{prom_coll:.0%} collection rate",
                expected_value=">50% for reliable outcome measurement",
                episode_type=ep_type, contract_id=contract_id,
                description=f"{ep_type}: PROM collection rate of {prom_coll:.0%} renders "
                            f"outcome measures unreliable",
                detail=f"With only {prom_coll:.0%} PROM collection, the reported improvement "
                       f"rate of {prom_improve:.1%} is measured on a biased sample. Compliant "
                       f"patients who return PROMs likely have better outcomes than "
                       f"non-responders. This is an operational/data capture problem, not a "
                       f"care quality problem — the provider lacks a systematic PROM "
                       f"collection workflow.",
                related_metrics={"prom_collection_rate": prom_coll,
                                 "prom_improvement_rate": prom_improve},
            ))

    # 7. Spinal fusion level distribution
    fus_12 = episodes_df[episodes_df["episode_type"] == "Spinal Fusion 1-2"]
    fus_3p = episodes_df[episodes_df["episode_type"] == "Spinal Fusion 3+"]
    if len(fus_12) > 0 and len(fus_3p) > 0:
        count_12 = fus_12.iloc[0].get("episode_count", 0)
        count_3p = fus_3p.iloc[0].get("episode_count", 0)
        total_fusions = count_12 + count_3p
        if total_fusions > 0:
            pct_3p = count_3p / total_fusions
            if pct_3p > 0.30:
                flags.append(ValidationFlag(
                    flag_id=_next_id(), severity="YELLOW", category="specialty",
                    metric_name="fusion_complexity_distribution",
                    metric_value=f"{pct_3p:.0%} are 3+ level fusions",
                    expected_value="<30% of fusions should be 3+ levels",
                    episode_type="Spinal Fusion", contract_id=contract_id,
                    description=f"Spinal fusion 3+ level cases are {pct_3p:.0%} of total "
                                f"fusions — potential case complexity concern",
                    detail=f"{count_3p} of {total_fusions} fusion cases ({pct_3p:.0%}) are "
                           f"3+ levels. If >30%, this warrants risk adjustment review to "
                           f"ensure benchmarks adequately account for case complexity.",
                    related_metrics={"fusion_1_2_count": count_12,
                                     "fusion_3_plus_count": count_3p,
                                     "pct_3_plus": round(pct_3p, 3)},
                ))

    return flags
