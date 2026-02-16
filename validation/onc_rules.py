"""Oncology-specific validation rules: clinical and financial logic for oncology episodes."""

import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from validation import ValidationFlag

_onc_counter = 0


def _next_id():
    global _onc_counter
    _onc_counter += 1
    return f"ONC-{_onc_counter:03d}"


# Biosimilar pair definitions: (brand_keyword, biosimilar_keyword)
BIOSIMILAR_PAIRS = [
    ("Trastuzumab (Herceptin)", "Trastuzumab-dkst"),
    ("Bevacizumab (Avastin)", "Bevacizumab-awwb"),
    ("Pegfilgrastim (Neulasta)", "Pegfilgrastim-jmdb"),
]

# Cancer type mapping to reference range keys
CANCER_TYPE_MAP = {
    "Breast": "breast",
    "Lung": "lung",
    "Colorectal": "colorectal",
    "Prostate": "prostate",
}


def validate_onc_rules(episodes_df: pd.DataFrame, quality_df: pd.DataFrame,
                       drugs_df: pd.DataFrame, reference_ranges: dict,
                       contract: dict) -> list[ValidationFlag]:
    """Run oncology-specific validation rules."""
    flags = []
    contract_id = contract["contract_id"]
    attributed_members = contract.get("attributed_members", 0)

    flags += _rule_pathway_adherence_cost(episodes_df, contract)
    flags += _rule_biosimilar_savings(drugs_df, episodes_df, contract)
    flags += _rule_site_of_service_savings(drugs_df, contract)
    flags += _rule_eol_acp_correlation(quality_df, contract)
    flags += _rule_novel_therapy_impact(drugs_df, episodes_df, contract)
    flags += _rule_volume_vs_incidence(episodes_df, reference_ranges, contract)
    flags += _rule_quality_gate_financial_impact(quality_df, episodes_df, contract)

    return flags


def _rule_pathway_adherence_cost(episodes_df, contract):
    """Rule 1: Pathway adherence cost correlation."""
    flags = []
    contract_id = contract["contract_id"]
    pathway_target = contract.get("pathway_adherence_target", 0.80)

    for _, row in episodes_df.iterrows():
        ep_label = f"{row.get('cancer_type', '')} {row.get('stage_group', '')} {row.get('line_of_therapy', '')}".strip()
        adherence = row.get("pathway_adherence_rate")
        avg_cost = row.get("avg_episode_cost")
        target = row.get("target_price")

        if pd.isna(adherence) or pd.isna(avg_cost) or pd.isna(target):
            continue
        if adherence >= pathway_target or avg_cost <= target:
            continue

        cost_overrun_pct = (avg_cost - target) / target
        if cost_overrun_pct <= 0.05:
            continue

        # Back-calculate: avg_cost = adherence * pathway_cost + (1-adherence) * non_pathway_cost
        pathway_cost_est = target
        if adherence < 1.0:
            non_pathway_cost_est = (avg_cost - adherence * pathway_cost_est) / (1 - adherence)
            cost_diff = non_pathway_cost_est - pathway_cost_est
            cost_diff_pct = cost_diff / pathway_cost_est if pathway_cost_est > 0 else 0

            if cost_diff_pct > 0.25:
                episode_count = row.get("episode_count", 0)
                non_pathway_count = int(episode_count * (1 - adherence))
                potential_savings = non_pathway_count * cost_diff

                flags.append(ValidationFlag(
                    flag_id=_next_id(), severity="RED", category="specialty",
                    metric_name="pathway_cost_correlation",
                    metric_value=f"adherence={adherence:.0%}, overrun={cost_overrun_pct:.1%}",
                    expected_value=f"pathway adherence >{pathway_target:.0%}",
                    episode_type=ep_label, contract_id=contract_id,
                    description=f"{ep_label}: Non-pathway regimens cost "
                                f"${non_pathway_cost_est:,.0f}/episode vs "
                                f"${pathway_cost_est:,.0f} pathway (+{cost_diff_pct:.0%}), "
                                f"driving ${potential_savings:,.0f} in excess cost",
                    detail=f"Back-calculation: ({adherence:.0%} x ${pathway_cost_est:,.0f}) + "
                           f"({1-adherence:.0%} x ${non_pathway_cost_est:,.0f}) = "
                           f"${adherence * pathway_cost_est + (1-adherence) * non_pathway_cost_est:,.0f} "
                           f"≈ ${avg_cost:,.0f}. The {1-adherence:.0%} non-pathway cases ({non_pathway_count} "
                           f"episodes) are the primary cost driver. Improving pathway adherence to "
                           f"{pathway_target:.0%} would save approximately "
                           f"${potential_savings:,.0f} across this episode type.",
                    related_metrics={
                        "pathway_adherence": adherence,
                        "avg_episode_cost": avg_cost,
                        "target_price": target,
                        "est_pathway_cost": pathway_cost_est,
                        "est_non_pathway_cost": round(non_pathway_cost_est),
                        "potential_savings": round(potential_savings),
                    },
                ))

    return flags


def _rule_biosimilar_savings(drugs_df, episodes_df, contract):
    """Rule 2: Biosimilar savings calculator."""
    flags = []
    contract_id = contract["contract_id"]
    total_episodes = episodes_df["episode_count"].sum()

    for brand_keyword, biosimilar_keyword in BIOSIMILAR_PAIRS:
        brand_rows = drugs_df[drugs_df["drug_name"].str.contains(brand_keyword.split("(")[0].strip(), na=False)
                              & ~drugs_df["is_biosimilar"].astype(str).str.lower().eq("true")]
        bio_rows = drugs_df[drugs_df["drug_name"].str.contains(biosimilar_keyword, na=False)]

        if len(brand_rows) == 0 or len(bio_rows) == 0:
            continue

        brand = brand_rows.iloc[0]
        bio = bio_rows.iloc[0]

        brand_claims = brand.get("total_claims", 0)
        bio_claims = bio.get("total_claims", 0)
        total_claims = brand_claims + bio_claims

        if total_claims == 0:
            continue

        brand_pct = brand_claims / total_claims
        if brand_pct <= 0.50:
            continue

        brand_cost = brand.get("avg_cost_per_claim", 0)
        bio_cost = bio.get("avg_cost_per_claim", 0)
        cost_diff = brand_cost - bio_cost
        potential_savings = brand_claims * cost_diff
        per_episode = potential_savings / total_episodes if total_episodes > 0 else 0

        severity = "RED" if brand_pct > 0.80 else "YELLOW"

        flags.append(ValidationFlag(
            flag_id=_next_id(), severity=severity, category="specialty",
            metric_name="biosimilar_savings_opportunity",
            metric_value=f"{brand.get('drug_name', '')}: {brand_pct:.0%} brand utilization",
            expected_value="brand utilization <50%",
            episode_type="Drug Detail", contract_id=contract_id,
            description=f"{brand.get('drug_name', '')}: {brand_claims} brand claims at "
                        f"${brand_cost:,.0f}/claim vs biosimilar at ${bio_cost:,.0f}/claim — "
                        f"${potential_savings:,.0f} savings opportunity (${per_episode:,.0f}/episode)",
            detail=f"Brand {brand.get('drug_name', '')} has {brand_claims} claims at "
                   f"${brand_cost:,.0f}/claim. Biosimilar {bio.get('drug_name', '')} has "
                   f"{bio_claims} claims at ${bio_cost:,.0f}/claim. Brand utilization is "
                   f"{brand_pct:.0%} ({brand_claims}/{total_claims}). If all brand claims "
                   f"switched to biosimilar, savings would be {brand_claims} x "
                   f"(${brand_cost:,.0f} - ${bio_cost:,.0f}) = ${potential_savings:,.0f}, "
                   f"or ${per_episode:,.0f} per episode across {total_episodes} total episodes.",
            related_metrics={
                "brand_drug": brand.get("drug_name"),
                "biosimilar_drug": bio.get("drug_name"),
                "brand_claims": brand_claims,
                "biosimilar_claims": bio_claims,
                "brand_cost_per_claim": brand_cost,
                "biosimilar_cost_per_claim": bio_cost,
                "potential_savings": round(potential_savings),
                "per_episode_impact": round(per_episode),
            },
        ))

    return flags


def _rule_site_of_service_savings(drugs_df, contract):
    """Rule 3: Site-of-service savings calculator."""
    flags = []
    contract_id = contract["contract_id"]

    for _, drug in drugs_df.iterrows():
        name = drug.get("drug_name", "")
        avg_cost = drug.get("avg_cost_per_claim", 0)
        hopd_pct = drug.get("site_of_service_hopd_pct", 0)
        total_claims = drug.get("total_claims", 0)
        is_bio = str(drug.get("is_biosimilar", False)).lower() == "true"

        if pd.isna(avg_cost) or avg_cost <= 2000:
            continue
        if pd.isna(hopd_pct) or hopd_pct <= 0.60:
            continue

        # Calculate excess HOPD claims above 40% target
        target_hopd_pct = 0.40
        excess_hopd_pct = hopd_pct - target_hopd_pct
        excess_hopd_claims = total_claims * excess_hopd_pct
        # HOPD costs ~2x office; facility fee markup is roughly 50% of drug cost
        est_savings = excess_hopd_claims * avg_cost * 0.50

        if est_savings < 5000:
            continue

        flags.append(ValidationFlag(
            flag_id=_next_id(), severity="YELLOW", category="specialty",
            metric_name="site_of_service_opportunity",
            metric_value=f"{name}: {hopd_pct:.0%} HOPD, ${avg_cost:,.0f}/claim",
            expected_value="HOPD <60% for office-administrable drugs",
            episode_type="Drug Detail", contract_id=contract_id,
            description=f"{name}: {hopd_pct:.0%} administered at HOPD vs "
                        f"{target_hopd_pct:.0%} target — estimated "
                        f"${est_savings:,.0f} in excess facility costs",
            detail=f"{name} has {total_claims} claims at ${avg_cost:,.0f}/claim "
                   f"with {hopd_pct:.0%} HOPD administration. Shifting the excess "
                   f"{excess_hopd_pct:.0%} ({excess_hopd_claims:.0f} claims) from HOPD to "
                   f"physician office could save an estimated ${est_savings:,.0f} in "
                   f"facility fees. HOPD infusion typically costs 2-3x physician office "
                   f"administration.",
            related_metrics={
                "drug_name": name,
                "hopd_pct": hopd_pct,
                "avg_cost_per_claim": avg_cost,
                "total_claims": total_claims,
                "excess_hopd_claims": round(excess_hopd_claims),
                "estimated_savings": round(est_savings),
            },
        ))

    return flags


def _rule_eol_acp_correlation(quality_df, contract):
    """Rule 4: EOL-ACP correlation — identify ACP as root cause."""
    flags = []
    contract_id = contract["contract_id"]

    # Check ACP rate first
    acp_rows = quality_df[quality_df["measure_id"] == "ONC-Q-009"]
    if len(acp_rows) == 0:
        return flags

    acp_rate = acp_rows.iloc[0].get("rate", 1.0)
    if pd.isna(acp_rate) or acp_rate >= 0.50:
        return flags

    # Count EOL failures
    eol_measures = {
        "ONC-Q-002": ("Chemo Within 14 Days of Death", "high_is_bad"),
        "ONC-Q-003": ("Hospice Enrollment", "low_is_bad"),
        "ONC-Q-004": ("Hospice >7 Days Before Death", "low_is_bad"),
        "ONC-Q-005": ("ICU Within 30 Days of Death", "high_is_bad"),
        "ONC-Q-006": ("ER Within 30 Days of Death", "high_is_bad"),
    }

    eol_failures = 0
    for _, row in quality_df.iterrows():
        mid = row.get("measure_id", "")
        if mid in eol_measures:
            rate = row.get("rate", 0)
            target = row.get("target", 0)
            _, direction = eol_measures[mid]
            if pd.notna(rate) and pd.notna(target):
                if direction == "high_is_bad" and rate > target:
                    eol_failures += 1
                elif direction == "low_is_bad" and rate < target:
                    eol_failures += 1

    if eol_failures >= 3:
        flags.append(ValidationFlag(
            flag_id=_next_id(), severity="RED", category="specialty",
            metric_name="acp_root_cause",
            metric_value=f"ACP rate {acp_rate:.1%}, {eol_failures}/5 EOL metrics failing",
            expected_value="ACP >50% to support EOL quality metrics",
            episode_type="End-of-Life Care", contract_id=contract_id,
            description=f"Advance Care Planning ({acp_rate:.1%}) is the root cause of "
                        f"systemic EOL metric failure — {eol_failures}/5 EOL measures failing",
            detail=f"The Advance Care Planning documentation rate of {acp_rate:.1%} "
                   f"(target >65%) is below the 50% threshold that predicts EOL metric "
                   f"failures. Without documented goals-of-care conversations, patients "
                   f"default to aggressive end-of-life treatment. This is both the "
                   f"highest-quality-impact and highest-cost issue: improving ACP is the "
                   f"single intervention that addresses all {eol_failures} failing EOL measures "
                   f"simultaneously. This is a process/workflow fix, not a clinical quality "
                   f"problem.",
            related_metrics={
                "acp_rate": acp_rate,
                "eol_failures": eol_failures,
            },
        ))

    return flags


def _rule_novel_therapy_impact(drugs_df, episodes_df, contract):
    """Rule 5: Novel therapy impact — identify carve-out eligible costs."""
    flags = []
    contract_id = contract["contract_id"]

    if not contract.get("novel_therapy_carveout", False):
        return flags

    lookback_months = contract.get("novel_therapy_lookback_months", 18)
    data_as_of_str = contract.get("data_as_of", "2025-01-15")

    try:
        data_as_of = datetime.strptime(data_as_of_str, "%Y-%m-%d")
    except (ValueError, TypeError):
        return flags

    cutoff_date = data_as_of - relativedelta(months=lookback_months)

    novel_drugs = []
    novel_total_cost = 0

    for _, drug in drugs_df.iterrows():
        if str(drug.get("is_novel_therapy", False)).lower() != "true":
            continue

        fda_date_str = drug.get("fda_approval_date", "")
        try:
            fda_date = datetime.strptime(str(fda_date_str), "%Y-%m-%d")
        except (ValueError, TypeError):
            continue

        if fda_date >= cutoff_date:
            cost = drug.get("total_cost", 0)
            novel_drugs.append({
                "name": drug.get("drug_name", ""),
                "fda_date": fda_date_str,
                "total_cost": cost,
                "claims": drug.get("total_claims", 0),
            })
            novel_total_cost += cost

    if not novel_drugs:
        return flags

    # Calculate savings impact if novel costs carved out
    total_cost = episodes_df["total_cost"].sum()
    total_target = episodes_df["total_target"].sum()
    savings_before = total_target - total_cost
    savings_after = (total_target - (total_cost - novel_total_cost))
    sharing_rate = contract.get("sharing_rate_savings", 0)
    impact = (savings_after - savings_before) * sharing_rate

    drug_list = "; ".join([f"{d['name']} (${d['total_cost']:,.0f})" for d in novel_drugs])

    flags.append(ValidationFlag(
        flag_id=_next_id(), severity="YELLOW", category="specialty",
        metric_name="novel_therapy_carveout",
        metric_value=f"${novel_total_cost:,.0f} in novel therapy costs",
        expected_value="These costs may be carved out per contract terms",
        episode_type="Drug Detail", contract_id=contract_id,
        description=f"Novel therapy carve-out: ${novel_total_cost:,.0f} in costs from "
                    f"{len(novel_drugs)} drug(s) approved within {lookback_months}-month "
                    f"lookback may be excluded from savings calculation",
        detail=f"Contract specifies novel therapy carve-out for drugs approved within "
               f"{lookback_months} months of {data_as_of_str}. Eligible drugs: {drug_list}. "
               f"If carved out, total cost decreases by ${novel_total_cost:,.0f}, changing "
               f"savings from ${savings_before:,.0f} to ${savings_after:,.0f}. "
               f"Impact on provider share: ${impact:,.0f}.",
        related_metrics={
            "novel_drugs": [d["name"] for d in novel_drugs],
            "novel_total_cost": novel_total_cost,
            "savings_before_carveout": round(savings_before),
            "savings_after_carveout": round(savings_after),
            "provider_share_impact": round(impact),
        },
    ))

    return flags


def _rule_volume_vs_incidence(episodes_df, reference_ranges, contract):
    """Rule 6: Episode volume vs expected incidence rates."""
    flags = []
    contract_id = contract["contract_id"]
    attributed_members = contract.get("attributed_members", 0)

    if attributed_members == 0:
        return flags

    incidence_rates = reference_ranges.get("incidence_rates_ma_per_1000", {})

    # Sum episodes by cancer type
    cancer_volumes = {}
    for _, row in episodes_df.iterrows():
        cancer = row.get("cancer_type", "")
        count = row.get("episode_count", 0)
        if pd.notna(count):
            cancer_volumes[cancer] = cancer_volumes.get(cancer, 0) + count

    for cancer_type, total_count in cancer_volumes.items():
        ref_key = CANCER_TYPE_MAP.get(cancer_type)
        if ref_key is None or ref_key not in incidence_rates:
            continue

        ref = incidence_rates[ref_key]
        rate_per_1000 = total_count / (attributed_members / 1000)
        expected = ref.get("expected", 0)
        min_rate = ref.get("min", 0)
        max_rate = ref.get("max", 0)

        if expected > 0 and rate_per_1000 > 2 * max_rate:
            flags.append(ValidationFlag(
                flag_id=_next_id(), severity="RED", category="specialty",
                metric_name="episode_volume_vs_incidence",
                metric_value=f"{cancer_type}: {rate_per_1000:.1f}/1,000",
                expected_value=f"{min_rate}-{max_rate}/1,000 (expected {expected}/1,000)",
                episode_type=f"{cancer_type} Volume", contract_id=contract_id,
                description=f"{cancer_type} episode rate of {rate_per_1000:.1f}/1,000 is "
                            f">{2*max_rate}/1,000 — potential attribution problem",
                detail=f"{total_count} {cancer_type} episodes for {attributed_members:,} "
                       f"members = {rate_per_1000:.1f}/1,000. Expected range is "
                       f"{min_rate}-{max_rate}/1,000. Rate exceeding 2x the maximum "
                       f"suggests a potential attribution algorithm issue or duplicated episodes.",
                related_metrics={
                    "cancer_type": cancer_type,
                    "episode_count": total_count,
                    "rate_per_1000": round(rate_per_1000, 1),
                    "expected_rate": expected,
                },
            ))
        elif expected > 0 and rate_per_1000 < 0.5 * min_rate:
            flags.append(ValidationFlag(
                flag_id=_next_id(), severity="YELLOW", category="specialty",
                metric_name="episode_volume_vs_incidence",
                metric_value=f"{cancer_type}: {rate_per_1000:.1f}/1,000",
                expected_value=f"{min_rate}-{max_rate}/1,000 (expected {expected}/1,000)",
                episode_type=f"{cancer_type} Volume", contract_id=contract_id,
                description=f"{cancer_type} episode rate of {rate_per_1000:.1f}/1,000 is "
                            f"below expected — potential access or underdiagnosis concern",
                detail=f"{total_count} {cancer_type} episodes for {attributed_members:,} "
                       f"members = {rate_per_1000:.1f}/1,000. Expected minimum is "
                       f"{min_rate}/1,000. Low rates may indicate access barriers, "
                       f"underdiagnosis, or attribution gaps.",
                related_metrics={
                    "cancer_type": cancer_type,
                    "episode_count": total_count,
                    "rate_per_1000": round(rate_per_1000, 1),
                    "expected_rate": expected,
                },
            ))

    return flags


def _rule_quality_gate_financial_impact(quality_df, episodes_df, contract):
    """Rule 7: Quality gate financial impact — identify cheapest path to passing."""
    flags = []
    contract_id = contract["contract_id"]
    gate_min = contract.get("quality_gate_minimum", 0)

    comp_row = quality_df[quality_df["measure_id"].str.contains("COMP", na=False)]
    if len(comp_row) == 0:
        return flags

    earned = comp_row.iloc[0].get("points_earned", 0)
    max_pts = comp_row.iloc[0].get("max_points", 0)
    if pd.isna(earned) or pd.isna(max_pts) or max_pts == 0:
        return flags

    composite_pct = (earned / max_pts) * 100
    gap = gate_min - composite_pct

    if gap <= 0 or gap > 5:
        return flags

    # Calculate total savings at risk
    total_savings = 0
    for _, row in episodes_df.iterrows():
        tc = row.get("total_cost", 0)
        tt = row.get("total_target", 0)
        if pd.notna(tc) and pd.notna(tt):
            total_savings += (tt - tc)

    sharing_rate = contract.get("sharing_rate_savings", 0)
    at_risk = max(0, total_savings) * sharing_rate

    # Find measures closest to improving
    non_comp = quality_df[~quality_df["measure_id"].str.contains("COMP", na=False)]
    improvement_candidates = []

    for _, row in non_comp.iterrows():
        current_pts = row.get("points_earned", 0)
        max_measure_pts = row.get("max_points", 0)
        gap_pts = max_measure_pts - current_pts
        if pd.notna(gap_pts) and gap_pts > 0:
            improvement_candidates.append({
                "measure": row.get("measure_name", ""),
                "current_points": current_pts,
                "max_points": max_measure_pts,
                "gap": gap_pts,
                "rate": row.get("rate"),
                "target": row.get("target"),
            })

    # Sort by gap (smallest first — easiest to improve)
    improvement_candidates.sort(key=lambda x: x["gap"])
    top_candidates = improvement_candidates[:3]

    candidate_text = "; ".join([
        f"{c['measure']} ({c['current_points']}/{c['max_points']}, gap={c['gap']}pts)"
        for c in top_candidates
    ])

    flags.append(ValidationFlag(
        flag_id=_next_id(), severity="RED", category="specialty",
        metric_name="quality_gate_improvement_path",
        metric_value=f"composite {composite_pct:.1f}%, need {gate_min}%, "
                     f"${at_risk:,.0f} at risk",
        expected_value=f"composite >= {gate_min}%",
        episode_type="Quality Gate", contract_id=contract_id,
        description=f"Quality gate {gap:.1f} points from passing — "
                    f"${at_risk:,.0f} at risk. Easiest improvement: {candidate_text}",
        detail=f"The quality composite of {earned}/{max_pts} ({composite_pct:.1f}%) "
               f"is {gap:.1f} points below the {gate_min}% gate. "
               f"Total shared savings at risk: ${at_risk:,.0f}. "
               f"Lowest-effort improvement candidates: {candidate_text}. "
               f"Closing this gap requires gaining {gap:.1f} percentage points, "
               f"equivalent to ~{gap * max_pts / 100:.0f} additional quality points.",
        related_metrics={
            "composite_pct": round(composite_pct, 1),
            "gate_minimum": gate_min,
            "gap_points": round(gap, 1),
            "savings_at_risk": round(at_risk),
            "improvement_candidates": top_candidates,
        },
    ))

    return flags
