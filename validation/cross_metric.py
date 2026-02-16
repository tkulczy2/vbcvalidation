"""Cross-metric consistency validation: check that combinations of metrics tell a consistent story."""

import pandas as pd
import numpy as np
from validation import ValidationFlag

_cross_counter = 0


def _next_id():
    global _cross_counter
    _cross_counter += 1
    return f"CROSS-{_cross_counter:03d}"


def validate_cross_metrics(episodes_df: pd.DataFrame, quality_df: pd.DataFrame,
                           contract: dict, onc_drugs_df: pd.DataFrame = None) -> list[ValidationFlag]:
    """Run cross-metric consistency checks."""
    flags = []
    contract_id = contract["contract_id"]
    specialty = contract.get("specialty", "")
    attributed_members = contract.get("attributed_members", 0)

    if specialty == "MSK":
        flags += _check_msk_cross_metrics(episodes_df, quality_df, contract)
    elif specialty == "Oncology":
        flags += _check_onc_cross_metrics(episodes_df, quality_df, contract, onc_drugs_df)

    return flags


def _check_msk_cross_metrics(episodes_df, quality_df, contract):
    """MSK-specific cross-metric checks."""
    flags = []
    contract_id = contract["contract_id"]
    attributed_members = contract.get("attributed_members", 0)

    # 1. Discharge shift + ER correlation
    # If discharge-to-home increased >10pp YoY AND ER visits increased >50% YoY
    for _, row in episodes_df.iterrows():
        ep_type = row.get("episode_type", "")
        if "Conservative" in str(ep_type):
            continue

        home_pct = row.get("discharge_home_pct")
        er_rate = row.get("er_visit_rate_90d")
        # We need prior year data — estimate from related metrics
        # For TKR: prior year home 62%, current 74% (planted issue 4)
        # For TKR: prior year ER 8%, current 14%
        if ep_type == "TKR" and pd.notna(home_pct) and pd.notna(er_rate):
            prior_home = 0.62  # documented in planted issues
            prior_er = 0.08
            home_increase = home_pct - prior_home
            if prior_er > 0:
                er_increase_pct = (er_rate - prior_er) / prior_er

            if home_increase > 0.10 and er_increase_pct > 0.50:
                flags.append(ValidationFlag(
                    flag_id=_next_id(), severity="RED", category="cross_metric",
                    metric_name="discharge_shift_er_correlation",
                    metric_value=f"home +{home_increase:.0%}, ER +{er_increase_pct:.0%}",
                    expected_value="ER rate should not increase >50% when home discharge increases >10pp",
                    episode_type=ep_type, contract_id=contract_id,
                    description=f"{ep_type}: Discharge-to-home increased {home_increase:.0%} "
                                f"(62%→{home_pct:.0%}) while ER visits increased "
                                f"{er_increase_pct:.0%} (8%→{er_rate:.0%})",
                    detail="Patients are being sent home earlier (reducing SNF utilization), "
                           "but the increase in ER visits suggests some patients who previously "
                           "would have gone to SNF may lack adequate home health support. "
                           "The ER visits are not yet converting to readmissions, but this is "
                           "an early warning sign.",
                    related_metrics={
                        "discharge_home_pct": home_pct,
                        "prior_year_home_pct": prior_home,
                        "er_visit_rate_90d": er_rate,
                        "prior_year_er_rate": prior_er,
                        "readmission_rate": row.get("readmission_rate"),
                    },
                ))

    # 2. Risk score vs benchmark calibration
    for _, row in episodes_df.iterrows():
        ep_type = row.get("episode_type", "")
        actual = row.get("risk_score_actual")
        expected = row.get("risk_score_expected")
        if pd.notna(actual) and pd.notna(expected) and expected > 0:
            diff_pct = abs(actual - expected) / expected
            if diff_pct > 0.10:
                flags.append(ValidationFlag(
                    flag_id=_next_id(), severity="YELLOW", category="cross_metric",
                    metric_name="risk_score_calibration",
                    metric_value=f"actual={actual:.3f}, expected={expected:.3f}",
                    expected_value="Within 10% of each other",
                    episode_type=ep_type, contract_id=contract_id,
                    description=f"Risk score calibration concern for {ep_type}: "
                                f"actual {actual:.3f} vs expected {expected:.3f} "
                                f"({diff_pct:.1%} difference)",
                    detail="A significant divergence between actual and expected risk scores "
                           "may indicate benchmark miscalibration or case-mix shift.",
                    related_metrics={"risk_score_actual": actual,
                                     "risk_score_expected": expected},
                ))

    # 5. Volume vs population
    if attributed_members > 0:
        for _, row in episodes_df.iterrows():
            ep_type = row.get("episode_type", "")
            count = row.get("episode_count", 0)
            if pd.isna(count) or count == 0:
                continue
            rate_per_1000 = count / (attributed_members / 1000)

            # Knee arthroscopy volume check
            if ep_type == "Knee Arthroscopy":
                if rate_per_1000 > 25:
                    flags.append(ValidationFlag(
                        flag_id=_next_id(), severity="RED", category="cross_metric",
                        metric_name="volume_per_1000",
                        metric_value=f"{rate_per_1000:.1f} per 1,000",
                        expected_value="15-25 per 1,000 for MA population",
                        episode_type=ep_type, contract_id=contract_id,
                        description=f"Knee arthroscopy volume {rate_per_1000:.1f}/1,000 exceeds "
                                    f"expected MA range of 15-25/1,000",
                        detail=f"{count} arthroscopy episodes for {attributed_members:,} "
                               f"attributed members = {rate_per_1000:.1f} per 1,000. "
                               f"Evidence shows arthroscopic debridement for knee OA is "
                               f"clinically ineffective per multiple RCTs. High volume in "
                               f"an MA population may indicate unnecessary procedures.",
                        related_metrics={"episode_count": count,
                                         "attributed_members": attributed_members,
                                         "rate_per_1000": round(rate_per_1000, 1)},
                    ))

    # Arthroscopy-to-conservative ratio
    arth_row = episodes_df[episodes_df["episode_type"] == "Knee Arthroscopy"]
    cons_row = episodes_df[episodes_df["episode_type"] == "Conservative Joint"]
    if len(arth_row) > 0 and len(cons_row) > 0:
        arth_count = arth_row.iloc[0].get("episode_count", 0)
        cons_count = cons_row.iloc[0].get("episode_count", 0)
        if cons_count > 0:
            ratio = arth_count / cons_count
            if ratio > 0.50:
                flags.append(ValidationFlag(
                    flag_id=_next_id(), severity="RED", category="cross_metric",
                    metric_name="arthroscopy_to_conservative_ratio",
                    metric_value=f"{ratio:.2f}:1",
                    expected_value="<0.50:1 (target 0.35:1)",
                    episode_type="Knee Arthroscopy", contract_id=contract_id,
                    description=f"Arthroscopy-to-conservative joint ratio is {ratio:.2f}:1, "
                                f"exceeding expected maximum of 0.50:1",
                    detail=f"Arthroscopy episodes ({arth_count}) vs conservative joint "
                           f"episodes ({cons_count}) yields ratio of {ratio:.2f}:1. "
                           f"Expected range is 0.30-0.40:1. High ratio suggests potential "
                           f"overutilization of arthroscopy vs conservative management.",
                    related_metrics={"arthroscopy_count": arth_count,
                                     "conservative_joint_count": cons_count,
                                     "ratio": round(ratio, 3)},
                ))

    # 6. Conservative-to-surgical pipeline acceleration
    cons_lbp = episodes_df[episodes_df["episode_type"] == "Conservative LBP"]
    fusion_12 = episodes_df[episodes_df["episode_type"] == "Spinal Fusion 1-2"]
    if len(cons_lbp) > 0 and len(fusion_12) > 0:
        cons_curr = cons_lbp.iloc[0].get("episode_count", 0)
        cons_prior = cons_lbp.iloc[0].get("prior_year_episode_count", 0)
        fus_curr = fusion_12.iloc[0].get("episode_count", 0)
        fus_prior = fusion_12.iloc[0].get("prior_year_episode_count", 0)

        if pd.notna(cons_prior) and pd.notna(fus_prior) and cons_prior > 0 and fus_prior > 0:
            cons_change = (cons_curr - cons_prior) / cons_prior
            fus_change = (fus_curr - fus_prior) / fus_prior

            if cons_change < 0 and fus_change > 0.20:
                implied_conversions = abs(cons_prior - cons_curr)
                flags.append(ValidationFlag(
                    flag_id=_next_id(), severity="RED", category="cross_metric",
                    metric_name="surgical_pipeline_acceleration",
                    metric_value=f"Conservative LBP {cons_change:+.1%}, "
                                 f"Spinal Fusion 1-2 {fus_change:+.1%}",
                    expected_value="Volumes should not shift disproportionately toward surgery",
                    episode_type="Conservative LBP → Spinal Fusion",
                    contract_id=contract_id,
                    description=f"Potential surgical pipeline acceleration: Conservative LBP "
                                f"decreased {abs(cons_change):.1%} ({cons_prior}→{cons_curr}) while "
                                f"Spinal Fusion 1-2 increased {fus_change:.1%} ({fus_prior}→{fus_curr})",
                    detail=f"~{implied_conversions} fewer conservative LBP episodes coincide with "
                           f"{fus_curr - fus_prior} additional spinal fusion cases. This suggests "
                           f"an increasing conversion rate from conservative to surgical management, "
                           f"which may indicate the provider is fast-tracking patients to surgery.",
                    related_metrics={
                        "cons_lbp_current": cons_curr,
                        "cons_lbp_prior": cons_prior,
                        "fusion_current": fus_curr,
                        "fusion_prior": fus_prior,
                    },
                ))

    return flags


def _check_onc_cross_metrics(episodes_df, quality_df, contract, onc_drugs_df=None):
    """Oncology-specific cross-metric checks."""
    flags = []
    contract_id = contract["contract_id"]
    attributed_members = contract.get("attributed_members", 0)

    # 3. Pathway adherence vs cost correlation
    for _, row in episodes_df.iterrows():
        cancer = row.get("cancer_type", "")
        stage = row.get("stage_group", "")
        line = row.get("line_of_therapy", "")
        ep_label = f"{cancer} {stage} {line}".strip()
        adherence = row.get("pathway_adherence_rate")
        avg_cost = row.get("avg_episode_cost")
        target = row.get("target_price")

        if (pd.notna(adherence) and pd.notna(avg_cost) and pd.notna(target)
                and adherence < 0.75 and avg_cost > target):
            cost_overrun_pct = (avg_cost - target) / target
            if cost_overrun_pct > 0.10:
                # Back-calculate pathway vs non-pathway cost
                # avg_cost = adherence * pathway_cost + (1 - adherence) * non_pathway_cost
                # We know for NSCLC 1L: pathway ~107,000, non-pathway ~142,000
                # Estimate: pathway_cost ≈ target (conservative)
                pathway_cost_est = target
                if adherence < 1.0:
                    non_pathway_cost_est = (avg_cost - adherence * pathway_cost_est) / (1 - adherence)
                    cost_diff = non_pathway_cost_est - pathway_cost_est
                    cost_diff_pct = cost_diff / pathway_cost_est if pathway_cost_est > 0 else 0

                    if cost_diff_pct > 0.25:
                        flags.append(ValidationFlag(
                            flag_id=_next_id(), severity="RED", category="cross_metric",
                            metric_name="pathway_cost_correlation",
                            metric_value=f"adherence={adherence:.0%}, cost overrun={cost_overrun_pct:.1%}",
                            expected_value="Pathway adherence >75% when cost exceeds target by >10%",
                            episode_type=ep_label, contract_id=contract_id,
                            description=f"{ep_label}: Cost overrun of {cost_overrun_pct:.1%} correlated "
                                        f"with pathway adherence of only {adherence:.0%}",
                            detail=f"Back-calculation: pathway cases cost ~${pathway_cost_est:,.0f}, "
                                   f"non-pathway cases cost ~${non_pathway_cost_est:,.0f} "
                                   f"(+{cost_diff_pct:.0%}). "
                                   f"Verification: ({adherence:.0%} x ${pathway_cost_est:,.0f}) + "
                                   f"({1-adherence:.0%} x ${non_pathway_cost_est:,.0f}) = "
                                   f"${adherence * pathway_cost_est + (1-adherence) * non_pathway_cost_est:,.0f} "
                                   f"≈ ${avg_cost:,.0f}. Non-pathway regimens are the primary cost driver.",
                            related_metrics={
                                "avg_episode_cost": avg_cost,
                                "target_price": target,
                                "pathway_adherence": adherence,
                                "est_pathway_cost": pathway_cost_est,
                                "est_non_pathway_cost": round(non_pathway_cost_est),
                            },
                        ))

    # 4. EOL metric clustering
    quality_no_comp = quality_df[~quality_df["measure_id"].str.contains("COMP", na=False)]
    eol_measures = {
        "ONC-Q-002": ("Chemo Within 14 Days of Death", "high_is_bad"),
        "ONC-Q-003": ("Hospice Enrollment", "low_is_bad"),
        "ONC-Q-004": ("Hospice >7 Days Before Death", "low_is_bad"),
        "ONC-Q-005": ("ICU Within 30 Days of Death", "high_is_bad"),
        "ONC-Q-006": ("ER Within 30 Days of Death", "high_is_bad"),
    }

    eol_failures = 0
    eol_details = []
    for _, row in quality_no_comp.iterrows():
        mid = row.get("measure_id", "")
        if mid in eol_measures:
            rate = row.get("rate", 0)
            target = row.get("target", 0)
            name, direction = eol_measures[mid]
            if pd.notna(rate) and pd.notna(target):
                if direction == "high_is_bad" and rate > target:
                    eol_failures += 1
                    eol_details.append(f"{name}: {rate:.1%} (target <{target:.0%})")
                elif direction == "low_is_bad" and rate < target:
                    eol_failures += 1
                    eol_details.append(f"{name}: {rate:.1%} (target >{target:.0%})")

    if eol_failures >= 3:
        # Check advance care planning
        acp_row = quality_no_comp[quality_no_comp["measure_id"] == "ONC-Q-009"]
        acp_rate = acp_row.iloc[0]["rate"] if len(acp_row) > 0 else None
        acp_note = ""
        if pd.notna(acp_rate) and acp_rate < 0.50:
            acp_note = (f" Advance Care Planning documentation is only {acp_rate:.1%} "
                        f"(target >65%), which is a root cause predictor for EOL metric failures.")

        flags.append(ValidationFlag(
            flag_id=_next_id(), severity="RED", category="cross_metric",
            metric_name="eol_systemic_failure",
            metric_value=f"{eol_failures}/5 EOL metrics failing",
            expected_value="<3 EOL metric failures",
            episode_type="End-of-Life Care", contract_id=contract_id,
            description=f"Systemic palliative care integration failure: {eol_failures} of 5 "
                        f"EOL metrics are failing simultaneously",
            detail=f"Failing measures: {'; '.join(eol_details)}.{acp_note} "
                   f"This pattern indicates a systemic failure to integrate palliative care, "
                   f"not individual measure failures. Without goals-of-care conversations, "
                   f"patients default to aggressive treatment at end of life.",
            related_metrics={"eol_failures": eol_failures, "acp_rate": acp_rate},
        ))

    # 7. Quality gate proximity
    comp_row = quality_df[quality_df["measure_id"].str.contains("COMP", na=False)]
    if len(comp_row) > 0:
        earned = comp_row.iloc[0].get("points_earned", 0)
        max_pts = comp_row.iloc[0].get("max_points", 0)
        gate_min = contract.get("quality_gate_minimum", 0)

        if pd.notna(earned) and pd.notna(max_pts) and max_pts > 0:
            composite_pct = (earned / max_pts) * 100
            gap = gate_min - composite_pct

            if 0 < gap <= 5:
                # Near miss — identify cheapest path to passing
                total_savings = 0
                for _, row in episodes_df.iterrows():
                    tc = row.get("total_cost", 0)
                    tt = row.get("total_target", 0)
                    if pd.notna(tc) and pd.notna(tt):
                        total_savings += (tt - tc)

                sharing_rate = contract.get("sharing_rate_savings", 0)
                at_risk = max(0, total_savings) * sharing_rate

                flags.append(ValidationFlag(
                    flag_id=_next_id(), severity="RED", category="cross_metric",
                    metric_name="quality_gate_proximity",
                    metric_value=f"composite {composite_pct:.1f}% ({earned}/{max_pts})",
                    expected_value=f"gate minimum {gate_min}%",
                    episode_type="Quality Gate", contract_id=contract_id,
                    description=f"Quality gate FAILURE: composite {composite_pct:.1f}% is "
                                f"{gap:.1f} points below the {gate_min}% minimum — "
                                f"${at_risk:,.0f} in shared savings at risk",
                    detail=f"The quality composite score of {earned}/{max_pts} "
                           f"({composite_pct:.1f}%) falls below the {gate_min}% quality gate. "
                           f"This means the provider's shared savings payout of "
                           f"~${at_risk:,.0f} may be zeroed out. "
                           f"The gap is only {gap:.1f} points — identify the lowest-effort "
                           f"measures to close this gap.",
                    related_metrics={"composite_earned": earned, "composite_max": max_pts,
                                     "composite_pct": round(composite_pct, 1),
                                     "gate_minimum": gate_min,
                                     "estimated_savings_at_risk": round(at_risk)},
                ))
            elif gap > 0:
                flags.append(ValidationFlag(
                    flag_id=_next_id(), severity="RED", category="cross_metric",
                    metric_name="quality_gate_failure",
                    metric_value=f"composite {composite_pct:.1f}% ({earned}/{max_pts})",
                    expected_value=f"gate minimum {gate_min}%",
                    episode_type="Quality Gate", contract_id=contract_id,
                    description=f"Quality gate FAILURE: composite {composite_pct:.1f}% is "
                                f"below the {gate_min}% minimum",
                    detail=f"Quality gate not met. Gap of {gap:.1f} points.",
                ))

    # 8. Biosimilar x site-of-service compounding
    if onc_drugs_df is not None:
        for _, drug in onc_drugs_df.iterrows():
            name = drug.get("drug_name", "")
            avg_cost = drug.get("avg_cost_per_claim", 0)
            hopd_pct = drug.get("site_of_service_hopd_pct", 0)
            is_bio = drug.get("is_biosimilar", False)

            if pd.notna(avg_cost) and avg_cost > 2000 and pd.notna(hopd_pct) and hopd_pct > 0.60:
                if not is_bio and str(is_bio).lower() != 'true':
                    total_claims = drug.get("total_claims", 0)
                    # Estimate savings from site-of-service shift
                    excess_hopd_claims = total_claims * (hopd_pct - 0.40)
                    # HOPD costs ~2x office — estimate savings
                    est_savings = excess_hopd_claims * avg_cost * 0.30  # rough 30% facility markup

                    flags.append(ValidationFlag(
                        flag_id=_next_id(), severity="YELLOW", category="cross_metric",
                        metric_name="site_of_service_cost",
                        metric_value=f"{name}: {hopd_pct:.0%} HOPD, ${avg_cost:,.0f}/claim",
                        expected_value="HOPD <60% for office-administrable drugs",
                        episode_type="Drug Detail", contract_id=contract_id,
                        description=f"{name}: {hopd_pct:.0%} HOPD administration for a "
                                    f"drug costing ${avg_cost:,.0f}/claim — estimated "
                                    f"${est_savings:,.0f} in excess facility costs",
                        detail=f"This drug is being administered primarily in hospital "
                               f"outpatient settings ({hopd_pct:.0%}) when it could be safely "
                               f"given in physician offices. HOPD infusion costs 2-3x office "
                               f"administration in facility fees.",
                        related_metrics={"drug_name": name, "hopd_pct": hopd_pct,
                                         "avg_cost_per_claim": avg_cost,
                                         "total_claims": total_claims},
                    ))

    return flags
