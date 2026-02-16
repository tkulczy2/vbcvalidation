"""Arithmetic reconciliation checks: verify internal mathematical consistency."""

import pandas as pd
import numpy as np
from validation import ValidationFlag

_arith_counter = 0


def _next_id():
    global _arith_counter
    _arith_counter += 1
    return f"ARITH-{_arith_counter:03d}"


def validate_arithmetic(episodes_df: pd.DataFrame, quality_df: pd.DataFrame,
                        contract: dict) -> list[ValidationFlag]:
    """Run arithmetic consistency checks on episode and quality data."""
    flags = []
    contract_id = contract["contract_id"]
    specialty = contract.get("specialty", "")

    # Determine episode type column
    ep_type_col = "episode_type" if "episode_type" in episodes_df.columns else "cancer_type"

    for _, row in episodes_df.iterrows():
        if ep_type_col == "cancer_type":
            ep_label = f"{row.get('cancer_type', '')} {row.get('stage_group', '')} {row.get('line_of_therapy', '')}".strip()
        else:
            ep_label = row.get("episode_type", "unknown")

        count = row.get("episode_count", 0)
        avg_cost = row.get("avg_episode_cost", 0)
        target = row.get("target_price", 0)
        total_cost = row.get("total_cost", 0)
        total_target = row.get("total_target", 0)
        variance_pct = row.get("variance_pct", 0)

        if pd.isna(count) or count == 0:
            continue

        # 1. Episode cost reconciliation: count * avg_cost ≈ total_cost
        if pd.notna(avg_cost) and pd.notna(total_cost):
            expected_total = count * avg_cost
            if total_cost != 0:
                diff_pct = abs(expected_total - total_cost) / total_cost
                if diff_pct > 0.01:
                    flags.append(ValidationFlag(
                        flag_id=_next_id(), severity="RED", category="arithmetic",
                        metric_name="episode_cost_reconciliation",
                        metric_value=f"count({count}) x avg(${avg_cost:,.0f}) = ${expected_total:,.0f}",
                        expected_value=f"total_cost = ${total_cost:,.0f}",
                        episode_type=ep_label, contract_id=contract_id,
                        description=f"Episode cost does not reconcile for {ep_label}: "
                                    f"${expected_total:,.0f} vs ${total_cost:,.0f} ({diff_pct:.1%} difference)",
                        detail=f"episode_count ({count}) x avg_episode_cost (${avg_cost:,.0f}) = "
                               f"${expected_total:,.0f}, but total_cost = ${total_cost:,.0f}. "
                               f"Difference of {diff_pct:.1%} exceeds 1% tolerance.",
                        related_metrics={"episode_count": count, "avg_episode_cost": avg_cost,
                                         "total_cost": total_cost},
                    ))

        # 2. Target reconciliation: count * target ≈ total_target
        if pd.notna(target) and pd.notna(total_target):
            expected_target = count * target
            if total_target != 0:
                diff_pct = abs(expected_target - total_target) / total_target
                if diff_pct > 0.01:
                    flags.append(ValidationFlag(
                        flag_id=_next_id(), severity="RED", category="arithmetic",
                        metric_name="target_reconciliation",
                        metric_value=f"count({count}) x target(${target:,.0f}) = ${expected_target:,.0f}",
                        expected_value=f"total_target = ${total_target:,.0f}",
                        episode_type=ep_label, contract_id=contract_id,
                        description=f"Target does not reconcile for {ep_label}",
                        detail=f"episode_count ({count}) x target_price (${target:,.0f}) = "
                               f"${expected_target:,.0f}, but total_target = ${total_target:,.0f}.",
                        related_metrics={"episode_count": count, "target_price": target,
                                         "total_target": total_target},
                    ))

        # 3. Variance calculation: (avg_cost - target) / target ≈ variance_pct
        if pd.notna(target) and pd.notna(avg_cost) and pd.notna(variance_pct) and target != 0:
            expected_var = (avg_cost - target) / target
            diff = abs(expected_var - variance_pct)
            if diff > 0.005:
                flags.append(ValidationFlag(
                    flag_id=_next_id(), severity="YELLOW", category="arithmetic",
                    metric_name="variance_calculation",
                    metric_value=f"reported variance = {variance_pct:.4f}",
                    expected_value=f"calculated variance = {expected_var:.4f}",
                    episode_type=ep_label, contract_id=contract_id,
                    description=f"Variance percentage does not match calculation for {ep_label}",
                    detail=f"(avg_cost - target) / target = ({avg_cost} - {target}) / {target} = "
                           f"{expected_var:.4f}, but variance_pct = {variance_pct:.4f}.",
                    related_metrics={"avg_episode_cost": avg_cost, "target_price": target,
                                     "variance_pct": variance_pct},
                ))

        # 4. Cost component sum (MSK)
        if specialty == "MSK" and "implant_cost_avg" in episodes_df.columns:
            components = ["implant_cost_avg", "facility_cost_avg", "professional_cost_avg",
                          "post_acute_cost_avg", "readmission_cost_avg"]
            comp_vals = {c: row.get(c, np.nan) for c in components}
            non_null_vals = {k: v for k, v in comp_vals.items() if pd.notna(v)}
            if non_null_vals and pd.notna(avg_cost) and avg_cost > 0:
                comp_sum = sum(non_null_vals.values())
                diff_pct = abs(comp_sum - avg_cost) / avg_cost
                if diff_pct > 0.05:
                    flags.append(ValidationFlag(
                        flag_id=_next_id(), severity="YELLOW", category="arithmetic",
                        metric_name="cost_component_sum",
                        metric_value=f"component sum = ${comp_sum:,.0f}",
                        expected_value=f"avg_episode_cost = ${avg_cost:,.0f} (within 5%)",
                        episode_type=ep_label, contract_id=contract_id,
                        description=f"Cost components sum to ${comp_sum:,.0f} vs avg cost "
                                    f"${avg_cost:,.0f} for {ep_label} ({diff_pct:.1%} difference)",
                        detail=f"Cost breakdown: {non_null_vals}. Sum = ${comp_sum:,.0f}. "
                               f"Difference of {diff_pct:.1%} exceeds 5% tolerance (may indicate "
                               f"uncategorized costs).",
                        related_metrics=non_null_vals,
                    ))

        # 4b. Cost component sum (Oncology)
        if specialty == "Oncology" and "drug_cost_avg" in episodes_df.columns:
            components = ["drug_cost_avg", "administration_cost_avg", "inpatient_cost_avg",
                          "er_cost_avg", "imaging_cost_avg", "lab_cost_avg",
                          "supportive_care_cost_avg", "other_cost_avg"]
            comp_vals = {c: row.get(c, np.nan) for c in components}
            non_null_vals = {k: v for k, v in comp_vals.items() if pd.notna(v)}
            if non_null_vals and pd.notna(avg_cost) and avg_cost > 0:
                comp_sum = sum(non_null_vals.values())
                diff_pct = abs(comp_sum - avg_cost) / avg_cost
                if diff_pct > 0.05:
                    flags.append(ValidationFlag(
                        flag_id=_next_id(), severity="YELLOW", category="arithmetic",
                        metric_name="cost_component_sum",
                        metric_value=f"component sum = ${comp_sum:,.0f}",
                        expected_value=f"avg_episode_cost = ${avg_cost:,.0f} (within 5%)",
                        episode_type=ep_label, contract_id=contract_id,
                        description=f"Cost components sum to ${comp_sum:,.0f} vs avg cost "
                                    f"${avg_cost:,.0f} for {ep_label} ({diff_pct:.1%} difference)",
                        detail=f"Cost breakdown: {non_null_vals}. Sum = ${comp_sum:,.0f}.",
                        related_metrics=non_null_vals,
                    ))

    # 5. Discharge disposition sum is checked in schema.py

    # 6. Quality score arithmetic
    quality_no_comp = quality_df[~quality_df["measure_id"].str.contains("COMP", na=False)]
    comp_row = quality_df[quality_df["measure_id"].str.contains("COMP", na=False)]

    if len(comp_row) > 0:
        reported_earned = comp_row.iloc[0].get("points_earned", 0)
        reported_max = comp_row.iloc[0].get("max_points", 0)
        calculated_earned = quality_no_comp["points_earned"].sum()
        calculated_max = quality_no_comp["max_points"].sum()

        if pd.notna(reported_earned) and calculated_earned != reported_earned:
            flags.append(ValidationFlag(
                flag_id=_next_id(), severity="YELLOW", category="arithmetic",
                metric_name="quality_points_sum",
                metric_value=f"reported = {reported_earned}",
                expected_value=f"sum of components = {calculated_earned}",
                episode_type="ALL", contract_id=contract_id,
                description=f"Composite quality points ({reported_earned}) does not match "
                            f"sum of component points ({calculated_earned})",
                detail=f"Individual measures sum to {calculated_earned}/{calculated_max}, "
                       f"but composite reports {reported_earned}/{reported_max}.",
            ))

    # 7. Rate calculation: numerator / denominator ≈ rate
    for _, row in quality_no_comp.iterrows():
        num = row.get("numerator", np.nan)
        denom = row.get("denominator", np.nan)
        rate = row.get("rate", np.nan)
        measure = row.get("measure_name", "unknown")

        if pd.notna(num) and pd.notna(denom) and pd.notna(rate) and denom > 0:
            calc_rate = num / denom
            diff = abs(calc_rate - rate)
            if diff > 0.005:
                flags.append(ValidationFlag(
                    flag_id=_next_id(), severity="RED", category="arithmetic",
                    metric_name="quality_rate_calculation",
                    metric_value=f"reported rate = {rate:.4f}",
                    expected_value=f"num/denom = {num}/{denom} = {calc_rate:.4f}",
                    episode_type="Quality", contract_id=contract_id,
                    description=f"Rate calculation mismatch for '{measure}': "
                                f"reported {rate:.3f} vs calculated {calc_rate:.3f}",
                    detail=f"numerator ({num}) / denominator ({denom}) = {calc_rate:.4f}, "
                           f"but reported rate = {rate:.4f}.",
                    related_metrics={"measure": measure, "numerator": num,
                                     "denominator": denom, "rate": rate},
                ))

    # 8. Member month check
    members = contract.get("attributed_members", 0)
    mm = contract.get("member_months", 0)
    if members > 0 and mm > 0:
        expected_mm = members * 12
        diff_pct = abs(expected_mm - mm) / mm
        if diff_pct > 0.05:
            flags.append(ValidationFlag(
                flag_id=_next_id(), severity="YELLOW", category="arithmetic",
                metric_name="member_months_check",
                metric_value=f"member_months = {mm}",
                expected_value=f"members x 12 = {expected_mm} (within 5%)",
                episode_type="ALL", contract_id=contract_id,
                description=f"Member months ({mm:,}) don't align with attributed members "
                            f"({members:,} x 12 = {expected_mm:,})",
                detail=f"Difference of {diff_pct:.1%} may indicate mid-year enrollment changes.",
            ))

    return flags
