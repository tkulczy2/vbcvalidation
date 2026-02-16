"""HTML report generator: assembles validation results into a professional report."""

import os
from datetime import datetime
from collections import defaultdict

import pandas as pd
from jinja2 import Environment, FileSystemLoader


def _severity_order(flag):
    """Sort key: RED first, then YELLOW, then GREEN."""
    order = {"RED": 0, "YELLOW": 1, "GREEN": 2}
    return order.get(flag.severity, 3)


def _calculate_financial_summary(episodes_df, contract):
    """Calculate financial summary for a contract."""
    total_cost = episodes_df["total_cost"].sum()
    total_target = episodes_df["total_target"].sum()
    total_episodes = episodes_df["episode_count"].sum()
    variance = total_cost - total_target
    variance_pct = variance / total_target if total_target > 0 else 0

    savings = total_target - total_cost  # positive = savings, negative = losses
    sharing_rate = contract.get("sharing_rate_savings", 0) if savings > 0 else contract.get("sharing_rate_losses", 0)
    provider_share = savings * sharing_rate

    return {
        "total_episodes": int(total_episodes),
        "total_cost": total_cost,
        "total_target": total_target,
        "variance": variance,
        "variance_pct": variance_pct,
        "savings": savings,
        "sharing_rate": sharing_rate,
        "provider_share": provider_share,
    }


def _get_quality_gate_status(quality_df, contract):
    """Determine quality gate pass/fail status."""
    comp_row = quality_df[quality_df["measure_id"].str.contains("COMP", na=False)]
    if len(comp_row) == 0:
        return {"pass": None, "composite_score": None, "gate_minimum": None}

    earned = comp_row.iloc[0].get("points_earned", 0)
    max_pts = comp_row.iloc[0].get("max_points", 0)
    gate_min = contract.get("quality_gate_minimum", 0)
    composite_pct = (earned / max_pts * 100) if max_pts > 0 else 0

    return {
        "pass": composite_pct >= gate_min,
        "composite_score": composite_pct,
        "composite_earned": earned,
        "composite_max": max_pts,
        "gate_minimum": gate_min,
    }


def generate_html_report(all_flags, diagnostics, contract_metadata,
                          msk_episodes, msk_quality,
                          onc_episodes, onc_quality, onc_drugs,
                          output_path="output/vbc_validation_report.html"):
    """Generate the self-contained HTML validation report."""

    contracts = contract_metadata.get("contracts", [])
    msk_contract = next((c for c in contracts if c["contract_id"] == "MSK-2024-001"), {})
    onc_contract = next((c for c in contracts if c["contract_id"] == "ONC-2024-001"), {})

    # Group flags by contract
    msk_flags = sorted([f for f in all_flags if f.contract_id == "MSK-2024-001"], key=_severity_order)
    onc_flags = sorted([f for f in all_flags if f.contract_id == "ONC-2024-001"], key=_severity_order)

    # Severity counts
    def count_sev(flags):
        counts = {"RED": 0, "YELLOW": 0, "GREEN": 0}
        for f in flags:
            counts[f.severity] = counts.get(f.severity, 0) + 1
        return counts

    msk_severity = count_sev(msk_flags)
    onc_severity = count_sev(onc_flags)
    total_severity = {k: msk_severity.get(k, 0) + onc_severity.get(k, 0) for k in ["RED", "YELLOW", "GREEN"]}

    # Group flags by episode type
    flags_by_episode = defaultdict(list)
    for f in all_flags:
        flags_by_episode[f.episode_type].append(f)

    # Financial summaries
    msk_financial = _calculate_financial_summary(msk_episodes, msk_contract)
    onc_financial = _calculate_financial_summary(onc_episodes, onc_contract)

    # Quality gate
    msk_quality_gate = _get_quality_gate_status(msk_quality, msk_contract)
    onc_quality_gate = _get_quality_gate_status(onc_quality, onc_contract)

    msk_financial["quality_gate"] = msk_quality_gate
    onc_financial["quality_gate"] = onc_quality_gate

    # If quality gate fails, calculate forfeited amount
    if onc_quality_gate.get("pass") is False and onc_financial["savings"] > 0:
        onc_financial["forfeited_savings"] = onc_financial["provider_share"]
        onc_financial["provider_share"] = 0
    else:
        onc_financial["forfeited_savings"] = 0

    if msk_quality_gate.get("pass") is False and msk_financial["savings"] > 0:
        msk_financial["forfeited_savings"] = msk_financial["provider_share"]
        msk_financial["provider_share"] = 0
    else:
        msk_financial["forfeited_savings"] = 0

    # Diagnostics by episode type
    diag_by_episode = {}
    for d in diagnostics:
        diag_by_episode[d.episode_type] = d

    # Convert DataFrames to dicts for template
    msk_episodes_data = msk_episodes.to_dict("records")
    onc_episodes_data = onc_episodes.to_dict("records")
    msk_quality_data = msk_quality.to_dict("records")
    onc_quality_data = onc_quality.to_dict("records")
    onc_drugs_data = onc_drugs.to_dict("records")

    # Episode types for iteration
    msk_episode_types = msk_episodes["episode_type"].tolist()
    onc_episode_labels = []
    for _, row in onc_episodes.iterrows():
        label = f"{row['cancer_type']} {row['stage_group']} {row['line_of_therapy']}".strip()
        onc_episode_labels.append(label)

    # Set up Jinja2
    template_dir = os.path.dirname(os.path.abspath(__file__))
    env = Environment(loader=FileSystemLoader(template_dir), autoescape=False)
    template = env.get_template("report_template.html")

    html = template.render(
        msk_contract=msk_contract,
        onc_contract=onc_contract,
        msk_flags=msk_flags,
        onc_flags=onc_flags,
        total_severity=total_severity,
        msk_severity=msk_severity,
        onc_severity=onc_severity,
        msk_financial=msk_financial,
        onc_financial=onc_financial,
        msk_episodes_data=msk_episodes_data,
        onc_episodes_data=onc_episodes_data,
        msk_quality_data=msk_quality_data,
        onc_quality_data=onc_quality_data,
        onc_drugs_data=onc_drugs_data,
        msk_episode_types=msk_episode_types,
        onc_episode_labels=onc_episode_labels,
        flags_by_episode=dict(flags_by_episode),
        diag_by_episode=diag_by_episode,
        all_flags=sorted(all_flags, key=_severity_order),
        generation_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
