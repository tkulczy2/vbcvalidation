"""VBC Validation Engine — Entry point orchestrating the full pipeline."""

import json
import sys
import pandas as pd

from validation.schema import validate_schema
from validation.arithmetic import validate_arithmetic
from validation.range_checks import validate_ranges
from validation.cross_metric import validate_cross_metrics
from validation.msk_rules import validate_msk_rules
from validation.onc_rules import validate_onc_rules
from diagnosis.ai_diagnostics import generate_all_diagnostics
from reporting.html_report import generate_html_report


def load_json(path):
    with open(path) as f:
        return json.load(f)


def get_contract(metadata, contract_id):
    for c in metadata["contracts"]:
        if c["contract_id"] == contract_id:
            return c
    raise ValueError(f"Contract {contract_id} not found")


def count_severity(flags, severity):
    return sum(1 for f in flags if f.severity == severity)


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

    # 3. Run validation pipeline
    all_flags = []

    # MSK validation
    print("Running MSK validation...")
    msk_contract = get_contract(contract_metadata, "MSK-2024-001")
    all_flags += validate_schema(msk_episodes, "msk_episodes", msk_contract)
    all_flags += validate_schema(msk_quality, "msk_quality", msk_contract)
    all_flags += validate_arithmetic(msk_episodes, msk_quality, msk_contract)
    all_flags += validate_ranges(msk_episodes, reference_ranges["msk"], msk_contract)
    all_flags += validate_cross_metrics(msk_episodes, msk_quality, msk_contract)
    all_flags += validate_msk_rules(msk_episodes, msk_quality, reference_ranges["msk"], msk_contract)

    # Oncology validation
    print("Running Oncology validation...")
    onc_contract = get_contract(contract_metadata, "ONC-2024-001")
    all_flags += validate_schema(onc_episodes, "onc_episodes", onc_contract)
    all_flags += validate_schema(onc_quality, "onc_quality", onc_contract)
    all_flags += validate_arithmetic(onc_episodes, onc_quality, onc_contract)
    all_flags += validate_ranges(onc_episodes, reference_ranges["oncology"], onc_contract)
    all_flags += validate_cross_metrics(onc_episodes, onc_quality, onc_contract, onc_drugs)
    all_flags += validate_onc_rules(onc_episodes, onc_quality, onc_drugs,
                                     reference_ranges["oncology"], onc_contract)

    print(f"Validation complete: {len(all_flags)} flags "
          f"(RED: {count_severity(all_flags, 'RED')}, "
          f"YELLOW: {count_severity(all_flags, 'YELLOW')}, "
          f"GREEN: {count_severity(all_flags, 'GREEN')})")

    # 4. AI diagnostics (optional)
    diagnostics = []
    try:
        print("Running AI diagnostics...")
        diagnostics = generate_all_diagnostics(all_flags, contract_metadata, {
            "msk_episodes": msk_episodes,
            "msk_quality": msk_quality,
            "onc_episodes": onc_episodes,
            "onc_quality": onc_quality,
            "onc_drugs": onc_drugs,
        })
        print(f"Generated {len(diagnostics)} diagnostic narratives.")
    except Exception as e:
        print(f"AI diagnostics unavailable: {e}. Proceeding with validation-only report.")

    # 5. Generate report
    print("Generating HTML report...")
    generate_html_report(
        all_flags, diagnostics, contract_metadata,
        msk_episodes, msk_quality,
        onc_episodes, onc_quality, onc_drugs,
        output_path="output/vbc_validation_report.html",
    )

    print(f"Report generated: output/vbc_validation_report.html")


if __name__ == "__main__":
    main()
