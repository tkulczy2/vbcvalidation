"""AI diagnostic layer: sends validation flags to Claude API for narrative interpretation."""

from dataclasses import dataclass
import json
import re
from collections import defaultdict

from validation import ValidationFlag
from diagnosis.prompt_templates import DIAGNOSTIC_SYSTEM_PROMPT, DIAGNOSTIC_PROMPT


@dataclass
class DiagnosticNarrative:
    episode_type: str
    diagnosis_summary: str
    probable_causes: list[dict]
    questions_for_provider: list[str]
    recommended_interventions: list[dict]
    contract_implications: str
    flags_addressed: list[str]  # flag_ids this narrative covers


def _group_flags_by_episode(flags: list[ValidationFlag]) -> dict[str, list[ValidationFlag]]:
    """Group ValidationFlag objects by their episode_type field."""
    grouped: dict[str, list[ValidationFlag]] = defaultdict(list)
    for flag in flags:
        grouped[flag.episode_type].append(flag)
    return dict(grouped)


def _format_flags(flags: list[ValidationFlag]) -> str:
    """Format a list of ValidationFlags into a readable string for the prompt."""
    lines = []
    for i, flag in enumerate(flags, 1):
        lines.append(f"Flag {i}:")
        lines.append(f"  Severity: {flag.severity}")
        lines.append(f"  Metric: {flag.metric_name}")
        lines.append(f"  Actual Value: {flag.metric_value}")
        lines.append(f"  Expected Value: {flag.expected_value}")
        lines.append(f"  Description: {flag.description}")
        lines.append(f"  Detail: {flag.detail}")
        lines.append("")
    return "\n".join(lines)


def _format_metrics(episode_type: str, report_data: dict) -> str:
    """Extract relevant metrics from report_data for the given episode_type."""
    lines = []

    for key in ["msk_episodes", "msk_quality", "onc_episodes", "onc_quality", "onc_drugs"]:
        df = report_data.get(key)
        if df is None:
            continue

        matched = None
        for col in ["episode_type", "cancer_type", "measure_name", "drug_name"]:
            if col in df.columns:
                matched = df[df[col].astype(str).str.contains(episode_type, case=False, na=False)]
                if not matched.empty:
                    break

        if matched is None or matched.empty:
            continue

        lines.append(f"--- {key} ---")
        for _, row in matched.iterrows():
            for col_name, value in row.items():
                lines.append(f"  {col_name}: {value}")
            lines.append("")

    if not lines:
        return "No additional metrics found for this episode type."

    return "\n".join(lines)


def _call_claude_api(prompt: str, system_prompt: str) -> dict:
    """Call the Claude API and return parsed JSON response."""
    try:
        import anthropic
    except ImportError:
        print("WARNING: anthropic package not installed. Install with: pip install anthropic")
        return {
            "diagnosis_summary": "AI diagnostics unavailable — anthropic package not installed.",
            "probable_causes": [],
            "questions_for_provider": [],
            "recommended_interventions": [],
            "contract_implications": "Unable to assess — AI diagnostics unavailable.",
        }

    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text

        # Handle JSON potentially wrapped in markdown code blocks
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = text.strip()

        return json.loads(json_str)

    except json.JSONDecodeError as e:
        print(f"WARNING: Failed to parse Claude response as JSON: {e}")
        return {
            "diagnosis_summary": f"AI response could not be parsed: {text[:500]}",
            "probable_causes": [],
            "questions_for_provider": [],
            "recommended_interventions": [],
            "contract_implications": "Unable to assess — response parsing failed.",
        }
    except Exception as e:
        print(f"WARNING: Claude API call failed: {e}")
        return {
            "diagnosis_summary": f"AI diagnostics unavailable — API error: {e}",
            "probable_causes": [],
            "questions_for_provider": [],
            "recommended_interventions": [],
            "contract_implications": "Unable to assess — AI diagnostics unavailable.",
        }


def _parse_diagnostic_response(response_dict: dict, episode_type: str,
                                flag_ids: list[str]) -> DiagnosticNarrative:
    """Create a DiagnosticNarrative from parsed JSON response."""
    return DiagnosticNarrative(
        episode_type=episode_type,
        diagnosis_summary=response_dict.get("diagnosis_summary", ""),
        probable_causes=response_dict.get("probable_causes", []),
        questions_for_provider=response_dict.get("questions_for_provider", []),
        recommended_interventions=response_dict.get("recommended_interventions", []),
        contract_implications=response_dict.get("contract_implications", ""),
        flags_addressed=flag_ids,
    )


def generate_all_diagnostics(flags: list[ValidationFlag], contract_metadata: dict,
                              report_data: dict) -> list[DiagnosticNarrative]:
    """Generate AI diagnostic narratives for all flagged issues.

    Groups related flags by episode type, sends each group to Claude API
    with context, and returns structured DiagnosticNarrative objects.
    """
    if not flags:
        return []

    grouped = _group_flags_by_episode(flags)
    contracts = contract_metadata.get("contracts", [])
    narratives: list[DiagnosticNarrative] = []

    for episode_type, episode_flags in grouped.items():
        try:
            contract_id = episode_flags[0].contract_id
            contract = next(
                (c for c in contracts if c.get("contract_id") == contract_id), None
            )
            if contract is None:
                print(f"WARNING: No contract found for '{contract_id}', "
                      f"skipping diagnostics for '{episode_type}'.")
                continue

            formatted_flags = _format_flags(episode_flags)
            formatted_metrics = _format_metrics(episode_type, report_data)
            flag_ids = [f.flag_id for f in episode_flags]

            prompt = DIAGNOSTIC_PROMPT.format(
                specialty=contract.get("specialty", "Unknown"),
                contract_name=contract.get("contract_name", "Unknown"),
                contract_type=contract.get("contract_type", "Unknown"),
                lob=contract.get("lob", "Unknown"),
                performance_period=contract.get("performance_period", "Unknown"),
                attributed_members=contract.get("attributed_members", 0),
                episode_type=episode_type,
                formatted_flags=formatted_flags,
                formatted_metrics=formatted_metrics,
            )

            response_dict = _call_claude_api(prompt, DIAGNOSTIC_SYSTEM_PROMPT)
            narrative = _parse_diagnostic_response(response_dict, episode_type, flag_ids)
            narratives.append(narrative)

        except Exception as e:
            print(f"WARNING: Failed to generate diagnostics for '{episode_type}': {e}")
            continue

    return narratives
