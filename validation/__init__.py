from dataclasses import dataclass, field
from typing import Any


@dataclass
class ValidationFlag:
    flag_id: str
    severity: str  # "RED", "YELLOW", "GREEN"
    category: str  # "schema", "arithmetic", "range", "cross_metric", "specialty"
    metric_name: str
    metric_value: Any
    expected_value: Any
    episode_type: str
    contract_id: str
    description: str
    detail: str
    related_metrics: dict = field(default_factory=dict)
