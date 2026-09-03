"""Study inputs that are small enough to audit directly in source control.

The expert judgments are methodological inputs, not record-level PCM scenario
data. Keeping them as explicit constants makes the AHP calculation reproducible
without distributing the private fused scenario table.
"""

from __future__ import annotations

from fractions import Fraction

import pandas as pd


EXPERT_PROFILES = [
    {
        "expert_id": "E1",
        "expert_profile": "Building energy / HVAC / building envelope / PCM in buildings",
        "main_relevance": "Building energy and PCM building applications",
    },
    {
        "expert_id": "E2",
        "expert_profile": "Structural engineering / concrete materials / durability",
        "main_relevance": "Durability, structural materials, and concrete systems",
    },
    {
        "expert_id": "E3",
        "expert_profile": "Building energy performance / retrofitting / HVAC-energy systems",
        "main_relevance": "Building energy efficiency, retrofitting, and energy systems",
    },
    {
        "expert_id": "E4",
        "expert_profile": "PU-PCM composites / thermal conductivity / AI multiscale modeling",
        "main_relevance": "PCM composites, thermal conductivity, and multiscale modeling",
    },
    {
        "expert_id": "E5",
        "expert_profile": "Computational mechanics / multiscale materials / material failure",
        "main_relevance": "Multiscale materials, heat transfer, and material failure",
    },
]


CONFIRMED_EXPERT_PAIRWISE = [
    ("C1", "Storage capacity and power", "Fast thermal response", {"E1": "2", "E2": "3", "E3": "3", "E4": "1", "E5": "1/2"}),
    ("C2", "Storage capacity and power", "Durability and low loss", {"E1": "1/2", "E2": "1/2", "E3": "1", "E4": "3", "E5": "1"}),
    ("C3", "Storage capacity and power", "Efficiency and load offset", {"E1": "1/3", "E2": "2", "E3": "1/3", "E4": "1/2", "E5": "2"}),
    ("C4", "Fast thermal response", "Durability and low loss", {"E1": "1/3", "E2": "1/5", "E3": "1/3", "E4": "3", "E5": "2"}),
    ("C5", "Fast thermal response", "Efficiency and load offset", {"E1": "1/5", "E2": "1/3", "E3": "1/7", "E4": "1/2", "E5": "3"}),
    ("C6", "Durability and low loss", "Efficiency and load offset", {"E1": "1/2", "E2": "3", "E3": "1/3", "E4": "1/5", "E5": "2"}),
]


def parse_saaty_value(value: object) -> float:
    """Convert a Saaty-scale number or fraction to a positive float."""
    text = str(value).strip()
    parsed = float(Fraction(text)) if "/" in text else float(text)
    if parsed <= 0:
        raise ValueError(f"Saaty values must be positive: {value!r}")
    return parsed


def expert_pairwise_frame() -> pd.DataFrame:
    """Return the confirmed expert judgments in long form."""
    rows: list[dict[str, object]] = []
    for comparison_id, left, right, values in CONFIRMED_EXPERT_PAIRWISE:
        for expert_id, value in values.items():
            rows.append(
                {
                    "expert_id": expert_id,
                    "comparison_id": comparison_id,
                    "left_factor": left,
                    "right_factor": right,
                    "saaty_value_left_over_right": value,
                    "numeric_value": parse_saaty_value(value),
                }
            )
    return pd.DataFrame(rows)
