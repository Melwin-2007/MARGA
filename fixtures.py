"""
Predefined Server-Side Fixtures for MPLADS Unified Engine Demo.
Guarantees reliable, deterministic judging presentation without live typing dependencies.
"""

from typing import Dict, Any

DEMO_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "clean": {
        "scenario_key": "clean",
        "title": "Scenario 1: Compliant Infrastructure (Clean)",
        "work_id": "WRK-MYS-001",
        "work_description": "Construction of community hall and approach road for public benefit at Mysore rural",
        "proposed_amount": 400000.0,
        "sanctioned_amount": 400000.0,
        "category": "Normal/Others",
        "ida": "UDUPI(DEPUTY COMMISSIONER UDUPI_IDA)",
        "latitude": 12.3120,
        "longitude": 76.6480,
        "expected_flow": "Compliance Cleared -> Cost Estimated (snapped to ₹4L) -> Low Risk (<40) -> Normal Execution (No Anchor)"
    },
    "inflation": {
        "scenario_key": "inflation",
        "title": "Scenario 2: Systemic Cost Inflation Anomaly",
        "work_id": "WRK-MYS-002",
        "work_description": "Construction of high level bridge across local stream in KR Nagar",
        "proposed_amount": 1500000.0,
        "sanctioned_amount": 12500000.0,
        "category": "Normal/Others",
        "ida": "UTTARA KANNADA(DEPUTY COMMISSIONER UTTARKANNADA KARWAR_IDA)",
        "latitude": 12.3385,
        "longitude": 76.6150,
        "expected_flow": "Compliance Cleared -> Cost Estimated (₹5L) -> +2400% Markup -> High Risk (Score >= 70) -> Automatically Anchored on Map"
    },
    "prohibited": {
        "scenario_key": "prohibited",
        "title": "Scenario 3: Prohibited Annexure II Work",
        "work_id": "WRK-MYS-003",
        "work_description": "Upgradation of facade and painting of the district collectorate building at Mysuru",
        "proposed_amount": 500000.0,
        "sanctioned_amount": 500000.0,
        "category": "Normal/Others",
        "ida": "DHARWAD(DEPUTY COMMISSIONER DHARWAR_IDA)",
        "latitude": 12.3020,
        "longitude": 76.6610,
        "expected_flow": "NLP Layer 2 Catch -> Annexure II Clause 6 Rejection -> Halts at FLAGGED_PROHIBITED -> Never Estimates Cost"
    }
}
