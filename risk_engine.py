import math
from typing import Dict, Any, List, Optional

class RiskEngine:
    """
    Explainable Composite Risk Scoring Engine for MPLADS projects.
    Combines NLP compliance flags, cost deviation statistical sigma,
    agency inflation history (modified Z-score), and operational indicators.
    """

    @staticmethod
    def score(work: Dict[str, Any]) -> Dict[str, Any]:
        """
        Computes composite risk score (0-100), risk level (LOW/MEDIUM/HIGH),
        and dynamically generates traceable factors from live field values.
        """
        # Case 1: Prohibited Works (immediate halt from NLP Compliance Engine)
        if work.get("compliance_flag") or work.get("status") == "FLAGGED_PROHIBITED":
            clause = work.get("violated_clause") or "Annexure II Prohibited Work Item"
            desc = work.get("work_description", "")
            return {
                "risk_score": 100,
                "risk_level": "HIGH",
                "factors": [
                    {
                        "label": f"Strict Annexure II Violation Detected: {clause}",
                        "weight": 1.0,
                        "source_field": "work_description",
                        "raw_value": desc,
                        "contribution": 100.0
                    }
                ]
            }

        factors: List[Dict[str, Any]] = []
        total_score = 0.0

        sanctioned_amount = float(work.get("sanctioned_amount") or work.get("proposed_amount") or 0.0)
        predicted_cost = float(work.get("predicted_cost") or work.get("ml_predicted_cost") or sanctioned_amount)
        cat_median = float(work.get("category_median") or 450000.0)
        cat_std = float(work.get("category_std") or 250000.0)
        category_name = work.get("category") or "Normal/Others"

        # 1. Cost Deviation Sigma (Statistical dispersion relative to category benchmark)
        sigma = (sanctioned_amount - cat_median) / max(cat_std, 1.0)
        cost_pts = 0.0
        if sigma > 0:
            if sigma >= 2.5:
                cost_pts = min(40.0, 25.0 + (sigma - 2.5) * 6.0)
            elif sigma >= 1.0:
                cost_pts = 12.0 + ((sigma - 1.0) / 1.5) * 13.0
            else:
                cost_pts = sigma * 12.0

        total_score += cost_pts
        factors.append({
            "label": f"Sanctioned budget is {sigma:+.1f}σ from {category_name} median (₹{sanctioned_amount:,.0f} vs ₹{cat_median:,.0f} baseline)",
            "weight": 0.35,
            "source_field": "sanctioned_amount",
            "raw_value": f"₹{sanctioned_amount:,.0f} (σ = {sigma:+.2f})",
            "contribution": round(cost_pts, 1)
        })

        # 2. AI Cost Engineering Deviation (Predicted vs Sanctioned)
        if predicted_cost > 0 and sanctioned_amount > 0:
            dev_pct = (sanctioned_amount - predicted_cost) / predicted_cost
            if dev_pct > 0.15:
                ai_pts = min(25.0, (dev_pct - 0.15) * 35.0)
                total_score += ai_pts
                factors.append({
                    "label": f"Sanction exceeds ML engineering estimate by {dev_pct * 100:+.1f}% (Predicted: ₹{predicted_cost:,.0f})",
                    "weight": 0.25,
                    "source_field": "deviation_pct",
                    "raw_value": f"{dev_pct * 100:+.1f}%",
                    "contribution": round(ai_pts, 1)
                })

        # 3. Agency Historical Inflation & Modified Z-Score
        agency_name = work.get("ida") or "Unknown Agency"
        mod_z = float(work.get("agency_mod_z_score") or 0.0)
        agency_med_dev = float(work.get("agency_median_deviation") or 0.0)

        agency_pts = 0.0
        if mod_z > 2.5:
            # Extreme systemic inflation track record
            agency_pts = min(30.0, 20.0 + (mod_z - 2.5) * 4.0)
            status_text = "Systemic High Inflation"
        elif mod_z > 0.5:
            agency_pts = (mod_z / 2.5) * 18.0
            status_text = "Moderate Inflation History"
        else:
            agency_pts = 0.0
            status_text = "Normal/Under-budget History"

        total_score += agency_pts
        factors.append({
            "label": f"Agency '{agency_name}' historical Mod Z-Score: {mod_z:+.2f} ({status_text}, historical markup {agency_med_dev * 100:+.1f}%)",
            "weight": 0.25,
            "source_field": "agency_mod_z_score",
            "raw_value": f"Mod Z: {mod_z:.2f}, Median Dev: {agency_med_dev * 100:.1f}%",
            "contribution": round(agency_pts, 1)
        })

        # 4. Workload Concentration & Operational Risk
        active_count = int(work.get("agency_active_work_count") or 1)
        active_pts = 0.0
        if active_count >= 15:
            active_pts = 10.0
        elif active_count >= 5:
            active_pts = 5.0

        total_score += active_pts
        factors.append({
            "label": f"Agency manages {active_count} concurrent active work allocations in district",
            "weight": 0.15,
            "source_field": "agency_active_work_count",
            "raw_value": active_count,
            "contribution": round(active_pts, 1)
        })

        # Final Score Normalization (0 - 100)
        final_score = int(round(min(100.0, max(0.0, total_score))))
        if final_score >= 70:
            level = "HIGH"
        elif final_score >= 40:
            level = "MEDIUM"
        else:
            level = "LOW"

        return {
            "risk_score": final_score,
            "risk_level": level,
            "factors": factors
        }
