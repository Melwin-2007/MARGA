# Prompt for Antigravity — Unify the Pipeline + Build the Composite Risk Score Engine

## Context (paste or link your repo first)

You are working in an existing FastAPI project called `marga_deployment`. It already contains working, independently-functioning ML components:

- `nlp_compliance.py` — `MPLADSComplianceEngine`: a three-layer hybrid NLP pipeline (Layer 1 regex/keyword filter for explicit prohibited-works phrases → Layer 2 TF-IDF cosine-similarity match against Annexure II violation clauses → Layer 3 lightweight NER pattern match for named-individual memorial violations).
- `train_mysore_estimator.py` — `MysoreCostEstimator`: a LightGBM regressor (text TF-IDF embeddings + asset-category one-hot + action-type one-hot, 392 features) that predicts a continuous project cost and snaps it to standardized Mysuru budget tranches (₹2L / 2.5L / 4L / 5L / 10L / 15L), with native TreeSHAP attribution for feature-level cost drivers.
- `main.py` — FastAPI app with lifespan cache loading of the trained model/vectorizer/feature-columns artifacts, an agency corruption/inflation audit routine (deviation_pct of sanctioned vs. ML-predicted cost, cross-referenced against `AUDITED_SANCTIONS.csv` using modified Z-scores, flags at `mod_z_score > 2.5`), and a geospatial audit-itinerary planner (`sklearn.neighbors.BallTree`, haversine metric, anchor-based clustering of high-risk "anchor" projects with surprise inspections in a 10km radius).
- `templates/index.html` — a dark-mode glassmorphism demo frontend with Leaflet.js maps and typewriter auto-fill demo scenarios.
- Trained artifacts already exported: `mysore_lgb_model.txt`, `mysore_tfidf_vectorizer.pkl`, `mysore_feature_columns.pkl`.
- `Dockerfile` and `requirements.txt` already set up.

**These four engines currently run as separate, disconnected calls.** Your job is NOT to build new ML models. Your job is to (1) wire the existing engines into one coherent lifecycle, and (2) build a new composite risk-scoring layer that combines their outputs into a single explainable score, surfaced in the UI the way a district auditor would actually see it.

---

## Task 1 — Unify the engines into one end-to-end work lifecycle

Implement a single stateful `Work` object that moves through these stages, calling the existing engines at the right point instead of leaving them as isolated demo buttons:

1. **Recommended** — MP submits `work_description`, `proposed_amount`, `category`, `location (lat/lon)`.
   - Immediately run `MPLADSComplianceEngine` on `work_description`. If flagged, the work is created in a `FLAGGED_PROHIBITED` status with the matched Annexure II clause attached — do not proceed to cost estimation.
   - If clean, run `MysoreCostEstimator` to produce `predicted_cost` + TreeSHAP `feature_contributions`. Store both on the `Work` record.

2. **Sanctioned** — DA submits `sanctioned_amount` + `implementing_agency_id`.
   - Compute `deviation_pct = (sanctioned_amount - predicted_cost) / predicted_cost`.
   - Run the existing agency inflation-audit logic against `AUDITED_SANCTIONS.csv` (modified Z-score) for this agency's rolling history.
   - Call the new `RiskEngine` (Task 2) to compute and store a composite risk score on the `Work` record at this point — this is the moment a real DA would need it.

3. **Flagged for inspection** — if composite risk score crosses your chosen "HIGH" threshold, add the work to the anchor pool used by the existing `BallTree` itinerary planner, so the next itinerary-generation call includes it as an anchor automatically. Don't require a manual step to add anchors.

4. **Itinerary generated** — expose the existing itinerary planner as an endpoint that pulls current anchors from step 3 live, rather than from a static/demo dataset.

Build this as real state transitions (a `status` field + timestamps), not just four independent function calls — the point is that the frontend and judges can watch one work item move through all four stages in a single demo run.

---

## Task 2 — Build the composite Risk Score engine

Create a new module `risk_engine.py` with a `RiskEngine.score(work) -> RiskResult` function.

**Inputs it consumes** (all from data you already compute elsewhere — do not build new models):
- `cost_deviation_sigma` — how many standard deviations the sanctioned amount is from the category+state median (you already compute category medians for the peer-benchmark tag; reuse that).
- `agency_mod_z_score` — the modified Z-score from the existing inflation-audit routine.
- `agency_active_work_count` — count of other active works currently tied to the same implementing agency (simple DB/CSV aggregation, not ML).
- `compliance_flag` — boolean + matched clause from the NLP engine (Task 1, step 1).
- `time_to_completion_ratio` — (optional, include only if you have a completion timestamp in your demo data) actual days-to-completion vs. category norm.

**Output — `RiskResult`:**
```python
{
  "risk_score": 87,              # 0-100
  "risk_level": "HIGH",          # LOW / MEDIUM / HIGH, thresholds you define e.g. <40/<70/>=70
  "factors": [
      {"label": "Cost 2.9σ above category+state median", "weight": 0.35, "source_field": "sanctioned_amount"},
      {"label": "Agency linked to 6 flagged works across 3 MPs", "weight": 0.30, "source_field": "agency_active_work_count"},
      {"label": "Completed in 11 days; category norm is 210 days", "weight": 0.20, "source_field": "completion_days"},
      {"label": "No completion photo on file", "weight": 0.15, "source_field": "photo_evidence"}
  ]
}
```

Each factor string must be generated from the actual field values on the `Work` record (not hardcoded text) so the evidence is traceable — this is what lets a DA click "View Evidence" and see each line sourced back to a real field, matching the audit-report style the project is designed around.

Use a simple, explainable weighted-sum or rule-based scoring (NOT a black-box model) — explainability is the whole point of this layer; a second ML model here would undermine the "we can quote the exact reason" pitch.

---

## Task 3 — Frontend: the Risk Alert Card

Add a card component to `templates/index.html` (reuse the existing glassmorphism style) that renders `RiskResult` like this:

```
Work #[id] — [category] — Risk Score: [score]/100 — [level]
• [factor 1 label]
• [factor 2 label]
• [factor 3 label]
[View Evidence]  [Mark Reviewed]  [Escalate to State]  [Dismiss as False Positive]
```

- **View Evidence** expands each factor to show the raw source field value it was computed from.
- Also render the TreeSHAP feature contributions from Task 1 as a simple horizontal bar chart (positive/negative bars) wherever the predicted cost is shown — this is your explainability visual, don't leave it as raw JSON.
- If the work was flagged by the NLP compliance engine, show the matched Annexure II clause text directly on the card, not just "flagged: true."

---

## Task 4 — Demo scenarios

Seed exactly 3 scripted `Work` records, wired into the existing typewriter auto-fill on the frontend, so each button press drives the full lifecycle through a different engine:

1. **Clean project** — nothing fires; ends at LOW risk, no anchor added.
2. **Cost-inflation case** — sanctioned amount deliberately set to trigger `agency_mod_z_score > 2.5`; ends at HIGH risk via the cost/agency path.
3. **Prohibited-work case** — `work_description` deliberately matches an Annexure II clause; halts at `FLAGGED_PROHIBITED` in step 1, never reaches cost estimation.

Hardcode these three payloads server-side (or in a fixtures file) so the demo never depends on live typing or network input during judging.

---

## Explicitly out of scope for this pass

Do not attempt any of the following — they're real parts of the larger vision but are multi-day builds and would dilute a focused MVP demo:
- IA mobile app / computer-vision ghost-asset photo classification
- The discrepancy reconciliation engine (aggregate vs. individual-record double-entry check)
- Role-separated dashboards for MP / State Nodal / MoSPI (build only the DA-facing view for now)
- Immutable evidence ledger / hash-chaining
- WhatsApp bot, asset catalog, smart-tranche disbursement

---

## Acceptance criteria

- [ ] A single `Work` record can be walked through Recommended → Sanctioned → (Flagged/anchored if high risk) → Itinerary-included, via real API calls, with no manual re-wiring between engines.
- [ ] `RiskResult` factors are always generated from live field values, never hardcoded strings, for any `Work` you construct.
- [ ] The 3 demo scenarios each visibly trigger a different engine when run end-to-end from the frontend.
- [ ] The risk alert card and SHAP bar chart render in the browser, not just in API responses.
- [ ] `docker-compose up` (or existing Dockerfile) still runs the whole thing with no new external dependencies beyond what's already in `requirements.txt`, unless unavoidable — if you add a dependency, note it explicitly.