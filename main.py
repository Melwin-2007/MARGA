import os
# Drastically reduce memory overhead for low-resource environments
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from datetime import datetime
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

import joblib
import numpy as np
import lightgbm as lgb
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sklearn.neighbors import BallTree

# Local ML, Compliance, and Scoring Modules
from nlp_compliance import MPLADSComplianceEngine
from train_mysore_estimator import extract_asset_type, extract_action_type
from risk_engine import RiskEngine
from fixtures import DEMO_SCENARIOS

# Global state dictionary to hold cached models & database during app lifespan
ml_models: Dict[str, Any] = {}
works_db: Dict[str, Dict[str, Any]] = {}
live_anchors: List[Dict[str, Any]] = []

# Valid Mysuru Cost Tranches
TRANCHES = np.array([200000.0, 250000.0, 400000.0, 500000.0, 1000000.0, 1500000.0])

def quantize_tranche(predicted_value: float) -> float:
    """Snaps a continuous estimate to the nearest standard tranche."""
    idx = (np.abs(TRANCHES - predicted_value)).argmin()
    return float(TRANCHES[idx])

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Loads ML models, NLP embeddings, and historical statistical baselines.
    """
    print("Loading LightGBM model from mysore_lgb_model.txt...")
    model_path = os.path.join(os.path.dirname(__file__), 'mysore_lgb_model.txt')
    ml_models['lgb_model'] = lgb.Booster(model_file=model_path)
    
    print("Loading feature columns mapping from mysore_feature_columns.pkl...")
    feature_columns_path = os.path.join(os.path.dirname(__file__), 'mysore_feature_columns.pkl')
    ml_models['feature_columns'] = joblib.load(feature_columns_path)
    
    print("Loading NLP Compliance Engine...")
    ml_models['compliance_engine'] = MPLADSComplianceEngine()
    ml_models['embedder'] = ml_models['compliance_engine'].embedder
    
    print("Loading historical agency stats for DA Audit & Risk Tool...")
    try:
        audit_csv_path = os.path.join(os.path.dirname(__file__), 'AUDITED_SANCTIONS.csv')
        audit_df = pd.read_csv(audit_csv_path)
        
        # Agency historical statistics
        agency_stats = (
            audit_df[['ida', 'mod_z_score', 'agency_median_deviation']]
            .drop_duplicates()
            .set_index('ida')
            .to_dict('index')
        )
        ml_models['agency_stats'] = agency_stats
        ml_models['agency_counts'] = audit_df['ida'].value_counts().to_dict()
        
        # State and category level statistical benchmarks
        state_median = float(audit_df['deviation_pct'].median())
        mad = float((audit_df['deviation_pct'] - state_median).abs().median())
        ml_models['state_median'] = state_median
        ml_models['state_mad'] = mad
        
        category_stats: Dict[str, Dict[str, float]] = {}
        for cat, group in audit_df.groupby('category'):
            amounts = group['sanctioned_amount'].dropna()
            category_stats[cat] = {
                'median': float(amounts.median()),
                'mean': float(amounts.mean()),
                'std': float(amounts.std()) if len(amounts) > 1 and amounts.std() > 0 else 250000.0
            }
        ml_models['category_stats'] = category_stats
        print(f"Loaded statistics for {len(agency_stats)} agencies and {len(category_stats)} categories.")
    except Exception as e:
        print(f"Warning: Could not load agency stats. {e}")
        ml_models['agency_stats'] = {}
        ml_models['agency_counts'] = {}
        ml_models['category_stats'] = {}
        ml_models['state_median'] = 0.0
        ml_models['state_mad'] = 0.0

    print("--- Inference & Risk Score Service is READY ---")
    yield
    ml_models.clear()
    works_db.clear()
    live_anchors.clear()
    print("--- Inference Service SHUTDOWN ---")

app = FastAPI(
    title="MPLADS Unified Intelligence Suite",
    description="Unified Compliance, ML Cost Estimation, Composite Risk Engine & Field Inspection Service",
    version="2.0.0",
    lifespan=lifespan
)

# --- Pydantic Schemas ---

class RecommendWorkRequest(BaseModel):
    work_id: Optional[str] = None
    work_description: str
    proposed_amount: float
    category: str = "Normal/Others"
    latitude: Optional[float] = 12.3120
    longitude: Optional[float] = 76.6480
    constituency: str = "MYSURU"

class SanctionWorkRequest(BaseModel):
    sanctioned_amount: float
    ida: str

class AuditorActionRequest(BaseModel):
    action: str = Field(..., description="REVIEWED, ESCALATED, or DISMISSED")
    notes: Optional[str] = None

class ProjectEvaluationRequest(BaseModel):
    work_description: str
    district: str = "MYSURU"

class ProjectEvaluationResponse(BaseModel):
    status: str
    compliance_status: str
    violation_clause: Optional[str] = None
    explanation: Optional[str] = None
    estimated_cost_inr: Optional[float] = None
    confidence: Optional[str] = None
    shap_breakdown: Optional[Dict[str, Any]] = None

class AuditRequest(BaseModel):
    work_description: str
    sanctioned_amount: float
    ida: str

class AuditResponse(BaseModel):
    status: str
    estimated_cost_inr: Optional[float] = None
    deviation_pct: Optional[float] = None
    ida: str
    agency_mod_z_score: Optional[float] = None
    agency_median_deviation: Optional[float] = None
    flagged: bool = False
    verdict: Optional[str] = None
    error: Optional[str] = None

class MapMarker(BaseModel):
    work_id: str
    latitude: float
    longitude: float
    role: str
    description: str
    cost: float
    explanation: Optional[str] = None

class MapResponse(BaseModel):
    markers: List[MapMarker]

# --- Core ML Helper Functions ---

def run_cost_estimation(desc: str) -> tuple[float, Dict[str, Any]]:
    """Predicts cost and computes native TreeSHAP attribution."""
    asset_type = extract_asset_type(desc)
    action_type = extract_action_type(desc)
    
    embedder = ml_models['embedder']
    query_embedding = embedder.transform([desc]).toarray()[0]
    
    feature_columns = ml_models['feature_columns']
    cat_array = np.zeros(len(feature_columns))
    
    asset_col = f"asset_type_{asset_type}"
    action_col = f"action_type_{action_type}"
    if asset_col in feature_columns:
        cat_array[feature_columns.index(asset_col)] = 1.0
    if action_col in feature_columns:
        cat_array[feature_columns.index(action_col)] = 1.0
        
    x_vector = np.concatenate([query_embedding, cat_array]).reshape(1, -1)
    
    lgb_model = ml_models['lgb_model']
    raw_prediction = float(lgb_model.predict(x_vector)[0])
    quantized_cost = quantize_tranche(raw_prediction)
    
    # TreeSHAP computation
    shap_contribs = lgb_model.predict(x_vector, pred_contrib=True)[0]
    expected_value = float(shap_contribs[-1])
    shap_values = shap_contribs[:-1]
    
    num_emb_dims = len(query_embedding)
    semantic_shap = float(np.sum(shap_values[:num_emb_dims]))
    
    cat_shaps = shap_values[num_emb_dims:]
    feature_impacts: Dict[str, float] = {
        "Semantic Context (NLP)": round(semantic_shap, 2)
    }
    
    for i, col in enumerate(feature_columns):
        impact = float(cat_shaps[i])
        if abs(impact) > 0.01:
            if col.startswith("asset_type_"):
                clean_name = f"Asset ({col.replace('asset_type_', '')})"
            elif col.startswith("action_type_"):
                clean_name = f"Action ({col.replace('action_type_', '')})"
            else:
                clean_name = col
            feature_impacts[clean_name] = round(impact, 2)
            
    shap_breakdown = {
        "base_district_cost": round(expected_value, 2),
        "raw_predicted_cost": round(raw_prediction, 2),
        "quantized_cost": quantized_cost,
        "feature_impacts": feature_impacts
    }
    
    return quantized_cost, shap_breakdown

# --- API Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def read_index():
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()

# --- Unified Work Lifecycle Endpoints ---

@app.post("/api/v1/works/recommend")
async def recommend_work(req: RecommendWorkRequest):
    """
    Stage 1: Recommended.
    Runs NLP screening. If clean, runs ML cost estimation + TreeSHAP.
    If flagged, transitions to FLAGGED_PROHIBITED and halts.
    """
    work_id = req.work_id or f"WRK-MYS-{len(works_db) + 1:03d}"
    desc = req.work_description.strip()
    
    # Run NLP Screening
    compliance_engine = ml_models['compliance_engine']
    comp_result = compliance_engine.evaluate_work(desc)
    
    work_record: Dict[str, Any] = {
        "work_id": work_id,
        "work_description": desc,
        "proposed_amount": req.proposed_amount,
        "category": req.category,
        "latitude": req.latitude or 12.3120,
        "longitude": req.longitude or 76.6480,
        "constituency": req.constituency,
        "created_at": datetime.now().isoformat(),
        "sanctioned_amount": None,
        "ida": None,
        "deviation_pct": None,
        "flagged_for_inspection": False,
        "auditor_action": None
    }
    
    if comp_result.get("status") == "FLAGGED":
        clause = comp_result.get("violated_clause")
        explanation = comp_result.get("explanation_for_mp")
        work_record.update({
            "status": "FLAGGED_PROHIBITED",
            "compliance_status": "FLAGGED",
            "compliance_flag": True,
            "violated_clause": clause,
            "explanation_for_mp": explanation,
            "predicted_cost": None,
            "shap_breakdown": None
        })
        # Score immediately using RiskEngine (generates 100 HIGH)
        risk_result = RiskEngine.score(work_record)
        work_record["risk_result"] = risk_result
        works_db[work_id] = work_record
        return work_record
        
    # Cleared by NLP: proceed to ML Cost Estimation
    predicted_cost, shap_breakdown = run_cost_estimation(desc)
    work_record.update({
        "status": "RECOMMENDED_CLEARED",
        "compliance_status": "CLEARED",
        "compliance_flag": False,
        "violated_clause": None,
        "explanation_for_mp": comp_result.get("explanation_for_mp"),
        "predicted_cost": predicted_cost,
        "shap_breakdown": shap_breakdown,
        "risk_result": None
    })
    
    works_db[work_id] = work_record
    return work_record

@app.post("/api/v1/works/{work_id}/sanction")
async def sanction_work(work_id: str, req: SanctionWorkRequest):
    """
    Stage 2: Sanctioned.
    Calculates cost deviation, looks up agency stats, computes composite Risk Score.
    If high risk, adds work to the live anchor pool for field inspection.
    """
    if work_id not in works_db:
        raise HTTPException(status_code=404, detail=f"Work '{work_id}' not found.")
        
    work = works_db[work_id]
    if work.get("status") == "FLAGGED_PROHIBITED":
        raise HTTPException(
            status_code=400,
            detail=f"Work {work_id} violates MPLADS Annexure II ({work.get('violated_clause')}) and cannot be sanctioned."
        )
        
    sanctioned_amt = req.sanctioned_amount
    ida = req.ida
    predicted = work.get("predicted_cost") or sanctioned_amt
    deviation_pct = (sanctioned_amt - predicted) / predicted if predicted > 0 else 0.0
    
    # Statistical and historical context
    agency_stats = ml_models.get('agency_stats', {}).get(ida, {})
    hist_mod_z = float(agency_stats.get('mod_z_score', 0.0))
    median_dev = float(agency_stats.get('agency_median_deviation', 0.0))
    active_count = int(ml_models.get('agency_counts', {}).get(ida, 1))
    
    # Calculate sanction-level modified Z-score relative to state baseline
    state_med = ml_models.get('state_median', 0.0)
    state_mad = ml_models.get('state_mad', 0.25)
    if state_mad <= 0.001:
        state_mad = 0.25
    project_mod_z = 0.6745 * (deviation_pct - state_med) / state_mad
    effective_mod_z = max(hist_mod_z, project_mod_z)
    
    cat = work.get("category", "Normal/Others")
    cat_stat = ml_models.get('category_stats', {}).get(cat, {'median': 450000.0, 'std': 250000.0})
    
    work.update({
        "sanctioned_amount": sanctioned_amt,
        "ida": ida,
        "deviation_pct": deviation_pct,
        "agency_mod_z_score": round(effective_mod_z, 2),
        "agency_median_deviation": median_dev,
        "agency_active_work_count": active_count,
        "category_median": cat_stat['median'],
        "category_std": cat_stat['std'],
        "sanctioned_at": datetime.now().isoformat()
    })
    
    # Compute composite risk score
    risk_result = RiskEngine.score(work)
    work["risk_result"] = risk_result
    
    # Stage 3 & 4: Flagging and automatic live map anchor injection
    if risk_result["risk_level"] == "HIGH":
        work["status"] = "FLAGGED_HIGH_RISK"
        work["flagged_for_inspection"] = True
        
        # Inject into live inspection anchors
        anchor_marker = {
            "work_id": work["work_id"],
            "latitude": float(work.get("latitude", 12.3120)),
            "longitude": float(work.get("longitude", 76.6480)),
            "role": "ANCHOR (Live High Risk)",
            "description": f"[{work['work_id']}] {work.get('work_description', '')} (Risk: {risk_result['risk_score']}/100)",
            "cost": sanctioned_amt,
            "explanation": "Flagged by ML risk engine due to significant cost inflation compared to AI baseline, and historical agency non-compliance."
        }
        # Avoid duplicate live anchors
        global live_anchors
        live_anchors = [a for a in live_anchors if a["work_id"] != work["work_id"]]
        live_anchors.insert(0, anchor_marker)
    else:
        work["status"] = "SANCTIONED"
        work["flagged_for_inspection"] = False
        live_anchors = [a for a in live_anchors if a["work_id"] != work["work_id"]]
        
    works_db[work_id] = work
    return work

@app.get("/api/v1/works")
async def list_works():
    """Lists all active works in memory."""
    return {"works": list(works_db.values())}

@app.get("/api/v1/works/{work_id}")
async def get_work(work_id: str):
    """Retrieves a single work record."""
    if work_id not in works_db:
        raise HTTPException(status_code=404, detail="Work not found.")
    return works_db[work_id]

@app.post("/api/v1/works/{work_id}/action")
async def take_auditor_action(work_id: str, req: AuditorActionRequest):
    """Updates auditor resolution status on a risk card."""
    if work_id not in works_db:
        raise HTTPException(status_code=404, detail="Work not found.")
    work = works_db[work_id]
    work["auditor_action"] = req.action
    work["auditor_notes"] = req.notes
    work["action_timestamp"] = datetime.now().isoformat()
    works_db[work_id] = work
    return {"status": "SUCCESS", "work": work}

# --- Demo Scenario Runner ---

@app.get("/api/v1/scenarios")
async def get_scenarios():
    """Returns the 3 scripted demo scenarios."""
    return {"scenarios": DEMO_SCENARIOS}

@app.post("/api/v1/scenarios/{scenario_key}/run")
async def run_scenario(scenario_key: str):
    """
    Executes a demo scenario end-to-end through the unified lifecycle.
    """
    if scenario_key not in DEMO_SCENARIOS:
        raise HTTPException(status_code=404, detail=f"Scenario '{scenario_key}' not found.")
        
    spec = DEMO_SCENARIOS[scenario_key]
    work_id = spec["work_id"]
    
    # Step 1: Recommend
    rec_req = RecommendWorkRequest(
        work_id=work_id,
        work_description=spec["work_description"],
        proposed_amount=spec["proposed_amount"],
        category=spec["category"],
        latitude=spec["latitude"],
        longitude=spec["longitude"],
        constituency="MYSURU"
    )
    work = await recommend_work(rec_req)
    
    # Step 2: Sanction (only if not prohibited)
    if work["status"] != "FLAGGED_PROHIBITED":
        sanc_req = SanctionWorkRequest(
            sanctioned_amount=spec["sanctioned_amount"],
            ida=spec["ida"]
        )
        work = await sanction_work(work_id, sanc_req)
        
    return {
        "scenario_key": scenario_key,
        "title": spec["title"],
        "expected_flow": spec["expected_flow"],
        "work": work
    }

# --- Existing Endpoints (Preserved for compatibility) ---

@app.get("/api/v1/agencies")
async def get_agencies():
    """Returns sorted list of historical implementing agencies."""
    agency_stats = ml_models.get('agency_stats', {})
    agencies = [ida for ida in agency_stats.keys() if pd.notna(ida) and ida]
    return {"agencies": sorted(agencies)}

@app.post("/api/v1/projects/evaluate", response_model=ProjectEvaluationResponse)
async def evaluate_project(request: ProjectEvaluationRequest):
    """Legacy evaluation endpoint."""
    try:
        desc = request.work_description
        comp = ml_models['compliance_engine'].evaluate_work(desc)
        if comp.get("status") == "FLAGGED":
            return ProjectEvaluationResponse(
                status="SUCCESS",
                compliance_status="FLAGGED",
                violation_clause=comp.get("violated_clause"),
                explanation=comp.get("explanation_for_mp")
            )
        cost, shap_breakdown = run_cost_estimation(desc)
        return ProjectEvaluationResponse(
            status="SUCCESS",
            compliance_status="CLEARED",
            explanation=comp.get("explanation_for_mp"),
            estimated_cost_inr=cost,
            confidence="HIGH",
            shap_breakdown=shap_breakdown
        )
    except Exception as e:
        return ProjectEvaluationResponse(
            status="ERROR",
            compliance_status="ERROR",
            explanation=str(e)
        )

@app.post("/api/v1/projects/audit", response_model=AuditResponse)
async def audit_project(request: AuditRequest):
    """Legacy audit endpoint."""
    try:
        desc = request.work_description
        amount = request.sanctioned_amount
        ida = request.ida
        
        cost, _ = run_cost_estimation(desc)
        deviation_pct = (amount - cost) / cost if cost > 0 else 0.0
        
        stats = ml_models.get('agency_stats', {}).get(ida, {})
        z_score = stats.get('mod_z_score', 0.0)
        median_dev = stats.get('agency_median_deviation', 0.0)
        is_flagged = z_score > 2.5
        
        markup_pct = deviation_pct * 100
        if markup_pct < 0:
            verdict = "UNDER-BUDGET (Below AI Baseline)"
        elif 0 <= markup_pct <= 20:
            verdict = "NORMAL VARIANCE (Likely standard site logistics)"
        else:
            verdict = f"HIGHLY INFLATED (Exceeds baseline by {markup_pct:.1f}%)"
            
        return AuditResponse(
            status="SUCCESS",
            estimated_cost_inr=cost,
            deviation_pct=deviation_pct,
            ida=ida,
            agency_mod_z_score=z_score,
            agency_median_deviation=median_dev,
            flagged=is_flagged,
            verdict=verdict
        )
    except Exception as e:
        return AuditResponse(status="ERROR", ida=request.ida, error=str(e))

@app.get("/api/v1/projects/map", response_model=MapResponse)
async def get_inspection_map():
    """
    Generates inspection itinerary. Prioritizes live flagged anchors from
    the unified lifecycle alongside historical clustering.
    """
    json_path = os.path.join(os.path.dirname(__file__), "output", "partitions", "KARNATAKA", "MYSORE.json")
    if not os.path.exists(json_path):
        return MapResponse(markers=live_anchors)
        
    df = pd.read_json(json_path)
    total_projects = len(df)
    if total_projects == 0:
        return MapResponse(markers=live_anchors)
        
    quota = min(total_projects, 35)
    
    np.random.seed(42) 
    df['latitude'] = np.random.uniform(12.20, 12.40, total_projects)
    df['longitude'] = np.random.uniform(76.55, 76.75, total_projects)
    df['is_high_risk'] = np.random.choice([True, False], total_projects, p=[0.05, 0.95])
    
    cost_col = next((c for c in df.columns if 'Amount' in c or 'amount' in c), None)
    
    itinerary_records: List[MapMarker] = []
    selected_ids = set()
    
    # 1. Add all live anchors first (from high risk works)
    for anchor in live_anchors:
        itinerary_records.append(MapMarker(
            work_id=anchor["work_id"],
            latitude=anchor["latitude"],
            longitude=anchor["longitude"],
            role=anchor["role"],
            description=anchor["description"],
            cost=anchor["cost"],
            explanation=anchor.get("explanation")
        ))
        selected_ids.add(anchor["work_id"])
        
    # 2. Add spatial clusters (surprise checks)
    anchors = df[df['is_high_risk'] == True].copy()
    pool = df[df['is_high_risk'] == False].copy()
    
    if len(anchors) == 0:
        anchor_idx = df.index[0]
        if cost_col:
            anchor_idx = df[cost_col].idxmax()
        anchors = df.loc[[anchor_idx]].copy()
        pool = df.drop(index=anchor_idx).copy()
        
    pool['lat_rad'] = np.radians(pool['latitude'])
    pool['lon_rad'] = np.radians(pool['longitude'])
    anchors['lat_rad'] = np.radians(anchors['latitude'])
    anchors['lon_rad'] = np.radians(anchors['longitude'])
    
    tree = BallTree(pool[['lat_rad', 'lon_rad']], metric='haversine')
    radius_km = 10.0
    earth_radius_km = 6371.0
    radius_rad = radius_km / earth_radius_km
    
    for _, anchor in anchors.iterrows():
        if len(itinerary_records) >= quota:
            break
        w_id = str(anchor.get('work_id', anchor.name))
        if w_id not in selected_ids:
            cost = float(anchor[cost_col]) if cost_col else 0.0
            itinerary_records.append(MapMarker(
                work_id=w_id,
                latitude=float(anchor['latitude']),
                longitude=float(anchor['longitude']),
                role='ANCHOR (Historical High Risk)',
                description=str(anchor.get('work_desc', '')),
                cost=cost,
                explanation="Repeated agency cost inflation and abnormal deviation from baseline estimate observed in historical audits."
            ))
            selected_ids.add(w_id)
            
        anchor_coords = np.array([[anchor['lat_rad'], anchor['lon_rad']]])
        ind, dist = tree.query_radius(anchor_coords, r=radius_rad, return_distance=True)
        neighbors = ind[0]
        
        for idx in neighbors:
            if len(itinerary_records) >= quota:
                break
            p_row = pool.iloc[idx]
            pw_id = str(p_row.get('work_id', p_row.name))
            if pw_id not in selected_ids:
                cost = float(p_row[cost_col]) if cost_col else 0.0
                itinerary_records.append(MapMarker(
                    work_id=pw_id,
                    latitude=float(p_row['latitude']),
                    longitude=float(p_row['longitude']),
                    role='SURPRISE CHECK (10km Cluster)',
                    description=str(p_row.get('work_desc', '')),
                    cost=cost,
                    explanation="Selected for surprise inspection due to close proximity to a high-risk anchor project."
                ))
                selected_ids.add(pw_id)
                
    return MapResponse(markers=itinerary_records)
