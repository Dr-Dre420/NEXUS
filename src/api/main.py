import os
import sys
import copy
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.api.data_store import store
from src.mechanics import step_forward

app = FastAPI(title="NEXUS Analytical API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SimulationRequest(BaseModel):
    borrower_id: str
    shock_magnitude: float  
    shock_duration_weeks: int = 4
    shock_type: str = "income_reduction"

class InterventionRequest(BaseModel):
    borrower_id: str
    intervention_type: str 
    amount: float
    duration_weeks: int = 4

@app.on_event("startup")
def startup_event():
    try:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        data_dir = os.path.join(project_root, 'data', 'frozen_m2c')
        store.load(data_dir=data_dir)
        print("Successfully loaded frozen M2C world data.")
    except Exception as e:
        print(f"Warning: Failed to load data store on startup: {e}")

@app.get("/health")
def health_check():
    return {"status": "ok", "loaded": store._loaded}

@app.get("/portfolio/summary")
def get_portfolio_summary():
    if not store._loaded:
        raise HTTPException(status_code=503, detail="Data store not loaded")
        
    df = store.world_state['next_df_prop']
    as_of = store.get_as_of_week()
    current_df = df[df['week'] == as_of]
    
    network_exposed_borrowers = int((current_df['borrower_propagation_exposure'] > 0).sum())
    total_exposure = float(current_df['borrower_propagation_exposure'].sum())
    
    return {
        "as_of_week": as_of,
        "total_eligible_borrowers": len(current_df),
        "network_exposed_borrowers": network_exposed_borrowers,
        "aggregate_network_exposure_index": total_exposure,
    }

@app.get("/borrower/{id}")
def get_borrower(id: str):
    if not store._loaded:
        raise HTTPException(status_code=503, detail="Data store not loaded")
    
    # Try integer first, if it fails try string format
    try:
        b_id = int(id.replace('B', ''))
    except:
        b_id = id
        
    row = store.get_borrower_state(b_id, store.get_as_of_week())
    if row is None:
        raise HTTPException(status_code=404, detail="Borrower not found or not eligible at current week")
        
    return {
        "borrower_id": f"B{int(row['borrower_id'])}" if isinstance(row['borrower_id'], (int, float)) else str(row['borrower_id']),
        "group_id": f"G{int(row['group_id'])}" if isinstance(row['group_id'], (int, float)) else str(row['group_id']),
        "financial_state": {
            "cash_buffer_mean_4w": float(row.get('cash_buffer_mean_4w', 0)),
            "weekly_income_mean_4w": float(row.get('weekly_income_mean_4w', 0)),
            "weekly_expenses_mean_4w": float(row.get('weekly_expenses_mean_4w', 0)),
            "current_stress": bool(row.get('current_stress', False)),
        },
        "operational_risk": {
            "score": float(row.get('model_c_score', 0)),
            "baseline_score": float(row.get('model_b_score', 0)),
            "risk_state": "High" if float(row.get('model_c_score', 0)) > 0.5 else "Moderate" if float(row.get('model_c_score', 0)) > 0.2 else "Low"
        },
        "network_evidence": {
            "borrower_propagation_exposure": float(row.get('borrower_propagation_exposure', 0)),
            "peer_stress_mean": float(row.get('peer_stress_mean_4w', 0)),
            "borrower_liability_share": float(row.get('borrower_liability_share', 1.0))
        }
    }

@app.get("/group/{id}")
def get_group(id: str):
    if not store._loaded:
        raise HTTPException(status_code=503, detail="Data store not loaded")
    
    try:
        g_id = int(id.replace('G', ''))
    except:
        g_id = id
        
    df = store.world_state['next_df_prop']
    as_of = store.get_as_of_week()
    current_df = df[df['week'] == as_of]
    
    group_members = current_df[current_df['group_id'] == g_id]
    if len(group_members) == 0:
        raise HTTPException(status_code=404, detail="Group not found or no eligible members")
        
    members_data = []
    for _, row in group_members.iterrows():
        members_data.append({
            "borrower_id": f"B{int(row['borrower_id'])}" if isinstance(row['borrower_id'], (int, float)) else str(row['borrower_id']),
            "current_stress": bool(row['current_stress']),
            "operational_risk_score": float(row.get('model_c_score', 0)),
            "borrower_propagation_exposure": float(row.get('borrower_propagation_exposure', 0)),
            "cash_buffer_mean_4w": float(row.get('cash_buffer_mean_4w', 0)),
            "liability_share": float(row.get('borrower_liability_share', 1.0))
        })
        
    return {
        "group_id": f"G{int(g_id)}" if isinstance(g_id, (int, float)) else str(g_id),
        "member_count": len(members_data),
        "members": members_data,
        "aggregate_group_exposure": sum(m['borrower_propagation_exposure'] for m in members_data),
        "group_coverage_utilization": 0.0 # Placeholder for actual utilization computation if available
    }

@app.get("/network")
def get_network():
    if not store._loaded:
        raise HTTPException(status_code=503, detail="Data store not loaded")
        
    df = store.world_state['next_df_prop']
    as_of = store.get_as_of_week()
    current_df = df[df['week'] == as_of]
    
    groups = current_df.groupby('group_id')
    
    result = []
    for gid, group_data in groups:
        exposure = float(group_data['borrower_propagation_exposure'].sum())
        stress_count = int(group_data['current_stress'].sum())
        member_count = len(group_data)
        result.append({
            "group_id": f"G{int(gid)}" if isinstance(gid, (int, float)) else str(gid),
            "aggregate_exposure": exposure,
            "member_count": member_count,
            "stressed_members": stress_count
        })
        
    return {
        "total_groups": len(result),
        "groups": result
    }

@app.post("/simulate")
def simulate(req: SimulationRequest):
    if not store._loaded:
        raise HTTPException(status_code=503, detail="Data store not loaded")
        
    state = store.get_baseline_state_copy()
    b_id = req.borrower_id
    if b_id not in state.borrower_to_household:
        raise HTTPException(status_code=404, detail="Borrower not found in simulation state")
        
    hh_id = state.borrower_to_household[b_id]
    
    # 1. Apply Shock to baseline
    hh = state.households[hh_id]
    if req.shock_type == "income_reduction":
        hh.weekly_income = max(0.0, hh.weekly_income - req.shock_magnitude)
    elif req.shock_type == "expense_increase":
        hh.weekly_expenses += req.shock_magnitude
    elif req.shock_type == "cash_shock":
        hh.cash_buffer = max(0.0, hh.cash_buffer - req.shock_magnitude)
        
    # 2. Step forward deterministic simulation
    baseline_deltas = []
    for w in range(req.shock_duration_weeks):
        state, shortfalls, group_coverage = step_forward(state)
        # Track borrower and group burden changes
        baseline_deltas.append({
            "week": w + 1,
            "shortfall": shortfalls.get(b_id, 0.0),
            "group_coverage_used": group_coverage.get(b_id, 0.0)
        })
        
    return {
        "status": "success",
        "scenario_type": req.shock_type,
        "borrower_id": b_id,
        "trajectory": baseline_deltas,
        "disclaimer": "Scenario simulation — not a guaranteed forecast. Outputs represent modeled downstream effects under strict deterministic assumptions."
    }

@app.post("/intervene")
def intervene(req: InterventionRequest):
    if not store._loaded:
        raise HTTPException(status_code=503, detail="Data store not loaded")
        
    state = store.get_baseline_state_copy()
    b_id = req.borrower_id
    if b_id not in state.borrower_to_household:
        raise HTTPException(status_code=404, detail="Borrower not found in simulation state")
        
    hh_id = state.borrower_to_household[b_id]
    hh = state.households[hh_id]
    
    # Identify loan
    loan = None
    for l_id, l in state.loans.items():
        if l.borrower_id == b_id:
            loan = l
            break
            
    if not loan:
        raise HTTPException(status_code=400, detail="No active loan found for borrower")

    # Apply intervention
    if req.intervention_type == "restructure":
        loan.weekly_instalment = max(0.0, loan.weekly_instalment - req.amount)
    elif req.intervention_type == "cash_injection":
        hh.cash_buffer += req.amount
    elif req.intervention_type == "payment_adjustment":
        loan.amount_due = max(0.0, loan.amount_due - req.amount)
        
    # Step forward
    baseline_deltas = []
    for w in range(req.duration_weeks):
        state, shortfalls, group_coverage = step_forward(state)
        baseline_deltas.append({
            "week": w + 1,
            "shortfall": shortfalls.get(b_id, 0.0),
            "amount_due": loan.amount_due
        })
        
    return {
        "status": "success",
        "scenario_type": req.intervention_type,
        "borrower_id": b_id,
        "trajectory": baseline_deltas,
        "disclaimer": "Under the stated model assumptions, this counterfactual intervention produces the above deterministic delta. It is not a causal prediction of real-world outcomes."
    }

@app.get("/evaluation")
def get_evaluation():
    if not store._loaded:
        raise HTTPException(status_code=503, detail="Data store not loaded")
    return store.get_evaluation_metrics()

@app.get("/assumptions")
def get_assumptions():
    if not store._loaded:
        raise HTTPException(status_code=503, detail="Data store not loaded")
        
    return {
        "world_seed": store.world_state['seed'],
        "generator_version": "1.0",
        "analytics_version": "M2C-FROZEN",
        "demo_as_of_timestamp": store.get_as_of_week(),
        "disclaimers": [
            "Model C did not demonstrate measurable incremental predictive value over Model B in the evaluated synthetic dataset.",
            "Network / Propagation evidence is diagnostic information only, not predictive."
        ]
    }
