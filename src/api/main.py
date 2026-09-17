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
        "as_of_week": int(as_of),
        "total_eligible_borrowers": int(len(current_df)),
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
        
    nodes = []
    member_ids = []
    for _, row in group_members.iterrows():
        b_id_str = f"B{int(row['borrower_id'])}" if isinstance(row['borrower_id'], (int, float)) else str(row['borrower_id'])
        member_ids.append(b_id_str)
        nodes.append({
            "id": b_id_str,
            "group_id": f"G{int(g_id)}" if isinstance(g_id, (int, float)) else str(g_id),
            "current_stress": bool(row['current_stress']),
            "operational_risk_score": float(row.get('model_c_score', 0)),
            "borrower_propagation_exposure": float(row.get('borrower_propagation_exposure', 0)),
            "cash_buffer_mean_4w": float(row.get('cash_buffer_mean_4w', 0)),
            "liability_share": float(row['borrower_liability_share'])
        })
        
    # Central JLG group node
    group_node_id = f"G{int(g_id)}" if isinstance(g_id, (int, float)) else str(g_id)
    nodes.append({
        "id": group_node_id,
        "type": "group",
        "current_stress": False,
        "operational_risk_score": 0.0,
        "borrower_propagation_exposure": 0.0,
        "cash_buffer_mean_4w": 0.0,
        "liability_share": 1.0
    })
        
    # Star graph topology representing Branch -> Centre -> JLG -> Borrowers
    edges = []
    for member_id in member_ids:
        # Edge from group liability mechanism to member borrower
        edges.append({
            "source": group_node_id,
            "target": member_id,
            "type": "group_liability"
        })
            
    return {
        "group_id": group_node_id,
        "member_count": len(member_ids),
        "nodes": nodes,
        "edges": edges,
        "aggregate_group_exposure": sum(float(row.get('borrower_propagation_exposure', 0)) for _, row in group_members.iterrows()),
        "group_coverage_utilization": 0.0 # Placeholder
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
        
    baseline_state = store.get_baseline_state_copy()
    scenario_state = store.get_baseline_state_copy()
    
    b_id = req.borrower_id
    if b_id not in baseline_state.borrower_to_household:
        raise HTTPException(status_code=404, detail="Borrower not found in simulation state")
        
    hh_id = scenario_state.borrower_to_household[b_id]
    
    # Identify group to track ripple effects
    group_id = None
    for l_id, l in scenario_state.loans.items():
        if l.borrower_id == b_id:
            group_id = l.group_id
            break
            
    group_members = []
    if group_id and group_id in scenario_state.groups:
        group_members = scenario_state.groups[group_id].members
    
    # 1. Apply Shock to scenario
    hh = scenario_state.households[hh_id]
    if req.shock_type == "income_reduction":
        hh.weekly_income = max(0.0, hh.weekly_income - req.shock_magnitude)
    elif req.shock_type == "expense_increase":
        hh.weekly_expenses += req.shock_magnitude
    elif req.shock_type == "cash_shock":
        hh.cash_buffer = max(0.0, hh.cash_buffer - req.shock_magnitude)
        
    # 2. Step forward deterministic simulation for BOTH
    trajectory = []
    
    for w in range(req.shock_duration_weeks):
        baseline_state, b_sf, b_cov = step_forward(baseline_state)
        scenario_state, s_sf, s_cov = step_forward(scenario_state)
        
        week_data = {
            "week": w + 1,
            "target_borrower": {
                "baseline": {
                    "cash_buffer": baseline_state.households[baseline_state.borrower_to_household[b_id]].cash_buffer,
                    "shortfall": b_sf.get(b_id, 0.0),
                    "group_coverage_used": b_cov.get(b_id, 0.0)
                },
                "scenario": {
                    "cash_buffer": scenario_state.households[scenario_state.borrower_to_household[b_id]].cash_buffer,
                    "shortfall": s_sf.get(b_id, 0.0),
                    "group_coverage_used": s_cov.get(b_id, 0.0)
                }
            },
            "group_members": {}
        }
        
        for gm in group_members:
            if gm != b_id:
                week_data["group_members"][gm] = {
                    "baseline": {
                        "cash_buffer": baseline_state.households[baseline_state.borrower_to_household[gm]].cash_buffer,
                        "shortfall": b_sf.get(gm, 0.0),
                        "group_coverage_used": b_cov.get(gm, 0.0)
                    },
                    "scenario": {
                        "cash_buffer": scenario_state.households[scenario_state.borrower_to_household[gm]].cash_buffer,
                        "shortfall": s_sf.get(gm, 0.0),
                        "group_coverage_used": s_cov.get(gm, 0.0)
                    }
                }
                
        trajectory.append(week_data)
        
    return {
        "status": "success",
        "scenario_type": req.shock_type,
        "borrower_id": b_id,
        "trajectory": trajectory,
        "disclaimer": "Scenario simulation — not a guaranteed forecast. Outputs represent modeled downstream effects under strict deterministic assumptions."
    }

@app.post("/intervene")
def intervene(req: InterventionRequest):
    if not store._loaded:
        raise HTTPException(status_code=503, detail="Data store not loaded")
        
    baseline_state = store.get_baseline_state_copy()
    scenario_state = store.get_baseline_state_copy()
    
    b_id = req.borrower_id
    if b_id not in baseline_state.borrower_to_household:
        raise HTTPException(status_code=404, detail="Borrower not found in simulation state")
        
    # Identify group to track ripple effects
    group_id = None
    for l_id, l in scenario_state.loans.items():
        if l.borrower_id == b_id:
            group_id = l.group_id
            break
            
    group_members = []
    if group_id and group_id in scenario_state.groups:
        group_members = scenario_state.groups[group_id].members
        
    # Apply intervention via dedicated module
    from src.interventions import apply_intervention
    try:
        apply_intervention(scenario_state, b_id, req.intervention_type, req.amount)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    # Step forward deterministic simulation for BOTH
    trajectory = []
    
    for w in range(req.duration_weeks):
        baseline_state, b_sf, b_cov = step_forward(baseline_state)
        scenario_state, s_sf, s_cov = step_forward(scenario_state)
        
        # Get target loan amount due for both states
        b_loan_base = next((l for l in baseline_state.loans.values() if l.borrower_id == b_id), None)
        b_loan_scen = next((l for l in scenario_state.loans.values() if l.borrower_id == b_id), None)
        
        week_data = {
            "week": w + 1,
            "target_borrower": {
                "baseline": {
                    "cash_buffer": baseline_state.households[baseline_state.borrower_to_household[b_id]].cash_buffer,
                    "shortfall": b_sf.get(b_id, 0.0),
                    "amount_due": b_loan_base.amount_due if b_loan_base else 0.0,
                    "group_coverage_used": b_cov.get(b_id, 0.0)
                },
                "scenario": {
                    "cash_buffer": scenario_state.households[scenario_state.borrower_to_household[b_id]].cash_buffer,
                    "shortfall": s_sf.get(b_id, 0.0),
                    "amount_due": b_loan_scen.amount_due if b_loan_scen else 0.0,
                    "group_coverage_used": s_cov.get(b_id, 0.0)
                }
            },
            "group_members": {}
        }
        
        for gm in group_members:
            if gm != b_id:
                week_data["group_members"][gm] = {
                    "baseline": {
                        "cash_buffer": baseline_state.households[baseline_state.borrower_to_household[gm]].cash_buffer,
                        "shortfall": b_sf.get(gm, 0.0),
                        "group_coverage_used": b_cov.get(gm, 0.0)
                    },
                    "scenario": {
                        "cash_buffer": scenario_state.households[scenario_state.borrower_to_household[gm]].cash_buffer,
                        "shortfall": s_sf.get(gm, 0.0),
                        "group_coverage_used": s_cov.get(gm, 0.0)
                    }
                }
                
        trajectory.append(week_data)
        
    return {
        "status": "success",
        "scenario_type": req.intervention_type,
        "borrower_id": b_id,
        "trajectory": trajectory,
        "disclaimer": "Modeled counterfactual under stated assumptions."
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
        "world_seed": int(store.world_state['seed']),
        "generator_version": "1.0",
        "analytics_version": "M2C-FROZEN",
        "demo_as_of_timestamp": int(store.get_as_of_week()),
        "temporal_evaluation_rules": "Purge gap minimum 4 weeks strictly enforced",
        "propagation_horizon": "4-week",
        "pv_threshold": ">= 0.30",
        "attribution_basis": "cumulative-shortfall",
        "operational_risk_score_definition": "CALIBRATED MODEL C SCORE",
        "network_propagation_evidence_definition": "DIAGNOSTIC EVIDENCE",
        "synthetic_data_disclaimer": "Results are specific to the synthetic worlds generated under the current NEXUS assumptions.",
        "disclaimers": [
            "Model C did not demonstrate measurable incremental predictive value over Model B in the evaluated synthetic dataset.",
            "Network / Propagation evidence is diagnostic information only, not predictive."
        ]
    }
