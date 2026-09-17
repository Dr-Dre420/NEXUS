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

def _risk_frame():
    """Frozen as-of cohort with descriptive percentile rank of the calibrated Model C score.

    The calibrated score is extremely low-variance at this base rate, so an absolute
    threshold cannot separate borrowers. Percentile rank is published as a DESCRIPTIVE
    ordering statistic only; the raw probability is always returned alongside it so the
    two are never confused.
    """
    df = store.world_state['next_df_prop']
    cur = df[df['week'] == store.get_as_of_week()].copy()
    cur['risk_percentile'] = cur['model_c_score'].rank(pct=True) * 100.0
    return cur


def _tier(pct: float) -> str:
    if pct >= 95.0:
        return "Elevated"
    if pct >= 80.0:
        return "Watch"
    return "Standard"


@app.get("/portfolio/summary")
def get_portfolio_summary():
    if not store._loaded:
        raise HTTPException(status_code=503, detail="Data store not loaded")

    cur = _risk_frame()
    n = int(len(cur))
    score = cur['model_c_score']
    exp = cur['borrower_propagation_exposure']

    tiers = cur['risk_percentile'].apply(_tier).value_counts().to_dict()

    top_risk = []
    for _, r in cur.nlargest(8, 'model_c_score').iterrows():
        top_risk.append({
            "borrower_id": str(r['borrower_id']),
            "group_id": str(r['group_id']),
            "operational_risk_score": float(r['model_c_score']),
            "risk_percentile": float(r['risk_percentile']),
            "risk_tier": _tier(float(r['risk_percentile'])),
            "borrower_propagation_exposure": float(r['borrower_propagation_exposure']),
            "current_stress": bool(r['current_stress']),
        })

    grp = cur.groupby('group_id').agg(
        aggregate_exposure=('borrower_propagation_exposure', 'sum'),
        mean_exposure=('borrower_propagation_exposure', 'mean'),
        stressed_members=('current_stress', 'sum'),
        member_count=('borrower_id', 'size'),
    ).reset_index()
    top_groups = [{
        "group_id": str(g['group_id']),
        "aggregate_exposure": float(g['aggregate_exposure']),
        "mean_exposure": float(g['mean_exposure']),
        "stressed_members": int(g['stressed_members']),
        "member_count": int(g['member_count']),
    } for _, g in grp.nlargest(8, 'aggregate_exposure').iterrows()]

    # Exposure concentration: share of total exposure held by the top decile of borrowers.
    srt = exp.sort_values(ascending=False)
    top_decile = float(srt.head(max(1, n // 10)).sum())
    total_exp = float(exp.sum())

    return {
        "as_of_week": int(store.get_as_of_week()),
        "world_seed": int(store.world_state['seed']),
        "analytics_version": "M2C-FROZEN",
        "total_borrowers_in_cohort": n,
        "currently_stressed_borrowers": int(cur['current_stress'].sum()),

        "operational_risk": {
            "definition": "Calibrated Model C probability of propagation vulnerability",
            "mean": float(score.mean()),
            "median": float(score.median()),
            "p90": float(score.quantile(0.90)),
            "max": float(score.max()),
            "distinct_values": int(score.nunique()),
            "tier_counts": {k: int(tiers.get(k, 0)) for k in ("Elevated", "Watch", "Standard")},
            "note": "Score variance is very low at this base rate; tiers are percentile bands, not absolute thresholds.",
        },

        "network_evidence": {
            "definition": "Diagnostic modeled propagation exposure — not a prediction",
            "borrowers_with_nonzero_exposure": int((exp > 0).sum()),
            "exposure_sum": total_exp,
            "exposure_mean": float(exp.mean()),
            "exposure_median": float(exp.median()),
            "exposure_max": float(exp.max()),
            "top_decile_share_of_exposure": float(top_decile / total_exp) if total_exp > 0 else 0.0,
            "groups_with_stressed_members": int((grp['stressed_members'] > 0).sum()),
            "total_groups": int(len(grp)),
        },

        "risk_distribution": [
            {"band": b, "count": int(((cur['risk_percentile'] >= lo) & (cur['risk_percentile'] < hi)).sum())}
            for b, lo, hi in [("0-50", 0, 50), ("50-80", 50, 80), ("80-95", 80, 95), ("95-100", 95, 100.1)]
        ],

        "top_risk_borrowers": top_risk,
        "top_exposure_groups": top_groups,
    }


@app.get("/borrower/{id}")
def get_borrower(id: str):
    if not store._loaded:
        raise HTTPException(status_code=503, detail="Data store not loaded")
    
    b_id = f"B{id}" if id.isdigit() else id
        
    row = store.get_borrower_state(b_id, store.get_as_of_week())
    if row is None:
        raise HTTPException(status_code=404, detail="Borrower not found or not eligible at current week")

    _rf = _risk_frame()
    _m = _rf[_rf['borrower_id'] == b_id]
    pct = float(_m.iloc[0]['risk_percentile']) if len(_m) else 0.0
        
    return {
        "borrower_id": b_id,
        "group_id": str(row['group_id']),
        "financial_state": {
            "cash_buffer_mean_4w": float(row['cash_buffer_mean_4w']),
            "weekly_income_mean_4w": float(row['weekly_income_mean_4w']),
            "weekly_expenses_mean_4w": float(row['weekly_expenses_mean_4w']),
            "debt_burden_ratio": float(row['debt_burden_ratio']),
            "amount_due_mean_4w": float(row['amount_due_mean_4w']),
            "principal_remaining": float(row['principal_remaining']),
            "days_past_due_max_4w": float(row['days_past_due_max_4w']),
            "shortfall_mean_4w": float(row['shortfall_mean_4w']),
            "buffer_trend_4w": float(row['buffer_trend_4w']),
            "current_stress": bool(row['current_stress']),
        },
        "operational_risk": {
            "score": float(row['model_c_score']),
            "baseline_score": float(row['model_b_score']),
            "risk_percentile": pct,
            "risk_tier": _tier(pct),
            "cohort_size": int(len(_rf)),
        },
        "network_evidence": {
            "borrower_propagation_exposure": float(row['borrower_propagation_exposure']),
            "peer_stress_mean": float(row['peer_predicted_stress_mean']),
            "borrower_liability_share": float(row['borrower_liability_share'])
        }
    }

@app.get("/group/{id}")
def get_group(id: str):
    if not store._loaded:
        raise HTTPException(status_code=503, detail="Data store not loaded")
    
    g_id = f"G{id}" if id.isdigit() else id
        
    df = store.world_state['next_df_prop']
    as_of = store.get_as_of_week()
    current_df = df[df['week'] == as_of]
    
    group_members = current_df[current_df['group_id'] == g_id]
    if len(group_members) == 0:
        raise HTTPException(status_code=404, detail="Group not found or no eligible members")
        
    _rf = _risk_frame().set_index('borrower_id')

    nodes = []
    member_ids = []
    for _, row in group_members.iterrows():
        b_id_str = str(row['borrower_id'])
        member_ids.append(b_id_str)
        nodes.append({
            "id": b_id_str,
            "group_id": str(g_id),
            "current_stress": bool(row['current_stress']),
            "operational_risk_score": float(row['model_c_score']),
            "risk_percentile": float(_rf.loc[b_id_str, 'risk_percentile']) if b_id_str in _rf.index else 0.0,
            "risk_tier": _tier(float(_rf.loc[b_id_str, 'risk_percentile'])) if b_id_str in _rf.index else "Standard",
            "borrower_propagation_exposure": float(row['borrower_propagation_exposure']),
            "cash_buffer_mean_4w": float(row['cash_buffer_mean_4w']),
            "weekly_income_mean_4w": float(row['weekly_income_mean_4w']),
            "liability_share": float(row['borrower_liability_share'])
        })
        
    # Central JLG group node
    group_node_id = str(g_id)
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
        "aggregate_group_exposure": float(group_members['borrower_propagation_exposure'].sum()),
        "mean_group_exposure": float(group_members['borrower_propagation_exposure'].mean()),
        "stressed_members": int(group_members['current_stress'].sum()),
        "group_buffer_total": float(group_members['cash_buffer_mean_4w'].sum()),
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
            "group_id": str(gid),
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
        
    if req.shock_magnitude < 0:
        raise HTTPException(status_code=400, detail="Shock magnitude must be >= 0")
    if not (1 <= req.shock_duration_weeks <= 52):
        raise HTTPException(status_code=400, detail="Duration must be between 1 and 52 weeks")
    if req.shock_type not in ["income_reduction", "expense_increase", "cash_shock"]:
        raise HTTPException(status_code=400, detail="Unknown shock type")

    baseline_state = store.get_baseline_state_copy()
    scenario_state = store.get_baseline_state_copy()
    
    b_id = f"B{req.borrower_id}" if req.borrower_id.isdigit() else req.borrower_id
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
        
    if req.amount < 0:
        raise HTTPException(status_code=400, detail="Amount must be >= 0")
    if not (1 <= req.duration_weeks <= 52):
        raise HTTPException(status_code=400, detail="Duration must be between 1 and 52 weeks")

    baseline_state = store.get_baseline_state_copy()
    scenario_state = store.get_baseline_state_copy()
    
    b_id = f"B{req.borrower_id}" if req.borrower_id.isdigit() else req.borrower_id
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
        
    eval_data = store.get_evaluation_metrics()
    conclusion = eval_data.get("final_scientific_conclusion", eval_data.get("results", [""])[0])
    
    return {
        "world_seed": int(store.world_state['seed']),
        "generator_version": "1.0",
        "analytics_version": "M2C-FROZEN",
        "as_of_week": int(store.get_as_of_week()),
        "temporal_evaluation_rules": "Purge gap minimum 4 weeks strictly enforced",
        "propagation_horizon": "4-week",
        "pv_threshold": ">= 0.30",
        "attribution_basis": "cumulative-shortfall",
        "operational_risk_score_definition": "CALIBRATED MODEL C SCORE",
        "network_propagation_evidence_definition": "DIAGNOSTIC EVIDENCE",
        "synthetic_data_disclaimer": "Results are specific to the synthetic worlds generated under the current NEXUS assumptions and should not be generalized to real microfinance populations.",
        "m2c_conclusion": conclusion,
        "disclaimers": [
            "Network / Propagation evidence is diagnostic information only, not predictive.",
            "This model does not establish causality between network topology and borrower stress."
        ]
    }
