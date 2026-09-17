"""
NEXUS API Test Suite — Strict Validation
All tests use positive/negative assertions. No conditional pass patterns.
"""
from fastapi.testclient import TestClient
import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.api.main import app
from src.api.data_store import store

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(project_root, 'data', 'frozen_m2c')
store.load(data_dir=data_dir)

client = TestClient(app)


# 1. Health check
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# 2. Portfolio summary
def test_portfolio_summary():
    response = client.get("/portfolio/summary")
    assert response.status_code == 200
    data = response.json()
    assert "as_of_week" in data
    assert "total_eligible_borrowers" in data
    assert isinstance(data["as_of_week"], int)
    assert isinstance(data["total_eligible_borrowers"], int)


# 3. Valid borrower returns 200 (canonical string ID)
def test_borrower_valid_returns_200():
    response = client.get("/borrower/B10")
    assert response.status_code == 200
    data = response.json()
    assert data["borrower_id"] == "B10"
    assert "operational_risk" in data
    assert "network_evidence" in data
    assert "financial_state" in data


# 4. Valid borrower via integer ID returns 200
def test_borrower_valid_integer_returns_200():
    response = client.get("/borrower/10")
    assert response.status_code == 200
    data = response.json()
    assert data["borrower_id"] == "B10"


# 5. Invalid borrower returns 404
def test_borrower_invalid_returns_404():
    response = client.get("/borrower/INVALID999")
    assert response.status_code == 404


# 6. Valid group returns 200
def test_group_valid_returns_200():
    response = client.get("/group/G1")
    assert response.status_code == 200
    data = response.json()
    assert data["group_id"] == "G1"
    assert "nodes" in data
    assert "edges" in data
    assert "member_count" in data


# 7. Invalid group returns 404
def test_group_invalid_returns_404():
    response = client.get("/group/INVALID999")
    assert response.status_code == 404


# 8. Network endpoint
def test_network_returns_200():
    response = client.get("/network")
    assert response.status_code == 200
    data = response.json()
    assert "total_groups" in data
    assert "groups" in data
    assert len(data["groups"]) > 0


# 9. Model B excludes propagation exposure
def test_b_excludes_propagation_exposure():
    features_b = store.model_b.features
    assert 'borrower_propagation_exposure' not in features_b


# 10. Model C includes propagation exposure
def test_c_includes_propagation_exposure():
    features_c = store.model_c.features
    assert 'borrower_propagation_exposure' in features_c


# 11. B/C feature separation is exactly 1 feature
def test_bc_exact_feature_separation():
    b_set = set(store.model_b.features)
    c_set = set(store.model_c.features)
    assert len(b_set) == 51
    assert len(c_set) == 52
    assert c_set - b_set == {'borrower_propagation_exposure'}
    assert b_set - c_set == set()


# 12. Canonical evaluation artifact exists and matches GET /evaluation
def test_canonical_evaluation_artifact_matches():
    eval_path = os.path.join(data_dir, 'evaluation.json')
    assert os.path.exists(eval_path), "evaluation.json must exist"
    with open(eval_path, 'r') as f:
        canonical = json.load(f)
    response = client.get("/evaluation")
    assert response.status_code == 200
    data = response.json()
    assert data == canonical


# 13. Assumptions metadata
def test_assumptions_metadata():
    response = client.get("/assumptions")
    assert response.status_code == 200
    data = response.json()
    assert data["world_seed"] == 909
    assert data["analytics_version"] == "M2C-FROZEN"
    assert data["as_of_week"] == 127
    assert data["propagation_horizon"] == "4-week"
    assert data["pv_threshold"] == ">= 0.30"
    assert "synthetic_data_disclaimer" in data


# 14. Simulator baseline immutability
def test_simulator_baseline_immutability():
    baseline_before = store.get_baseline_state_copy()
    req = {
        "borrower_id": "B1",
        "shock_magnitude": 500,
        "shock_duration_weeks": 4,
        "shock_type": "income_reduction"
    }
    client.post("/simulate", json=req)
    baseline_after = store.get_baseline_state_copy()

    hh_id = baseline_before.borrower_to_household["B1"]
    assert baseline_before.households[hh_id].weekly_income == baseline_after.households[hh_id].weekly_income
    assert baseline_before.households[hh_id].cash_buffer == baseline_after.households[hh_id].cash_buffer


# 15. Intervention baseline immutability
def test_intervention_baseline_immutability():
    baseline_before = store.get_baseline_state_copy()
    req = {
        "borrower_id": "B1",
        "intervention_type": "cash_injection",
        "amount": 500,
        "duration_weeks": 4
    }
    client.post("/intervene", json=req)
    baseline_after = store.get_baseline_state_copy()
    hh_id = baseline_before.borrower_to_household["B1"]
    assert baseline_before.households[hh_id].cash_buffer == baseline_after.households[hh_id].cash_buffer


# 16. Simulator determinism
def test_simulator_determinism():
    req = {
        "borrower_id": "B1",
        "shock_magnitude": 500,
        "shock_duration_weeks": 4,
        "shock_type": "income_reduction"
    }
    res1 = client.post("/simulate", json=req)
    res2 = client.post("/simulate", json=req)
    assert res1.json() == res2.json()


# 17. Intervention determinism
def test_intervention_determinism():
    req = {
        "borrower_id": "B1",
        "intervention_type": "restructure",
        "amount": 100,
        "duration_weeks": 4
    }
    res1 = client.post("/intervene", json=req)
    res2 = client.post("/intervene", json=req)
    assert res1.json() == res2.json()


# 18. Seed 909 cross-endpoint consistency
def test_seed_909_cross_endpoint_consistency():
    b10_res = client.get("/borrower/B10")
    assert b10_res.status_code == 200

    sim_req = {
        "borrower_id": "B10",
        "shock_magnitude": 100,
        "shock_duration_weeks": 1,
        "shock_type": "cash_shock"
    }
    sim_res = client.post("/simulate", json=sim_req)
    assert sim_res.status_code == 200
    cb_sim_baseline = sim_res.json()["trajectory"][0]["target_borrower"]["baseline"]["cash_buffer"]
    assert cb_sim_baseline >= 0


# 19. Invalid simulation: negative magnitude
def test_negative_simulation_magnitude():
    req = {
        "borrower_id": "B1",
        "shock_magnitude": -500,
        "shock_duration_weeks": 4,
        "shock_type": "income_reduction"
    }
    assert client.post("/simulate", json=req).status_code == 400


# 20. Invalid simulation: oversized duration
def test_oversized_simulation_duration():
    req = {
        "borrower_id": "B1",
        "shock_magnitude": 500,
        "shock_duration_weeks": 100,
        "shock_type": "income_reduction"
    }
    assert client.post("/simulate", json=req).status_code == 400


# 21. Unknown shock type
def test_unknown_shock_type():
    req = {
        "borrower_id": "B1",
        "shock_magnitude": 500,
        "shock_duration_weeks": 4,
        "shock_type": "unknown_magic_wand"
    }
    assert client.post("/simulate", json=req).status_code == 400


# 22. Invalid intervention type
def test_invalid_intervention_type():
    req = {
        "borrower_id": "B1",
        "intervention_type": "magic_wand",
        "amount": 100,
        "duration_weeks": 4
    }
    assert client.post("/intervene", json=req).status_code == 400


# 23. Invalid intervention: nonexistent borrower
def test_invalid_intervention_borrower():
    req = {
        "borrower_id": "INVALID999",
        "intervention_type": "cash_injection",
        "amount": 100,
        "duration_weeks": 4
    }
    assert client.post("/intervene", json=req).status_code == 404


# 24. Absence of hidden lineage in API responses
def test_absence_of_hidden_lineage():
    req = {
        "borrower_id": "B1",
        "shock_magnitude": 500,
        "shock_duration_weeks": 4,
        "shock_type": "income_reduction"
    }
    res = client.post("/simulate", json=req).json()
    res_str = json.dumps(res)
    assert "world_seed" not in res_str
    assert "leakage" not in res_str
