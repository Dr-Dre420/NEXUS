import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.api.main import app
from src.api.data_store import store

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(project_root, 'data', 'frozen_m2c')
store.load(data_dir=data_dir)

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_portfolio_summary():
    response = client.get("/portfolio/summary")
    assert response.status_code == 200
    data = response.json()
    assert "as_of_week" in data
    assert "total_eligible_borrowers" in data
    
def test_borrower():
    # Attempt to fetch borrower B1
    response = client.get("/borrower/B1")
    if response.status_code == 200:
        data = response.json()
        assert data["borrower_id"] == "B1"
        assert "operational_risk" in data
        assert "network_evidence" in data
    else:
        assert response.status_code == 404

def test_group():
    # Attempt to fetch group G1
    response = client.get("/group/G1")
    if response.status_code == 200:
        data = response.json()
        assert data["group_id"] == "G1"
        assert "members" in data
    else:
        assert response.status_code == 404

def test_network():
    response = client.get("/network")
    assert response.status_code == 200
    data = response.json()
    assert "total_groups" in data
    assert "groups" in data

def test_simulate():
    req = {
        "borrower_id": "B1",
        "shock_magnitude": 500,
        "shock_duration_weeks": 4,
        "shock_type": "income_reduction"
    }
    response = client.post("/simulate", json=req)
    if response.status_code == 200:
        data = response.json()
        assert data["status"] == "success"
        assert "trajectory" in data
    else:
        assert response.status_code == 404

def test_intervene():
    req = {
        "borrower_id": "B1",
        "intervention_type": "cash_injection",
        "amount": 500,
        "duration_weeks": 4
    }
    response = client.post("/intervene", json=req)
    if response.status_code == 200:
        data = response.json()
        assert data["status"] == "success"
        assert "trajectory" in data
    else:
        assert response.status_code in [400, 404]

def test_evaluation():
    response = client.get("/evaluation")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "M0" in data["models"]
    assert "Model C calibrated" in data["models"]
    assert "results" in data

def test_assumptions():
    response = client.get("/assumptions")
    assert response.status_code == 200
    data = response.json()
    assert data["world_seed"] == 909
    assert data["analytics_version"] == "M2C-FROZEN"
    assert "temporal_evaluation_rules" in data
    assert "propagation_horizon" in data

def test_intervene():
    req = {
        "borrower_id": "B1",
        "intervention_type": "restructure",
        "amount": 100,
        "duration_weeks": 4
    }
    # Test identical requests return identical deterministic results
    res1 = client.post("/intervene", json=req)
    res2 = client.post("/intervene", json=req)
    
    assert res1.status_code == 200
    assert res1.json() == res2.json()
    
    data = res1.json()
    assert data["scenario_type"] == "restructure"
    assert len(data["trajectory"]) == 4
    assert "target_borrower" in data["trajectory"][0]
    
    # Test invalid borrower fails cleanly
    req_inv_b = req.copy()
    req_inv_b["borrower_id"] = "INVALID999"
    res_inv_b = client.post("/intervene", json=req_inv_b)
    assert res_inv_b.status_code == 404
    
    # Test invalid intervention type fails cleanly
    req_inv_i = req.copy()
    req_inv_i["intervention_type"] = "magic_wand"
    res_inv_i = client.post("/intervene", json=req_inv_i)
    assert res_inv_i.status_code == 400
