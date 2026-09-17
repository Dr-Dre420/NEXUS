import pytest
from src.mechanics import (
    WorldState, HouseholdState, LoanState, GroupState, 
    process_household_cash_flow, process_repayments, apply_group_liability_coverage
)

def test_mechanics_immutability():
    hh = HouseholdState(
        household_id="HH1", cash_buffer=100.0, non_liquid_reserve=0.0, weekly_income=50.0, 
        weekly_expenses=20.0, weekly_external_debt=0.0
    )
    state = WorldState(
        households={"HH1": hh}, loans={}, groups={}, borrower_to_household={}
    )
    
    new_state = process_household_cash_flow(state)
    
    # Original state shouldn't be mutated
    assert state.households["HH1"].cash_buffer == 100.0
    # New state should be mutated (100 + 50 - 20 = 130)
    assert new_state.households["HH1"].cash_buffer == 130.0

def test_process_repayments_full_payment():
    hh = HouseholdState(
        household_id="HH1", cash_buffer=500.0, non_liquid_reserve=0.0, weekly_income=0.0, 
        weekly_expenses=0.0, weekly_external_debt=0.0
    )
    loan = LoanState(
        loan_id="L1", borrower_id="B1", group_id="G1", 
        principal_remaining=1000.0, weekly_instalment=100.0, 
        amount_due=100.0, days_past_due=7
    )
    state = WorldState(
        households={"HH1": hh}, loans={"L1": loan}, groups={}, 
        borrower_to_household={"B1": "HH1"}
    )
    
    new_state, shortfalls = process_repayments(state)
    
    assert new_state.households["HH1"].cash_buffer == 400.0
    assert new_state.loans["L1"].amount_due == 0.0
    assert new_state.loans["L1"].days_past_due == 0
    assert shortfalls["B1"] == 0.0

def test_process_repayments_shortfall():
    hh = HouseholdState(
        household_id="HH1", cash_buffer=50.0, non_liquid_reserve=0.0, weekly_income=0.0, 
        weekly_expenses=0.0, weekly_external_debt=0.0
    )
    loan = LoanState(
        loan_id="L1", borrower_id="B1", group_id="G1", 
        principal_remaining=1000.0, weekly_instalment=100.0, 
        amount_due=100.0, days_past_due=0
    )
    state = WorldState(
        households={"HH1": hh}, loans={"L1": loan}, groups={}, 
        borrower_to_household={"B1": "HH1"}
    )
    
    new_state, shortfalls = process_repayments(state)
    
    assert new_state.households["HH1"].cash_buffer == 0.0
    assert new_state.loans["L1"].amount_due == 50.0
    assert new_state.loans["L1"].days_past_due == 7
    assert shortfalls["B1"] == 50.0

def test_apply_group_liability_coverage():
    hh1 = HouseholdState("HH1", cash_buffer=0.0, non_liquid_reserve=0.0, weekly_income=0.0, weekly_expenses=0.0, weekly_external_debt=0.0)
    hh2 = HouseholdState("HH2", cash_buffer=100.0, non_liquid_reserve=0.0, weekly_income=0.0, weekly_expenses=0.0, weekly_external_debt=0.0)
    loan1 = LoanState("L1", "B1", "G1", 1000.0, 100.0, 50.0, 7)
    loan2 = LoanState("L2", "B2", "G1", 1000.0, 100.0, 0.0, 0)
    group = GroupState("G1", ["B1", "B2"], 100.0)
    
    state = WorldState(
        households={"HH1": hh1, "HH2": hh2},
        loans={"L1": loan1, "L2": loan2},
        groups={"G1": group},
        borrower_to_household={"B1": "HH1", "B2": "HH2"}
    )
    
    shortfalls = {"B1": 50.0, "B2": 0.0}
    new_state, covered = apply_group_liability_coverage(state, shortfalls)
    
    assert covered["B1"] == 25.0
    assert new_state.loans["L1"].amount_due == 25.0
    assert new_state.loans["L1"].days_past_due == 7
