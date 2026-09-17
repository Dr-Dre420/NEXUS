import copy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

@dataclass
class HouseholdState:
    household_id: str
    cash_buffer: float
    non_liquid_reserve: float
    weekly_income: float
    weekly_expenses: float
    weekly_external_debt: float
    liability_weight: float = 1.0

@dataclass
class LoanState:
    loan_id: str
    borrower_id: str
    group_id: str
    principal_remaining: float
    weekly_instalment: float
    amount_due: float  # Current accumulated amount due
    days_past_due: int
    is_defaulted: bool = False
    is_closed: bool = False

@dataclass
class GroupState:
    group_id: str
    members: List[str]
    capacity: float  # Configured total capacity of the group to cover shortfalls

@dataclass
class WorldState:
    households: Dict[str, HouseholdState]
    loans: Dict[str, LoanState]
    groups: Dict[str, GroupState]
    borrower_to_household: Dict[str, str]

def process_household_cash_flow(state: WorldState, max_liquid_buffer_weeks: float = 12.0) -> WorldState:
    """
    Applies weekly income, expenses, and external debt to the cash buffer.
    Sweeps any excess cash beyond max_liquid_buffer_weeks * weekly_expenses into non_liquid_reserve.
    Returns a new mutated state object (caller's state is preserved).
    """
    new_state = copy.deepcopy(state)
    for hh_id, hh in new_state.households.items():
        net_cash_flow = hh.weekly_income - hh.weekly_expenses - hh.weekly_external_debt
        hh.cash_buffer = max(0.0, hh.cash_buffer + net_cash_flow)
        
        max_buffer = hh.weekly_expenses * max_liquid_buffer_weeks
        if hh.cash_buffer > max_buffer:
            excess = hh.cash_buffer - max_buffer
            hh.cash_buffer = max_buffer
            hh.non_liquid_reserve += excess
            
    return new_state

def apply_loan_instalments(state: WorldState) -> WorldState:
    """
    Adds weekly instalment to the amount_due for all active loans.
    """
    new_state = copy.deepcopy(state)
    for loan_id, loan in new_state.loans.items():
        if loan.principal_remaining > 0 and not loan.is_defaulted and not loan.is_closed:
            loan.amount_due += loan.weekly_instalment
    return new_state

def process_repayments(state: WorldState, max_days_past_due: int = 180) -> Tuple[WorldState, Dict[str, float]]:
    """
    Borrowers attempt to pay amount_due from their household cash buffer.
    Returns the new state and a dictionary mapping borrower_id to their shortfall.
    """
    new_state = copy.deepcopy(state)
    shortfalls: Dict[str, float] = {}

    for loan_id, loan in new_state.loans.items():
        if loan.is_defaulted or loan.is_closed:
            continue
            
        if loan.amount_due <= 0:
            loan.days_past_due = 0
            # Check for closure
            if loan.principal_remaining <= 0:
                loan.is_closed = True
            continue

        hh_id = new_state.borrower_to_household[loan.borrower_id]
        hh = new_state.households[hh_id]

        if hh.cash_buffer >= loan.amount_due:
            # Full payment
            paid = loan.amount_due
            hh.cash_buffer -= paid
            loan.principal_remaining = max(0.0, loan.principal_remaining - paid)
            loan.amount_due = 0.0
            loan.days_past_due = 0
            shortfalls[loan.borrower_id] = 0.0
            
            # Check for closure
            if loan.principal_remaining <= 0:
                loan.is_closed = True
        else:
            # Partial or no payment
            paid = hh.cash_buffer
            hh.cash_buffer = 0.0
            loan.principal_remaining = max(0.0, loan.principal_remaining - paid)
            loan.amount_due -= paid
            shortfalls[loan.borrower_id] = loan.amount_due
            
            # Increment days past due by 7 (weekly tick) if amount due is positive
            if loan.amount_due > 0:
                loan.days_past_due += 7
                
            # Check for default
            if loan.days_past_due >= max_days_past_due:
                loan.is_defaulted = True

    return new_state, shortfalls

def apply_group_liability_coverage(
    state: WorldState, 
    shortfalls: Dict[str, float],
    coverage_fraction: float = 0.50
) -> Tuple[WorldState, Dict[str, float]]:
    """
    Groups attempt to cover member shortfalls.
    Returns new state and a dictionary of covered amounts per borrower.
    """
    new_state = copy.deepcopy(state)
    covered_amounts: Dict[str, float] = {}

    for group_id, group in new_state.groups.items():
        # Identify members with shortfalls and healthy members
        group_shortfalls = {
            b_id: shortfalls.get(b_id, 0.0)
            for b_id in group.members
            if shortfalls.get(b_id, 0.0) > 0
        }
        
        healthy_members = [
            b_id for b_id in group.members
            if shortfalls.get(b_id, 0.0) == 0.0
        ]
        
        total_shortfall = sum(group_shortfalls.values())
        if total_shortfall == 0 or not healthy_members:
            continue
            
        # The total group coverage obligation is bounded by coverage_fraction
        total_group_coverage_obligation = total_shortfall * coverage_fraction
        
        # Calculate normalized liability weights for healthy members
        weights = {}
        for b_id in healthy_members:
            hh_id = new_state.borrower_to_household[b_id]
            weights[b_id] = new_state.households[hh_id].liability_weight
            
        sum_weights = sum(weights.values())
        if sum_weights > 0:
            normalized_weights = {b: w / sum_weights for b, w in weights.items()}
        else:
            normalized_weights = {b: 1.0 / len(healthy_members) for b in healthy_members}
            
        total_actual_covered = 0.0
        member_contributions = {}
        
        # Each member attempts to pay their weighted share of the total obligation
        for b_id in healthy_members:
            hh_id = new_state.borrower_to_household[b_id]
            hh = new_state.households[hh_id]
            
            target_contribution = total_group_coverage_obligation * normalized_weights[b_id]
            actual_contribution = min(target_contribution, hh.cash_buffer)
            
            member_contributions[b_id] = actual_contribution
            total_actual_covered += actual_contribution
            
            hh.cash_buffer = max(0.0, hh.cash_buffer - actual_contribution)
            
        if total_actual_covered == 0:
            continue
            
        # Distribute the total_actual_covered to the loans with shortfalls
        coverage_ratio = total_actual_covered / total_shortfall
        
        for b_id, sf in group_shortfalls.items():
            covered = sf * coverage_ratio
            covered_amounts[b_id] = covered
            
            # Apply coverage to the loan
            for loan in new_state.loans.values():
                if loan.borrower_id == b_id and loan.amount_due > 0:
                    actual_coverage = min(covered, loan.amount_due)
                    loan.amount_due -= actual_coverage
                    loan.principal_remaining = max(0.0, loan.principal_remaining - actual_coverage)
                    if loan.amount_due <= 0:
                        loan.days_past_due = 0
                    break

    return new_state, covered_amounts

def step_forward(state: WorldState, coverage_fraction: float = 0.50, max_liquid_buffer_weeks: float = 12.0, max_days_past_due: int = 180) -> Tuple[WorldState, Dict[str, float], Dict[str, float]]:
    """
    Performs one weekly tick.
    Returns (new_state, shortfalls, group_covered_amounts)
    """
    s1 = process_household_cash_flow(state, max_liquid_buffer_weeks)
    s2 = apply_loan_instalments(s1)
    
    # Pre-calculate theoretical shortfalls (what members CANNOT pay)
    anticipated_shortfalls = {}
    for loan_id, loan in s2.loans.items():
        if loan.amount_due > 0 and not loan.is_defaulted and not loan.is_closed:
            hh_id = s2.borrower_to_household[loan.borrower_id]
            hh = s2.households[hh_id]
            sf = max(0.0, loan.amount_due - hh.cash_buffer)
            if sf > 0:
                anticipated_shortfalls[loan.borrower_id] = sf
                
    # Group steps in BEFORE bank collects
    s3, covered_amounts = apply_group_liability_coverage(s2, anticipated_shortfalls, coverage_fraction)
    
    # Bank collects individual repayments
    s4, shortfalls = process_repayments(s3, max_days_past_due)
    
    return s4, shortfalls, covered_amounts
