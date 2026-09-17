def apply_intervention(state, borrower_id: str, intervention_type: str, amount: float):
    """
    Applies an intervention to the provided state deep-copy.
    Does NOT return a new state; it mutates the state passed in.
    """
    if borrower_id not in state.borrower_to_household:
        raise ValueError(f"Borrower {borrower_id} not found in state.")
        
    hh_id = state.borrower_to_household[borrower_id]
    hh = state.households[hh_id]
    
    # Identify loan
    loan = None
    for l_id, l in state.loans.items():
        if l.borrower_id == borrower_id:
            loan = l
            break
            
    if not loan:
        raise ValueError(f"No active loan found for borrower {borrower_id}")
        
    if intervention_type == "restructure":
        loan.weekly_instalment = max(0.0, loan.weekly_instalment - amount)
    elif intervention_type == "cash_injection":
        hh.cash_buffer += amount
    elif intervention_type == "payment_adjustment":
        loan.amount_due = max(0.0, loan.amount_due - amount)
    else:
        raise ValueError(f"Unsupported intervention type: {intervention_type}")
