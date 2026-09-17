import random
import copy
import pandas as pd
from typing import List, Dict, Any, Tuple
from .mechanics import WorldState, HouseholdState, LoanState, GroupState, step_forward, apply_loan_instalments, process_household_cash_flow, process_repayments

class SyntheticWorldGenerator:
    def __init__(self, seed: int, n_borrowers: int = 400, coverage_fraction: float = 0.50, slack_regime: str = "base"):
        self.seed = seed
        self.n_borrowers = n_borrowers
        self.coverage_fraction = coverage_fraction
        self.slack_regime = slack_regime
        random.seed(self.seed)
        self.history = []
        self.lineage = []
        self.episode_lineage = []
        self.world_state = None
        
    def generate_world(self):
        """
        Creates the static hierarchy: Branch -> Centre -> JLG -> Borrower -> Household
        """
        households = {}
        loans = {}
        groups = {}
        borrower_to_hh = {}
        
        borrowers_per_jlg = 5
        jlgs_per_centre = 5
        centres_per_branch = 4
        
        n_jlgs = self.n_borrowers // borrowers_per_jlg
        n_centres = n_jlgs // jlgs_per_centre
        
        borrower_id_counter = 1
        
        for g in range(1, n_jlgs + 1):
            group_id = f"G{g}"
            members = []
            
            for _ in range(borrowers_per_jlg):
                b_id = f"B{borrower_id_counter}"
                hh_id = f"HH{borrower_id_counter}"
                members.append(b_id)
                borrower_to_hh[b_id] = hh_id
                
                base_expenses = random.uniform(1000, 1500)
                base_ext_debt = random.uniform(0, 300)
                
                # Active loan
                principal = random.uniform(20000, 50000)
                instalment = principal / 200.0  # 4 year loan so it spans the 156 week simulation
                
                # Assign heterogeneous financial profile
                profile = random.choices(["vulnerable", "moderate", "resilient"], weights=[0.3, 0.5, 0.2])[0]
                
                # Adjust based on slack_regime
                if self.slack_regime == "conservative":
                    margin_multipliers = {"vulnerable": 0.0, "moderate": 0.05, "resilient": 0.1}
                    buffer_multipliers = {"vulnerable": 0.0, "moderate": 1.5, "resilient": 3.0}
                elif self.slack_regime == "high":
                    margin_multipliers = {"vulnerable": 0.1, "moderate": 0.3, "resilient": 0.6}
                    buffer_multipliers = {"vulnerable": 2.0, "moderate": 4.0, "resilient": 8.0}
                else: # base
                    margin_multipliers = {"vulnerable": 0.0, "moderate": 0.15, "resilient": 0.3}
                    buffer_multipliers = {"vulnerable": 0.5, "moderate": 2.5, "resilient": 6.0}
                    
                min_required = base_expenses + base_ext_debt + instalment
                margin = min_required * margin_multipliers[profile]
                base_income = random.uniform(min_required + margin, min_required + margin + 100)
                
                cash_buffer = base_expenses * buffer_multipliers[profile]
                
                # Assign heterogeneous liability weight
                liability_weight = random.uniform(0.1, 1.0)
                
                households[hh_id] = HouseholdState(
                    household_id=hh_id,
                    cash_buffer=cash_buffer,
                    non_liquid_reserve=0.0,
                    weekly_income=base_income,
                    weekly_expenses=base_expenses,
                    weekly_external_debt=base_ext_debt,
                    liability_weight=liability_weight
                )
                
                loans[f"L{borrower_id_counter}"] = LoanState(
                    loan_id=f"L{borrower_id_counter}",
                    borrower_id=b_id,
                    group_id=group_id,
                    principal_remaining=principal,
                    weekly_instalment=instalment,
                    amount_due=0.0,
                    days_past_due=0,
                    is_defaulted=False,
                    is_closed=False
                )
                
                borrower_id_counter += 1
                
            groups[group_id] = GroupState(
                group_id=group_id,
                members=members,
                capacity=random.uniform(5000, 10000) # Configurable group capacity
            )
            
        self.world_state = WorldState(
            households=households,
            loans=loans,
            groups=groups,
            borrower_to_household=borrower_to_hh
        )
        
    def _record_history(self, week: int, state: WorldState, shortfalls: Dict[str, float], covered: Dict[str, float], scenario_label: str = "none"):
        for loan in state.loans.values():
            hh_id = state.borrower_to_household[loan.borrower_id]
            hh = state.households[hh_id]
            self.history.append({
                "week": week,
                "borrower_id": loan.borrower_id,
                "group_id": loan.group_id,
                "cash_buffer": hh.cash_buffer,
                "non_liquid_reserve": hh.non_liquid_reserve,
                "weekly_income": hh.weekly_income,
                "weekly_expenses": hh.weekly_expenses,
                "principal_remaining": loan.principal_remaining,
                "amount_due": loan.amount_due,
                "days_past_due": loan.days_past_due,
                "is_defaulted": loan.is_defaulted,
                "is_closed": loan.is_closed,
                "shortfall": shortfalls.get(loan.borrower_id, 0.0),
                "group_covered_amount": covered.get(loan.borrower_id, 0.0),
                "scenario_family": scenario_label # Generator metadata, not for detector
            })

    def _step_forward_no_network(self, state: WorldState, max_liquid_buffer_weeks: float = 12.0, max_days_past_due: int = 180) -> Tuple[WorldState, Dict[str, float]]:
        s1 = process_household_cash_flow(state, max_liquid_buffer_weeks)
        s2 = apply_loan_instalments(s1)
        s3, shortfalls = process_repayments(s2, max_days_past_due)
        return s3, shortfalls

    def simulate(self, n_weeks: int = 156):
        self.generate_world()
        
        active_shocks = {}
        self.states = {}
        self.scenario_history = {}
        self.scenario_params = {}
        self.shortfall_history = {}
        
        self.states[0] = copy.deepcopy(self.world_state)
        
        shock_schedule = self._generate_shock_schedule(n_weeks)
        
        for week in range(1, n_weeks + 1):
            self.states[week] = copy.deepcopy(self.world_state)
            
            # 1. Apply Shocks
            scenario, params = self._apply_shocks(week, shock_schedule, self.world_state, active_shocks)
            self.scenario_history[week] = scenario
            self.scenario_params[week] = params
            
            # 2. Compute counterfactual (No group liability THIS WEEK)
            cf_state = copy.deepcopy(self.world_state)
            cf_state, cf_shortfalls, _ = step_forward(
                cf_state, 
                coverage_fraction=0.0, 
                max_liquid_buffer_weeks=12.0,
                max_days_past_due=180
            )
            
            # 3. Step forward actual world
            next_state, actual_shortfalls, group_coverage = step_forward(
                self.world_state, 
                coverage_fraction=self.coverage_fraction,
                max_liquid_buffer_weeks=12.0,
                max_days_past_due=180
            )
            
            self.shortfall_history[week] = actual_shortfalls

            # 4. Record State and Lineage
            self._record_history(week, next_state, actual_shortfalls, group_coverage, scenario)
            self._compute_marginal_lineage(week, cf_shortfalls, actual_shortfalls)
            
            # 5. Revert temporary shocks
            self._revert_shocks(week, self.world_state, active_shocks)
            
            self.world_state = next_state
            
        self.states[n_weeks + 1] = copy.deepcopy(self.world_state)
        
        # 6. Compute Episode Network Contribution (4-week windows)
        self._compute_episode_contributions(n_weeks)

        history_df = pd.DataFrame(self.history)
        lineage_df = pd.DataFrame(self.lineage)
        ep_lineage_df = pd.DataFrame(self.episode_lineage)
        return history_df, lineage_df, ep_lineage_df

    def _generate_shock_schedule(self, n_weeks: int):
        schedule = {}
        for w in range(5, n_weeks, 5):
            # Mostly propagating stress to generate the specific target events we need,
            # with occasional isolated shocks to test false positives.
            # We avoid seasonal_downturn wiping out the whole network.
            scenario_type = random.choice([
                "propagating_stress", 
                "propagating_stress", 
                "propagating_stress", 
                "isolated_shock",
                "seasonal_downturn"
            ])
            schedule[w] = scenario_type
        return schedule
        
    def _apply_shocks(self, week: int, schedule: dict, state: WorldState, active_shocks: dict, shock_params: dict = None) -> Tuple[str, dict]:
        scenario = schedule.get(week, "none")
        if scenario == "none":
            return scenario, {}
            
        params_out = {}
            
        if scenario == "seasonal_downturn":
            # Apply to all, use a fixed probability threshold or exact list if provided
            if shock_params and 'hit_list' in shock_params:
                hit_list = shock_params['hit_list']
            else:
                hit_list = [hh.household_id for hh in state.households.values() if random.random() < 0.10]
                params_out['hit_list'] = hit_list
                
            for hh_id in hit_list:
                hh = state.households[hh_id]
                hh.cash_buffer = max(0.0, hh.cash_buffer - 500)
                
        elif scenario == "isolated_shock":
            if shock_params and 'hh_id' in shock_params:
                hh_id = shock_params['hh_id']
                magnitude = shock_params['magnitude']
                duration = shock_params['duration']
            else:
                hh_id = random.choice(list(state.households.keys()))
                magnitude = random.uniform(1000, 3000)
                duration = random.randint(2, 6)
                params_out['hh_id'] = hh_id
                params_out['magnitude'] = magnitude
                params_out['duration'] = duration
                
            hh = state.households[hh_id]
            hh.cash_buffer = max(0.0, hh.cash_buffer - magnitude/2)
            hh.weekly_expenses += magnitude
            active_shocks[week + duration] = active_shocks.get(week + duration, []) + [(hh_id, magnitude)]
            
        elif scenario == "individual_deterioration":
            if shock_params and 'hh_id' in shock_params:
                hh_id = shock_params['hh_id']
                magnitude = shock_params['magnitude']
            else:
                hh_id = random.choice(list(state.households.keys()))
                magnitude = random.uniform(300, 800)
                params_out['hh_id'] = hh_id
                params_out['magnitude'] = magnitude
                
            hh = state.households[hh_id]
            hh.weekly_expenses += magnitude
            
        elif scenario == "propagating_stress":
            if shock_params and 'group_id' in shock_params:
                group_id = shock_params['group_id']
                magnitude = shock_params['magnitude']
            else:
                group_id = random.choice(list(state.groups.keys()))
                magnitude = random.uniform(8000, 15000) # Larger, front-loaded shock
                params_out['group_id'] = group_id
                params_out['magnitude'] = magnitude
                
            g = state.groups[group_id]
            source_b = g.members[0]
            hh_id = state.borrower_to_household[source_b]
            state.households[hh_id].cash_buffer = 0
            
            # Massive ongoing expense (representing an emergency medical or business shock)
            state.households[hh_id].weekly_expenses += magnitude
            # Severe income loss
            state.households[hh_id].weekly_income *= random.uniform(0.0, 0.2)
            
        return scenario, params_out
        
    def _revert_shocks(self, week: int, state: WorldState, active_shocks: dict):
        if week in active_shocks:
            for hh_id, amt in active_shocks[week]:
                state.households[hh_id].weekly_expenses -= amt

    def _compute_marginal_lineage(self, week: int, cf_shortfalls: Dict[str, float], actual_shortfalls: Dict[str, float]):
        for b_id, actual_sf in actual_shortfalls.items():
            if actual_sf > 0:
                cf_sf = cf_shortfalls.get(b_id, 0.0)
                increase_due_to_network = actual_sf - cf_sf
                if increase_due_to_network > 0:
                    share = increase_due_to_network / actual_sf
                    self.lineage.append({
                        "week": week,
                        "destination_borrower": b_id,
                        "mechanism": "marginal_weekly_network_contribution",
                        "contribution_share": share
                    })
                    
    def _compute_episode_contributions(self, n_weeks: int, horizon: int = 4):
        # Episode covers [t, t+horizon) which is exactly the prediction window.
        all_borrowers = list(self.states[0].borrower_to_household.keys())
        
        for t in range(1, n_weeks - horizon + 1):
            cf_state = copy.deepcopy(self.states[t])
            cf_shortfalls_history = {}
            
            # Run CF forward `horizon` weeks with Network OFF
            for w in range(t, t + horizon):
                scenario = self.scenario_history[w]
                params = self.scenario_params[w]
                
                # Apply exactly the same shocks
                if scenario == "seasonal_downturn":
                    for hh_id in params['hit_list']:
                        cf_state.households[hh_id].cash_buffer = max(0.0, cf_state.households[hh_id].cash_buffer - 500)
                elif scenario == "isolated_shock":
                    hh_id = params['hh_id']
                    mag = params['magnitude']
                    cf_state.households[hh_id].cash_buffer = max(0.0, cf_state.households[hh_id].cash_buffer - mag/2)
                    cf_state.households[hh_id].weekly_expenses += mag
                elif scenario == "individual_deterioration":
                    hh_id = params['hh_id']
                    cf_state.households[hh_id].weekly_expenses += params['magnitude']
                elif scenario == "propagating_stress":
                    hh_id = cf_state.borrower_to_household[cf_state.groups[params['group_id']].members[0]]
                    cf_state.households[hh_id].cash_buffer = 0
                    cf_state.households[hh_id].weekly_expenses += params['magnitude']
                    cf_state.households[hh_id].weekly_income *= 0.1
                
                cf_state, cf_sf, _ = step_forward(
                    cf_state, 
                    coverage_fraction=0.0, 
                    max_liquid_buffer_weeks=12.0,
                    max_days_past_due=180
                )
                cf_shortfalls_history[w] = cf_sf
                
                # Revert temporary shocks
                if scenario == "isolated_shock":
                    hh_id = params['hh_id']
                    cf_state.households[hh_id].weekly_expenses -= params['magnitude']
                    
            # Compare downstream burdens at t+horizon
            fw_state_end = self.states[t + horizon]
            
            for b_id in all_borrowers:
                hh_id = cf_state.borrower_to_household[b_id]
                
                fw_cum_sf = sum(self.shortfall_history[w].get(b_id, 0.0) for w in range(t, t + horizon))
                cf_cum_sf = sum(cf_shortfalls_history[w].get(b_id, 0.0) for w in range(t, t + horizon))
                
                fw_burden = fw_cum_sf
                cf_burden = cf_cum_sf
                
                network_incremental_burden = max(0.0, fw_burden - cf_burden)
                
                if fw_burden > 1e-6:
                    share = network_incremental_burden / fw_burden
                else:
                    share = 0.0
                    
                if share > 0:
                    self.episode_lineage.append({
                        "week": t, # the start of the episode (prediction time)
                        "destination_borrower": b_id,
                        "episode_network_contribution": share,
                        "fw_burden": fw_burden,
                        "cf_burden": cf_burden,
                        "network_incremental_burden": network_incremental_burden
                    })
