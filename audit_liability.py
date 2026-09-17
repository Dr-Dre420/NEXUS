import os
import sys
sys.path.append(os.path.abspath("."))
from src.api.data_store import store

store.load()
df = store.world_state['next_df_prop']
as_of = store.get_as_of_week()
current_df = df[df['week'] == as_of]

print(f"Data columns: {current_df.columns.tolist()}")

if 'borrower_liability_share' in current_df.columns:
    has_explicit = current_df['borrower_liability_share'].notna().sum()
    missing = current_df['borrower_liability_share'].isna().sum()
    print(f"Dataframe borrower_liability_share: explicit={has_explicit}, missing={missing}")
    if missing > 0:
        print(f"Sample of explicit: {current_df['borrower_liability_share'].dropna().head().tolist()}")
else:
    print("borrower_liability_share not in DataFrame!")

state = store.baseline_state
weights = []
for hh in state.households.values():
    weights.append(hh.liability_weight)

print(f"Household liability_weights: {len(weights)} total, explicit values: {set(weights)}")
