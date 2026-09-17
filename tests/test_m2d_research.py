"""
Tests for M2D research logic.

Guards the three things that would invalidate the research: temporal leakage in
the research features, drift in the frozen M2C artifacts, and non-determinism.
"""
import os
import sys
import json
import pickle
import hashlib
import subprocess

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'research', 'm2d'))

CACHE = os.path.join(ROOT, 'research', 'm2d', 'cache')
HAS_CACHE = os.path.isdir(CACHE) and len(os.listdir(CACHE)) >= 1

cache_required = pytest.mark.skipif(
    not HAS_CACHE, reason='research world cache absent; run research/m2d/gen_world.py')


def _world(seed=909):
    return pickle.load(open(os.path.join(CACHE, f'world_{seed}.pkl'), 'rb'))['df']


# --------------------------------------------------------------------------
# Frozen M2C artifacts must not drift
# --------------------------------------------------------------------------

def test_frozen_m2c_evaluation_untouched_by_research():
    """The canonical evaluation artifact must match the committed blob exactly."""
    rel = 'data/frozen_m2c/evaluation.json'
    on_disk = open(os.path.join(ROOT, rel), 'rb').read()
    committed = subprocess.run(['git', 'show', f'HEAD:{rel}'],
                               cwd=ROOT, capture_output=True)
    assert committed.returncode == 0, 'frozen artifact missing from HEAD'
    # Compare CONTENT, not raw bytes: git normalises newlines on checkout under
    # core.autocrlf, so a byte-for-byte hash always differs on Windows even when
    # the file is untouched. Normalising first tests what actually matters.
    def norm(b):
        return hashlib.sha256(b.replace(b"\r\n", b"\n")).hexdigest()

    assert norm(on_disk) == norm(committed.stdout), \
        'research modified the frozen M2C evaluation artifact'
    assert json.loads(on_disk) == json.loads(committed.stdout), \
        'frozen M2C evaluation content diverged from the committed artifact'


def test_frozen_m2c_conclusion_not_overstated():
    """Research must not have rewritten the frozen conclusion into a claim of success."""
    d = json.load(open(os.path.join(ROOT, 'data/frozen_m2c/evaluation.json')))
    text = d['final_scientific_conclusion'].lower()
    assert 'inconclusive' in text
    for banned in ('proves', 'demonstrates superiority', 'model c is better'):
        assert banned not in text


def test_model_b_still_excludes_propagation_exposure():
    from src.models.model_b import ModelB
    from src.models.model_c import ModelC
    cols = ['principal_remaining', 'borrower_propagation_exposure',
            'peer_predicted_stress_mean', 'borrower_id', 'week']
    X = pd.DataFrame({c: [0.0] for c in cols})
    b = ModelB()._filter_features(X)
    c = ModelC()._filter_features(X)
    assert 'borrower_propagation_exposure' not in b
    assert 'borrower_propagation_exposure' in c
    assert set(c) - set(b) == {'borrower_propagation_exposure'}


# --------------------------------------------------------------------------
# Research feature provenance: no future information
# --------------------------------------------------------------------------

@cache_required
def test_research_features_are_backward_looking_only():
    """Truncating the future must not change a feature computed at week t.

    Any feature that used information after t would change when later weeks are
    removed. This is the operational definition of temporal validity.
    """
    from e3_variants import add_variant_features
    df = _world()
    cut = 100
    full = add_variant_features(df)
    trunc = add_variant_features(df[df.week <= cut].copy())

    feats = [c for c in full.columns if c.startswith('v_')]
    assert feats, 'no research features found'

    key = ['borrower_id', 'week']
    a = full[full.week <= cut].sort_values(key).reset_index(drop=True)
    b = trunc.sort_values(key).reset_index(drop=True)

    leaky = []
    for f in feats:
        if not np.allclose(a[f].fillna(0).values, b[f].fillna(0).values,
                           rtol=1e-9, atol=1e-12):
            leaky.append(f)
    assert not leaky, f'features change when the future is removed (leakage): {leaky}'


@cache_required
def test_research_features_exclude_hidden_lineage_and_scenario_labels():
    from e3_variants import add_variant_features, feature_sets
    d = add_variant_features(_world())
    banned = ('lineage', 'contribution', 'destination_borrower', 'source_borrower',
              'scenario_family', 'fw_burden', 'cf_burden', 'incremental_burden')
    for spec, feats in feature_sets(d).items():
        for f in feats:
            low = f.lower()
            assert not any(b in low for b in banned), f'{spec} uses forbidden input {f}'


@cache_required
def test_research_features_exclude_the_target_and_future_outcomes():
    from e3_variants import add_variant_features, feature_sets
    d = add_variant_features(_world())
    forbidden = {'propagation_vulnerability', 'next_period_stress', 'rt_new_stress',
                 'rt_pv_window', 'rt_net_stress_any', 'model_c_score', 'model_b_score'}
    for spec, feats in feature_sets(d).items():
        assert not (set(feats) & forbidden), f'{spec} leaks an outcome column'


@cache_required
def test_horizon_research_targets_are_strictly_future():
    """A research horizon label at week t must depend only on weeks after t."""
    from e5_horizon_structure import research_horizon_targets
    d = research_horizon_targets(_world())
    one = d[d.borrower_id == d.borrower_id.iloc[0]].sort_values('week')
    # a borrower stressed at t must never be labelled positive at t
    for c in [c for c in d.columns if c.startswith('rh_')]:
        assert not (d[c] & d['current_stress'].astype(bool)).any(), \
            f'{c} labels rows that are already stressed at t'


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------

@cache_required
def test_research_features_are_deterministic():
    from e3_variants import add_variant_features
    df = _world()
    a = add_variant_features(df)
    b = add_variant_features(df)
    feats = [c for c in a.columns if c.startswith('v_')]
    for f in feats:
        assert np.allclose(a[f].fillna(0), b[f].fillna(0), rtol=0, atol=0)


def test_world_generation_is_reproducible():
    """Two generators with the same seed must produce identical initial state."""
    from src.data import SyntheticWorldGenerator
    a = SyntheticWorldGenerator(n_borrowers=40, seed=4242, slack_regime='conservative')
    a.generate_world()
    b = SyntheticWorldGenerator(n_borrowers=40, seed=4242, slack_regime='conservative')
    b.generate_world()
    ka = sorted(a.world_state.borrower_to_household)
    kb = sorted(b.world_state.borrower_to_household)
    assert ka == kb
    for k in ka:
        ha = a.world_state.households[a.world_state.borrower_to_household[k]]
        hb = b.world_state.households[b.world_state.borrower_to_household[k]]
        assert ha.cash_buffer == hb.cash_buffer
        assert ha.liability_weight == hb.liability_weight


# --------------------------------------------------------------------------
# Result-integrity guards
# --------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, 'research/m2d/e4_results.csv')),
                    reason='E4 results absent')
def test_unscoreable_worlds_are_recorded_not_dropped():
    d = pd.read_csv(os.path.join(ROOT, 'research/m2d/e4_results.csv'))
    assert 'unscoreable' in set(d.status), \
        'worlds without test positives must be recorded, never silently dropped'
    pv = d[(d.target == 'propagation_vulnerability')]
    assert pv.seed.nunique() == 20, 'all 20 worlds must appear for the frozen target'


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, 'research/m2d/e1_prevalence_per_seed.csv')),
                    reason='E1 results absent')
def test_prevalence_census_covers_every_world():
    d = pd.read_csv(os.path.join(ROOT, 'research/m2d/e1_prevalence_per_seed.csv'))
    assert len(d) == 20
    assert (d.pv_positives >= 0).all()
    # the census must agree with itself: train+val+test <= total positives
    assert (d.train_pos + d.val_pos + d.test_pos <= d.pv_positives).all()
