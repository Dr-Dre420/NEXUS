"""M2D RESEARCH — run the experiment matrix over a seed list."""
import sys, os, json, argparse
sys.path.append(os.path.abspath('.')); sys.path.append(os.path.abspath('research/m2d'))
import numpy as np, pandas as pd
from experiment import load, feature_sets, fit_eval, CACHE
from features_research import CANDIDATE_SETS

HYPOTHESES = {
 'A':   'Individual financial features only; no peer/network context. Lower bound.',
 'B':   'Individual + network context (M2C Model B, 51 features). The baseline to beat.',
 'P0_m2c_exposure':   'M2C Model C. Exposure = peer_predicted_stress_mean * borrower_liability_share; both factors already in B, so no new information is expected.',
 'P1_liab_channel':   'Routing peer stress through the CONTRACTUAL liability weight (the generator transmission channel) rather than a wealth share should add information B lacks.',
 'P2_concentration':  'Mean peer stress cannot distinguish one severe peer from several mild ones; max/count/HHI restore that structure.',
 'P3_persistence':    'A peer stressed for several consecutive weeks drains the group pot cumulatively; decay and velocity capture duration the contemporaneous mean discards.',
 'P4_dest_fragility': 'Absolute buffer adequacy (buffer / obligation) measures capacity to absorb a liability call; B only has a RELATIVE wealth share.',
 'P5_process':        'Propagation as a process: source stress x contractual channel x persistence x destination vulnerability, in one interpretable scalar.',
 'P6_full':           'All research propagation features together; upper bound on what this family can contribute.',
}

def main(seeds, targets, tag, outdir):
    os.makedirs(outdir, exist_ok=True)
    rows = []
    for seed in seeds:
        df = load(seed)
        A, B = feature_sets(df)
        for target in targets:
            specs = [('A', A), ('B', B)]
            for name, extra in CANDIDATE_SETS.items():
                specs.append((f'B+{name}', B + [c for c in extra if c in df.columns]))
            for exp_name, feats in specs:
                r = fit_eval(df, feats, target)
                base = dict(experiment_id=f'{tag}|{target}|{exp_name}|seed{seed}',
                            tag=tag, target=target, model_spec=exp_name, seed=seed,
                            model='LightGBM(binary,lr0.05,leaves31,depth5,seed42)',
                            hypothesis=HYPOTHESES.get(exp_name.replace('B+',''), HYPOTHESES.get(exp_name,'')),
                            features=';'.join(feats) if len(feats) < 20 else f'{len(feats)} features',
                            calibration='Platt on validation split only; test labels unused')
                if r is None:
                    base.update(dict(status='unscoreable_no_test_positives'))
                else:
                    r.pop('preds', None)
                    base.update(r); base['status'] = 'ok'
                rows.append(base)
        print(f'  seed {seed} done', flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(outdir, f'{tag}_results.csv'), index=False)
    d.to_json(os.path.join(outdir, f'{tag}_results.json'), orient='records', indent=2)
    print(f'wrote {len(d)} rows -> {outdir}/{tag}_results.csv')
    return d

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs='+', type=int, required=True)
    ap.add_argument('--targets', nargs='+', default=['propagation_vulnerability','rt_new_stress'])
    ap.add_argument('--tag', required=True)
    ap.add_argument('--outdir', default='research/m2d')
    a = ap.parse_args()
    main(a.seeds, a.targets, a.tag, a.outdir)
