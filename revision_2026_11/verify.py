"""Recompute the published numerical evidence without training or symbolic search."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import tempfile
import shutil

import numpy as np
import pandas as pd

from reproduce import ROOT, EXP, load, read


def checksums():
    entries = read(ROOT / 'SHA256SUMS.json')
    for name, expected in entries.items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        assert actual == expected, name
    return len(entries)


def controls():
    load('public_control_check', ROOT / 'code/run.py').verify_results()
    return {'pairs': 20, 'variants': 40}


def budgets():
    folder = ROOT / 'results/matched_budget'
    summaries = [read(p) for p in sorted((folder / 'records').glob('*/*/seed_*/support_*.json'))]
    assert len(summaries) == 320
    table = pd.read_csv(folder / 'tables/aggregate.csv')
    count = 0
    for row in table[table.budget == 10000].itertuples():
        selected = [s for s in summaries if (s['system'], s['method'], s['support']) == (row.system, row.method, row.support)]
        assert len(selected) == 5
        values = [next(m['query_rmse'] for m in s['milestones'] if m['step'] == 10000) for s in selected]
        np.testing.assert_allclose([statistics.mean(values), statistics.stdev(values)], [row.rmse_mean, row.rmse_sd], rtol=1e-12, atol=1e-12)
        assert all(s['early_stopping'] is False and s['query_selected_final'] is False for s in selected)
        count += 1
    assert count == 64
    return {'records': len(summaries), 'aggregate_rows': count}


def wasserstein():
    folder = EXP / 'wasserstein'
    mod = load('public_wasserstein_check', folder / 'scripts/original_wasserstein.py')
    with tempfile.TemporaryDirectory(prefix='dssdl-w-') as tmp:
        mod.ROOT = Path(tmp)
        shutil.copytree(folder / 'data', mod.ROOT / 'data')
        (mod.ROOT / 'tables').mkdir()
        actual, _, _ = mod.load_and_compute(read(folder / 'data/source_manifest.json'))
    expected = pd.read_csv(ROOT / 'results/wasserstein/reference.csv')
    for row in actual.itertuples():
        ref = expected[(expected.system == row.system) & (expected.method == row.method)].iloc[0]
        assert row.status == ref.status
        np.testing.assert_allclose(row.sliced_wasserstein_1, ref.sliced_wasserstein_1, rtol=1e-12, atol=1e-12, equal_nan=True)
    return {'trajectories': len(actual), 'complete': int((actual.status == 'complete').sum())}


def online():
    count, max_difference, aggregates = 0, 0., {}
    for system in ('fhn', 'lorenz'):
        for method in ('pysr44', 'paper_no_symbol', 'sindy_source'):
            errors = []
            for seed in range(3):
                folder = ROOT / 'results/online' / system / method / f'seed_{seed:03d}'
                table = pd.read_csv(folder / 'window_metrics.csv')
                table = table[(table.horizon_s == .5) & (table['mode'] == 'online')].sort_values('origin_s')
                with np.load(folder / 'forecasts.npz', allow_pickle=False) as data:
                    np.testing.assert_allclose(data['origin_index'] * .01, table.origin_s)
                    delta = data['online'][:, :50] - data['truth'][:, :50]
                    actual = np.sqrt(np.mean(delta ** 2, axis=(1, 2)))
                np.testing.assert_allclose(actual, table.rmse, rtol=1e-9, atol=1e-10)
                assert np.isfinite(actual).all() and len(actual) == 45
                max_difference = max(max_difference, float(np.max(np.abs(actual - table.rmse))))
                errors.extend(actual.tolist())
                count += len(actual)
            aggregates[system + '/' + method] = float(np.mean(errors))
    expected = {'lorenz/pysr44': .502997, 'lorenz/paper_no_symbol': .523199, 'lorenz/sindy_source': .385173,
                'fhn/pysr44': .008703, 'fhn/paper_no_symbol': .009424, 'fhn/sindy_source': .007420}
    for key, value in expected.items():
        assert abs(aggregates[key] - value) < 5e-7, (key, aggregates[key], value)
    return {'forecast_windows': count, 'maximum_rmse_difference': max_difference, 'means': aggregates}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checksums', action='store_true')
    args = parser.parse_args()
    report = {}
    if args.checksums:
        report['checked_files'] = checksums()
    report.update(controls=controls(), matched_budget=budgets(), wasserstein=wasserstein(), online=online())
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
