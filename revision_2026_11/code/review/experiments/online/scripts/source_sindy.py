"""Source-only support selection and causal, regularized coefficient updates."""
from pathlib import Path
import hashlib
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


def library(x, powers):
    return np.prod(x[:, None, :] ** powers[None, :, :], axis=-1)


def update(x, derivative, previous, artifact, alpha=None):
    design = library(x, artifact['powers']) / artifact['feature_scale']
    y = derivative / artifact['output_scale']
    result = np.zeros_like(previous)
    penalty = artifact['alpha'] if alpha is None else alpha
    for j, mask in enumerate(artifact['support']):
        a = design[:, mask]
        if a.shape[1] == 0:
            continue
        if penalty == 0:
            result[j, mask] = np.linalg.lstsq(a, y[:, j], rcond=1e-10)[0]
        else:
            stacked = np.vstack([a / np.sqrt(len(a)), np.sqrt(penalty) * np.eye(a.shape[1])])
            target = np.r_[y[:, j] / np.sqrt(len(a)), np.sqrt(penalty) * previous[j, mask]]
            result[j, mask] = np.linalg.lstsq(stacked, target, rcond=1e-10)[0]
    return result


def physical_coefficients(normalized, artifact):
    return normalized * artifact['output_scale'][:, None] / artifact['feature_scale']


def load_artifact(folder):
    with np.load(folder / 'artifact.npz') as a:
        result = {k: a[k].copy() for k in a.files}
    info = json.loads((folder / 'selection.json').read_text())
    result['alpha'] = info['alpha']
    return result


def fit_source(system, project, folder, forecast):
    import pysindy as ps
    if (folder / 'selection.json').exists():
        return load_artifact(folder)
    folder.mkdir(parents=True, exist_ok=True)
    source = project / 'compare2/DSSDL/datasets' / ('FHN' if system == 'fhn' else 'lorenz') / 'train'
    tasks, hashes = [], {}
    # File names are used only to pair state and derivative observations.
    for path in sorted((source / 'features').glob('*.txt')):
        label = source / 'labels' / path.name
        x, y = np.loadtxt(path), np.loadtxt(label)
        tasks.append((x, y, int(.7 * len(x))))
        for p in [path, label]:
            hashes[str(p.resolve())] = hashlib.sha256(p.read_bytes()).hexdigest()
    dim = tasks[0][0].shape[1]
    poly = PolynomialFeatures(3).fit(np.zeros((1, dim)))
    powers = poly.powers_
    tx = np.concatenate([x[:cut] for x, _, cut in tasks])
    ty = np.concatenate([y[:cut] for _, y, cut in tasks])
    feature_scale = np.maximum(np.sqrt(np.mean(library(tx, powers)**2, axis=0)), 1e-12)
    output_scale = np.maximum(np.sqrt(np.mean(ty**2, axis=0)), 1e-12)
    artifact = dict(powers=powers, feature_scale=feature_scale, output_scale=output_scale)
    choices, records = [], []
    for threshold in [.001, .003, .01, .03, .1, .2]:
        masks = []
        for x, y, cut in tasks:
            optimizer = ps.STLSQ(threshold=threshold, alpha=1e-6, max_iter=50, unbias=False)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                optimizer.fit(library(x[:cut], powers) / feature_scale, y[:cut] / output_scale)
            masks.append(np.abs(optimizer.coef_) > 1e-12)
        support = np.mean(masks, axis=0) >= .5
        candidate = artifact | dict(support=support, alpha=1e-6)
        fitted, errors = [], []
        for x, y, cut in tasks:
            coef = update(x[:cut], y[:cut], np.zeros((dim, len(powers))), candidate)
            pred = library(x[cut:], powers) / feature_scale @ coef.T
            fitted.append(coef)
            errors.append(np.mean((pred-y[cut:]/output_scale)**2))
        score = float(np.mean(errors))
        records.append(dict(threshold=threshold, active_terms=int(support.sum()), validation_mse=score))
        choices.append((score, int(support.sum()), threshold, candidate, np.asarray(fitted)))
    best = min(c[0] for c in choices)
    selected = min((c for c in choices if c[0] <= best*1.05+1e-12), key=lambda c:(c[1], c[0], -c[2]))
    _, _, threshold, artifact, fitted = selected
    artifact['initial'] = np.median(fitted, axis=0)
    alpha_scores = []
    limit = 100. if system == 'fhn' else 500.
    # Only source validation tails are forecast; no target files are read here.
    for alpha in [0., 1e-6, 1e-4, 1e-3, 1e-2, .1]:
        initial_states, coefs, truths = [], [], []
        for x, _, cut in tasks:
            previous = artifact['initial'].copy()
            for origin in range(cut, len(x)-50, 50):
                history = x[origin-50:origin+1]
                previous = update(history[:-1], np.diff(history, axis=0)/.01, previous, artifact, alpha)
                initial_states.append(x[origin]); coefs.append(physical_coefficients(previous, artifact))
                truths.append(x[origin+1:origin+51])
        pred = forecast(np.asarray(initial_states), np.asarray(coefs), powers, 50, limit)
        truth = np.asarray(truths)
        failed = ~np.isfinite(pred).all((1, 2))
        score = float(np.sqrt(np.mean(((pred-truth)/output_scale)**2))) if not failed.any() else float('inf')
        alpha_scores.append(dict(alpha=alpha, failed=int(failed.sum()), normalized_state_rmse=score))
    alpha_choice = min(alpha_scores, key=lambda a:(a['failed'], a['normalized_state_rmse'], a['alpha']))
    artifact['alpha'] = alpha_choice['alpha']
    pd.DataFrame(records).to_csv(folder / 'structure_candidates.csv', index=False)
    pd.DataFrame(alpha_scores).to_csv(folder / 'temporal_regularization_candidates.csv', index=False)
    np.savez_compressed(folder / 'artifact.npz', **{k:v for k,v in artifact.items() if k != 'alpha'})
    terms = poly.get_feature_names_out(['v','w'] if system=='fhn' else ['x','y','z'])
    selection = dict(system=system, source_task_count=len(tasks), threshold=threshold, alpha=artifact['alpha'],
                     support=[[str(t) for t in terms[mask]] for mask in artifact['support']],
                     source_hashes=hashes, physical_parameter_labels=False, true_equation_template=False,
                     parameter_names_parsed=False, target_data_used_for_selection=False,
                     source_refit='first70percent only; validation tails never refitted',
                     regularization_objective='mean squared normalized derivative error + alpha * squared change in normalized active coefficients')
    (folder / 'selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    print('SOURCE SELECTED', system, selection['support'], 'alpha',artifact['alpha'],flush=True)
    return artifact
