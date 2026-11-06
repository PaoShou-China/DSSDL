# Experiment 42: parameter-label-free PySR closure

Locked before execution, 2026-09-04. This is an exploratory implementation audit, not a final multi-seed claim.

## Information boundary

- FHN and Lorenz start from the same existing no-symbol source checkpoints used by experiment 41.
- Observed source states, derivative targets, and opaque task IDs are available. Physical parameter values, parameter-bearing filename parsing, physical decoders, analytic equations, and historical discovered equations are forbidden inputs to discovery/training.
- Source checkpoint pretraining already saw the source tasks/data. Held-out source partitions below assess symbolic distillation and continuation, not fully unseen-task generalization.
- No physical coordinate calibration or post-hoc scaling is performed. Learned coordinates remain q1, q2, q3.

## Automatic loop

1. Query the current full neural-plus-symbolic model on shared source-state probes for every learned source task vector.
2. Split the response into the mean over search-training tasks and the task-dependent remainder. This generic decomposition prevents the common response from hiding smaller task effects; it uses no equations or physical labels.
3. PySR searches common response from states and remainder from states plus learned coordinates. Operators are +, -, *. Standardization is estimated on search-training data only and inverted algebraically, not physically calibrated.
4. Expand PySR expressions with SymPy. Candidate terms must have been emitted by PySR; no analytic polynomial library is supplied. Refit coefficients and remove small contribution terms with a fixed threshold grid. Choose a sparse expression within 5% of the minimum held-out normalized RMSE. Save every candidate and selection decision. No integer snapping.
5. Inject the resulting differentiable expression into the original MetaNet interface and original loss: MSE(net+symbol, labels) + lambda MSE(symbol, labels), lambda=1. Resume source training with the original two-step Adam pattern (task mapper, then all model weights).
6. Repeat discovery on the updated model for a fixed two-round pilot budget. Save all rounds, including worse ones. Select the final available checkpoint by source validation derivative RMSE only, allowing round 0 to win. This fixed-budget pilot does not claim convergence.

## Fixed splits / budgets

- Seed 0, one source checkpoint per system. Two discovery/feedback rounds, 60 source continuation epochs per round, batch 512, Adam lr 0.001, weight decay 0.0001.
- Source continuation uses the first 80% of each source trajectory; last 20% is validation. Best post-update epoch per round is chosen by full-model source validation MSE.
- 384 common state probes, drawn from first-80% source states. A fixed 80% source-task partition is used for task-varying symbolic regression, and a fixed 80% probe partition for common-response regression.
- PySR per component/output: up to 100 iterations, 8 populations of 32, maxsize 25, serial execution, fixed seed, 45-second search timeout after Julia setup. Actual elapsed time and equation archives are saved.
- Sparsity thresholds are relative contribution levels: 0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03. Coefficients are refitted in normalized candidate columns.

## Comparisons and evaluation

- Initial no-symbol checkpoint; both closed-loop rounds; no-symbol continuation at the same cumulative epoch budgets. The no-symbol control shares data, splits, optimizer settings and seeds; it does not receive a symbolic penalty.
- Report full-model and symbol-only derivative RMSE in source normalization and raw units. Record teacher-distillation error separately; low distillation error is not proof of a true physical equation.
- Static target: adapt only task vector on the existing support trajectory, report the existing query derivative error. No query values affect search, training, adaptation or selection.
- Time-varying exploratory check: experiment-41 fast-sine target, seed 0, W=0.5 s, H=0.5 s, same 45 causal origins and support-only LBFGS settings. Reuse t/states only, never parameter_reference. Report state RMSE and failed forecasts; compare selected closure, initial no-symbol, and matched no-symbol continuation. The FHN amplitude remains that of experiment 41, not experiment 38.
- State forecasts reset to the observed state at each origin, freeze the adapted vector over the forecast, and do not read future observations until scoring.
- No target metric selects a round or search setting. No multi-seed robustness, full window grid, noise sweep, or physical recovery claim is made in this pilot.

## Verification

Runtime tripwires reject known parameter parsers/decoder fitters. Save checkpoint/input hashes, source split IDs, search inputs, expressions, training logs and evaluation arrays. Numerically compare differentiable symbolic evaluation to NumPy, test gradients, and verify protected inputs are unchanged.
