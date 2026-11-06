# Experiment-to-evidence map

| Evidence | Location | Reproduction | Interpretation |
|---|---|---|---|
| SI Table 13: dimension selection | `results/controls/pairs/gate` | `code/run.py gate` | 8 candidate coordinates; five paired seeds; pendulum selects 1, quadrotor 3. Accuracy need not improve uniformly. |
| SI Table 14: symbolic reconstruction | `results/controls/pairs/separation` | `code/run.py regularizer-pendulum` / `regularizer-lorenz` | Fixed teacher/template; lambda=1 versus 0; both retain the symbolic term. |
| SI Table 15: adaptation budgets | `results/matched_budget` | `reproduce.py budget` | 2 systems x 4 neural methods x 5 records x 8 support sizes; final 10,000-update comparison. |
| SI Note 3.5: symbolic discovery | `code/review/experiments/label_free_discovery`, `symbolic_feedback`, `measured_pendulum_closure` | `code/run.py discover`, `feedback`, `pendulum` | Search, consolidation and feedback have separate records; physical parameter labels are not fitted. |
| Figures 5e and 6e: distribution distance | `results/wasserstein`, `code/review/experiments/wasserstein/data` | `reproduce.py wasserstein` | Original fixed-parameter trajectory exports; 128 projections, seed 20260720; one saved trajectory per method. |
| Figures 5f-g, 6f-g and SI Table 12: online deployment | `results/online`, `code/review/experiments/online` | `reproduce.py online` | Fixed offline models; 0.5-second trailing window; mean state RMSE over 135 forecasts per method/system. |

## Component controls

All paired comparisons use identical observations and initial conditions within
a pair. Source validation is separate from target adaptation. The target task
coordinate is fitted for 500 Adam updates at learning rate 0.03; the final iterate
is evaluated on the query observations. Query data do not stop adaptation.

For the measured pendulum, source files 1-7 and target file 8 retain the original
scaling, with angular velocity divided by 20 and acceleration divided by 100.
Fifty exact support/query duplicates are excluded from the new query evaluation,
leaving 2,118 observations. Estimated lengths are not training labels.

The Lorenz regularizer target uses new observations at a parameter configuration
already represented among the sources. This control is not parameter extrapolation.
Its source split is temporal 80/20 and its retained checkpoint minimizes complete-
model source-validation error. The pendulum control uses its recorded random
80/20 split and final feedback epoch. These choices are documented separately.

## Matched budgets

Support sizes are 1, 5, 10, 20, 50, 100, 200 and 500, with historical records 1-5.
The saved 320 summaries include the 10,000-update milestone and any subsequent
support-diagnostic extension. Aggregate checks explicitly select the 10,000
milestone rather than the last available budget.

Lorenz DSSDL-symbol source regularization is 10 for records 1-4, except record 1
at one support observation; it is 1 for record 5 and that exception. Thus the
five-record standard deviation is not pure seed variability under one common
source hyperparameter. The original stopping-rule replay and fixed-budget
comparison use paired new runs. Historical outputs are not reproduced bit for
bit: 310/320 replay errors agree within absolute RMSE 2e-5. All records are retained.

## Trajectory metrics

The static FHN export uses parameters [0.85, 1.05, 0.85], initial state [-1, 1]
and 5 seconds; the Lorenz export uses [9.9, 29.5, 2.9], initial state [8, -8, 27]
and 3.5 seconds. These are the original trajectory-error configurations, not the
sample-count configurations above. The incomplete Lorenz GPLearn trajectory is
marked failed; its short prefix is not assigned a full-interval distance.

The online systems use the selected saved scenarios, not newly generated test
conditions. The plotted coefficient readout is an offset for FHN or a multiplier
of x in the Lorenz y equation. No decoder fitted to physical labels, post-hoc
time shift, or amplitude correction is applied. The neural-only model is compared
through state prediction, not assigned an arbitrary physical coefficient.

The main 0.5-second forecast averages are:

| System | DSSDL (feedback checkpoint) | w/o symbol | Source-structured SINDy |
|---|---:|---:|---:|
| FHN | 0.008703 | 0.009424 | 0.007420 |
| Lorenz | 0.502997 | 0.523199 | 0.385173 |

These averages describe the displayed online tests and are not a claim that
DSSDL beats SINDy on every prediction metric.
