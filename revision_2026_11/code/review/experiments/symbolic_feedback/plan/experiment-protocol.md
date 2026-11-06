# Experiment 44: human-assisted symbolic coefficient collapse

Locked before execution on 2026-09-04 following the user's explicit request.
The user permits human simplification but explicitly forbids filling missing
terms from the known system. This is NOT fully automatic structure discovery.

## Input and scope

- First label-free GPLearn expressions from experiment 37 and first label-free
  PySR expressions from experiment 42, FHN and Lorenz.
- Learned source task coordinates from the original no-symbol checkpoint.
- Existing source states and derivative supervision. No source/target physical
  parameter labels, physical decoders, target data or analytic coefficient values.
- Preserve the original code, old checkpoints/results, and manuscript. All new
  work lives here. Experiment 43 remains prepared but unexecuted.

## Manual structural choices (not truth-free automatic selections)

Numbers are not snapped. All shared numerical coefficients start at the mean
of their discovered coefficient function on search-training source tasks.
The choices below are manually selected from already present state factors.
They must NOT be described as uniquely minimal or algebraically equivalent
to the complete first-round expression.

FHN, both engines:

    dv = a*v + b*v^3 + c*w + Q1
    dw = d*v + Q2 + Q3*w

Q1 is the complete discovered state-independent coefficient of dv. Q2 is the
state-independent coefficient of dw. Q3 is the coefficient of w in dw.
Other state factors are removed. Weak q dependence in the coefficients of
v/v^3/w in dv and v in dw is projected to shared scalars a,b,c,d.

Lorenz, GPLearn:

    dx = Q1*(y-x)
    dy = Q2*x + a*x*z
    dz = b*x*y + c*z + Q3

Q1 = (coefficient_y(dx) - coefficient_x(dx))/2. This manually ties two nearly
opposite coefficients; the leftover common-sign component is removed.
Q2 = coefficient_x(dy). Q3 = the complete additive coefficient of dz.
No y term is inserted in dy, and c*z remains a shared, not task-dependent,
coefficient. Other first-round factors/biases are removed.

Lorenz, PySR:

    dx = Q1*(y-x)
    dy = Q2*x + a*x*z + b*y
    dz = c*x*y + Q3*z

Q1 is tied as above; Q2 = coefficient_x(dy); Q3 = coefficient_z(dz), which was
present and task-dependent in the FIRST PySR round. The task dependence of
the retained x*z and y coefficients is projected to shared scalars. Additional
state factors and additive bias functions are removed, even if their effect
is non-negligible; report the initial projection error explicitly.

## Coordinate and optimization changes

- The three new Q functions are evaluated on old learned source q coordinates
  to initialize three new task coordinates. No physical labels are used.
- For numerical conditioning only, use Q_j = mean_j + std_j*z_j with moments
  computed from these discovered coefficient values on source search-training
  tasks. These statistics are not physical calibration. Equations are reported
  in Q coordinates; model inputs use z coordinates.
- Treating Q functions as independent coordinates relaxes their old algebraic
  dependencies. This is reparameterization, not just renaming notation.
- Keep the original MetaNet hidden widths and three-dimensional task vector.
  Warm-start existing network weights; replace source task-vector values with
  normalized Q values. No trained inverse/physical decoder is fitted.
- Shared numerical coefficients are additional trainable scalar parameters in
  the new prior module, initialized only from first-round expressions. This is
  a disclosed extension; it is not the unchanged original fixed-prior method.
- Original two-step optimization and loss are reused:
  MSE(network + symbol, derivative) + MSE(symbol, derivative).
  100 source epochs, lr .001, batch 512; best source-time-validation full-model
  RMSE epoch. First80% source trajectories train, last20% validate. Pretraining
  already saw these sources: no unseen-neural-task claim.

## Next symbolic search and reporting

- The compact prior is fed back for 100 epochs. Save before/after exact prior,
  learned task coordinates, coefficient values, source symbol-only errors,
  pruning residual and retained/deleted state-factor records.
- A following symbolic search, when run, must use the same engine for its
  branch. It receives model responses, observed states and new z coordinates,
  never the known true equation. Save raw outputs without manually claiming
  they stayed compact. Search budgets must be recorded explicitly.
- Distinguish first-round simplification, feedback optimization and following
  symbolic rediscovery. A feedback checkpoint alone is not a second search.
- Do not make prediction-superiority or fully-automatic-recovery claims.
- Verify protected hashes, permitted state support, coefficient-function
  initialization, differentiable/numpy agreement, finite gradients, selection
  epoch, and source-only information access.
