# Third-party code

The two `cavia-master` subdirectories contain `regression/cavia_model.py` from
the project's existing CAVIA baseline. Each copy retains its original MIT
license (Copyright 2019 Luisa M. Zintgraf). The model is included so that the
matched-budget CAVIA entry point does not depend on an external checkout.

Scientific Python packages are installed separately and retain their own
licenses. The archival modules in `code/baseline_reference/GPlearn` and `SINDy`
also require their corresponding packages if used directly; neither is required
to recompute the distances from saved trajectories. Their presence does not
imply that every historical baseline script is a supported standalone launcher.
Use the commands in the package README for the checked reproduction paths.
