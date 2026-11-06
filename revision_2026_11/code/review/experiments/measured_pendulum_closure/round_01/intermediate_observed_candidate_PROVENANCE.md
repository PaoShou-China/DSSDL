# Intermediate Candidate Provenance

`intermediate_observed_candidate.csv` is a transcription of the complexity-16 row that the agent read from the running first-round output0 `hall_of_fame.csv`, before this run finished. It is not the final Pareto candidate at complexity 16: later search replaced that entry with a lower-error mixed-angle sine. The source text and loss appeared in the tool output of this conversation. This transcription is explicitly distinguished from PySR's final native `pysr/*/hall_of_fame.csv` and `candidates.json` artifacts.

No expression term or coefficient was inferred from a reference pendulum equation. The intermediate row already contains sin(theta), u, q-dependent amplitudes, and u^2. The agent selects this actual emitted intermediate row for interpretability, under the user's authorization for human intervention. Its score is recomputed on archived search_data before any simplification. The final automatic choice remains separately archived.

The original running file was `round_01/output0/pysr/20260905_172710_AdPnt6/hall_of_fame.csv`. That native file now contains the final, replaced entry. This separate transcription does not modify it. Recomputed training MSE is 0.0018175566313107302, agreeing with the observed native float32 loss 0.0018175564.

The feedback result is therefore human-assisted candidate selection and approximate simplification, not fully automatic final-Pareto selection. Reproducing the downstream feedback uses this saved candidate directly and does not depend on re-capturing transient search files.
