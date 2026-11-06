# Recorded symbolic discovery and feedback

The first-round PySR searches use state observations and learned task coordinates,
not physical parameter labels. The frozen neural model supplies the responses
for the search. FHN and Lorenz search mean and task-dependent responses separately;
the candidate terms are refitted and selected using the recorded source validation
and complexity rule. Search settings and expressions are in
`records/published_tables/search_settings.tex` and `first_round_expressions.tex`.

## Consolidation

For a term `a(q)*psi(x)`, the substitution `Q=a(q)` is exact. Removing terms,
tying coefficients or making previously tied coefficients independent changes
the model and is recorded as approximate consolidation. The feedback templates
were proposed with LLM assistance and confirmed by the authors. The code in
`symbolic_feedback/scripts/run_feedback.py` implements those recorded choices;
it is not a general-purpose autonomous simplifier.

- FHN retains `v`, `v**3`, `w` and an offset in the first equation, and `v`, `w`
  and an offset in the second. Learned coefficient functions initialize new task
  coordinates; shared numerical coefficients remain trainable.
- Lorenz retains `Q1*(y-x)`, `Q2*x+c1*x*z+c2*y` and `c3*x*y+Q3*z`. Tying the two
  linear coefficients in the first equation is approximate, not an identity.
- The measured pendulum uses an earlier candidate from the same search with
  separate `sin(theta)` and velocity terms, rather than the final automatic
  candidate containing `sin(theta+0.70709*u)`. Its quadratic velocity and constant
  terms are removed, and initially tied slopes may vary independently.

After consolidation, the structure stays fixed while task coordinates, shared
numerical coefficients and neural weights are fitted to observations. The saved
before/after models and equations preserve the sequence. Feedback errors assess
these chosen templates; repeated fixed-template training is not repeated discovery.

## General LLM prompt

The following prompt was tested retrospectively on anonymized first-round fits.
It was not the automated selection rule used to obtain all feedback templates.

```text
Read the fitted expressions as if you had to explain them to a colleague
who does not know the system. What is the simplest account that still
explains the distinctions visible in the input?

First rewrite without approximation: group repeated patterns and name a
combination only when the name makes a relationship easier to see.
Then consider what each proposed simplification would lose. Does it
merely change notation, or would two cases the fit distinguishes now
look the same? Use the supplied examples to judge that trade-off.
Prefer the simpler account when the distinction has little support;
a small number alone is not evidence of a small effect.

Give one preferred form for each output, define any new symbols, and
show what was discarded (original minus proposal). Briefly explain the
most consequential choice and what observation could overturn it.
Do not assume an underlying law or add terms absent from the input.
```

The prompt check retained extra terms and did not reproduce all author-confirmed
templates. For example, `a(q)*x+b(q)*y` can be collected exactly into
`Q*(y-x)+D*(x+y)`, with `Q=(b-a)/2` and `D=(a+b)/2`. Dropping the second term is
a separate approximation. This example explains the mapping; it was not supplied
as a hint in the anonymized prompt test.
