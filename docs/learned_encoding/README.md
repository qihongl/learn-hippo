# Reproducing the learned event-boundary encoding experiment

This directory documents the isolated `boundary_em` experiment added to the legacy
`learn-hippo` repository. The primary entry point is the
[self-contained HTML report](experiment_report.html). Its source is
[experiment_report.md](experiment_report.md); all quantitative claims trace to JSON
under `outputs/learned_encoding/`.

## Install and verify

From the repository root, install the modern environment and run its tests:

```sh
uv sync --extra dev --extra report
uv run pytest
uv run ruff check .
```

The new package is isolated under `src/boundary_em`; the commands above do not run or
modernize the historical Python stack used for the 2022 paper. The Ruff `include`
contract in `pyproject.toml` therefore covers the isolated package, its tests, and
the reproducible figure sources rather than the untouched legacy files.

## Reproduce the oracle precondition

This evaluates every binary write schedule over the four semantic study states under
both declared capacity conventions:

```sh
uv run python -m boundary_em.run_oracle \
  --config configs/learned_encoding/oracle.yaml
```

The command writes `outputs/learned_encoding/oracle_results.json`. Verify that
`oracle_precondition.fixed.passed` and `oracle_precondition.historical.passed` are
both true before training a policy.

## Reproduce confirmatory policy training

The reported analysis uses all model seeds 100–114. Each invocation writes one JSON
file and one ignored checkpoint. The seeds may be dispatched independently; the
simple sequential version is:

```sh
for seed in $(seq 100 114); do
  uv run python -m boundary_em.run_policy \
    --config configs/learned_encoding/reported.yaml \
    --seed "$seed"
done
```

Aggregate only after all 15 seed files exist:

```sh
uv run python -m boundary_em.aggregate \
  --config configs/learned_encoding/reported.yaml \
  --input-directory outputs/learned_encoding/reported \
  --output outputs/learned_encoding/reported_summary.json
```

The aggregator rejects missing seeds and mismatched configuration hashes, computes
the fixed-seed bootstrap intervals, and records every preregistered success check in
`success_audit`. Model weights remain frozen during all held-out evaluations.

## Reproduce post-confirmatory mechanism analyses

This stage requires the locally generated checkpoints from the previous step:

```sh
uv run python -m boundary_em.run_mechanism \
  --config configs/learned_encoding/mechanism.yaml
```

It writes `outputs/learned_encoding/mechanism_results.json`. These analyses use new
episode seeds and are exploratory: they test gate-input ablations, write probability
by semantic progress, retrieval-temperature sensitivity, and a latest-trace oracle.

## Rebuild figures and the standalone paper

```sh
for script in outputs/learned_encoding/figures/make_fig_*.py; do
  uv run python "$script"
done

scripts/build_learned_encoding_html.sh
```

Every figure script reads versioned JSON rather than hard-coded result values and
exports PDF, editable SVG, and 300-DPI PNG. The HTML build embeds the SVGs, CSS, and
MathML equations into one portable file and fails if a runtime resource remains
external.

## Evidence and design documents

- `experiment_design.md`: hypotheses, staged protocol, criteria, and failure logic.
- `data_contract.md`: generator, split, leakage, and provenance contract.
- `decision_log.md`: chronological scientific and architectural choices.
- `RESULTS.md`: concise stage-by-stage results and limitations.
- `experiment_report.md`: full small-paper narrative used to generate the HTML.
- `outputs/learned_encoding/figures/figure_contract.md`: figure-level claim and
  uncertainty specifications.

All reported values are measured executions of a controlled synthetic task. They are
not human data. Human review by a domain expert is strongly recommended before using
the report for scientific publication.
