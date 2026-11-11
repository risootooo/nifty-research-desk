# Contributing

Thanks for looking under the hood. This repo is a daily research desk
for Indian markets — its one non-negotiable rule is that **no number
appears in a published brief unless an engine produced it**. Correctness
and honest reporting matter more than features.

## Run the tests first
```bash
pip install -e .
python -m pytest -q
```
Every change must keep the full suite green, and new behaviour needs
new tests.

## Standards held in this repo
- **The number guard is sacred.** Brief prose is generated from the
  day's snapshot by the agents in `src/nifty_desk/agents.py`, and
  `verify_numbers` must pass on every composed brief. Never hardcode a
  market figure into a template; if a sentence needs a number, the
  number comes from the snapshot or it doesn't get printed.
- **Engines are textbook-faithful.** The vendored models (Hamilton
  regime-switching, GARCH, chain/Greeks/gamma machinery) are implemented
  from their papers — changes to an estimator need a test that shows it
  still recovers known behaviour.
- **Real data or labelled synthetic** — say which, with provenance and
  the fetch route recorded. Bhavcopy settlement marks and model
  estimates must be described as what they are.
- **Disagreements are published, not smoothed.** The sceptic agent
  exists to flag internal tensions between panels. If your change makes
  a panel disagree with another, the brief should say so.
- **Research language only.** Nothing in this repo is investment
  advice, a signal service, or a recommendation — wording that drifts
  that way will be asked to change.
- **Seeded and typed** — every simulation takes a seed; public
  functions carry type hints and docstrings.

## Pull requests
Small, single-purpose PRs with a description of what changed and what
the tests show. Discussion first for anything that changes an engine,
a brief template, or the number guard.
