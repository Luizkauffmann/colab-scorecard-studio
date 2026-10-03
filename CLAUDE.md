# Notes for Claude

- `Luizkauffmann/scorecard-binning` is the original Dataiku webapp. It is a read-only
  reference: never modify it. All work for this project happens in this repo.
- Engine code lives in `src/scorecard_studio/`. It has no Dataiku imports.
- Variable-type detection must only come from `scorecard_studio.dtypes.detect_dtype`.
- Any change to scoring rules must keep `tests/test_exports_parity.py` passing. The
  engine, `scorer.py` and SQL must always agree.
- Conventions (WOE sign, `(lower, upper]`, Missing = group 0, Special = group -1) are
  documented in README.md. Don't change them silently.
- Run `pytest` before every commit.
