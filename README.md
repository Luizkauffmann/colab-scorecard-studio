# Colab Scorecard Studio

An open, copy-and-run Google Colab template for building **binary-target scorecards**:
interactive optimal binning, WOE, logistic regression, scorecard scaling, validation and
export. The flagship use case is the **application (onboarding) PD scorecard**, and the
same framework works for fraud, AML and Kaggle-style binary classification.

> **Status:** Milestone 1 of 7 is done. The `scorecard_studio` engine is packaged and
> tested. The Colab notebook and the interactive binning app arrive in Milestone 2.

## Install

In a Colab cell:

```python
!pip -q install git+https://github.com/Luizkauffmann/colab-scorecard-studio
```

## Quick start (engine only)

```python
from scorecard_studio import BinningEngine, make_credit_application_data
from scorecard_studio.datasets import CREDIT_DEMO_SPECIAL_CODES

df = make_credit_application_data()                       # synthetic demo data
train = df[df.APPLICATION_DATE < "2024-07-01"]            # fit on TRAIN only

engine = BinningEngine(train, target_col="DEFAULT_12M",
                       special_codes=CREDIT_DEMO_SPECIAL_CODES,
                       exclude=["APPLICATION_ID", "APPLICATION_DATE"])
engine.fit_all(max_bins=6, monotonic="auto")
engine.get_iv_summary()                                   # IV / Gini / KS per variable
engine.get_result("MONTHLY_INCOME").summary()             # binning table incl. Missing bin

engine.adjust_cutoffs("AGE", [25, 35, 50, 65])            # manual refinement
woe_df = engine.transform(df, metrics=["woe"])            # apply to Train/Test/OOT

bundle = engine.build_scoring_bundle(name="app_scorecard")
bundle.save_json("bundle.json")                           # reload with ScoringBundle.load_json
bundle.save_python("scorer.py")                           # stdlib-only scorer
bundle.save_sql("transform.sql", dialect="bigquery")      # standard | spark | bigquery
```

## Conventions

| Topic | Rule |
|---|---|
| Event | `target == 1` (default, fraud, SAR...) |
| WOE | `ln(%events / %non-events)`. Positive WOE means riskier than average. |
| Numerical bins | `(lower, upper]`. A value equal to a cutoff falls in the lower bin. |
| Missing | Always its own bin (group `0`). If training has no missing values, its WOE is 0 and you get a warning. |
| Special codes | Pooled into one Special bin (group `-1`) and excluded from the cutoff search. |
| Unseen categories | Scored as Missing. |
| Gini | `2 * AUC - 1`, computed from the binned WOE. |
| `min_bin_size` | Share of **all** rows, including missing and special. |
| Scorecard points | `-factor * beta * WOE + (offset - factor * b0) / n`; needs the model coefficients |

`scorer.py`, the SQL export and `engine.transform` are tested to produce identical output,
including at exact cutoffs, on special codes, missing values and unseen categories.

## Domain presets

```python
from scorecard_studio import get_preset
p = get_preset("fraud")          # credit_pd | fraud | aml | kaggle
engine.fit_all(**p["fit_params"])
```

Fraud and AML presets set `min_bin_n_event`, so every bin holds enough events for its WOE
to be meaningful at low event rates.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Roadmap

1. **Engine package**: done
2. Interactive binning app running inside Colab, with state saved to Google Drive
3. Model-ready output dataset (WOE or bin dummies) for Train / Test / OOT
4. Variable selection, logistic regression, scorecard scaling
5. Validation report and deployment exports
6. Public template, sample datasets, walkthrough
7. Medium article

Ported from the author's Dataiku webapp
[`scorecard-binning`](https://github.com/Luizkauffmann/scorecard-binning), which remains
a separate project.

## License

Apache 2.0
