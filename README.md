# Colab Scorecard Studio

An open, copy-and-run Google Colab template for building **binary-target scorecards**:
interactive optimal binning, WOE, logistic regression, scorecard scaling, validation and
export. The flagship use case is the **application (onboarding) PD scorecard**, and the
same framework works for fraud, AML and Kaggle-style binary classification.

> **Status:** Milestone 1 (engine) is done, and notebook sections 0–3 (setup, data intake,
> sample design, univariate screening) run end to end. The interactive binning app arrives in
> Milestone 2.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Luizkauffmann/colab-scorecard-studio/blob/m1b-intake-screening/notebooks/scorecard_studio.ipynb)

## Install

In a Colab cell:

```python
!pip -q install git+https://github.com/Luizkauffmann/colab-scorecard-studio
```

## The notebook

`notebooks/scorecard_studio.ipynb` is the template. Copy it to your Drive, edit the one
config cell, and run all. Each section writes to a versioned folder on Drive
(`scorecard_studio/run_YYYYMMDD_HHMMSS/`).

The default data is the Kaggle [Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
(`laotse/credit-risk-dataset`, target `loan_status`), loaded with no Kaggle account. The demo
config excludes `loan_grade` and `loan_int_rate` (lender-assigned, so circular in an application
model) and sends impossible ages and employment lengths to the Special bin.

## Quick start: intake, split, screening

```python
import scorecard_studio as ss

cfg = ss.StudioConfig(
    data_source="demo:credit_risk", target="loan_status", preset="credit_pd",
    exclude=["loan_grade", "loan_int_rate"],
    force_include=["loan_percent_income", "person_income"],   # reviewed high-IV variables
    iv_min=0.10, iv_max=0.50,
    plausibility={"person_age": (18, 100), "person_emp_length": (0, 60)},
)
intake = ss.run_intake(cfg)                 # load, check names, 0/1 target, plausibility rules
sample = ss.make_split(intake.df, cfg.target, date_col=cfg.date_col, seed=cfg.seed)
train = intake.df[sample == "train"]        # screening and binning see Train only

screen = ss.screen_variables(
    train, cfg.target, iv_min=cfg.resolved_iv_min, iv_max=cfg.resolved_iv_max,
    fit_params=cfg.fit_params, exclude=cfg.exclude, force_include=cfg.force_include,
    special_codes=intake.special_codes, id_col=intake.id_col)
screen.table                                # status and reason for every candidate
screen.selected                             # shortlist for the binning app
screen.save("outputs/screening")            # shortlist.json, table CSV, HTML report
```

Screening statuses: `selected`, `forced` (in `FORCE_INCLUDE`), `review` (IV above `IV_MAX`:
not kept until you force-include or exclude it), `low_iv`, `excluded`, `id_like`, `constant`,
`fit_error`. A high IV is treated as a question, not a prize.

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
| Screening | On Train only, IV from an automatic fit with the preset's constraints. `credit_pd`: `IV_MIN` 0.10; fraud/AML/Kaggle: 0.02. `IV_MAX` 0.50 holds for review, never drops silently. |
| Infeasible constraints | If no binning satisfies `min_bin_size` / `min_bin_n_event` / `max_bins`, `fit` raises instead of returning one bin. |
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

1. **Engine package**: done. Intake, sample design and screening (notebook sections 0–3): done
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
