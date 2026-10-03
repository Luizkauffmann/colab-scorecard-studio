# Colab Scorecard Studio

An open, copy-and-run Google Colab template for building **binary-target scorecards**:
interactive optimal binning, WOE, logistic regression, scorecard scaling, validation and
export. The flagship use case is the **application (onboarding) PD scorecard**, and the
same framework works for fraud, AML and Kaggle-style binary classification.

> **Status:** notebook sections 0–8 run end to end: setup, data intake, sample design,
> univariate screening, the interactive binning app, logistic regression, the Siddiqi
> scorecard, rescoring, and the scorecard alignment app with exported Python/SQL scoring code.
> Bin-stability and full validation reports come next.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Luizkauffmann/colab-scorecard-studio/blob/main/notebooks/scorecard_studio.ipynb)

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

## The binning app

Section 4 opens an interactive binning app inside the notebook output (Flask in a background
thread, served through Colab's own port proxy: no public tunnel). It is a port of the
[Dataiku webapp](https://github.com/Luizkauffmann/scorecard-binning):

* drag cutoffs on the variable's histogram, double-click to add one, type exact values
* merge adjacent bins, split a bin at its median, regroup categories with chips
* re-run optimal binning with other settings, or reset to the automatic result
* Missing and Special are their own bins by default; merging them into a regular bin (or
  Special with Missing) is a modeling choice made in the bin table, applied identically by
  `transform`, `scorer.py` and SQL
* several candidate targets (`ALT_TARGETS`): a target selector, one set of bins per target,
  and every target column excluded as a predictor for the others
* flags for empty, tiny, zero-event and non-monotonic bins
* every change is saved to Drive (`04_binning/binning_config.json`) and reloaded after a
  runtime reset when the Train sample is the same

**Create output dataset** writes `model_dataset.parquet` with every sample (Train, Test, OOT)
and, per selected variable, the original column, `opt_<var>` (bin, ordered category) and
`woe_<var>`. Test and OOT get the Train bins through the same scoring code as the exported
`scorer.py` and SQL (tested row for row).

```python
from scorecard_studio.app import launch_app
app = launch_app(train, "loan_status", store=store, full=df, sample=sample,
                 screen=screen, special_codes=intake.special_codes, fit_params=cfg.fit_params)
app.save_output()          # or the button in the app
app.engine                 # the fitted BinningEngine
```

## Model, scorecard, rescoring

```python
bundle = app.engine.build_scoring_bundle(output["variables"])
model = ss.fit_logistic(model_df, "loan_status",
                        {"person_income": "woe", "person_home_ownership": "bins", "loan_amnt": "raw"},
                        bundle, selection="backward")        # fit on Train rows only
model.summary, model.flags, model.performance                # p-values, VIF, signs, Gini/KS by sample

card = ss.build_scorecard(model, bundle, ss.ScalingParams(pdo=20, base_score=600, base_odds=50),
                          plausibility=cfg.plausibility_spec())
card.table()                                                 # points per bin (Siddiqi)
card.set_points("person_home_ownership", 1, 160)             # manual judgment, flagged in the table
card.save("scorecard.json")

ss.load_scorecard("scorecard.json").score(new_df)            # plausibility -> bins -> points -> score, pd
```

Each variable enters the model as `"woe"` (one coefficient, expected positive), `"bins"`
(dummies, most populous bin as reference; a separate Missing bin is a missing flag) or `"raw"`
(linear; refused when the variable has missing values or special codes). Points follow
Siddiqi: Factor = PDO/ln2, Offset = Score − Factor·ln(Odds),
points = −(β·WOE + α/n)·Factor + Offset/n. Before rounding, a record's points add up exactly to
the model's score (tested).

## Scorecard alignment app (section 8)

A second app, inside the notebook, on the final scorecard and every scored row:

* **Strategy**: cutoff by maximum profit (benefit of a good, cost of a bad), target approval
  rate, maximum bad rate, or manual; exact curves over every distinct score; KPIs, profit and
  rate curves, score distribution, ROC, decision matrix. The cutoff is set on Test (or OOT),
  approval rates count every row and bad rates only rows with a known outcome. The
  profit-optimal cutoff is compared with the scale's break-even score as a calibration check.
* **Scorecard**: editable points per bin with the model's points kept, a reason per override,
  flags for overrides set under another scale and for bins whose points break the risk order.
* **Statistics**: AUC, Gini (= 2·AUC − 1), KS per sample, gains table in PDO-wide bands,
  observed vs expected bad rate, realized PDO and odds, score PSI, each variable's share of the
  score range.
* **Export**: the final table (`pts_<var>`, `score`, `pd`, `decision` for every row) and the
  scoring code: a standalone Python script (`python scorer.py in.csv out.csv`) or SQL
  (standard, Spark, BigQuery). Both include the data-quality rules, the bins with the
  Missing/Special choices and the points after overrides, and are tested to reproduce the
  final table exactly.

```python
from scorecard_studio.align import launch_alignment
align = launch_alignment(card, df, "loan_status", store=store, sample=sample,
                         cost_bad=1000, benefit_good=100)
align.finalize()                  # or the button
print(align.code("sql", dialect="bigquery", table="applications"))
```

`GOOD_CLASS` says which target value is good (0 by default); a higher score always means a
higher probability of good, so AUC and cutoffs can't be inverted by the coding.

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
| Missing | Its own bin (group `0`) by default. If training has no missing values, its WOE is 0 and you get a warning. Can be merged into a regular bin (`missing_to`). |
| Special codes | Pooled into one Special bin (group `-1`) and excluded from the cutoff search. Can be merged into a regular bin or combined with Missing (`special_to`). |
| Merged Missing/Special | Scored with the target bin's group, WOE and label in `transform`, `scorer.py` and SQL. |
| Unseen categories | Scored like Missing. |
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
2. Interactive binning app running inside Colab, with state saved to Google Drive: done
3. Model-ready output dataset (`opt_` bins and `woe_` columns) for Train / Test / OOT: done; bin stability report next
4. Variable selection, logistic regression, scorecard scaling, rescoring: done
5. Validation report and deployment exports
6. Public template, sample datasets, walkthrough
7. Medium article

Ported from the author's Dataiku webapp
[`scorecard-binning`](https://github.com/Luizkauffmann/scorecard-binning), which remains
a separate project.

## License

Apache 2.0
