# Builds notebooks/scorecard_studio.ipynb. Edit this file, then run it.
import os

import nbformat as nbf

REF = "m4c-alignment-app"
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(f"""
# Colab Scorecard Studio

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Luizkauffmann/colab-scorecard-studio/blob/{REF}/notebooks/scorecard_studio.ipynb)

A copy-and-run template that takes a **binary-target table** (1 = event: default, fraud, SAR...) to a scorecard. The flagship case is an **application PD scorecard**.

**How to use it**
1. *File → Save a copy in Drive*, so your changes and outputs are yours.
2. Edit **one cell**, the config cell in section 0. Everything else reads from it.
3. *Runtime → Run all*. Each section explains what it does, why, and what to check before moving on.

Every section writes its output to a versioned folder on your Google Drive (`scorecard_studio/run_YYYYMMDD_HHMMSS/`), so a runtime reset never loses work.

| # | Section | Status |
|---|---|---|
| 0 | Setup and config | ✅ |
| 1 | Data intake | ✅ |
| 2 | Sample design (Train / Test / OOT) | ✅ |
| 3 | Univariate screening (Train only) | ✅ |
| 4 | Interactive optimal binning (web app) and the model-ready dataset | ✅ |
| 5 | Logistic regression (raw / bins / WOE per variable) | ✅ |
| 6 | Scorecard (Siddiqi scaling), performance | ✅ |
| 7 | Rescoring new data | ✅ |
| 8 | Scorecard alignment app: cutoff strategy, points overrides, statistics, final table, Python/SQL scoring code | ✅ |
| – | Bin stability report, full validation report, deployment exports | coming |

**Default data:** the Kaggle [Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) (32,581 loans, target `loan_status`). It is *simulated* bureau-style data with a ~22% default rate, far above a real prime portfolio. It's good for learning the workflow, not for drawing conclusions about real lending.
""")

md("""
## 0. Setup

Installs the `scorecard_studio` engine from GitHub and mounts Google Drive for the outputs. Colab starts from a fresh machine every session, so this cell always runs first. It takes about a minute (optbinning brings in Google OR-Tools, the solver behind optimal binning).
""")
code(f"""
STUDIO_REF = "{REF}"   # git branch or tag of the engine to install

import sys
if "google.colab" in sys.modules:
    !pip -q install "git+https://github.com/Luizkauffmann/colab-scorecard-studio@{{STUDIO_REF}}"
""")
code("""
import os, warnings
os.environ.setdefault("GLOG_minloglevel", "2")   # quiet OR-Tools solver logs
warnings.filterwarnings("ignore", module="cvxpy")

import pandas as pd
from IPython.display import HTML, display

import scorecard_studio as ss
from scorecard_studio.align import launch_alignment
from scorecard_studio.app import launch_app
from scorecard_studio.store import RunStore, in_colab, mount_drive

pd.set_option("display.max_columns", 50)
pd.set_option("display.max_colwidth", 90)
print("scorecard_studio", ss.__version__, "| Colab:", in_colab())
""")

md("""
### Config cell: the only cell you need to edit

| Setting | What it does |
|---|---|
| `DATA_SOURCE` | `"demo:credit_risk"` (default), `"demo:synthetic"` (has dates and an OOT period), `"upload"` (file picker), `"drive:folder/file.csv"` (path under *My Drive*), an `https://` URL, `"kaggle:owner/dataset"` (needs `KAGGLE_USERNAME` / `KAGGLE_KEY` in Colab *Secrets*), or a local path. CSV, Parquet, Excel and JSON are read by extension. |
| `TARGET`, `EVENT_VALUE` | The outcome column. It must be 0/1 with **1 = event**. Otherwise set `EVENT_VALUE` to the value that means event (e.g. `"Charged Off"`). Rows with a missing target are dropped and counted. |
| `ALT_TARGETS` | Other candidate targets (e.g. a 24-month default flag). Each gets its own screening, bins and output; the binning app has a target selector. Every target column is excluded as a predictor for the others. |
| `ID_COL`, `DATE_COL` | Optional. With a date, the most recent period becomes an out-of-time (OOT) sample. |
| `PRESET` | `credit_pd`, `fraud`, `aml` or `kaggle`: binning constraints and screening defaults per domain. |
| `EXCLUDE` | Columns that must not enter the model. **Type the names.** They're still profiled, so you can see what you gave up. |
| `FORCE_INCLUDE` | Columns kept whatever their IV (policy variables, or high-IV variables you have reviewed). |
| `IV_MIN`, `IV_MAX` | Screening thresholds on Information Value. `None` uses the preset (credit PD: 0.10 / 0.50). |
| `SPECIAL_CODES` | Values with a meaning of their own, e.g. `{"months_since_delinq": [-999]}`. They get a separate Special bin. |
| `PLAUSIBILITY` | `{column: (min, max)}`. Values outside are data errors, routed per `IMPLAUSIBLE_ACTION`: `"special"` (to the Special bin), `"missing"` or `"flag"` (report only). The scorecard stores these rules and applies them when rescoring new data. |
| `REPRESENTATION_DEFAULT`, `REPRESENTATION` | How each variable enters the logistic regression: `"woe"`, `"bins"` (dummies) or `"raw"`. |
| `SELECTION`, `P_MAX`, `KEEP_IN_MODEL` | `"none"` or `"backward"` elimination (wrong-sign WOE first, then p-value > `P_MAX`); `KEEP_IN_MODEL` is never removed. |
| `PDO`, `BASE_SCORE`, `BASE_ODDS`, `BASE_POINTS` | Scorecard scaling: `BASE_SCORE` points at `BASE_ODDS`:1 good:bad odds, `PDO` points to double the odds. `BASE_POINTS`: `"spread"` over variables or `"separate"` row. |
| `GOOD_CLASS` | Which target value is *good* (0 by default: target 1 is the bad / event). A higher score always means a higher probability of good. |
| `COST_OF_BAD`, `BENEFIT_OF_GOOD` | Economics for the cutoff (section 8): average credit loss on a loan that goes bad, average margin on a loan that pays. |
| `ALIGN_DATASETS` | Other tables with the same raw columns to analyse in the alignment app (same source options as `DATA_SOURCE`). The app's *Dataset* and *Target* selectors switch between them and any 0/1 column. |

**Switching to your own data:** change `DATA_SOURCE` and `TARGET`, and empty `EXCLUDE`, `FORCE_INCLUDE` and `PLAUSIBILITY` (they name the demo's columns). A name that isn't in your data stops the run with a *did you mean* suggestion.
""")
code("""
# ── Data ────────────────────────────────────────────────────────────────
DATA_SOURCE = "demo:credit_risk"
TARGET      = "loan_status"        # 1 = event (default)
ALT_TARGETS = []                   # other candidate targets, e.g. ["default_24m"]
EVENT_VALUE = None
ID_COL      = None                 # None: a row_id is generated
DATE_COL    = None                 # None: stratified random split, no OOT
PRESET      = "credit_pd"

# ── Variables ───────────────────────────────────────────────────────────
EXCLUDE = [
    "loan_grade",     # assigned by the lender from an earlier risk assessment: circular in an application PD model
    "loan_int_rate",  # priced from the grade: same problem
]
FORCE_INCLUDE = [
    # Reviewed: both exceed IV_MAX, both are known at application time, neither is a post-outcome field.
    "loan_percent_income",  # requested amount / declared income: affordability
    "person_income",        # declared income
]

# ── Screening ───────────────────────────────────────────────────────────
IV_MIN = 0.10           # keep variables with IV >= IV_MIN (None = preset default)
IV_MAX = 0.50           # IV above this is held for review, not kept automatically
CORR_THRESHOLD = 0.70   # flag numerical pairs with |Spearman| >= this

# ── Data quality ────────────────────────────────────────────────────────
SPECIAL_CODES = {}
PLAUSIBILITY = {"person_age": (18, 100), "person_emp_length": (0, 60)}
IMPLAUSIBLE_ACTION = "special"

# ── Sample design ───────────────────────────────────────────────────────
TEST_SIZE = 0.30
OOT_START = None        # e.g. "2024-07-01"; None = last OOT_SHARE of rows by date
OOT_SHARE = 0.20
SEED = 42

# ── Model (section 5) ───────────────────────────────────────────────────
REPRESENTATION_DEFAULT = "woe"     # "woe" | "bins" | "raw" for every variable...
REPRESENTATION = {}                # ...except these, e.g. {"person_home_ownership": "bins"}
SELECTION = "backward"             # "none" | "backward"
P_MAX = 0.05
KEEP_IN_MODEL = []                 # never removed by backward selection

# ── Scorecard (section 6) ───────────────────────────────────────────────
PDO = 20                # points to double the good:bad odds
BASE_SCORE = 600        # score at BASE_ODDS
BASE_ODDS = 50          # good:bad odds at BASE_SCORE (50:1)
BASE_POINTS = "spread"  # "spread" over the variables (Siddiqi) or "separate"
GOOD_CLASS = 0          # target value that means good (0: target 1 = bad / event)

# ── Alignment (section 8) ───────────────────────────────────────────────
COST_OF_BAD = 1000      # average credit loss on a loan that goes bad
BENEFIT_OF_GOOD = 100   # average margin on a loan that pays
ALIGN_DATASETS = {}     # extra tables to analyse in the app, e.g. {"2025 vintage": "drive:data/apps_2025.csv"}

# ── Outputs ─────────────────────────────────────────────────────────────
OUTPUT_DIR = "/content/drive/MyDrive/scorecard_studio"
""")
code("""
cfg = ss.StudioConfig(
    data_source=DATA_SOURCE, target=TARGET, alt_targets=ALT_TARGETS, event_value=EVENT_VALUE, id_col=ID_COL, date_col=DATE_COL,
    preset=PRESET, exclude=EXCLUDE, force_include=FORCE_INCLUDE, iv_min=IV_MIN, iv_max=IV_MAX,
    special_codes=SPECIAL_CODES, plausibility=PLAUSIBILITY, implausible_action=IMPLAUSIBLE_ACTION,
    test_size=TEST_SIZE, oot_start=OOT_START, oot_share=OOT_SHARE, corr_threshold=CORR_THRESHOLD,
    seed=SEED, output_dir=OUTPUT_DIR,
).validate()                      # settings only; column names are checked once the data is loaded

if OUTPUT_DIR.startswith("/content/drive"):
    if not mount_drive():         # running outside Colab: write next to the notebook instead
        cfg.output_dir = "scorecard_studio_runs"
store = RunStore(cfg.output_dir)
store.write_json("config", "config.json", cfg.to_dict())
print("Outputs for this run:", store.root)
""")

md("""
## 1. Data intake

**What it does:** loads the table, checks every column name in the config against it, makes the target a clean 0/1, applies the plausibility rules, and writes a data dictionary.

**Why it matters:** a scorecard is only as good as its modeling table. Two things deserve attention here:

* **Unknown outcomes.** Rows without a target (still inside the performance window, indeterminate) can't be events or non-events. They are dropped and counted, never guessed.
* **Impossible values.** The demo data has applicants aged 123 and 144 and 123 years of employment. These are recording errors, not risky customers. Capping them would hide the problem, and dropping the rows would bias the sample. Here they're routed to an explicit code that lands in the **Special** bin, so you can see them. The same rule must run on new applications before scoring; the export stage will include it.

**Check before moving on:** the event rate is what you expect for your default definition, missing rates look sensible, and every column has the right role.
""")
code("""
intake = ss.run_intake(cfg)
df, ID = intake.df, intake.id_col

display(pd.Series(intake.report, name="value").to_frame())
print("\\nPlausibility rules:")
display(intake.plausibility)
print("Special codes used for binning:", intake.special_codes)
""")
code("""
display(intake.dictionary)

store.write_frame("intake", "raw.parquet", df)
store.write_frame("intake", "data_dictionary.csv", intake.dictionary.astype({"examples": str}))
store.write_json("intake", "intake_report.json", {**intake.report, "special_codes": intake.special_codes,
                                                 "plausibility": intake.plausibility.to_dict("records")})
""")
md("""
**About duplicates:** the report counts rows that are identical on every column. They aren't removed. In the demo data, 12 coarse columns and no applicant ID mean two different people can look identical. With your own data, an ID column makes real duplicates visible: investigate them (a loan booked twice, or the same applicant in two samples) before modeling.
""")

md("""
## 2. Sample design

**What it does:** splits the rows into **Train**, **Test** and, when there is a date, **OOT** (out-of-time: the most recent period).

**Why it matters:** everything that *learns* from the target (screening, bins, WOE, the model) uses **Train only**. Test and OOT are only for measuring. Choosing variables with IV computed on the full table looks harmless, but it quietly uses the validation data and makes validation optimistic.

* **With a date:** OOT is the latest period. A model has to work on applications that arrive after it was built, so OOT is the closest you get to production.
* **Without a date** (as in the demo data): a stratified random split, with the same event rate in Train and Test. Test then measures generalisation to *similar* applicants, not stability over time. Population drift won't show up.

**Check before moving on:** every sample has enough events (a few hundred is comfortable), and the event rates are close. A very different OOT rate usually points to the performance window or the default definition.
""")
code("""
sample = ss.make_split(df, cfg.target, date_col=cfg.date_col, test_size=cfg.test_size,
                       oot_start=cfg.oot_start, oot_share=cfg.oot_share, seed=cfg.seed)
summary = ss.split_summary(df, cfg.target, sample, cfg.date_col)
display(summary)
for w in ss.split_warnings(summary):
    print("⚠", w)

store.write_frame("split", "split.parquet", pd.DataFrame({ID: df[ID], "sample": sample}))
store.write_frame("split", "split_summary.csv", summary)
train = df[sample == "train"]
""")

md("""
## 3. Univariate screening (Train only)

**What it does:** for each candidate variable, fits an automatic optimal binning **on Train** with the preset's constraints (the same engine and settings the binning app uses next) and reports missing rate, cardinality, IV, Gini, KS and the event rate per bin.

**Why IV from optimal bins:** IV depends on the binning. Rough deciles understate U-shaped variables, and one bin per category overstates high-cardinality ones (an ID-like column can look excellent). Measuring it the way the app will bin keeps this number close to what you'll see there.

**How variables are sorted:**

| Status | Meaning | Goes to the binning app? |
|---|---|---|
| `selected` | `IV_MIN` ≤ IV ≤ `IV_MAX` | yes |
| `forced` | in `FORCE_INCLUDE` | yes |
| `review` | IV > `IV_MAX` | **no, until you decide** |
| `low_iv` | IV < `IV_MIN` | no |
| `excluded` | in `EXCLUDE` (still profiled) | no |
| `id_like`, `constant`, `fit_error` | not usable as is | no |

**A high IV is a question, not a prize.** An IV above 0.5 often means the variable encodes the outcome: a collections flag, a field filled after default, a lender decision. Before keeping one, ask *"was this known, in this form, at the moment of the decision?"* If yes, add it to `FORCE_INCLUDE`. If not, add it to `EXCLUDE`. In the demo, `loan_grade` and `loan_int_rate` (excluded) show what a circular variable looks like. `loan_percent_income` and `person_income` were reviewed and kept.

**Thresholds are not exact.** IV is an estimate from one sample, so a variable at 0.098 is not really weaker than one at 0.102. The report marks anything within 10% of `IV_MIN` or `IV_MAX` as *borderline*. In the demo, `loan_intent` sits at about 0.10 and lands on either side depending on the random split. Decide those on business grounds: keep it with `FORCE_INCLUDE` or leave it out, but don't tune `IV_MIN` until it falls your way.

**Check before moving on:** nothing is left in `review`, borderline variables have a deliberate decision, the event-rate plots make business sense (more debt relative to income → more defaults), and the correlated pairs below don't surprise you.
""")
code("""
# One screening per target (Train rows with a known outcome; other targets are never predictors).
screens = ss.screen_all_targets(
    train, cfg.targets, iv_min=cfg.resolved_iv_min, iv_max=cfg.resolved_iv_max,
    fit_params=cfg.fit_params, exclude=cfg.exclude, force_include=cfg.force_include,
    special_codes=intake.special_codes, id_col=ID, date_col=cfg.date_col,
    corr_threshold=cfg.corr_threshold)
screen = screens[cfg.target]
for t in cfg.alt_targets:
    print(f"[{t}] shortlist:", screens[t].selected)

print("Status:", screen.status_counts().to_dict())
borderline = screen.table.loc[screen.table["reason"].str.contains("borderline"), "variable"].tolist()
if borderline:
    print("Borderline (IV within 10% of a threshold):", borderline)
if screen.needs_review:
    print(f"⚠ Held for review (IV > {cfg.resolved_iv_max}): {screen.needs_review}. "
          "Add each to FORCE_INCLUDE or EXCLUDE in the config cell and re-run.")

colors = {"selected": "#d9f0d9", "forced": "#d6e6f5", "review": "#fde2b8"}
display(screen.table.style
        .apply(lambda r: [f"background-color: {colors.get(r['status'], '')}"] * len(r), axis=1)
        .format({"missing_rate": "{:.1%}", "top_share": "{:.1%}", "iv": "{:.3f}",
                 "gini": "{:.3f}", "ks": "{:.3f}", "bins": "{:.0f}"}, na_rep=""))
""")
md("""
**Event rate by bin** for the variables going forward (and any held for review). Bars are the share of Train rows in each bin, the red line is the event rate, and grey bars are the Missing and Special bins. Look for trends that make business sense and for Special/Missing bins that behave differently from their neighbours. Those are the bins the app lets you treat explicitly.
""")
code("""
fig = screen.plot_grid()
""")
md("""
**Correlated predictors.** Pairs of numerical variables with |Spearman| ≥ `CORR_THRESHOLD`. Nothing is dropped here. Two correlated variables can both earn a place, and redundancy is better judged later, on WOE and with the model in view (correlation and VIF at variable selection).
""")
code("""
display(screen.correlated_pairs if len(screen.correlated_pairs) else "No pairs above the threshold.")
""")
code("""
paths = screen.save(store.folder("screening"),
                    extra_html=summary.to_html(index=False, border=0))
SHORTLIST = screen.selected
print("Shortlist for the binning app:", SHORTLIST)
for k, p in paths.items():
    print(f"{k:>10}: {p}")
""")
md("""
### Before moving on

* `shortlist.json` lists the variables passed to the binning app and the reason every other variable was dropped. Keep it with your model documentation.
* `screening_report.html` (open it from Drive) is the same table with every event-rate plot, ready to share.
* Changed your mind about a variable? Edit `EXCLUDE` / `FORCE_INCLUDE` / `IV_MIN` in the config cell and re-run from section 0. Each run gets its own folder, so earlier runs stay intact.

""")

md("""
## 4. Interactive binning

**What it does:** opens the binning app on the **Train** sample. Each shortlisted variable starts from the automatic optimal binning you saw at screening. You then shape the bins by hand:

* **Numerical:** drag a red cutoff line, double-click the histogram to add one, click a line's label to type an exact value (or remove it), tick adjacent bins and **Merge**, or **⊕ split** a bin at its median.
* **Categorical:** in *Group categories*, select chips, then **← Move here** on the target group and **Apply grouping**. Each chip shows that category's event rate and count.
* **Re-run optimal binning** with other settings (max bins, monotonic trend, minimum bin size or events), or **Reset to auto**.
* The checkbox next to each variable decides whether it goes into the output dataset. *Show other candidates* lets you add back a `low_iv` variable. `review` and `excluded` stay decisions for the config cell.
* With `ALT_TARGETS`, the **Target** selector at the top left switches between targets. Each one has its own bins, WOE and output file.

**Missing and Special are a modeling choice.** By default each is its own bin. Use the dropdown on their rows (or select them together with a bin and **Merge**) to merge Missing into a regular bin, Special into a bin, or Special with Missing. The merged bin's WOE is computed on all its rows, and the exported scoring code follows the same choice. A separate Missing bin is the classic choice when "no value" carries information: with the *bins* representation it becomes its own dummy, i.e. a missing flag. Merge it when it is small or behaves like one of the bins.

**Why by hand:** optimal binning maximises IV under constraints. It doesn't know that a cutoff at 25,000 is a policy line, that a WOE reversal between two neighbouring bins is noise, or that two categories belong together for business reasons. Bins that a person can explain are bins a validator will accept and that stay stable in production.

**Check before moving on, for every variable:**
* No bin is flagged red (empty, or zero events / non-events). Merge it.
* Bins hold at least the minimum share (5% for credit PD), unless you have a reason.
* WOE moves in one direction for ordinal variables (the *Monotonic* box), or the shape has a business explanation.
* Missing and Special behave plausibly. A Special bin built from three rows has a WOE you shouldn't trust.

Every change is saved to `04_binning/<target>/binning_config.json` on Drive. After a runtime reset, re-run the notebook: this cell reloads your bins, as long as the Train sample is the same (same data, `SEED` and split settings).

If the app doesn't appear (some corporate browsers block the embedded frame), run `app.open_in_tab()` in a new cell.
""")
code("""
app = launch_app(
    train, cfg.target, alt_targets=cfg.alt_targets, screens=screens, store=store,
    full=df, sample=sample, id_col=ID, date_col=cfg.date_col,
    special_codes=intake.special_codes, fit_params=cfg.fit_params)
""")

md("""
### The model-ready dataset

When the bins look right, click **Create output dataset** in the app, or run the cell below. It writes one table with **every sample** (Train, Test and OOT if any) and, for each selected variable, three columns:

| Column | Content |
|---|---|
| `<var>` | the original value |
| `opt_<var>` | its bin, as an ordered category: regular bins, then Special, then Missing |
| `woe_<var>` | the bin's weight of evidence (computed on Train) |

The bins are fit on Train only, and Test and OOT receive the same bins and WOE values, through the same scoring code that the exported `scorer.py` and SQL use. At the modeling step you choose, per variable, whether the logistic regression uses the original value, the bins (as dummies) or the WOE.

Also written to `04_binning/<target>/`: `bundle.json` (the scoring rules) and `binning_tables.csv` (every bin of every variable, for the model documentation). With `ALT_TARGETS`, `app.save_output(target="...")` writes the dataset for another target.
""")
code("""
output = app.save_output()
model_df = pd.read_parquet(output["paths"]["dataset"])
print(f"{output['rows']:,} rows × {output['columns']} columns | samples: {output['samples']}")
print("Variables:", output["variables"])
display(model_df.head())
""")
code("""
display(app.session.binning_tables())
""")

md("""
## 5. Logistic regression

**What it does:** fits a logistic regression of the target on the selected variables, on the **Train** rows of the model-ready dataset. Test (and OOT) are only scored.

**How each variable enters the model** (`REPRESENTATION_DEFAULT`, overridden per variable in `REPRESENTATION`):

| Representation | Columns | When to use it |
|---|---|---|
| `"woe"` | `woe_<var>`: one coefficient | The classic scorecard. One parameter per variable, monotonic by construction, robust with few events. The coefficient should be **positive** (positive WOE = riskier). |
| `"bins"` | one dummy per bin of `opt_<var>` (the most populous bin is the reference) | When the bins' effect isn't proportional to their WOE once other variables are in. A separate Missing bin becomes a missing flag. Costs one parameter per bin. |
| `"raw"` | the original value | Linear effects you trust. Not allowed when the variable has missing values or special codes (a linear term would treat −99999 as a number). It gives a points *formula*, not a points table. |

**Selection:** with `SELECTION = "backward"`, variables are removed one at a time, first any WOE variable with a non-positive coefficient (its effect is reversed by the others), then the least significant above `P_MAX` (joint Wald test for bins). `KEEP_IN_MODEL` protects policy variables.

**Check before moving on:** no flags you can't explain (sign, p-value, VIF > 5, correlated pairs), Gini on Test close to Train (a large drop means overfitting), and every variable has a business story.
""")
code("""
bundle = app.engine.build_scoring_bundle(output["variables"])
representations = {v: REPRESENTATION.get(v, REPRESENTATION_DEFAULT) for v in output["variables"]}
model = ss.fit_logistic(model_df, cfg.target, representations, bundle,
                        selection=SELECTION, p_max=P_MAX, keep=KEEP_IN_MODEL)

for step in model.selection_log:
    print("•", step)
print("Flags:" if model.flags else "No flags.")
for f in model.flags:
    print("⚠", f)
display(model.summary.style.format({"Coefficient": "{:.4g}", "p-value": "{:.3g}", "Max VIF": "{:.2f}", "IV": "{:.3f}"}, na_rep="—"))
display(model.performance.style.format({"event_rate": "{:.2%}", "AUC": "{:.3f}", "Gini": "{:.3f}", "KS": "{:.3f}"}))
""")
code("""
display(model.coefficients.style.format({"Coefficient": "{:.4g}", "Std. error": "{:.3g}", "z": "{:.2f}",
                                          "p-value": "{:.3g}", "VIF": "{:.2f}"}, na_rep="—"))
store.write_frame("model", "coefficients.csv", model.coefficients)
store.write_frame("model", "summary.csv", model.summary)
store.write_frame("model", "performance.csv", model.performance)
store.write_json("model", "model.json", {"representations": model.representations, "intercept": model.intercept,
                                         "coef": model.coef, "bin_coefs": model.bin_coefs,
                                         "reference": model.reference, "flags": model.flags,
                                         "selection_log": model.selection_log})
""")

md("""
## 6. Scorecard

**What it does:** turns the model into points with Siddiqi's scaling (*Credit Risk Scorecards*, the same formula as optbinning's `pdo_odds` method):

* Factor = PDO / ln(2), Offset = BASE_SCORE − Factor · ln(BASE_ODDS)
* Score = Offset − Factor · logit(PD): higher score = lower risk, and every `PDO` points double the good:bad odds.
* Points per bin = −(β · WOE + α / n) · Factor + Offset / n for WOE variables (β of the bin's dummy for bins variables). The intercept α and the offset are spread evenly over the n variables, or put into one base-points row with `BASE_POINTS = "separate"`.

Points are rounded to integers. Before rounding, a record's points add up exactly to the model's score. That property is tested, so the scorecard *is* the model.

**Manual judgment:** `card.set_points(variable, group, points)` overrides a bin's points (the table keeps the model's value in *Model points* and flags *Manual*). Rescoring uses the edited points. A separate points-adjustment app will build on this.

**Check before moving on:** points move in the expected direction across each variable's bins, no single variable dominates the score range, and the event rate falls steadily across score bands on Test as well as Train.
""")
code("""
scaling = ss.ScalingParams(pdo=PDO, base_score=BASE_SCORE, base_odds=BASE_ODDS)
card = ss.build_scorecard(model, bundle, scaling, base_points=BASE_POINTS,
                          plausibility=cfg.plausibility_spec(), good_class=GOOD_CLASS)
print(f"Factor {scaling.factor:.4f} · Offset {scaling.offset:.4f}")
display(card.table())
""")
code("""
scored = card.score(df, keep=[ID, cfg.target])
scored["sample"] = sample.values
display(ss.performance(scored[cfg.target], scored["score"], scored["sample"])
        .style.format({"event_rate": "{:.2%}", "AUC": "{:.3f}", "Gini": "{:.3f}", "KS": "{:.3f}"}))
display(ss.score_bands(scored[cfg.target], scored["score"], scored["sample"]))

card_path = card.save(store.path("scorecard", "scorecard.json"))
store.write_frame("scorecard", "scorecard_table.csv", card.table())
store.write_frame("scorecard", "scored.parquet", scored)
print("Scorecard saved to", card_path)
""")

md("""
## 7. Rescoring new data

`scorecard.json` holds the whole scoring logic: the plausibility rules, the bins of every variable (including your Missing/Special choices), the representation and the points. Any table with the same raw columns can be rescored from it, in this notebook or anywhere `scorecard_studio` is installed:

```python
card = ss.load_scorecard("…/06_scorecard/scorecard.json")
scores = card.score(new_df, keep=["application_id"])   # pts_<var>, score, pd
```

For the bins and WOE only (e.g. to feed another model), use `ss.apply_bins(new_df, bundle)` with the `bundle.json` from section 4. It gives the same `opt_<var>` / `woe_<var>` columns as the model-ready dataset.

Set `NEW_DATA` below to a file (same options as `DATA_SOURCE`, e.g. `"upload"` or `"drive:folder/new_apps.csv"`). Left as `None`, the cell rescores the first rows of the demo data, so you can see the output.
""")
code("""
NEW_DATA = None   # e.g. "upload" or "drive:folder/new_applications.csv"

if NEW_DATA:
    new_df, source = ss.load_table(NEW_DATA)
else:
    new_df, source = ss.load_table(cfg.data_source)
    new_df = new_df.head(1000)
reloaded = ss.load_scorecard(card_path)
rescored = reloaded.score(new_df)
out_path = store.write_frame("rescore", "rescored.csv", pd.concat([new_df, rescored], axis=1))
print(f"Rescored {len(rescored):,} rows from {source} -> {out_path}")
display(pd.concat([new_df, rescored], axis=1).head())
""")

md("""
## 8. Scorecard alignment

**What it does:** opens the alignment app on the scorecard from section 6 and every scored row. The settings on the left start from the config cell and can be changed in the app: dataset (the model dataset, or any table in `ALIGN_DATASETS`), target (any 0/1 column, e.g. another default definition), good class, sample used to set the cutoff, PDO / base score / base odds, base points, cost of a bad, benefit of a good.

Switching dataset or target never changes the scorecard: the same bins and points are applied, and only the outcomes you measure them against change. That's how you check a cutoff on a newer vintage, or how the score ranks a 24-month default flag it wasn't built on. Rows with an unknown outcome (e.g. rejects in a through-the-door file) count in the approval rate but not in bad rates or profit.

| Tab | What you do there |
|---|---|
| **Strategy** | Pick the cutoff (approve if score ≥ cutoff). The suggestion maximises profit = benefit × goods approved − cost × bads approved. You can instead target an **approval rate**, a **maximum bad rate** among approved, or set it **manually**. KPIs, profit curve, approval/bad-rate curves, score distribution, ROC and the decision matrix update at once. |
| **Scorecard** | Every variable and bin with WOE, coefficient, **model points** and the **points in use**. Type new points to override a bin (manual judgment) and give the reason. Overrides are kept through rescaling but flagged when set under another scale; an *order* flag warns when a riskier bin gets more points than a safer one. |
| **Statistics** | AUC, Gini (= 2·AUC − 1) and KS per sample; the gains table in PDO-wide bands; observed vs expected bad rate per band, and the **realized PDO and odds**; each variable's share of the score range; score PSI. |
| **Export** | **Create final table**: every row with `pts_<var>` per variable, `score`, `pd` and `decision`. **Scoring code** in Python (standalone script) or SQL (standard, Spark, BigQuery) with everything after your overrides. |

**Why these choices:**
* The cutoff is set on **Test** (or OOT when you have one), not Train: an in-sample cutoff looks better than it will be.
* **Approval rates count every row; bad rates only rows with a known outcome.** In onboarding the rows without an outcome are mostly past rejects.
* The profit-optimal cutoff should sit near the **break-even score** the scale implies (odds = cost ÷ benefit). A large gap means the scale's odds don't hold on this data, i.e. the scorecard needs recalibration before its PD can be used for pricing or provisioning.
* Every number in the app (KPIs, final table, code) comes from the same points table, so the points always add up to the score.

**Check before moving on:** the cutoff's approval rate and bad rate are acceptable for the business; every override has a reason and no unexplained *order* flag; the realized PDO and odds are close to the design on Test; the score PSI from Train is small.

Every change is saved to `08_alignment/alignment_state.json` and reloaded after a runtime reset.
""")
code("""
extra = {name: ss.load_table(src)[0] for name, src in ALIGN_DATASETS.items()}
align = launch_alignment(card, df, cfg.target, store=store, sample=sample, id_col=ID,
                         cost_bad=COST_OF_BAD, benefit_good=BENEFIT_OF_GOOD,
                         dataset_name="model dataset", datasets=extra)
""")
md("""
### Final table and scoring code

The **Create final table** button in the app, or the cell below, writes to `08_alignment/`:

* `final_scored.parquet` / `.csv`: every row of every sample with the original variables, `pts_<var>` for each variable, `score`, `pd` (probability of bad) and `decision`;
* `final_scorecard.json` and `final_scorecard_table.csv`: the final scorecard with overrides, reasons and the cutoff;
* `scorer.py` and `scorer_standard.sql`: the scoring code.

**Using the scoring code:** `python scorer.py new_applications.csv scored.csv` (no installs needed), or `from scorer import score_record`. The SQL reads a table with the same columns (`input_table` by default; pick the name and dialect in the app) and returns its columns plus the points, score, pd and decision. Both apply the data-quality rules, the bins (with your Missing/Special choices) and the points in use, and are tested to give exactly the scores of the final table.
""")
code("""
final_info = align.finalize()
final = align.final_table()
print(f"{final_info['rows']:,} rows · cutoff {final_info['cutoff']:g} · approval {final_info['approval_rate']:.1%}")
display(final.head())
for lang in ("python", "sql"):
    print("Saved", align.session.save_code(lang))
""")
code("""
print(align.code("python")[:1200], "...")
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "colab": {"provenance": [], "toc_visible": True},
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python"}})
nbf.write(nb, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scorecard_studio.ipynb"))
print(len(cells), "cells")
