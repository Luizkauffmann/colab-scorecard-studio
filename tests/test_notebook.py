"""The template notebook runs top to bottom (sections 0-3) on Credit Risk-shaped data."""

import json
import os

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")
pytest.importorskip("ipykernel")

from conftest import make_credit_risk_like  # noqa: E402

NOTEBOOK = os.path.join(os.path.dirname(__file__), "..", "notebooks", "scorecard_studio.ipynb")


def test_notebook_runs_end_to_end(tmp_path):
    csv = tmp_path / "credit_risk_dataset.csv"
    make_credit_risk_like(n=4000).to_csv(csv, index=False)
    nb = nbformat.read(NOTEBOOK, as_version=4)
    replaced = 0
    for cell in nb.cells:
        if cell.cell_type == "code" and 'DATA_SOURCE = "demo:credit_risk"' in cell.source:
            cell.source = (cell.source
                           .replace('"demo:credit_risk"', repr(str(csv)))
                           .replace('"/content/drive/MyDrive/scorecard_studio"', repr(str(tmp_path / "runs"))))
            replaced += 1
    assert replaced == 1, "config cell not found"

    nbclient.NotebookClient(nb, timeout=600, kernel_name="python3",
                            resources={"metadata": {"path": str(tmp_path)}}).execute()

    errors = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", [])
              if o.output_type == "error"]
    assert not errors, errors
    runs = os.listdir(tmp_path / "runs")
    assert len(runs) == 1
    root = tmp_path / "runs" / runs[0]
    for f in ["00_config/config.json", "01_intake/raw.parquet", "01_intake/data_dictionary.csv",
              "02_split/split.parquet", "03_screening/shortlist.json",
              "03_screening/screening_report.html", "04_binning/loan_status/binning_config.json",
              "04_binning/loan_status/model_dataset.parquet", "04_binning/loan_status/bundle.json",
              "05_model/coefficients.csv", "06_scorecard/scorecard.json",
              "06_scorecard/scorecard_table.csv", "07_rescore/rescored.csv",
              "08_alignment/alignment_state.json", "08_alignment/final_scored.parquet",
              "08_alignment/final_scorecard.json", "08_alignment/scorer.py",
              "08_alignment/scorer_standard.sql"]:
        assert (root / f).exists(), f
    shortlist = json.loads((root / "03_screening/shortlist.json").read_text())
    assert "loan_grade" in shortlist["dropped"] and shortlist["selected"]
    assert shortlist["dropped"]["loan_grade"]["status"] == "excluded"
    import pandas as pd
    model = pd.read_parquet(root / "04_binning/loan_status/model_dataset.parquet")
    for v in shortlist["selected"]:
        assert {v, f"opt_{v}", f"woe_{v}"} <= set(model.columns)
    import scorecard_studio as ss
    card = ss.load_scorecard(str(root / "06_scorecard/scorecard.json"))
    scored = card.score(pd.read_csv(csv))
    assert scored["score"].notna().all() and (scored["score"] % 1 == 0).all()
