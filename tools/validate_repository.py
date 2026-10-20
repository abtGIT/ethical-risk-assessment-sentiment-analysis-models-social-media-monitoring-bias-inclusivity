"""Offline, read-only integrity checks for the public MSc thesis repository.

This is a structural guard, not a reproduction of the experimental results.
Run: python tools/validate_repository.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CORE = [
    "FINAL_00_Config_and_Freeze.ipynb",
    "FINAL_01_Data_Preparation_and_Subgroups.ipynb",
    "FINAL_02_Model_Development_and_Lock.ipynb",
    "FINAL_03_RQ1_Subgroup_Performance.ipynb",
    "FINAL_04_RQ2_Disparity_Audit.ipynb",
    "FINAL_05_RQ3_Errors_Confidence_XAI.ipynb",
    "FINAL_06_v2_RQ4_External_Robustness.ipynb",
    "FINAL_07_Ethical_Risk_Synthesis.ipynb",
]
CHARTS = [
    "Fig3_6_class_distribution", "Fig3_7_post_length",
    "Fig3_8_markers", "Fig5_1_baseline_validation",
    "Fig5_2_subgroup_macro_f1", "Fig5_3_cpd",
    "Fig5_4_error_directions", "Fig5_5_hcer_ece",
    "Fig5_6_cross_dataset",
]


def read_json(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def check_notebooks() -> int:
    paths = sorted(
        [* (ROOT / "EDA").glob("*.ipynb"),
         * (ROOT / "Implementation").glob("*.ipynb"),
         * (ROOT / "manifests").glob("*.ipynb"),
         * (ROOT / "archive").rglob("*.ipynb")]
    )
    assert paths, "No notebooks found"
    for path in paths:
        notebook = read_json(str(path.relative_to(ROOT)))
        assert notebook.get("nbformat") == 4, f"Not a v4 notebook: {path}"
        assert isinstance(notebook.get("cells"), list), f"No cells: {path}"
        for idx, cell in enumerate(notebook["cells"]):
            assert cell.get("cell_type") in {"code", "markdown", "raw"}, (path, idx)
            assert isinstance(cell.get("source"), (str, list)), (path, idx)
            meta = cell.get("metadata") or {}
            assert not (meta.get("executionInfo") or {}).get("user"), (
                f"Public Colab executor identity present: {path}, cell {idx}"
            )
    for name in CORE:
        assert (ROOT / "Implementation" / name).is_file(), name
    return len(paths)


def check_freeze() -> None:
    run = read_json("manifests/FINAL_RUN_MANIFEST.json")
    standalone = read_json("manifests/SUBGROUP_FREEZE_CONFIG.json")
    assert run["subgroup_freeze"] == standalone, "Frozen subgroup manifests disagree"
    assert run["seed"] == 42, "Unexpected seed"


def check_figures() -> int:
    with (ROOT / "figures/FIGURE_VALUES.csv").open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames == ["figure", "item", "value"]
        values = list(reader)
    assert len(values) == 63, f"Expected 63 logged values; found {len(values)}"
    expected_ids = {"3.6", "3.7", "3.8", *(f"5.{i}" for i in range(1, 7))}
    assert {row["figure"] for row in values} == expected_ids, "Unexpected chart inventory"
    for name in CHARTS:
        for ext in (".png", ".svg"):
            assert (ROOT / "figures" / (name + ext)).is_file(), name + ext
    source = ROOT / "figures/diagrams/source"
    for dot_file in source.glob("Fig*.dot"):
        name = dot_file.stem
        for ext in (".png", ".svg"):
            assert (ROOT / "figures/diagrams" / (name + ext)).is_file(), name + ext
    return len(values)


def check_figure_notebook_style() -> None:
    nb = read_json("Implementation/FIGURES_Regenerate.ipynb")
    code = "\n".join(
        "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        for c in nb["cells"] if c["cell_type"] == "code"
    )
    assert "legend overlaps plot" in code, "Figure overlap guard was removed"
    assert "(a) Train vs test" in code, "Figure 3.7 panel title changed"
    assert "(b) By sentiment class, all splits pooled" in code, "Figure 3.7 panel title changed"
    assert "Fig3_8_markers.png" in code, "Figure 3.8 was removed"


def main() -> None:
    notebook_count = check_notebooks()
    check_freeze()
    row_count = check_figures()
    check_figure_notebook_style()
    print(
        f"Repository checks passed: {notebook_count} active notebooks, "
        f"frozen manifest agreement, {row_count} figure values, "
        "all expected chart asset pairs."
    )


if __name__ == "__main__":
    main()
