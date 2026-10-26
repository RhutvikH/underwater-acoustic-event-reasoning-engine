"""Ten-species call-library evaluation.

The headline is recording-held-out macro-F1. Test scores do not choose the
model, the penalty, or which species stay in the table. A missed
demonstration bar is still a completed run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from uaere.data.species import WATKINS_CITATION, SpeciesDataError, load_species_split
from uaere.math.metrics import ece
from uaere.models.species_head import _matrix, save_species_checkpoint, select_species_head

CAVEAT = (
    "This is call-library identification on isolated, already-cut recordings, "
    "not detection in a continuous underwater soundscape. Cuts that share a tape "
    "or groups.csv id stay on one side of the split. A high score is not a "
    "raw-recording detector and is not the toadfish result."
)
DEMO_MACRO_F1 = 0.70
DEMO_MIN_TEST = 15


def _predict_names(clf, features: np.ndarray, mu: np.ndarray, sd: np.ndarray, labels: list[str]) -> list[str]:
    xn = (features - mu) / sd
    pred = np.asarray(clf.predict(xn))
    if np.issubdtype(pred.dtype, np.integer):
        return [labels[int(i)] for i in pred]
    return [str(p) for p in pred]


def _max_proba(clf, features: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    xn = (features - mu) / sd
    proba = np.asarray(clf.predict_proba(xn), dtype=np.float64)
    return proba.max(axis=1)


def run_species_suite(
    root: str | Path,
    out_dir: str | Path = "artifacts/paper",
    seed: int = 0,
    train_steps: int = 2000,
    max_per_class: int = 500,
    min_files: int = 40,
    min_groups: int = 8,
    min_classes: int = 10,
) -> dict:
    if not root:
        raise SpeciesDataError(
            "uaere eval --suite species requires --root. "
            "It does not fall back to the synthetic twin or the toadfish head."
        )
    print("species suite: loading the recording-locked split", flush=True)
    train, val, test, meta = load_species_split(
        root,
        seed=seed,
        max_per_class=max_per_class,
        min_files=min_files,
        min_groups=min_groups,
        min_classes=min_classes,
        require_recording_ids=True,
    )
    head, selection = select_species_head(train, val, steps=train_steps, seed=seed)
    models = list(selection.pop("_models"))
    labels = list(head.labels or [])
    public_selection = {k: v for k, v in selection.items() if k != "_models"}
    meta = dict(meta)
    meta["selection"] = public_selection
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_dir = out.parent / "models"
    save_species_checkpoint(head, model_dir / "species_head.joblib", meta=meta, selection=public_selection)
    save_species_checkpoint(head, out / "species_head.joblib", meta=meta, selection=public_selection)

    gold = [r.taxon or r.event_class.value for r in test]
    # Features once per representation. Test does not go back into model choice.
    by_feature: dict[str, np.ndarray] = {}
    comparisons = []
    for row in models:
        if row["feature"] not in by_feature:
            by_feature[row["feature"]] = _matrix(test, row["feature"])[0]
        names = _predict_names(row["clf"], by_feature[row["feature"]], row["mu"], row["sd"], labels)
        comparisons.append(
            {
                "kind": row["kind"],
                "feature": row["feature"],
                "alpha": row["alpha"],
                "test_macro_f1": float(
                    f1_score(gold, names, labels=labels, average="macro", zero_division=0)
                ),
                "chosen": row["kind"] == public_selection["winner"]
                and row["feature"] == public_selection["feature"]
                and row["alpha"] == public_selection.get("alpha"),
            }
        )
    feature = str(public_selection["feature"])
    pred = _predict_names(head.sklearn, by_feature[feature], head._feat_mu, head._feat_sd, labels)
    conf = _max_proba(head.sklearn, by_feature[feature], head._feat_mu, head._feat_sd)
    correct = np.array([int(a == b) for a, b in zip(pred, gold, strict=True)], dtype=np.float64)
    precision, recall, f1, support = precision_recall_fscore_support(
        gold, pred, labels=labels, zero_division=0
    )
    matrix = confusion_matrix(gold, pred, labels=labels)
    per_class = [
        {
            "label": labels[i],
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(labels))
    ]
    train_counts: dict[str, int] = {}
    for rec in train:
        name = rec.taxon or rec.event_class.value
        train_counts[name] = train_counts.get(name, 0) + 1
    majority = max(train_counts, key=lambda name: (train_counts[name], name))
    majority_pred = [majority] * len(gold)
    macro = float(f1_score(gold, pred, labels=labels, average="macro", zero_division=0))
    accuracy = float(np.mean(correct)) if len(correct) else 0.0
    majority_macro = float(f1_score(gold, majority_pred, labels=labels, average="macro", zero_division=0))
    majority_accuracy = float(np.mean([g == majority for g in gold])) if gold else 0.0
    bar_met = bool(macro >= DEMO_MACRO_F1 and per_class and all(row["support"] >= DEMO_MIN_TEST for row in per_class))
    report = {
        "caveat": CAVEAT,
        "citation": meta.get("citation") or WATKINS_CITATION,
        "task": "call_library_species_identification",
        "split_unit": meta.get("split_unit"),
        "split_note": meta.get("split_note"),
        "working_fs": meta.get("working_fs"),
        "cutoff_hz": meta.get("cutoff_hz"),
        "seed": int(seed),
        "seeds_not_run": [1, 2],
        "max_per_class": max_per_class,
        "cap_unit": "files",
        "floors": {
            "min_files": min_files,
            "min_groups": min_groups,
            "min_classes": min_classes,
        },
        "n_windows": meta.get("n_windows"),
        "n_groups": meta.get("n_groups"),
        "counts": meta.get("counts"),
        "labels": labels,
        "selection": public_selection,
        "model_comparison_test": comparisons,
        "note_on_comparison": (
            "Validation macro-F1 chose the checkpoint. The test column is a report. "
            "It did not change the winner, the alpha, or the species list."
        ),
        "headline": {
            "macro_f1": macro,
            "accuracy": accuracy,
            "majority_label": majority,
            "majority_accuracy": majority_accuracy,
            "majority_macro_f1": majority_macro,
            "ece_max_probability": float(ece(correct, conf)) if len(correct) else None,
            "demonstration_bar": {
                "macro_f1_at_least": DEMO_MACRO_F1,
                "min_test_files_per_species": DEMO_MIN_TEST,
                "met": bar_met,
            },
        },
        "per_class": per_class,
        "confusion": {"labels": labels, "matrix": matrix.tolist()},
    }
    (out / "species_suite.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "species_suite.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _markdown(report: dict) -> str:
    h = report["headline"]
    bar = h["demonstration_bar"]
    lines = [
        "# Species call-library result",
        "",
        report["caveat"],
        "",
        f"Citation: {report['citation']}",
        "",
        f"Split: `{report['split_unit']}`. {report['split_note']}",
        "",
        f"Working rate: {report['working_fs']} Hz. Cutoff: {report['cutoff_hz']} Hz. Seed {report['seed']}. Seeds {report['seeds_not_run']} were not run.",
        "",
        f"Chosen model: `{report['selection']['winner']}` on `{report['selection']['feature']}` features. Validation macro-F1 {report['selection']['val_macro_f1']:.6f}. Solver iterations {report['selection'].get('n_iter')} of {report['selection'].get('max_iter')} (converged: {report['selection'].get('converged')}).",
        "",
        "## Headline (test, scored once)",
        "",
        f"- Macro-F1: {h['macro_f1']:.6f}",
        f"- Accuracy: {h['accuracy']:.6f}",
        f"- Majority class `{h['majority_label']}` accuracy: {h['majority_accuracy']:.6f}",
        f"- Majority class macro-F1: {h['majority_macro_f1']:.6f}",
        f"- ECE of the maximum class probability: {h['ece_max_probability']:.6f}",
        f"- Demonstration bar (macro-F1 ≥ {bar['macro_f1_at_least']} and ≥ {bar['min_test_files_per_species']} test files per species): {'met' if bar['met'] else 'not met'}",
        "",
        "## Per class",
        "",
        "| Label | Precision | Recall | F1 | Support |",
        "|-------|-----------|--------|----|---------|",
    ]
    for row in report["per_class"]:
        lines.append(
            f"| {row['label']} | {row['precision']:.6f} | {row['recall']:.6f} | {row['f1']:.6f} | {row['support']} |"
        )
    lines.extend(
        [
            "",
            "## Validation choice",
            "",
            "| Kind | Features | Alpha | Val macro-F1 | Iterations | Converged |",
            "|------|----------|-------|--------------|------------|-----------|",
        ]
    )
    for row in report["selection"]["candidates"]:
        alpha = "—" if row["alpha"] is None else f"{row['alpha']:.0e}"
        lines.append(
            f"| {row['kind']} | {row['feature']} | {alpha} | {row['val_macro_f1']:.6f} | "
            f"{row.get('n_iter')}/{row.get('max_iter')} | {row.get('converged')} |"
        )
    lines.extend(["", "## Test scores of every candidate (not used to choose)", ""])
    for row in report["model_comparison_test"]:
        mark = "chosen" if row["chosen"] else "report only"
        lines.append(f"- `{row['kind']}` alpha={row['alpha']}: test macro-F1 {row['test_macro_f1']:.6f} ({mark})")
    lines.extend(["", "Confusion labels: " + ", ".join(report["confusion"]["labels"]), ""])
    for row in report["confusion"]["matrix"]:
        lines.append(" ".join(str(int(v)) for v in row))
    lines.append("")
    return "\n".join(lines)
