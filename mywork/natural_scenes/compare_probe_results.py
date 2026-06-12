#!/usr/bin/env python3
"""Merge cross-image probe JSON outputs from both notebooks and export a CSV table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

DEFAULT_RLHF_JSON = Path(__file__).resolve().parent / "cross_image_probe_results.json"
DEFAULT_OV_JSON = Path(__file__).resolve().parent / "llava_ov_probe_results.json"
DEFAULT_CSV = Path(__file__).resolve().parent / "probe_comparison.csv"

MODEL_COLUMNS = [
    ("sft", "SFT+"),
    ("rlhf", "RLHF"),
    ("llava_ov", "LLaVA-OV"),
]

PROBE_ORDER = [
    "T1", "T2", "C1", "C2", "C3", "E1", "E2", "E3",
    "K1", "K2", "K3", "O1", "O2", "O3", "A1", "A2",
]


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def iter_per_probe_rows(data: dict) -> list[dict]:
    """Return per_probe rows; tolerate legacy formats when schema v1.0 is missing."""
    if data.get("schema_version") == "1.0" and "per_probe" in data:
        return data["per_probe"]

    probes = data.get("probes", [])
    probe_by_id = {p["id"]: p for p in probes}
    rows: list[dict] = []

    if "results_sft" in data or "results_rlhf" in data:
        for pid in PROBE_ORDER:
            meta = probe_by_id.get(pid, {"id": pid})
            answers = {}
            if "results_sft" in data:
                answers["sft"] = data["results_sft"].get(pid, "")
            if "results_rlhf" in data:
                answers["rlhf"] = data["results_rlhf"].get(pid, "")
            rows.append({
                "id": pid,
                "image": meta.get("image", ""),
                "type": meta.get("type", ""),
                "question": meta.get("question", ""),
                "gt": meta.get("gt", ""),
                "note": meta.get("note", ""),
                "answers": answers,
                "correct": {k: None for k in answers},
                "verdict": {k: "" for k in answers},
            })
        return rows

    if "results_ov" in data:
        for pid in PROBE_ORDER:
            meta = probe_by_id.get(pid, {"id": pid})
            rows.append({
                "id": pid,
                "image": meta.get("image", ""),
                "type": meta.get("type", ""),
                "question": meta.get("question", ""),
                "gt": meta.get("gt", ""),
                "note": meta.get("note", ""),
                "answers": {"llava_ov": data["results_ov"].get(pid, "")},
                "correct": {"llava_ov": None},
                "verdict": {"llava_ov": ""},
            })
        return rows

    if "per_probe" in data:
        legacy = data["per_probe"]
        if legacy and "answers" not in legacy[0]:
            for row in legacy:
                rows.append({
                    "id": row["id"],
                    "image": row.get("image", ""),
                    "type": row.get("type", ""),
                    "question": row.get("question", ""),
                    "gt": row.get("gt", ""),
                    "note": row.get("note", ""),
                    "answers": {"llava_ov": row.get("answer", "")},
                    "correct": {"llava_ov": row.get("correct")},
                    "verdict": {"llava_ov": row.get("verdict", "")},
                })
            return rows

    raise ValueError(f"Unrecognized benchmark JSON format: {data.keys()}")


def merge_runs(rlhf_data: dict, ov_data: dict) -> list[dict]:
    merged: dict[str, dict] = {}

    for payload in (rlhf_data, ov_data):
        for row in iter_per_probe_rows(payload):
            pid = row["id"]
            entry = merged.setdefault(
                pid,
                {
                    "id": pid,
                    "image": row.get("image", ""),
                    "type": row.get("type", ""),
                    "question": row.get("question", ""),
                    "gt": row.get("gt", ""),
                    "note": row.get("note", ""),
                },
            )
            for model_key, answer in row.get("answers", {}).items():
                entry[f"answer_{model_key}"] = answer
                correct = row.get("correct", {}).get(model_key)
                if correct is not None:
                    entry[f"correct_{model_key}"] = correct
                verdict = row.get("verdict", {}).get(model_key)
                if verdict:
                    entry[f"verdict_{model_key}"] = verdict

    rows = sorted(merged.values(), key=lambda r: PROBE_ORDER.index(r["id"]) if r["id"] in PROBE_ORDER else 999)
    for row in rows:
        correct_flags = []
        for model_key, _ in MODEL_COLUMNS:
            flag = row.get(f"correct_{model_key}")
            if flag is not None:
                correct_flags.append((model_key, flag))
        if not correct_flags:
            row["best_models"] = ""
        else:
            winners = [key for key, ok in correct_flags if ok]
            row["best_models"] = ",".join(winners) if winners else "none"
        row["all_correct"] = bool(correct_flags) and all(ok for _, ok in correct_flags)
    return rows


def write_csv(rows: list[dict], output_path: Path) -> None:
    fieldnames = ["id", "image", "type", "question", "gt", "note"]
    for model_key, _ in MODEL_COLUMNS:
        fieldnames.extend([f"answer_{model_key}", f"correct_{model_key}", f"verdict_{model_key}"])
    fieldnames.extend(["best_models", "all_correct"])

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_summary(rows: list[dict]) -> None:
    totals = {key: {"correct": 0, "total": 0} for key, _ in MODEL_COLUMNS}
    for row in rows:
        for model_key, _ in MODEL_COLUMNS:
            col = f"correct_{model_key}"
            if col in row and row[col] is not None:
                totals[model_key]["total"] += 1
                totals[model_key]["correct"] += int(bool(row[col]))

    print("Model accuracy on shared probes:")
    for model_key, label in MODEL_COLUMNS:
        stats = totals[model_key]
        if stats["total"] == 0:
            continue
        acc = stats["correct"] / stats["total"]
        print(f"  {label:<10} {stats['correct']}/{stats['total']} ({acc:.1%})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rlhf-json", type=Path, default=DEFAULT_RLHF_JSON)
    parser.add_argument("--ov-json", type=Path, default=DEFAULT_OV_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()

    rows = merge_runs(load_json(args.rlhf_json), load_json(args.ov_json))
    write_csv(rows, args.output_csv)
    print_summary(rows)
    print(f"\n[OK] Comparison saved to {args.output_csv}")


if __name__ == "__main__":
    main()
