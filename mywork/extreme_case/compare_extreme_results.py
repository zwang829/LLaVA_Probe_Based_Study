#!/usr/bin/env python3
"""Compare extreme OOD probe JSON outputs and export a CSV table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

DEFAULT_RLHF_JSON = Path(__file__).resolve().parent / "llava_rlhf_extreme_results.json"
DEFAULT_OV_JSON = Path(__file__).resolve().parent / "llava_ov_extreme_results.json"
DEFAULT_CSV = Path(__file__).resolve().parent / "extreme_comparison.csv"

PROBE_ORDER = ["A1", "A2", "B1", "B2", "C1", "C2", "D1"]
MODEL_COLUMNS = [
    ("llava_rlhf", "LLaVA-RLHF"),
    ("llava_ov", "OneVision-1.5"),
]
RLHF_KEY = "llava_rlhf"
OV_KEY = "llava_ov"
QUALITATIVE_IDS = {"A2"}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def iter_rows(data: dict) -> list[dict]:
    if "per_probe" in data:
        return data["per_probe"]
    raise ValueError(f"Expected schema v1.0 with per_probe in {data.keys()}")


def verdict_for_row(row: dict, model_key: str) -> str:
    pid = row["id"]
    if pid in QUALITATIVE_IDS:
        return "(qualitative)"
    correct = row.get("correct", {}).get(model_key)
    if correct is None:
        return "n/a"
    return "correct" if correct else "wrong"


def truncate(text: str, width: int) -> str:
    text = " ".join(text.split())
    if len(text) <= width:
        return text
    return text[: width - 3] + "..."


def merge_results(rlhf_data: dict, ov_data: dict) -> list[dict]:
    merged: dict[str, dict] = {}
    probe_meta: dict[str, dict] = {}

    for payload in (rlhf_data, ov_data):
        for probe in payload.get("probes", []):
            probe_meta[probe["id"]] = probe
        for row in iter_rows(payload):
            pid = row["id"]
            meta = probe_meta.get(pid, row)
            entry = merged.setdefault(
                pid,
                {
                    "id": pid,
                    "image": meta.get("image", row.get("image", "")),
                    "question": meta.get("question", row.get("question", "")),
                    "gt": meta.get("gt", row.get("gt", "")),
                    "type": meta.get("type", row.get("type", "")),
                    "note": meta.get("note", row.get("note", "")),
                },
            )
            for model_key, _ in MODEL_COLUMNS:
                if model_key in row.get("answers", {}):
                    entry[f"answer_{model_key}"] = row["answers"][model_key]
                    correct = row.get("correct", {}).get(model_key)
                    if correct is not None:
                        entry[f"correct_{model_key}"] = correct
                    verdict = row.get("verdict", {}).get(model_key)
                    if verdict:
                        entry[f"verdict_{model_key}"] = verdict

    rows = [merged[pid] for pid in PROBE_ORDER if pid in merged]
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


def print_table(rows: list[dict]) -> None:
    print("=" * 160)
    print(
        f"{'ID':<4} {'Image':<20} {'Question':<42} {'GT':<6} "
        f"{'LLaVA-RLHF':<28} {'OneVision-1.5':<28} {'Verdict'}"
    )
    print("-" * 160)

    for row in rows:
        rlhf_ans = row.get(f"answer_{RLHF_KEY}", "")
        ov_ans = row.get(f"answer_{OV_KEY}", "")
        rlhf_v = verdict_for_row(
            {"id": row["id"], "correct": {RLHF_KEY: row.get(f"correct_{RLHF_KEY}")}},
            RLHF_KEY,
        )
        ov_v = verdict_for_row(
            {"id": row["id"], "correct": {OV_KEY: row.get(f"correct_{OV_KEY}")}},
            OV_KEY,
        )
        verdict = f"RLHF={rlhf_v}, OV={ov_v}"

        if row["id"] in QUALITATIVE_IDS:
            rlhf_disp = truncate(rlhf_ans, 28)
            ov_disp = truncate(ov_ans, 28)
        else:
            rlhf_disp = truncate(rlhf_ans, 28)
            ov_disp = truncate(ov_ans, 28)

        print(
            f"{row['id']:<4} {row['image'][:19]:<20} {truncate(row['question'], 42):<42} "
            f"{row['gt'][:5]:<6} {rlhf_disp:<28} {ov_disp:<28} {verdict}"
        )

    print("-" * 160)
    print("\nA2 full responses (qualitative, not auto-scored):")
    for row in rows:
        if row["id"] != "A2":
            continue
        print(f"  LLaVA-RLHF:     {row.get(f'answer_{RLHF_KEY}', '')}")
        print(f"  OneVision-1.5:  {row.get(f'answer_{OV_KEY}', '')}")


def print_summary(rows: list[dict]) -> None:
    scored_ids = [pid for pid in PROBE_ORDER if pid not in QUALITATIVE_IDS]
    totals = {
        RLHF_KEY: {"correct": 0, "total": 0},
        OV_KEY: {"correct": 0, "total": 0},
    }
    both_correct = both_wrong = rlhf_only = ov_only = 0

    for row in rows:
        if row["id"] not in scored_ids:
            continue
        rlhf_ok = bool(row.get(f"correct_{RLHF_KEY}"))
        ov_ok = bool(row.get(f"correct_{OV_KEY}"))
        totals[RLHF_KEY]["total"] += 1
        totals[OV_KEY]["total"] += 1
        totals[RLHF_KEY]["correct"] += int(rlhf_ok)
        totals[OV_KEY]["correct"] += int(ov_ok)
        if rlhf_ok and ov_ok:
            both_correct += 1
        elif not rlhf_ok and not ov_ok:
            both_wrong += 1
        elif rlhf_ok:
            rlhf_only += 1
        else:
            ov_only += 1

    print("\nSummary (scored probes only; A2 excluded):")
    for key, label in ((RLHF_KEY, "LLaVA-RLHF"), (OV_KEY, "OneVision-1.5")):
        stats = totals[key]
        acc = stats["correct"] / max(stats["total"], 1)
        print(f"  {label:<16} {stats['correct']}/{stats['total']} ({acc:.1%})")

    print(
        f"\nPairwise: both correct={both_correct}, both wrong={both_wrong}, "
        f"RLHF only={rlhf_only}, OV only={ov_only}"
    )
    print(f"Qualitative probe: A2 (open description on pure noise)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rlhf-json", type=Path, default=DEFAULT_RLHF_JSON)
    parser.add_argument("--ov-json", type=Path, default=DEFAULT_OV_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()

    rows = merge_results(load_json(args.rlhf_json), load_json(args.ov_json))
    write_csv(rows, args.output_csv)
    print_table(rows)
    print_summary(rows)
    print(f"\n[OK] Comparison saved to {args.output_csv}")


if __name__ == "__main__":
    main()
