#!/usr/bin/env python3
"""Summarize and sample raw GRPO rollout audit JSONL records."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


FORMAT_FLAGS = {
    "empty_output",
    "missing_action",
    "invalid_action",
    "missing_thought",
    "unbalanced_think_tags",
    "multiple_think_blocks",
    "multiple_actions_in_turn",
    "hallucinated_observation",
    "unknown_tool",
}


def load_records(path: Path) -> List[Dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(item, dict):
            raise SystemExit(f"Expected object at {path}:{line_number}")
        records.append(item)
    if not records:
        raise SystemExit(f"No rollout records found in {path}")
    return records


def summarize(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(records)
    anomaly_counts: Counter[str] = Counter()
    task_samples: Counter[str] = Counter()
    task_anomalies: Counter[str] = Counter()
    groups: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    format_anomalous = 0

    for row in rows:
        flags = set(row.get("active_anomalies") or [])
        task_id = str(row.get("task_id"))
        task_samples[task_id] += 1
        anomaly_counts.update(flags)
        if flags:
            task_anomalies[task_id] += 1
        if flags & FORMAT_FLAGS:
            format_anomalous += 1
        groups[(row.get("batch_index"), row.get("group_index"))].append(row)

    zero_std_groups = 0
    duplicate_groups = 0
    for group_rows in groups.values():
        if abs(float(group_rows[0].get("group_reward_std") or 0.0)) <= 1e-12:
            zero_std_groups += 1
        completions = {
            json.dumps(row.get("raw_assistant_turns"), ensure_ascii=False, sort_keys=True)
            for row in group_rows
        }
        if len(completions) == 1:
            duplicate_groups += 1

    group_count = len(groups)
    anomalous_samples = sum(bool(row.get("active_anomalies")) for row in rows)
    return {
        "sample_count": len(rows),
        "group_count": group_count,
        "anomalous_sample_count": anomalous_samples,
        "anomalous_sample_fraction": anomalous_samples / len(rows),
        "format_anomalous_sample_count": format_anomalous,
        "format_anomalous_sample_fraction": format_anomalous / len(rows),
        "zero_reward_std_group_count": zero_std_groups,
        "zero_reward_std_group_fraction": zero_std_groups / group_count if group_count else 0.0,
        "exact_duplicate_group_count": duplicate_groups,
        "exact_duplicate_group_fraction": duplicate_groups / group_count if group_count else 0.0,
        "anomaly_counts": dict(anomaly_counts.most_common()),
        "tasks": {
            task_id: {
                "sample_count": count,
                "anomalous_sample_count": task_anomalies[task_id],
                "anomalous_sample_fraction": task_anomalies[task_id] / count,
            }
            for task_id, count in sorted(task_samples.items())
        },
    }


def print_report(summary: Dict[str, Any]) -> None:
    print("=== GRPO raw rollout audit ===")
    print(f"samples: {summary['sample_count']}")
    print(f"groups: {summary['group_count']}")
    print(
        "anomalous samples: "
        f"{summary['anomalous_sample_count']} "
        f"({summary['anomalous_sample_fraction']:.1%})"
    )
    print(
        "format anomalies: "
        f"{summary['format_anomalous_sample_count']} "
        f"({summary['format_anomalous_sample_fraction']:.1%})"
    )
    print(
        "zero-reward-std groups: "
        f"{summary['zero_reward_std_group_count']} "
        f"({summary['zero_reward_std_group_fraction']:.1%})"
    )
    print(
        "exact-duplicate groups: "
        f"{summary['exact_duplicate_group_count']} "
        f"({summary['exact_duplicate_group_fraction']:.1%})"
    )
    print("anomaly counts:")
    if not summary["anomaly_counts"]:
        print("  (none)")
    for name, count in summary["anomaly_counts"].items():
        print(f"  {name}: {count}")


def print_examples(records: List[Dict[str, Any]], limit: int) -> None:
    anomalous = [row for row in records if row.get("active_anomalies")]
    anomalous.sort(key=lambda row: (float(row.get("reward") or 0.0), row.get("task_id", "")))
    if not anomalous or limit <= 0:
        return
    print(f"\n=== first {min(limit, len(anomalous))} anomalous samples ===")
    for row in anomalous[:limit]:
        print(
            f"\ntask={row.get('task_id')} group={row.get('group_index')} "
            f"generation={row.get('generation_index')} reward={row.get('reward')}"
        )
        print(f"flags={row.get('active_anomalies')}")
        for turn_index, turn in enumerate(row.get("raw_assistant_turns") or [], 1):
            print(f"--- assistant turn {turn_index} ---")
            print(turn)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="rollout_traces.jsonl")
    parser.add_argument(
        "--show", type=int, default=10,
        help="print up to N anomalous raw generations (default: 10)",
    )
    parser.add_argument(
        "--summary_out", type=Path, default=None,
        help="optionally write the recomputed aggregate summary as JSON",
    )
    args = parser.parse_args()

    records = load_records(args.trace)
    summary = summarize(records)
    print_report(summary)
    print_examples(records, args.show)
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nsummary written: {args.summary_out}")


if __name__ == "__main__":
    main()
