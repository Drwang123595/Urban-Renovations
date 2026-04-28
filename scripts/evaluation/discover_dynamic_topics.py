from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.core import (
    summarize_dynamic_binary_recommendations,
    summarize_dynamic_fixed_crosswalk,
    summarize_dynamic_topic_candidates,
    summarize_dynamic_topic_distribution,
    summarize_dynamic_topic_quality,
)
from src.urban.dynamic_topic_discovery import DynamicTopicConfig, DynamicTopicDiscovery


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append local dynamic topic evidence to an urban-renewal prediction workbook."
    )
    parser.add_argument("--pred", required=True, type=Path, help="Prediction workbook to enrich")
    parser.add_argument("--output", type=Path, default=None, help="Enriched workbook path")
    parser.add_argument("--report", type=Path, default=None, help="Dynamic topic report workbook path")
    parser.add_argument("--include-full-corpus", action="store_true", help="Cluster all rows as background evidence")
    parser.add_argument("--min-topic-size", type=int, default=20, help="Minimum preferred samples per dynamic topic")
    parser.add_argument("--max-topics", type=int, default=60, help="Maximum dynamic topics")
    parser.add_argument(
        "--keyword-fallback-only",
        action="store_true",
        help="Disable sklearn clustering and use deterministic keyword buckets",
    )
    return parser.parse_args()


def _default_output_path(pred_path: Path) -> Path:
    return pred_path.with_name(f"{pred_path.stem}_dynamic_topics.xlsx")


def _default_report_path(output_path: Path) -> Path:
    report_dir = output_path.parent.parent / "reports" if output_path.parent.name == "predictions" else output_path.parent
    return report_dir / f"{output_path.stem}_dynamic_topic_report.xlsx"


def run(args: argparse.Namespace) -> dict[str, Path]:
    pred_path = args.pred
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction workbook does not exist: {pred_path}")

    output_path = args.output or _default_output_path(pred_path)
    report_path = args.report or _default_report_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    pred_df = pd.read_excel(pred_path, engine="openpyxl")
    discovery = DynamicTopicDiscovery(
        DynamicTopicConfig(
            min_topic_size=args.min_topic_size,
            max_topics=args.max_topics,
            include_full_corpus=args.include_full_corpus,
            prefer_sklearn=not args.keyword_fallback_only,
        )
    )
    enriched = discovery.enrich(pred_df, include_full_corpus=args.include_full_corpus)
    enriched.to_excel(output_path, index=False, engine="openpyxl")

    source_name = output_path.stem
    with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
        summarize_dynamic_topic_quality(enriched, source_name).to_excel(
            writer,
            sheet_name="Dynamic Topic Quality",
            index=False,
        )
        summarize_dynamic_topic_distribution(enriched, source_name).to_excel(
            writer,
            sheet_name="Dynamic Topic Distribution",
            index=False,
        )
        summarize_dynamic_fixed_crosswalk(enriched, source_name).to_excel(
            writer,
            sheet_name="Dynamic Fixed Crosswalk",
            index=False,
        )
        summarize_dynamic_topic_candidates(enriched, source_name).to_excel(
            writer,
            sheet_name="Dynamic Topic Candidates",
            index=False,
        )
        summarize_dynamic_binary_recommendations(enriched, source_name).to_excel(
            writer,
            sheet_name="Dynamic Binary Recommendations",
            index=False,
        )

    return {"output": output_path, "report": report_path}


def main() -> None:
    outputs = run(parse_args())
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
