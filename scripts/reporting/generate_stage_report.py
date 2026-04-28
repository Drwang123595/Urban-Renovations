from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
from xml.sax.saxutils import escape

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pipeline.run_stable_release import DEFAULT_DATASET_ID, DEFAULT_TAG, build_paths
from src.runtime.project_paths import PROJECT_ROOT


REPORT_DEPENDENCY_MESSAGE = (
    "Report generation dependencies are missing. Install them with "
    "`python -m pip install -e .[report]` or `python -m pip install -e .[dev]`."
)


@dataclass(frozen=True)
class StageReportInputs:
    dataset_id: str
    tag: str
    run_dir: Path
    prediction_file: Path
    eval_summary_file: Path
    run_summary_file: Path
    unknown_review_file: Path
    output_dir: Path
    table_file: Path
    pdf_file: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the stable urban-renewal stage report.")
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID, help="Dataset identifier")
    parser.add_argument("--tag", default=DEFAULT_TAG, help="Run tag under runs/stable_release/")
    parser.add_argument("--pred", type=Path, default=None, help="Prediction workbook override")
    parser.add_argument("--report-dir", type=Path, default=None, help="Directory containing Eval_Summary.xlsx")
    parser.add_argument("--eval-summary", type=Path, default=None, help="Eval_Summary.xlsx override")
    parser.add_argument("--run-summary", type=Path, default=None, help="Stable_Run_Summary.json override")
    parser.add_argument("--unknown-review", type=Path, default=None, help="Unknown review workbook override")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output" / "pdf")
    parser.add_argument("--tables", type=Path, default=None, help="Output workbook for report tables")
    parser.add_argument("--pdf", type=Path, default=None, help="Output PDF path")
    parser.add_argument("--no-pdf", action="store_true", help="Only export the source tables workbook")
    return parser.parse_args()


def resolve_report_inputs(args: argparse.Namespace) -> StageReportInputs:
    stable_paths = build_paths(dataset_id=args.dataset_id, tag=args.tag)
    report_dir = Path(args.report_dir) if args.report_dir else stable_paths.result_dir
    eval_summary_file = Path(args.eval_summary) if args.eval_summary else report_dir / "Eval_Summary.xlsx"
    output_dir = Path(args.output_dir)
    stem = f"urban_renovation_stable_report_{args.tag}"
    return StageReportInputs(
        dataset_id=args.dataset_id,
        tag=args.tag,
        run_dir=stable_paths.run_dir,
        prediction_file=Path(args.pred) if args.pred else stable_paths.prediction_file,
        eval_summary_file=eval_summary_file,
        run_summary_file=Path(args.run_summary) if args.run_summary else stable_paths.run_summary_file,
        unknown_review_file=Path(args.unknown_review) if args.unknown_review else stable_paths.unknown_review_file,
        output_dir=output_dir,
        table_file=Path(args.tables) if args.tables else output_dir / f"{stem}_tables.xlsx",
        pdf_file=Path(args.pdf) if args.pdf else output_dir / f"{stem}.pdf",
    )


def validate_report_inputs(inputs: StageReportInputs) -> None:
    required = {
        "prediction workbook": inputs.prediction_file,
        "evaluation summary": inputs.eval_summary_file,
        "stable run summary": inputs.run_summary_file,
    }
    missing = [f"{label}: {path}" for label, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required report inputs:\n" + "\n".join(missing))


def load_run_summary(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_summary_tables(path: Path) -> Dict[str, pd.DataFrame]:
    workbook = pd.ExcelFile(path, engine="openpyxl")
    tables: Dict[str, pd.DataFrame] = {}
    for sheet_name in workbook.sheet_names:
        tables[sheet_name] = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
    return tables


def _first_matching_row(frame: pd.DataFrame, **filters: Any) -> Dict[str, Any]:
    if frame.empty:
        return {}
    filtered = frame
    for column, value in filters.items():
        if column not in filtered.columns:
            return {}
        filtered = filtered[filtered[column].astype(str) == str(value)]
    if filtered.empty:
        return {}
    return filtered.iloc[0].to_dict()


def _metric_value(metrics: Dict[str, Any], eval_row: Dict[str, Any], key: str, default: Any = "") -> Any:
    if key in metrics and metrics[key] not in (None, ""):
        return metrics[key]
    title_key = "".join(part.capitalize() if index else part for index, part in enumerate(key.split("_")))
    return eval_row.get(title_key, eval_row.get(key, default))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def collect_report_facts(
    inputs: StageReportInputs,
    tables: Dict[str, pd.DataFrame],
    run_summary: Dict[str, Any],
) -> Dict[str, Any]:
    metrics = dict(run_summary.get("metrics") or {})
    thresholds = dict(run_summary.get("thresholds") or {})
    file_stem = str(metrics.get("file") or inputs.prediction_file.stem)

    all_metrics = tables.get("All Metrics", pd.DataFrame())
    urban_row = _first_matching_row(all_metrics, File=file_stem, Metric="Urban Renewal")
    if not urban_row and not all_metrics.empty:
        urban_candidates = all_metrics[all_metrics.get("Metric", "").astype(str) == "Urban Renewal"]
        if not urban_candidates.empty:
            urban_row = urban_candidates.iloc[0].to_dict()

    unknown_rate = tables.get("Unknown Rate", pd.DataFrame())
    unknown_row = _first_matching_row(unknown_rate, File=file_stem)
    decision_sources = tables.get("Decision Source Metrics", pd.DataFrame()).copy()
    if not decision_sources.empty and "File" in decision_sources.columns:
        decision_sources = decision_sources[decision_sources["File"].astype(str) == file_stem].copy()
    explainability_quality = tables.get("Explainability Quality", pd.DataFrame()).copy()
    if not explainability_quality.empty and "File" in explainability_quality.columns:
        explainability_quality = explainability_quality[explainability_quality["File"].astype(str) == file_stem].copy()
    explainability_row = explainability_quality.iloc[0].to_dict() if not explainability_quality.empty else {}
    evidence_balance_metrics = tables.get("Evidence Balance Metrics", pd.DataFrame()).copy()
    if not evidence_balance_metrics.empty and "File" in evidence_balance_metrics.columns:
        evidence_balance_metrics = evidence_balance_metrics[evidence_balance_metrics["File"].astype(str) == file_stem].copy()
    dynamic_topic_quality = tables.get("Dynamic Topic Quality", pd.DataFrame()).copy()
    if not dynamic_topic_quality.empty and "File" in dynamic_topic_quality.columns:
        dynamic_topic_quality = dynamic_topic_quality[dynamic_topic_quality["File"].astype(str) == file_stem].copy()
    dynamic_topic_row = dynamic_topic_quality.iloc[0].to_dict() if not dynamic_topic_quality.empty else {}
    dynamic_topic_distribution = tables.get("Dynamic Topic Distribution", pd.DataFrame()).copy()
    if not dynamic_topic_distribution.empty and "File" in dynamic_topic_distribution.columns:
        dynamic_topic_distribution = dynamic_topic_distribution[dynamic_topic_distribution["File"].astype(str) == file_stem].copy()
    dynamic_binary_recommendations = tables.get("Dynamic Binary Recommendations", pd.DataFrame()).copy()
    if not dynamic_binary_recommendations.empty and "File" in dynamic_binary_recommendations.columns:
        dynamic_binary_recommendations = dynamic_binary_recommendations[
            dynamic_binary_recommendations["File"].astype(str) == file_stem
        ].copy()
    possible_fn_count = 0
    possible_fp_count = 0
    high_priority_binary_review_count = 0
    if not dynamic_binary_recommendations.empty:
        action = dynamic_binary_recommendations.get("Candidate Action", pd.Series(dtype=object)).astype(str)
        totals = pd.to_numeric(dynamic_binary_recommendations.get("Total", pd.Series(dtype=float)), errors="coerce").fillna(0)
        possible_fn_count = _safe_int(totals[action == "possible_false_negative_cluster"].sum())
        possible_fp_count = _safe_int(totals[action == "possible_false_positive_cluster"].sum())
        priority = dynamic_binary_recommendations.get("Review Priority", pd.Series(dtype=object)).astype(str)
        high_priority_binary_review_count = _safe_int(totals[priority == "high"].sum())

    pred_df = pd.read_excel(inputs.prediction_file, engine="openpyxl")
    llm_used_sum = _safe_int(pd.to_numeric(pred_df.get("llm_used", pd.Series(dtype=int)), errors="coerce").fillna(0).sum())
    llm_attempted_sum = _safe_int(
        pd.to_numeric(pred_df.get("llm_attempted", pd.Series(dtype=int)), errors="coerce").fillna(0).sum()
    )

    def prediction_coverage(column: str) -> float:
        if column not in pred_df.columns or len(pred_df) == 0:
            return 0.0
        normalized = pred_df[column].fillna("").astype(str).str.strip().str.lower()
        return round(float((~normalized.isin({"", "nan", "none", "null"})).mean()), 6)

    return {
        "dataset_id": inputs.dataset_id,
        "tag": inputs.tag,
        "file": file_stem,
        "rows": _safe_int(metrics.get("rows"), len(pred_df)),
        "accuracy": _safe_float(_metric_value(metrics, urban_row, "accuracy", urban_row.get("Accuracy", 0.0))),
        "correct": _safe_int(_metric_value(metrics, urban_row, "correct", urban_row.get("Correct", 0))),
        "total": _safe_int(_metric_value(metrics, urban_row, "total", urban_row.get("Total", len(pred_df)))),
        "tp": _safe_int(_metric_value(metrics, urban_row, "tp", urban_row.get("TP", 0))),
        "tn": _safe_int(_metric_value(metrics, urban_row, "tn", urban_row.get("TN", 0))),
        "fp": _safe_int(_metric_value(metrics, urban_row, "fp", urban_row.get("FP", 0))),
        "fn": _safe_int(_metric_value(metrics, urban_row, "fn", urban_row.get("FN", 0))),
        "precision": _safe_float(_metric_value(metrics, urban_row, "precision", urban_row.get("Precision", 0.0))),
        "recall": _safe_float(_metric_value(metrics, urban_row, "recall", urban_row.get("Recall", 0.0))),
        "f1": _safe_float(_metric_value(metrics, urban_row, "f1", urban_row.get("F1", 0.0))),
        "predicted_unknown_count": _safe_int(
            metrics.get("predicted_unknown_count", unknown_row.get("Predicted Unknown Count", 0))
        ),
        "predicted_unknown_rate": _safe_float(
            metrics.get("predicted_unknown_rate", unknown_row.get("Predicted Unknown Rate", 0.0))
        ),
        "unknown_hint_resolution_total": _safe_int(metrics.get("unknown_hint_resolution_total", 0)),
        "unknown_hint_resolution_accuracy": _safe_float(metrics.get("unknown_hint_resolution_accuracy", 0.0)),
        "unknown_review_total": _safe_int(metrics.get("unknown_review_total", 0)),
        "llm_used_sum": llm_used_sum,
        "llm_attempted_sum": llm_attempted_sum,
        "explanation_coverage": _safe_float(
            metrics.get(
                "explanation_coverage",
                explainability_row.get("Decision Explanation Coverage", prediction_coverage("decision_explanation")),
            )
        ),
        "rule_stack_coverage": _safe_float(
            metrics.get(
                "rule_stack_coverage",
                explainability_row.get("Rule Stack Coverage", prediction_coverage("decision_rule_stack")),
            )
        ),
        "binary_evidence_coverage": _safe_float(
            metrics.get(
                "binary_evidence_coverage",
                explainability_row.get("Binary Evidence Coverage", prediction_coverage("binary_decision_evidence")),
            )
        ),
        "review_trigger_rate": _safe_float(explainability_row.get("Review Trigger Rate", 0.0)),
        "near_threshold_count": _safe_int(explainability_row.get("Near Threshold Count", 0)),
        "conflict_count": _safe_int(explainability_row.get("Conflict Count", 0)),
        "dynamic_topic_coverage": _safe_float(dynamic_topic_row.get("Dynamic Topic Coverage", 0.0)),
        "unknown_dynamic_coverage": _safe_float(dynamic_topic_row.get("Unknown Dynamic Coverage", 0.0)),
        "dynamic_topic_count": _safe_int(dynamic_topic_row.get("Dynamic Topic Count", 0)),
        "candidate_new_urban_topic_count": _safe_int(dynamic_topic_row.get("Candidate New Urban Topic Count", 0)),
        "dynamic_possible_fn_count": possible_fn_count,
        "dynamic_possible_fp_count": possible_fp_count,
        "dynamic_high_priority_binary_review_count": high_priority_binary_review_count,
        "dynamic_topic_quality": dynamic_topic_quality,
        "dynamic_topic_distribution": dynamic_topic_distribution,
        "dynamic_binary_recommendations": dynamic_binary_recommendations,
        "gate_status": str(run_summary.get("gate_status", "")),
        "gate_failures": list(run_summary.get("gate_failures") or []),
        "generated_at": str(run_summary.get("generated_at", "")),
        "thresholds": thresholds,
        "decision_sources": decision_sources,
        "explainability_quality": explainability_quality,
        "evidence_balance_metrics": evidence_balance_metrics,
    }


def _key_value_frame(mapping: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([{"Field": key, "Value": value} for key, value in mapping.items()])


def build_export_tables(
    inputs: StageReportInputs,
    facts: Dict[str, Any],
    source_tables: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    thresholds = facts.get("thresholds") or {}
    export = {
        "Stable Metrics": _key_value_frame(
            {
                "dataset_id": facts["dataset_id"],
                "tag": facts["tag"],
                "file": facts["file"],
                "rows": facts["rows"],
                "accuracy": facts["accuracy"],
                "precision": facts["precision"],
                "recall": facts["recall"],
                "f1": facts["f1"],
                "tp": facts["tp"],
                "tn": facts["tn"],
                "fp": facts["fp"],
                "fn": facts["fn"],
                "predicted_unknown_count": facts["predicted_unknown_count"],
                "predicted_unknown_rate": facts["predicted_unknown_rate"],
                "unknown_hint_resolution_accuracy": facts["unknown_hint_resolution_accuracy"],
                "llm_attempted_sum": facts["llm_attempted_sum"],
                "llm_used_sum": facts["llm_used_sum"],
                "explanation_coverage": facts["explanation_coverage"],
                "rule_stack_coverage": facts["rule_stack_coverage"],
                "binary_evidence_coverage": facts["binary_evidence_coverage"],
                "review_trigger_rate": facts["review_trigger_rate"],
                "near_threshold_count": facts["near_threshold_count"],
                "conflict_count": facts["conflict_count"],
                "dynamic_topic_coverage": facts["dynamic_topic_coverage"],
                "unknown_dynamic_coverage": facts["unknown_dynamic_coverage"],
                "dynamic_topic_count": facts["dynamic_topic_count"],
                "candidate_new_urban_topic_count": facts["candidate_new_urban_topic_count"],
                "dynamic_possible_fn_count": facts["dynamic_possible_fn_count"],
                "dynamic_possible_fp_count": facts["dynamic_possible_fp_count"],
                "dynamic_high_priority_binary_review_count": facts["dynamic_high_priority_binary_review_count"],
                "gate_status": facts["gate_status"],
            }
        ),
        "Gate Thresholds": _key_value_frame(thresholds),
        "Artifact Paths": _key_value_frame(
            {
                "run_dir": str(inputs.run_dir),
                "prediction_file": str(inputs.prediction_file),
                "eval_summary_file": str(inputs.eval_summary_file),
                "run_summary_file": str(inputs.run_summary_file),
                "unknown_review_file": str(inputs.unknown_review_file),
            }
        ),
    }
    export["Explainability Quality"] = (
        facts["explainability_quality"].copy()
        if not facts["explainability_quality"].empty
        else pd.DataFrame(
            [
                {
                    "File": facts["file"],
                    "Decision Explanation Coverage": facts["explanation_coverage"],
                    "Rule Stack Coverage": facts["rule_stack_coverage"],
                    "Binary Evidence Coverage": facts["binary_evidence_coverage"],
                    "Review Trigger Rate": facts["review_trigger_rate"],
                    "Near Threshold Count": facts["near_threshold_count"],
                    "Conflict Count": facts["conflict_count"],
                }
            ]
        )
    )
    export["Evidence Balance Metrics"] = (
        facts["evidence_balance_metrics"].copy()
        if not facts["evidence_balance_metrics"].empty
        else pd.DataFrame(columns=["File", "Evidence Balance", "Total", "Accuracy", "FP", "FN"])
    )
    export["Dynamic Topic Quality"] = (
        facts["dynamic_topic_quality"].copy()
        if not facts["dynamic_topic_quality"].empty
        else pd.DataFrame(
            [
                {
                    "File": facts["file"],
                    "Dynamic Topic Coverage": facts["dynamic_topic_coverage"],
                    "Unknown Dynamic Coverage": facts["unknown_dynamic_coverage"],
                    "Dynamic Topic Count": facts["dynamic_topic_count"],
                    "Candidate New Urban Topic Count": facts["candidate_new_urban_topic_count"],
                }
            ]
        )
    )
    export["Dynamic Topic Distribution"] = (
        facts["dynamic_topic_distribution"].copy()
        if not facts["dynamic_topic_distribution"].empty
        else pd.DataFrame(
            columns=[
                "File",
                "Dynamic Topic ID",
                "Dynamic Topic Name Zh",
                "Mapping Status",
                "Fixed Topic Candidate",
                "Count",
                "Mean Confidence",
                "Keywords",
            ]
        )
    )
    export["Dynamic Binary Recommendations"] = (
        facts["dynamic_binary_recommendations"].copy()
        if not facts["dynamic_binary_recommendations"].empty
        else pd.DataFrame(
            columns=[
                "File",
                "Candidate Action",
                "Candidate Label",
                "Review Priority",
                "Total",
                "Share",
                "Current Positive Rate",
                "Mean Candidate Confidence",
            ]
        )
    )
    for sheet_name in [
        "All Metrics",
        "Unknown Rate",
        "Decision Source Metrics",
        "Topic Distribution",
        "Boundary Bucket Metrics",
        "Unknown Conflict Analysis",
        "Explainability Quality",
        "Evidence Balance Metrics",
        "Dynamic Topic Quality",
        "Dynamic Topic Distribution",
        "Dynamic Fixed Crosswalk",
        "Dynamic Topic Candidates",
        "Dynamic Binary Recommendations",
        "Guardrails",
        "Run Metadata",
        "Protocol",
        "Comparability",
    ]:
        if sheet_name in source_tables:
            export[sheet_name] = source_tables[sheet_name]
    return export


def write_table_exports(tables: Dict[str, pd.DataFrame], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, table in tables.items():
            table.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    return output_path


def _load_reportlab():
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError as exc:
        raise RuntimeError(REPORT_DEPENDENCY_MESSAGE) from exc
    return {
        "colors": colors,
        "TA_CENTER": TA_CENTER,
        "TA_LEFT": TA_LEFT,
        "A4": A4,
        "ParagraphStyle": ParagraphStyle,
        "getSampleStyleSheet": getSampleStyleSheet,
        "cm": cm,
        "PageBreak": PageBreak,
        "Paragraph": Paragraph,
        "SimpleDocTemplate": SimpleDocTemplate,
        "Spacer": Spacer,
        "Table": Table,
        "TableStyle": TableStyle,
    }


def _fmt_pct(value: float, *, already_percent: bool = False) -> str:
    scaled = value if already_percent else value * 100.0
    return f"{scaled:.2f}%"


def _paragraph(text: Any, style: Any, Paragraph: Any) -> Any:
    return Paragraph(escape(str(text)), style)


def _table_from_rows(rows: list[list[Any]], styles: Dict[str, Any], rl: Dict[str, Any], widths: list[float]) -> Any:
    Paragraph = rl["Paragraph"]
    Table = rl["Table"]
    TableStyle = rl["TableStyle"]
    colors = rl["colors"]

    rendered = []
    for row_index, row in enumerate(rows):
        style = styles["table_header"] if row_index == 0 else styles["table_body"]
        rendered.append([_paragraph(value, style, Paragraph) for value in row])
    table = Table(rendered, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324D")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D8E0E8")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def build_pdf(inputs: StageReportInputs, facts: Dict[str, Any], table_export_path: Path) -> Path:
    rl = _load_reportlab()
    colors = rl["colors"]
    ParagraphStyle = rl["ParagraphStyle"]
    getSampleStyleSheet = rl["getSampleStyleSheet"]
    SimpleDocTemplate = rl["SimpleDocTemplate"]
    Spacer = rl["Spacer"]
    Paragraph = rl["Paragraph"]
    PageBreak = rl["PageBreak"]
    cm = rl["cm"]

    inputs.pdf_file.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(inputs.pdf_file),
        pagesize=rl["A4"],
        leftMargin=1.45 * cm,
        rightMargin=1.45 * cm,
        topMargin=1.25 * cm,
        bottomMargin=1.25 * cm,
    )
    sample = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "StageTitle",
            parent=sample["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#17324D"),
            alignment=rl["TA_LEFT"],
        ),
        "h1": ParagraphStyle(
            "StageH1",
            parent=sample["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#17324D"),
            spaceBefore=8,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "StageBody",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor("#17324D"),
            spaceAfter=5,
        ),
        "table_header": ParagraphStyle(
            "TableHeader",
            parent=sample["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8.2,
            leading=10,
            textColor=colors.white,
            alignment=rl["TA_CENTER"],
        ),
        "table_body": ParagraphStyle(
            "TableBody",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=8.0,
            leading=10,
            textColor=colors.HexColor("#17324D"),
        ),
        "small": ParagraphStyle(
            "Small",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=9,
            textColor=colors.HexColor("#6E7C8C"),
        ),
    }

    metric_rows = [
        ["Metric", "Value", "Gate / Note"],
        ["Accuracy", _fmt_pct(float(facts["accuracy"]), already_percent=True), f">= {facts['thresholds'].get('min_accuracy', '')}"],
        ["Precision", f"{facts['precision']:.6f}", f">= {facts['thresholds'].get('min_precision', '')}"],
        ["Recall", f"{facts['recall']:.6f}", f">= {facts['thresholds'].get('min_recall', '')}"],
        ["F1", f"{facts['f1']:.6f}", f">= {facts['thresholds'].get('min_f1', '')}"],
        ["FP / FN", f"{facts['fp']} / {facts['fn']}", f"FP <= {facts['thresholds'].get('max_fp', '')}; FN <= {facts['thresholds'].get('max_fn', '')}"],
        ["Predicted Unknown", str(facts["predicted_unknown_count"]), f"<= {facts['thresholds'].get('max_unknown_count', '')}"],
        ["LLM attempted / used", f"{facts['llm_attempted_sum']} / {facts['llm_used_sum']}", "stable release requires used = 0"],
        [
            "Explanation coverage",
            _fmt_pct(float(facts["explanation_coverage"])),
            f">= {_fmt_pct(float(facts['thresholds'].get('min_explanation_coverage', 1.0)))}",
        ],
        [
            "Rule stack coverage",
            _fmt_pct(float(facts["rule_stack_coverage"])),
            f">= {_fmt_pct(float(facts['thresholds'].get('min_rule_stack_coverage', 1.0)))}",
        ],
        [
            "Binary evidence coverage",
            _fmt_pct(float(facts["binary_evidence_coverage"])),
            f">= {_fmt_pct(float(facts['thresholds'].get('min_binary_evidence_coverage', 1.0)))}",
        ],
    ]

    decision_rows = [["Decision Source", "Total", "Accuracy", "Precision", "Recall", "F1"]]
    decision_df = facts["decision_sources"]
    if not decision_df.empty:
        for _, row in decision_df.iterrows():
            decision_rows.append(
                [
                    row.get("Decision Source", ""),
                    row.get("Total", ""),
                    row.get("Accuracy", ""),
                    row.get("Precision", ""),
                    row.get("Recall", ""),
                    row.get("F1", ""),
                ]
            )

    explainability_rows = [
        ["Metric", "Value"],
        ["Decision explanation coverage", _fmt_pct(float(facts["explanation_coverage"]))],
        ["Rule stack coverage", _fmt_pct(float(facts["rule_stack_coverage"]))],
        ["Binary evidence coverage", _fmt_pct(float(facts["binary_evidence_coverage"]))],
        ["Review trigger rate", _fmt_pct(float(facts["review_trigger_rate"]))],
        ["Near-threshold samples", facts["near_threshold_count"]],
        ["Conflict samples", facts["conflict_count"]],
    ]

    evidence_rows = [["Evidence Balance", "Total", "Accuracy", "FP", "FN"]]
    evidence_df = facts["evidence_balance_metrics"]
    if not evidence_df.empty:
        for _, row in evidence_df.iterrows():
            evidence_rows.append(
                [
                    row.get("Evidence Balance", ""),
                    row.get("Total", ""),
                    row.get("Accuracy", ""),
                    row.get("FP", ""),
                    row.get("FN", ""),
                ]
            )
    else:
        evidence_rows.append(["not available", 0, "", "", ""])

    dynamic_rows = [
        ["Metric", "Value"],
        ["Dynamic topic count", facts["dynamic_topic_count"]],
        ["Dynamic topic coverage", _fmt_pct(float(facts["dynamic_topic_coverage"]))],
        ["Unknown dynamic coverage", _fmt_pct(float(facts["unknown_dynamic_coverage"]))],
        ["Candidate new urban topic rows", facts["candidate_new_urban_topic_count"]],
        ["Possible false-negative review rows", facts["dynamic_possible_fn_count"]],
        ["Possible false-positive review rows", facts["dynamic_possible_fp_count"]],
        ["High-priority binary review rows", facts["dynamic_high_priority_binary_review_count"]],
    ]
    dynamic_distribution_rows = [["Dynamic Topic", "Status", "Count", "Keywords"]]
    dynamic_distribution_df = facts["dynamic_topic_distribution"]
    if not dynamic_distribution_df.empty:
        for _, row in dynamic_distribution_df.head(8).iterrows():
            dynamic_distribution_rows.append(
                [
                    row.get("Dynamic Topic ID", ""),
                    row.get("Mapping Status", ""),
                    row.get("Count", ""),
                    row.get("Keywords", ""),
                ]
            )
    else:
        dynamic_distribution_rows.append(["not available", "", 0, ""])

    path_rows = [
        ["Artifact", "Path"],
        ["Prediction", inputs.prediction_file],
        ["Eval Summary", inputs.eval_summary_file],
        ["Run Summary", inputs.run_summary_file],
        ["Unknown Review", inputs.unknown_review_file],
        ["Table Export", table_export_path],
    ]

    story = [
        _paragraph("Urban Renovation Stable Release Report", styles["title"], Paragraph),
        Spacer(1, 0.18 * cm),
        _paragraph(
            f"Dataset: {facts['dataset_id']} | Tag: {facts['tag']} | Gate: {facts['gate_status']}",
            styles["body"],
            Paragraph,
        ),
        _paragraph(
            "This report is generated from the canonical stable-release artifacts: "
            "Stable_Run_Summary.json, Eval_Summary.xlsx, and the locked prediction workbook.",
            styles["body"],
            Paragraph,
        ),
        _paragraph("Stable Metrics", styles["h1"], Paragraph),
        _table_from_rows(metric_rows, styles, rl, [4.6 * cm, 4.0 * cm, 8.2 * cm]),
        Spacer(1, 0.2 * cm),
        _paragraph("LLM Role", styles["h1"], Paragraph),
        _paragraph(
            "The hybrid pipeline may call the LLM only to collect a family-level hint for Unknown cases. "
            "The stable contract keeps llm_used at 0, so the LLM hint is evidence for conservative recovery, "
            "not an online primary decision source that overwrites topic_final.",
            styles["body"],
            Paragraph,
        ),
        _paragraph("Explainability Quality", styles["h1"], Paragraph),
        _paragraph(
            "Explanations are generated deterministically from rule, score, guard, and evidence fields. "
            "No LLM-generated narrative is used in the stable evidence chain.",
            styles["body"],
            Paragraph,
        ),
        _table_from_rows(explainability_rows, styles, rl, [8.0 * cm, 5.0 * cm]),
        Spacer(1, 0.2 * cm),
        _paragraph("Evidence Balance", styles["h1"], Paragraph),
        _table_from_rows(evidence_rows, styles, rl, [5.4 * cm, 2.2 * cm, 3.0 * cm, 2.0 * cm, 2.0 * cm]),
        Spacer(1, 0.2 * cm),
        _paragraph("Dynamic Topic Evidence", styles["h1"], Paragraph),
        _paragraph(
            "Dynamic topics are a local post-processing evidence layer. They do not overwrite topic_final, "
            "urban_flag, or final_label, and they are not part of the stable gate.",
            styles["body"],
            Paragraph,
        ),
        _table_from_rows(dynamic_rows, styles, rl, [8.0 * cm, 5.0 * cm]),
        Spacer(1, 0.2 * cm),
        _table_from_rows(dynamic_distribution_rows, styles, rl, [3.2 * cm, 4.2 * cm, 2.0 * cm, 7.0 * cm]),
        Spacer(1, 0.2 * cm),
        _paragraph("Decision Source Breakdown", styles["h1"], Paragraph),
        _table_from_rows(decision_rows, styles, rl, [5.0 * cm, 2.0 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm]),
        PageBreak(),
        _paragraph("Artifact Trace", styles["h1"], Paragraph),
        _table_from_rows(path_rows, styles, rl, [4.0 * cm, 13.0 * cm]),
        Spacer(1, 0.2 * cm),
        _paragraph(
            f"Generated from run summary timestamp: {facts.get('generated_at') or 'unknown'}",
            styles["small"],
            Paragraph,
        ),
    ]
    if facts["gate_failures"]:
        story.insert(
            4,
            _paragraph("Gate failures: " + "; ".join(map(str, facts["gate_failures"])), styles["body"], Paragraph),
        )

    doc.build(story)
    return inputs.pdf_file


def generate_report(args: argparse.Namespace) -> Dict[str, Path]:
    inputs = resolve_report_inputs(args)
    validate_report_inputs(inputs)
    run_summary = load_run_summary(inputs.run_summary_file)
    source_tables = load_summary_tables(inputs.eval_summary_file)
    facts = collect_report_facts(inputs, source_tables, run_summary)
    export_tables = build_export_tables(inputs, facts, source_tables)
    table_path = write_table_exports(export_tables, inputs.table_file)
    result = {"tables": table_path}
    if not args.no_pdf:
        result["pdf"] = build_pdf(inputs, facts, table_path)
    return result


def main() -> None:
    outputs = generate_report(parse_args())
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
