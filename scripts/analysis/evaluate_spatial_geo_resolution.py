import argparse
import json
from pathlib import Path

import pandas as pd

from src.runtime.config import Schema
from src.strategies.geo_resolver import GeoResolver


def _is_positive(value: object) -> bool:
    return str(value).strip() in {"1", "1.0", "true", "True"}


def evaluate(input_path: Path, output_xlsx: Path, output_json: Path) -> dict:
    df = pd.read_excel(input_path)
    resolver = GeoResolver()
    rows = []
    for index, row in df.iterrows():
        is_spatial = _is_positive(row.get(Schema.IS_SPATIAL, "0"))
        area = row.get(Schema.SPATIAL_DESC, "")
        llm_scale = row.get(Schema.SPATIAL_LEVEL, "")
        result = resolver.resolve(area, llm_scale_level=str(llm_scale)) if is_spatial else None
        payload = {
            "row_index": index,
            "is_spatial": int(is_spatial),
            "area": area,
            "old_scale": llm_scale,
            "mapped_scale": result.mapped_spatial_scale_level if result else "",
            "geo_resolution_status": result.geo_resolution_status if result else "not_applicable",
            "scale_decision_source": result.scale_decision_source if result else "",
            "resolved_study_area": result.resolved_study_area if result else "",
            "resolved_geo_id": result.resolved_geo_id if result else "",
            "area_hierarchy_path": result.area_hierarchy_path if result else "",
            "geo_source": result.geo_source if result else "",
        }
        rows.append(payload)
    out = pd.DataFrame(rows)
    positives = out[out["is_spatial"] == 1].copy()
    matched_mask = positives["geo_resolution_status"].astype(str).str.startswith("matched")
    override_mask = positives["scale_decision_source"].eq("mapping_override_llm")
    summary = {
        "input": str(input_path),
        "rows": int(len(out)),
        "spatial_positive_rows": int(len(positives)),
        "geo_matched_rows": int(matched_mask.sum()),
        "geo_matched_share": round(float(matched_mask.mean()) if len(positives) else 0.0, 4),
        "mapping_override_llm_rows": int(override_mask.sum()),
        "unresolved_or_ambiguous_rows": int(
            positives["geo_resolution_status"].eq("unresolved_or_ambiguous").sum()
        ),
        "status_counts": positives["geo_resolution_status"].value_counts(dropna=False).to_dict(),
        "scale_decision_source_counts": positives["scale_decision_source"].value_counts(dropna=False).to_dict(),
    }

    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        pd.DataFrame([summary]).to_excel(writer, sheet_name="Summary", index=False)
        out.to_excel(writer, sheet_name="All_Rows", index=False)
        positives[positives["geo_resolution_status"].eq("unresolved_or_ambiguous")].to_excel(
            writer,
            sheet_name="Unresolved_Review",
            index=False,
        )
        positives[positives["scale_decision_source"].eq("mapping_override_llm")].to_excel(
            writer,
            sheet_name="Scale_Overrides",
            index=False,
        )
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate gazetteer resolution over spatial output.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-xlsx", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    summary = evaluate(args.input, args.output_xlsx, args.output_json)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
