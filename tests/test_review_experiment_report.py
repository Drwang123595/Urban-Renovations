from __future__ import annotations

from pathlib import Path

import pandas as pd
from docx import Document

from src.review_experiment_report import REPORT_TITLE, generate_review_experiment_report


def _build_sample_review_frame() -> pd.DataFrame:
    urban_flag = "\u9884\u6d4b_\u662f\u5426\u5c5e\u4e8e\u57ce\u5e02\u66f4\u65b0\u7814\u7a76"
    spatial_flag = "\u9884\u6d4b_\u7a7a\u95f4\u7814\u7a76/\u975e\u7a7a\u95f4\u7814\u7a76"
    spatial_level = "\u9884\u6d4b_\u7a7a\u95f4\u7b49\u7ea7"
    space_detail = "\u9884\u6d4b_\u5177\u4f53\u7a7a\u95f4\u63cf\u8ff0"
    urban_confidence = "\u57ce\u5e02\u66f4\u65b0\u5224\u5b9a\u7f6e\u4fe1\u5ea6(confidence)"
    reasoning = "\u7a7a\u95f4\u63d0\u53d6\u4f9d\u636e(Reasoning)"
    spatial_confidence = "\u7a7a\u95f4\u63d0\u53d6\u7f6e\u4fe1\u5ea6(Confidence)"

    return pd.DataFrame(
        [
            {
                "Article Title": "Paper A",
                "Publication Year": 2001,
                "Keywords Plus": "renewal",
                "Abstract": "A",
                "WoS Categories": "Urban Studies",
                "Research Areas": "Geography",
                urban_flag: "1",
                spatial_flag: "1",
                spatial_level: "7. Single-city / Municipal Scale",
                space_detail: "Shanghai",
                "topic_final": "U1",
                "topic_final_name_en": "old neighborhood renewal",
                "topic_final_name_zh": "\u8001\u65e7\u4f4f\u533a\u66f4\u65b0",
                urban_confidence: 0.95,
                reasoning: "district level evidence",
                spatial_confidence: "High",
                "review_flag": 0,
                "review_reason": "",
            },
            {
                "Article Title": "Paper B",
                "Publication Year": 2003,
                "Keywords Plus": "governance",
                "Abstract": "B",
                "WoS Categories": "Urban Studies",
                "Research Areas": "Policy",
                urban_flag: "0",
                spatial_flag: "0",
                spatial_level: "Not mentioned",
                space_detail: "Not mentioned",
                "topic_final": "N3",
                "topic_final_name_en": "general urban governance",
                "topic_final_name_zh": "\u4e00\u822c\u57ce\u5e02\u6cbb\u7406",
                urban_confidence: 0.68,
                reasoning: "",
                spatial_confidence: "",
                "review_flag": 0,
                "review_reason": "",
            },
            {
                "Article Title": "Paper C",
                "Publication Year": 2018,
                "Keywords Plus": "gentrification",
                "Abstract": "C",
                "WoS Categories": "Urban Studies",
                "Research Areas": "Sociology",
                urban_flag: "1",
                spatial_flag: "1",
                spatial_level: "8. District / County Scale",
                space_detail: "Shenzhen",
                "topic_final": "U12",
                "topic_final_name_en": "gentrification exclusion and neighborhood change",
                "topic_final_name_zh": "\u7ec5\u58eb\u5316\u3001\u6392\u65a5\u4e0e\u793e\u533a\u53d8\u5316",
                urban_confidence: 0.89,
                reasoning: "district scale evidence",
                spatial_confidence: "High",
                "review_flag": 0,
                "review_reason": "",
            },
            {
                "Article Title": "Paper D",
                "Publication Year": 2019,
                "Keywords Plus": "policy",
                "Abstract": "D",
                "WoS Categories": "Urban Studies",
                "Research Areas": "Economics",
                urban_flag: "1",
                spatial_flag: "1",
                spatial_level: "3. National / Single-country Scale",
                space_detail: "China",
                "topic_final": "U10",
                "topic_final_name_en": "renewal finance and policy tools",
                "topic_final_name_zh": "\u66f4\u65b0\u878d\u8d44\u4e0e\u653f\u7b56\u5de5\u5177",
                urban_confidence: 0.93,
                reasoning: "national evidence",
                spatial_confidence: "Medium",
                "review_flag": 0,
                "review_reason": "",
            },
            {
                "Article Title": "Paper E",
                "Publication Year": 2020,
                "Keywords Plus": "gentrification",
                "Abstract": "E",
                "WoS Categories": "Urban Studies",
                "Research Areas": "Sociology",
                urban_flag: "1",
                spatial_flag: "1",
                spatial_level: "7. Single-city / Municipal Scale",
                space_detail: "Hong Kong",
                "topic_final": "U12",
                "topic_final_name_en": "gentrification exclusion and neighborhood change",
                "topic_final_name_zh": "\u7ec5\u58eb\u5316\u3001\u6392\u65a5\u4e0e\u793e\u533a\u53d8\u5316",
                urban_confidence: 0.92,
                reasoning: "city scale evidence",
                spatial_confidence: "High",
                "review_flag": 0,
                "review_reason": "",
            },
            {
                "Article Title": "Paper F",
                "Publication Year": 2022,
                "Keywords Plus": "renewal",
                "Abstract": "F",
                "WoS Categories": "Urban Studies",
                "Research Areas": "Geography",
                urban_flag: "1",
                spatial_flag: "1",
                spatial_level: "9. Micro / Neighborhood / Block Scale",
                space_detail: "Shanghai",
                "topic_final": "U1",
                "topic_final_name_en": "old neighborhood renewal",
                "topic_final_name_zh": "\u8001\u65e7\u4f4f\u533a\u66f4\u65b0",
                urban_confidence: 0.91,
                reasoning: "block level evidence",
                spatial_confidence: "High",
                "review_flag": 0,
                "review_reason": "",
            },
        ]
    )


def test_generate_review_experiment_report_creates_docx_with_images(tmp_path):
    workbook_path = tmp_path / "review.xlsx"
    output_path = tmp_path / "report.docx"
    _build_sample_review_frame().to_excel(workbook_path, sheet_name="Sheet1", index=False, engine="openpyxl")

    result = generate_review_experiment_report(workbook_path, output_path=output_path)

    assert result == output_path
    assert output_path.exists()

    document = Document(output_path)
    full_text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    assert REPORT_TITLE in full_text
    assert "核心结论" in full_text
    assert "图 1 Year-Topic Share" in full_text
    assert len(document.inline_shapes) >= 6
