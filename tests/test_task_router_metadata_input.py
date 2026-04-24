import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import Schema
from src.task_router import TaskRouter


def test_read_input_preserves_optional_metadata_columns(tmp_path):
    input_path = tmp_path / "demo.xlsx"
    pd.DataFrame(
        [
            {
                Schema.TITLE: "Demo title",
                Schema.ABSTRACT: "Demo abstract",
                Schema.AUTHOR_KEYWORDS: "urban renewal; governance",
                Schema.KEYWORDS_PLUS: "redevelopment",
                Schema.WOS_CATEGORIES: "Urban Studies",
                Schema.RESEARCH_AREAS: "Public Administration",
            }
        ]
    ).to_excel(input_path, index=False, engine="openpyxl")

    router = TaskRouter.__new__(TaskRouter)
    df = TaskRouter._read_input(router, input_path)

    assert list(df.columns) == [
        Schema.TITLE,
        Schema.ABSTRACT,
        Schema.AUTHOR_KEYWORDS,
        Schema.KEYWORDS_PLUS,
        Schema.KEYWORDS,
        Schema.WOS_CATEGORIES,
        Schema.RESEARCH_AREAS,
    ]
    assert df.loc[0, Schema.AUTHOR_KEYWORDS] == "urban renewal; governance"

