import numpy as np
import pandas as pd

from scripts.analysis.evaluate_specter2_urban_ablation import run_ablation
from src.runtime.config import Schema
from src.urban.specter2.encoder import EncodingResult, Specter2Encoder
from src.urban.specter2.evaluator import evaluate_specter2_ablation


def test_small_fixture_generates_baseline_specter2_only_and_hybrid_metrics():
    truth = pd.Series([1, 1, 1, 1, 0, 0, 0, 0])
    baseline_pred = pd.Series([1, 0, 1, 1, 0, 1, 0, 0])
    embeddings = np.array(
        [
            [1.00, 0.05],
            [0.95, 0.10],
            [0.90, 0.15],
            [0.85, 0.20],
            [0.10, 0.90],
            [0.15, 0.85],
            [0.20, 0.80],
            [0.25, 0.75],
        ],
        dtype=np.float32,
    )

    result = evaluate_specter2_ablation(
        truth=truth,
        baseline_pred=baseline_pred,
        embeddings=embeddings,
        random_state=20260513,
    )

    metrics = result.metrics
    assert set(metrics["Group"]) == {"baseline", "specter2_only", "hybrid"}
    assert set(["TP", "TN", "FP", "FN", "Precision", "Recall", "F1"]).issubset(metrics.columns)
    assert metrics["Total"].tolist() == [8, 8, 8]
    assert {"baseline_pred", "specter2_only_pred", "hybrid_pred"}.issubset(result.predictions.columns)


def test_ablation_script_writes_baseline_outputs_when_specter2_is_unavailable(tmp_path, monkeypatch):
    truth_path = tmp_path / "truth.xlsx"
    pred_path = tmp_path / "pred.xlsx"
    output_dir = tmp_path / "out"
    pd.DataFrame(
        {
            Schema.TITLE: ["A", "B", "C"],
            Schema.ABSTRACT: ["Urban renewal", "Transit model", "Brownfield reuse"],
            Schema.IS_URBAN_RENEWAL: [1, 0, 1],
        }
    ).to_excel(truth_path, index=False)
    pd.DataFrame(
        {
            Schema.TITLE: ["A", "B", "C"],
            Schema.IS_URBAN_RENEWAL: [1, 0, 0],
        }
    ).to_excel(pred_path, index=False)

    def fake_encode(self, records):
        return EncodingResult(
            status="specter2_unavailable",
            embeddings=np.zeros((0, 0), dtype=np.float32),
            reason="missing test dependency",
        )

    monkeypatch.setattr(Specter2Encoder, "encode", fake_encode)

    manifest = run_ablation(
        truth_workbook=truth_path,
        prediction_workbook=pred_path,
        output_dir=output_dir,
    )

    assert manifest["encoding_status"] == "specter2_unavailable"
    assert manifest["groups_evaluated"] == ["baseline"]
    assert (output_dir / "Specter2_Ablation_Summary.xlsx").exists()
    assert (output_dir / "specter2_ablation_manifest.json").exists()
