import argparse
import pandas as pd

def _to_binary(series):
    s = pd.to_numeric(series, errors="coerce")
    return s

def compute_score(df, truth_col, pred_col):
    truth = _to_binary(df[truth_col])
    pred = _to_binary(df[pred_col])
    valid = truth.notna() & pred.notna()
    if valid.sum() == 0:
        return 0.0, 0, 0
    correct = (truth[valid].astype(int) == pred[valid].astype(int)).sum()
    total = int(valid.sum())
    score = correct / total * 100.0
    return score, correct, total

def main():
    parser = argparse.ArgumentParser(description="Evaluate Urban Renewal Extraction Results")
    parser.add_argument("--input", default=r"Data\Result\Result1_实验对比评估.xlsx")
    parser.add_argument("--truth-col", default="是否属于城市更新研究(人工)")
    parser.add_argument("--single-col", default="是否属于城市更新研究（一次性）")
    parser.add_argument("--step-col", default="是否属于城市更新研究（分步骤）")
    parser.add_argument("--output", default=r"Data\Result\Result1_实验对比评估_评分.xlsx")
    args = parser.parse_args()

    df = pd.read_excel(args.input, engine="openpyxl")

    single_score, single_correct, single_total = compute_score(df, args.truth_col, args.single_col)
    step_score, step_correct, step_total = compute_score(df, args.truth_col, args.step_col)

    summary = pd.DataFrame(
        [
            {
                "指标": "一次性",
                "正确": single_correct,
                "总数": single_total,
                "得分(100分制)": round(single_score, 2),
            },
            {
                "指标": "分步骤",
                "正确": step_correct,
                "总数": step_total,
                "得分(100分制)": round(step_score, 2),
            },
        ]
    )

    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="原始数据")
        summary.to_excel(writer, index=False, sheet_name="评分结果")

    print(f"一次性：正确 {single_correct}/{single_total}，得分 {round(single_score,2)}")
    print(f"分步骤：正确 {step_correct}/{step_total}，得分 {round(step_score,2)}")
    print(f"结果已保存：{args.output}")

if __name__ == "__main__":
    main()
