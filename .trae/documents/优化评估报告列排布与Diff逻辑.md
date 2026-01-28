# 优化评估报告格式

根据您的需求，我们将调整 `evaluate.py` 生成的 Excel 报告中 "Detail Comparison" Sheet 的列排布和内容。

## 1. 需求分析

### 1.1 剔除冗余列

* 移除 `Article Title_pred`、`Abstract_pred`：因为我们是基于 Title 对齐的，且输入内容一致，只需保留真值文件中的 `Article Title` 和 `Abstract` 即可（通常合并后保留左侧的）。

* 移除 `_key`：这是内部对齐用的辅助列，用户不需要看到。

### 1.2 调整列顺序（相邻存放）

* 目标是将同一指标的 Truth 和 Pred 列放在一起。

* **期望顺序示例**：
  `Article Title`, `Abstract`,
  `是否属于城市更新研究_truth`, `是否属于城市更新研究_pred`, `Diff_Urban Renewal`,
  `空间研究/非空间研究_truth`, `空间研究/非空间研究_pred`, `Diff_Spatial Study`...

### 1.3 调整 Diff 值定义

* **当前**：不匹配 = 1，匹配 = 0。

* **新要求**：**正确匹配 = 1，不匹配 = 0**。
  *(注：用户原话是“正确匹配为1，不匹配为0”。请确认是否执行“匹配=1, 不匹配=0”。通常逻辑是1=Success, 0=Fail*

  * **确认**：用户明确说 "正确匹配为1，不匹配为0"。我们将严格执行此逻辑。

## 2. 修改计划 (`evaluate.py`)

### 2.1 修改 `evaluate_single_file` 函数

* **数据清洗**：

  * 在 `pd.merge` 之后，删除 `_key`。

  * 删除 `Article Title_pred` 和 `Abstract_pred`（如果存在）。

  * 重命名 `Article Title_truth` -> `Article Title`, `Abstract_truth` -> `Abstract`（如果 Pandas 自动加了后缀）。

* **Diff 计算逻辑更新**：

  * `detail_df[f"Diff_{desc}"] = np.where(truth_vals == pred_vals, 1, 0)`

* **列重排**：

  * 构建一个新的列顺序列表。

  * 首先放入基础列：`Article Title`, `Abstract`。

  * 然后遍历每个映射指标，依次放入 `[Truth列, Pred列, Diff列]`。

  * 最后放入其他未处理的列（如果有）。

## 3. 实施步骤

1. **修改代码**：更新 `evaluate.py` 中的 `evaluate_single_file` 函数。

