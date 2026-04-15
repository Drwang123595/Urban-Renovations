# Evaluate 脚本优化计划

您希望修改 `evaluate.py` 中 `Diff` 值的计算逻辑，特别是针对“待确定”或“不确定”的情况。

## 1. 需求分析
*   **目标**: 修改 `evaluate_single_file` 函数中的 Diff 计算逻辑。
*   **规则**:
    1.  **Truth == Pred**: Diff = 1
    2.  **Truth != Pred**: Diff = 0
    3.  **Pred == "待确定" (或类似的不确定状态)**: Diff = 0 (直接判定为错误/不匹配)

## 2. 代码现状 (`evaluate.py`)
目前代码 (Line 138) 使用的是简单的 numpy 比较：
```python
merged[diff_col_name] = np.where(truth_vals == pred_vals, 1, 0)
```
这对于“待确定”的情况，如果真值也是“待确定”，它会判为 1 (匹配)。如果真值是 0 或 1，它判为 0。

## 3. 修改方案
我们需要显式处理预测值为“待确定”的情况。

### 3.1 逻辑修改
```python
# 伪代码
if pred_val == "待确定":
    diff = 0
else:
    diff = 1 if truth_val == pred_val else 0
```

### 3.2 实施细节
在 `scripts/evaluate.py` 的循环中，针对每一行应用这个逻辑。考虑到 Pandas 的向量化操作，我们可以使用 `np.where` 的嵌套或者 `apply`。

**更稳健的写法**:
```python
# 1. 基础比较：是否相等
matches = (truth_vals == pred_vals)

# 2. 惩罚项：如果预测值是 "待确定"，则强制为 False
# 注意：truth_vals 和 pred_vals 此时可能是数字或字符串混合，需统一处理
# 对于 "Urban Renewal" 指标，pred_vals 可能是 1, 0, "待确定"

# 统一转为字符串进行比较（最安全）
t_str = merged[t_col].fillna("").astype(str).str.strip()
p_str = merged[p_col].fillna("").astype(str).str.strip()

# 逻辑：(Truth == Pred) AND (Pred != "待确定")
condition = (t_str == p_str) & (p_str != "待确定")

merged[diff_col_name] = np.where(condition, 1, 0)
```

**注意**: 用户提到的“不确定”可能包括 `待确定` 字符串。我们在之前的 Prompt 修改中统一使用了 `待确定` 这个词。

## 4. 实施步骤
1.  修改 `scripts/evaluate.py`。
2.  在计算 `metrics` 和 `diff_col_name` 之前，统一将列转换为字符串进行比较，以处理混合类型。
3.  应用新的 Diff 逻辑。

这样修改后，即使模型预测结果与真值一致（都为“待确定”），Diff 也会是 0。或者如果模型预测“待确定”而真值是 1，Diff 也是 0。符合“预测值为不确定时，直接就是 0”的要求。
