# Evaluate 脚本逻辑修正计划

根据您的最新指示：“如果同时为1或者同时为0，diff输出1，否则都是0”。

这意味着：
1.  **Truth=1, Pred=1** -> **Diff=1**
2.  **Truth=0, Pred=0** -> **Diff=1**
3.  **Truth=1, Pred=0** -> **Diff=0**
4.  **Truth=1, Pred=待确定** -> **Diff=0**
5.  **Truth=待确定, Pred=待确定** -> **Diff=0** (因为不是“同时为1或同时为0”)

**新逻辑**:
只有当 Truth 和 Pred **都为 1** 或 **都为 0** 时，Diff 才为 1。其他任何情况（包括一方或双方为“待确定”）均为 0。

这实际上将是一个**严格的二元匹配**，排除了所有非二元状态的得分。

## 实施步骤
修改 `scripts/evaluate.py` 中的 Diff 计算逻辑：

```python
# 归一化处理（处理 1.0, "1", 1 等）
def normalize(val):
    s = str(val).strip()
    if s.replace('.0', '') == '1': return 1
    if s.replace('.0', '') == '0': return 0
    return -1 # 其他情况（待确定、None、空等）

# 应用归一化
t_norm = merged[t_col].apply(normalize)
p_norm = merged[p_col].apply(normalize)

# 核心逻辑：(Truth == Pred) AND (Pred IN [0, 1])
# 因为如果 Pred 是 0 或 1，且 Truth == Pred，那么 Truth 也必然是 0 或 1。
# 这就满足了 "同时为1 或 同时为0" 的条件。
condition = (t_norm == p_norm) & (p_norm.isin([0, 1]))

merged[diff_col_name] = np.where(condition, 1, 0)
```

这个逻辑非常清晰且健壮，能完美覆盖您的要求。
