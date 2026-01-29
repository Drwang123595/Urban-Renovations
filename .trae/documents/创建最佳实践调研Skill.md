# 创建新 Skill: `best-practice-researcher`

根据您的需求，我将创建一个名为 `best-practice-researcher` (最佳实践调研员) 的 Skill。

## Skill 定义

*   **名称**: `best-practice-researcher`
*   **功能**: 在编写代码前，强制执行“需求分析 -> 深度联网搜索 -> 案例对比 -> 方案制定”的流程。
*   **触发时机**: 当用户布置复杂的编程任务、功能开发或重构任务时。

## 实施步骤

1.  **创建目录**: `.trae/skills/best-practice-researcher/`
2.  **创建配置文件**: `SKILL.md`，内容如下：

```markdown
---
name: "best-practice-researcher"
description: "Before coding, extracts task keys, searches online for cases, compares mature schemes, and plans. Invoke for complex dev tasks."
---

# Best Practice Researcher

This skill enforces a "Research-First" workflow to ensure code quality and adherence to industry standards.

## Workflow

1.  **Extract Key Information**:
    *   Analyze the user's request to identify the core technical challenge, target technology stack, and constraints.
    *   Summarize the "Search Keywords" to be used.

2.  **Deep Web Search**:
    *   Use `WebSearch` to find existing implementations, libraries, or design patterns.
    *   Look for:
        *   GitHub repositories with similar features.
        *   Official documentation best practices.
        *   Tech blogs or StackOverflow discussions comparing solutions.

3.  **Case Comparison & Analysis**:
    *   Compare the found "Mature Schemes" against the current project context.
    *   Identify pros/cons of each approach.
    *   Select the most suitable solution.

4.  **Implementation Planning**:
    *   Draft a step-by-step coding plan based on the selected solution.
    *   Present this plan to the user for confirmation.

5.  **Execution**:
    *   Once approved, proceed with coding (using other tools).

## When to Invoke
*   When the user asks to "implement a new feature".
*   When the user asks to "refactor" a module.
*   When the user asks for "best practices" or "optimization".
*   **Do NOT invoke** for simple tasks like variable renaming or typo fixes.
```

## 预期效果
创建后，当您下次要求“帮我实现一个XX功能”时，系统会自动调用此 Skill，先去网上查资料、找案例，然后告诉您“我找到了业界通用的做法是XXX，我们准备这样写...”，最后才开始写代码。
