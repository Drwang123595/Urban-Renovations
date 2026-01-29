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
