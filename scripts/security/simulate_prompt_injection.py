import sys
import argparse
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prompting.generator import PromptGenerator
from src.prompting.strategy_registry import PromptStrategyRegistry
from src.runtime.config import Config
from src.runtime.memory import ConversationMemory


def _collect_common_strategy_names(registry: PromptStrategyRegistry):
    themes = ["urban_renewal", "spatial"]
    enabled_by_theme = []
    for theme in themes:
        enabled_by_theme.append(set(registry.list_enabled_strategies(theme=theme)))
    common = sorted(enabled_by_theme[0].intersection(enabled_by_theme[1]))
    return common


def _collect_registry_rows(registry: PromptStrategyRegistry):
    rows = []
    for definition in registry.strategies.values():
        rows.append(
            {
                "key": definition.key,
                "theme": definition.theme,
                "name": definition.name,
                "template_file": definition.template_file,
                "enabled": definition.enabled,
            }
        )
    rows.sort(key=lambda item: item["key"])
    return rows


def _build_urban_messages(shot_mode: str, title: str, abstract: str):
    prompt_gen = PromptGenerator(shot_mode=shot_mode, default_theme="urban_renewal")
    system_prompt = prompt_gen.get_step_system_prompt()
    memory = ConversationMemory(system_prompt=system_prompt, skip_index=True)
    user_prompt = prompt_gen.get_step_prompt(1, title, abstract, include_context=True)
    memory.add_user_message(user_prompt)
    return {
        "theme": "urban_renewal",
        "shot_mode": shot_mode,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "messages": memory.get_messages(),
    }


def _build_spatial_messages(shot_mode: str, title: str, abstract: str):
    prompt_gen = PromptGenerator(shot_mode=shot_mode, default_theme="spatial")
    system_prompt = prompt_gen.get_spatial_system_prompt()
    memory = ConversationMemory(system_prompt=system_prompt, skip_index=True)
    user_prompt = prompt_gen.get_spatial_user_prompt(title, abstract)
    memory.add_user_message(user_prompt)
    return {
        "theme": "spatial",
        "shot_mode": shot_mode,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "messages": memory.get_messages(),
    }


def _format_messages(messages):
    lines = []
    for index, item in enumerate(messages, start=1):
        role = item.get("role", "")
        content = item.get("content", "")
        lines.append(f"{index}. role={role}")
        lines.append(content)
        lines.append("")
    return "\n".join(lines).strip()


def _summarize_diff(urban_payload, spatial_payload):
    urban_system = urban_payload["system_prompt"]
    spatial_system = spatial_payload["system_prompt"]
    urban_user = urban_payload["user_prompt"]
    spatial_user = spatial_payload["user_prompt"]

    summary = {
        "system_len_urban": len(urban_system),
        "system_len_spatial": len(spatial_system),
        "user_len_urban": len(urban_user),
        "user_len_spatial": len(spatial_user),
        "same_system": urban_system == spatial_system,
        "same_user": urban_user == spatial_user,
        "messages_roles_same": [m["role"] for m in urban_payload["messages"]]
        == [m["role"] for m in spatial_payload["messages"]],
        "urban_system_head": urban_system.strip().splitlines()[0] if urban_system.strip() else "",
        "spatial_system_head": spatial_system.strip().splitlines()[0] if spatial_system.strip() else "",
        "urban_user_head": urban_user.strip().splitlines()[0] if urban_user.strip() else "",
        "spatial_user_head": spatial_user.strip().splitlines()[0] if spatial_user.strip() else "",
    }
    return summary


def generate_injection_audit_md(
    output_path: Path,
    urban_shot: str = None,
    spatial_shot: str = None,
    compare_all: bool = True,
):
    registry_path = PROJECT_ROOT / "src" / "templates" / "strategy_registry.yaml"
    registry = PromptStrategyRegistry.load_from_file(registry_path)
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    title = "Urban renewal and neighborhood change in Shenzhen"
    abstract = "This study evaluates regeneration interventions in old neighborhoods and compares spatial outcomes across districts."

    default_shot = Config.DEFAULT_SHOT_MODE
    urban_shot = urban_shot or default_shot
    spatial_shot = spatial_shot or default_shot

    common_strategy_names = _collect_common_strategy_names(registry) if compare_all else []
    urban = _build_urban_messages(urban_shot, title, abstract)
    spatial = _build_spatial_messages(spatial_shot, title, abstract)
    urban_prompt_gen = PromptGenerator(shot_mode=urban_shot, default_theme="urban_renewal")
    spatial_prompt_gen = PromptGenerator(shot_mode=spatial_shot, default_theme="spatial")
    urban_template = urban_prompt_gen.registry.get_template_file("urban_renewal", urban_prompt_gen.shot_mode)
    spatial_template = spatial_prompt_gen.registry.get_template_file("spatial", spatial_prompt_gen.shot_mode)

    compare_blocks = []
    if compare_all:
        for strategy_name in common_strategy_names:
            urban_payload = _build_urban_messages(strategy_name, title, abstract)
            spatial_payload = _build_spatial_messages(strategy_name, title, abstract)
            compare_blocks.append(
                {
                    "strategy": strategy_name,
                    "urban": urban_payload,
                    "spatial": spatial_payload,
                    "diff": _summarize_diff(urban_payload, spatial_payload),
                }
            )
    rows = _collect_registry_rows(registry)

    text = []
    text.append("# 两类提示词实际注入模拟报告（无模型调用）")
    text.append("")
    text.append("## 1. 运行摘要")
    text.append(f"- 生成时间: {run_time}")
    text.append("- 是否调用大模型: No")
    text.append("- LLM_CALLED: false")
    text.append("- 模拟范围: urban_renewal + spatial + both(串行顺序)")
    text.append(f"- 模拟模式: {'compare_all' if compare_all else 'single_effective'}")
    if compare_all:
        text.append(f"- 同名策略并排范围: {', '.join(common_strategy_names)}")
    else:
        text.append(f"- 本次策略: urban={urban_shot}, spatial={spatial_shot}")
    text.append("")
    text.append("## 1.1 本次生效策略证明")
    text.append(f"- urban requested_shot: {urban_shot}")
    text.append(f"- urban resolved_shot: {urban_prompt_gen.shot_mode}")
    text.append(f"- urban template_file: {urban_template}")
    text.append(f"- spatial requested_shot: {spatial_shot}")
    text.append(f"- spatial resolved_shot: {spatial_prompt_gen.shot_mode}")
    text.append(f"- spatial template_file: {spatial_template}")
    text.append("- 说明: 若 requested 与 resolved 不一致，表示发生了别名解析或非法回退。")
    text.append("")
    text.append("## 2. 注入执行链路")
    text.append("- 入口: scripts/pipeline/main_py313.py")
    text.append("- 路由: src/task_router.py")
    text.append("- 模板解析: src/prompts.py + src/templates/strategy_registry.yaml")
    text.append("- 消息组装: src/memory.py")
    text.append("")
    text.append("## 3. 城市更新注入明细")
    text.append(f"- 主题: {urban['theme']}")
    text.append(f"- 策略: {urban['shot_mode']}")
    text.append("")
    text.append("### 3.1 System Prompt")
    text.append("```text")
    text.append(urban["system_prompt"])
    text.append("```")
    text.append("")
    text.append("### 3.2 User Prompt")
    text.append("```text")
    text.append(urban["user_prompt"])
    text.append("```")
    text.append("")
    text.append("### 3.3 最终注入 messages")
    text.append("```text")
    text.append(_format_messages(urban["messages"]))
    text.append("```")
    text.append("")
    text.append("## 4. 空间分析注入明细")
    text.append(f"- 主题: {spatial['theme']}")
    text.append(f"- 策略: {spatial['shot_mode']}")
    text.append("")
    text.append("### 4.1 System Prompt")
    text.append("```text")
    text.append(spatial["system_prompt"])
    text.append("```")
    text.append("")
    text.append("### 4.2 User Prompt")
    text.append("```text")
    text.append(spatial["user_prompt"])
    text.append("```")
    text.append("")
    text.append("### 4.3 最终注入 messages")
    text.append("```text")
    text.append(_format_messages(spatial["messages"]))
    text.append("```")
    text.append("")
    text.append("## 5. both 模式执行顺序（模拟）")
    text.append("- Phase A: urban_renewal 注入并处理")
    text.append("- Phase B: spatial 注入并处理")
    text.append("- 隔离方式: strict serial")
    text.append("")
    if compare_all:
        text.append("## 6. 同名策略并排差异总表")
        text.append("| strategy | urban_system_len | spatial_system_len | urban_user_len | spatial_user_len | same_system | same_user | messages_roles_same |")
        text.append("|---|---:|---:|---:|---:|---|---|---|")
        for block in compare_blocks:
            diff = block["diff"]
            text.append(
                f"| {block['strategy']} | {diff['system_len_urban']} | {diff['system_len_spatial']} | {diff['user_len_urban']} | {diff['user_len_spatial']} | {diff['same_system']} | {diff['same_user']} | {diff['messages_roles_same']} |"
            )
        text.append("")
        text.append("## 7. 同名策略逐项并排")
        for block in compare_blocks:
            strategy = block["strategy"]
            urban_payload = block["urban"]
            spatial_payload = block["spatial"]
            diff = block["diff"]
            text.append(f"### 7.{common_strategy_names.index(strategy)+1} 策略 `{strategy}`")
            text.append("- urban_renewal 与 spatial 并排摘要")
            text.append("")
            text.append("| 维度 | urban_renewal | spatial |")
            text.append("|---|---|---|")
            text.append(f"| system 首行 | {diff['urban_system_head']} | {diff['spatial_system_head']} |")
            text.append(f"| user 首行 | {diff['urban_user_head']} | {diff['spatial_user_head']} |")
            text.append(f"| messages 角色序列一致 | {diff['messages_roles_same']} | {diff['messages_roles_same']} |")
            text.append("")
            text.append("#### Urban system/user 摘要")
            text.append("```text")
            text.append(urban_payload["system_prompt"][:800])
            text.append("")
            text.append("---")
            text.append(urban_payload["user_prompt"])
            text.append("```")
            text.append("")
            text.append("#### Spatial system/user 摘要")
            text.append("```text")
            text.append(spatial_payload["system_prompt"][:800])
            text.append("")
            text.append("---")
            text.append(spatial_payload["user_prompt"])
            text.append("```")
            text.append("")
            text.append("#### 差异结论")
            text.append(f"- system 是否完全相同: {diff['same_system']}")
            text.append(f"- user 是否完全相同: {diff['same_user']}")
            text.append("- 人工检查项: 关注任务目标是否独立、输出约束是否只服务本任务。")
            text.append("")
    else:
        text.append("## 6. 单策略生效检查")
        text.append("- 已关闭并排比较，仅验证预设策略是否真实生效。")
        text.append("- 检查 requested/resolved/template_file 三组字段。")
        text.append("")

    text.append("## 8. 注册表映射")
    text.append("| key | theme | name | template_file | enabled |")
    text.append("|---|---|---|---|---|")
    for row in rows:
        text.append(
            f"| {row['key']} | {row['theme']} | {row['name']} | {row['template_file']} | {row['enabled']} |"
        )
    text.append("")
    text.append("## 9. 人工复核提示")
    text.append("- 检查 urban_renewal 注入是否只使用 urban_renewal 目录模板")
    text.append("- 检查 spatial 注入是否只使用 spatial 目录模板")
    text.append("- 检查根目录 legacy 模板未进入本次注入链路")
    text.append("- 若怀疑默认 few-shot，请先看 1.1 的 requested/resolved/template_file 证据。")
    text.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(text), encoding="utf-8")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Simulate prompt injection without model calls")
    parser.add_argument("--urban-shot", type=str, default=None)
    parser.add_argument("--spatial-shot", type=str, default=None)
    parser.add_argument("--compare-all", action="store_true")
    parser.add_argument("--single-mode", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    compare_all = True
    if args.single_mode:
        compare_all = False
    elif args.compare_all:
        compare_all = True
    output = PROJECT_ROOT / "doc" / "两类提示词注入无模型模拟报告.md"
    path = generate_injection_audit_md(
        output_path=output,
        urban_shot=args.urban_shot,
        spatial_shot=args.spatial_shot,
        compare_all=compare_all,
    )
    print(f"Report generated: {path}")


if __name__ == "__main__":
    main()
