# Urban Renovation 项目结构迁移方案

本文档记录当前项目的目标目录、模块职责、旧路径到新路径映射、保留清单、缓存清理范围和验证命令。迁移原则是不新增业务能力、不改变稳定分类流程、不改变既有 Excel 合同，并保留旧 CLI 与旧 import 的兼容入口。

## 目标目录树

```text
src/
  runtime/              # 配置、路径、LLM client、会话记忆
  prompting/            # prompt 生成、策略注册、manifest
  tasks/                # 任务路由、输入输出合并、urban/spatial 任务执行
    urban/
    spatial/
  urban/                # 城市更新识别核心
    taxonomy/
    rules/
    topic_model/
    hybrid/
    dynamic/
    specter2/
  spatial/              # 空间尺度/空间信息抽取核心
  evaluation/           # 对齐、指标、统计检验、summary sheets
  reporting/            # 报告、review workbook、指标命名
  templates/            # urban/spatial prompt templates

scripts/
  pipeline/             # 主运行入口、稳定发布入口
  evaluation/           # 官方评估脚本
  analysis/
    urban/
    spatial/
    specter2/
  reporting/
  data/
  dev/
  security/
  reference/

tests/
  urban/
  spatial/
  evaluation/
  pipeline/
  runtime/
  prompting/
  reporting/
  fixtures/

doc/
  current/              # 当前合同、运行说明、论文方法文档
  archives/             # 历史实验归档
```

## 模块职责

| 目录 | 职责 |
| --- | --- |
| `src/runtime/` | 集中维护运行配置、路径约定、LLM client 和会话记忆。 |
| `src/prompting/` | 维护 prompt 生成、策略注册、prompt manifest，不承载任务业务逻辑。 |
| `src/tasks/` | 负责任务路由、批处理输入输出、urban/spatial 结果合并。 |
| `src/urban/taxonomy/` | 固定城市更新主题体系、主题名、主题组、打分定义。 |
| `src/urban/rules/` | 元数据规则过滤、边界排除规则、硬负类规则。 |
| `src/urban/topic_model/` | BERTopic 服务、本地主题分类器、主题族 gate。 |
| `src/urban/hybrid/` | 三阶段 hybrid 分类器与二分类最终策略。 |
| `src/urban/dynamic/` | 动态主题发现和动态二分类修正。 |
| `src/urban/specter2/` | SPECTER2 离线 A/B 评估增强，默认不进入稳定流程。 |
| `src/spatial/` | 空间信息抽取策略和地理尺度解析。 |
| `src/evaluation/` | truth/pred 对齐、指标计算、评估输出结构。 |
| `src/reporting/` | 报告、review workbook、指标命名与展示。 |

## 迁移映射

| 旧路径 | 新路径 | 兼容方式 |
| --- | --- | --- |
| `src/tasks/task_router.py` | `src/tasks/router.py` | 旧文件保留 wrapper。 |
| `src/evaluation/core.py` | `src/evaluation/metrics.py` | 旧文件保留 wrapper。 |
| `src/strategies/spatial.py` | `src/spatial/extraction.py` | 旧文件保留 wrapper。 |
| `src/strategies/geo_resolver.py` | `src/spatial/geo_resolver.py` | 旧文件保留 wrapper。 |
| `src/urban/urban_topic_taxonomy.py` | `src/urban/taxonomy/core.py` | 旧文件保留 wrapper。 |
| `src/urban/urban_rule_filter.py` | `src/urban/rules/metadata_filter.py` | 旧文件保留 wrapper。 |
| `src/urban/urban_topic_classifier.py` | `src/urban/topic_model/local_classifier.py` | 旧文件保留 wrapper。 |
| `src/urban/urban_bertopic_service.py` | `src/urban/topic_model/bertopic_service.py` | 旧文件保留 wrapper。 |
| `src/urban/urban_family_gate.py` | `src/urban/topic_model/family_gate.py` | 旧文件保留 wrapper。 |
| `src/urban/urban_hybrid_classifier.py` | `src/urban/hybrid/classifier.py` | 旧文件保留 wrapper。 |
| `src/urban/urban_binary_policy_v2.py` | `src/urban/hybrid/binary_policy_v2.py` | 旧文件保留 wrapper。 |
| `src/urban/dynamic_topic_discovery.py` | `src/urban/dynamic/topic_discovery.py` | 旧文件保留 wrapper。 |
| `src/urban/dynamic_binary_refinement.py` | `src/urban/dynamic/binary_refinement.py` | 旧文件保留 wrapper。 |
| `scripts/analysis/evaluate_specter2_urban_ablation.py` | `scripts/analysis/specter2/evaluate_urban_ablation.py` | 旧脚本保留 wrapper。 |
| `scripts/analysis/*spatial*.py` | `scripts/analysis/spatial/*.py` | 旧脚本保留 wrapper。 |

## 禁止删除清单

- `Data/`
- `Result/`
- `output/`
- `doc/` 与 `docs/` 中的人工文档
- `.venv-bertopic313/`
- `scripts/main.py`
- `scripts/pipeline/main_py313.py`
- `scripts/pipeline/run_stable_release.py`
- 根级兼容 import：`src.config`、`src.task_router`、`src.urban_hybrid_classifier`、`src.urban_rule_filter`、`src.urban_topic_taxonomy`
- 稳定发布预测、评估 workbook、SPECTER2 本地缓存和模型缓存

## 可清理缓存

只清理不会影响业务结果的临时缓存：

- `__pycache__/`
- `.pytest_cache/`
- `.pytest-tmp/`
- `pytest-cache-files-*`
- `.codex-tmp/`

## 验证命令

```powershell
.venv-bertopic313\Scripts\python.exe -m compileall src scripts tests
.venv-bertopic313\Scripts\python.exe -m pytest tests -q
.venv-bertopic313\Scripts\python.exe scripts\main.py --help
.venv-bertopic313\Scripts\python.exe scripts\evaluation\evaluate.py --help
.venv-bertopic313\Scripts\python.exe scripts\pipeline\run_stable_release.py --skip-classification
```
