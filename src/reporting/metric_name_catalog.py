from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from ..runtime.config import Schema


@dataclass(frozen=True)
class MetricNameSpec:
    stage: str
    source_field: str
    display_name: str
    definition: str
    used_for_final_binary: bool = False


METRIC_NAME_SPECS: tuple[MetricNameSpec, ...] = (
    MetricNameSpec("输入信息", Schema.TITLE, "输入文章标题(Article Title)", "文献标题，用于规则、主题和裁决证据。"),
    MetricNameSpec("输入信息", Schema.ABSTRACT, "输入摘要(Abstract)", "文献摘要，用于判断研究对象、更新动作和风险语境。"),
    MetricNameSpec("输入信息", Schema.AUTHOR_KEYWORDS, "输入作者关键词(Author Keywords)", "作者关键词，用于主题识别和动态主题命名。"),
    MetricNameSpec("输入信息", Schema.KEYWORDS_PLUS, "输入扩展关键词(Keywords Plus)", "扩展关键词，用于补充主题证据。"),
    MetricNameSpec("输入信息", Schema.WOS_CATEGORIES, "输入学科类别(WoS Categories)", "Web of Science 学科类别背景。"),
    MetricNameSpec("输入信息", Schema.RESEARCH_AREAS, "输入研究领域(Research Areas)", "研究领域背景。"),
    MetricNameSpec("输入信息", "Publication Year", "输入发表年份(Publication Year)", "文献发表年份。"),
    MetricNameSpec("最终二分类", Schema.IS_URBAN_RENEWAL, "最终二分类标签(是否属于城市更新研究)", "城市更新二分类最终标签。", True),
    MetricNameSpec("最终二分类", "final_label", "最终二分类标签(final_label)", "统一后的最终二分类标签。", True),
    MetricNameSpec("最终二分类", "urban_flag", "最终二分类标志(urban_flag)", "与 final_label 保持一致的运行标志。", True),
    MetricNameSpec("最终二分类", "confidence", "最终二分类置信度(confidence)", "城市更新判定置信度或规则融合置信度。"),
    MetricNameSpec("最终二分类", "Predicted Positive Rate", "预测正例率(Predicted Positive Rate)", "预测为城市更新研究的样本比例。"),
    MetricNameSpec("最终二分类", "Truth Positive Rate", "真值正例率(Truth Positive Rate)", "有标签数据中的城市更新正例比例。"),
    MetricNameSpec("空间提取", Schema.IS_SPATIAL, "空间研究标签(空间研究/非空间研究)", "是否属于空间研究。"),
    MetricNameSpec("空间提取", Schema.SPATIAL_LEVEL, "空间等级标签(空间等级)", "空间研究尺度等级。"),
    MetricNameSpec("空间提取", Schema.SPATIAL_DESC, "具体空间描述(具体空间描述)", "抽取到的具体空间对象。"),
    MetricNameSpec("空间提取", "Reasoning", "空间提取依据(Reasoning)", "空间判断依据。"),
    MetricNameSpec("空间提取", "Confidence", "空间提取置信度(Confidence)", "空间判断置信度。"),
    MetricNameSpec("空间提取", Schema.SPATIAL_VALIDATION_STATUS, "空间验证状态(spatial_validation_status)", "空间结果验证状态。"),
    MetricNameSpec("空间提取", Schema.SPATIAL_VALIDATION_REASON, "空间验证原因(spatial_validation_reason)", "空间验证原因。"),
    MetricNameSpec("空间提取", Schema.SPATIAL_AREA_EVIDENCE, "空间区域证据(spatial_area_evidence)", "支持空间抽取的区域证据。"),
    MetricNameSpec("固定主题体系", "topic_final", "固定主题标签(topic_final)", "固定 topic taxonomy 输出的最终主题标签。", True),
    MetricNameSpec("固定主题体系", "topic_final_group", "固定主题组(topic_final_group)", "固定主题所属组：urban、nonurban 或 unknown。", True),
    MetricNameSpec("固定主题体系", "topic_final_name", "固定主题名称(topic_final_name)", "固定主题英文名称。"),
    MetricNameSpec("固定主题体系", "topic_final_name_en", "固定主题英文名(topic_final_name_en)", "固定主题英文名称。"),
    MetricNameSpec("固定主题体系", "topic_final_name_zh", "固定主题中文名(topic_final_name_zh)", "固定主题中文名称。"),
    MetricNameSpec("固定主题体系", "taxonomy_coverage_status", "主题覆盖状态(taxonomy_coverage_status)", "样本与固定 taxonomy 的覆盖关系。", True),
    MetricNameSpec("规则证据链", "decision_explanation", "规则判定说明(decision_explanation)", "最终二分类的规则解释文本。"),
    MetricNameSpec("规则证据链", "primary_positive_evidence", "主要正向证据(primary_positive_evidence)", "支持城市更新判定的主要证据。"),
    MetricNameSpec("规则证据链", "primary_negative_evidence", "主要负向证据(primary_negative_evidence)", "排除城市更新判定的主要证据。"),
    MetricNameSpec("规则证据链", "evidence_balance", "证据倾向(evidence_balance)", "正负证据综合倾向。", True),
    MetricNameSpec("规则证据链", "decision_rule_stack", "规则链路(decision_rule_stack)", "判定过程中触发的规则路径。"),
    MetricNameSpec("规则证据链", "binary_decision_evidence", "二分类证据(binary_decision_evidence)", "二分类打分和阈值依据。"),
    MetricNameSpec("规则证据链", "urban_probability_score", "城市更新概率分(urban_probability_score)", "城市更新二分类概率或规则融合分。", True),
    MetricNameSpec("规则证据链", "binary_decision_threshold", "二分类阈值(binary_decision_threshold)", "当前二分类判定阈值。"),
    MetricNameSpec("规则证据链", "binary_decision_source", "二分类判定来源(binary_decision_source)", "基础二分类的来源链。", True),
    MetricNameSpec("规则证据链", "unknown_recovery_path", "Unknown恢复路径(unknown_recovery_path)", "Unknown 样本的本地恢复路径。"),
    MetricNameSpec("规则证据链", "unknown_recovery_evidence", "Unknown恢复证据(unknown_recovery_evidence)", "Unknown 恢复所依据的证据。"),
    MetricNameSpec("规则证据链", "review_flag", "复核标志(review_flag)", "是否触发复核。"),
    MetricNameSpec("规则证据链", "review_reason", "复核原因(review_reason)", "触发复核的原因。"),
    MetricNameSpec("动态主题层", "dynamic_topic_id", "动态主题编号(dynamic_topic_id)", "动态主题簇编号。"),
    MetricNameSpec("动态主题层", "dynamic_topic_name_zh", "动态主题名称(dynamic_topic_name_zh)", "规则生成的动态主题中文名。"),
    MetricNameSpec("动态主题层", "dynamic_topic_keywords", "动态主题关键词(dynamic_topic_keywords)", "动态主题簇关键词。"),
    MetricNameSpec("动态主题层", "dynamic_topic_size", "动态主题样本量(dynamic_topic_size)", "动态主题簇样本数量。"),
    MetricNameSpec("动态主题层", "dynamic_topic_confidence", "动态主题置信度(dynamic_topic_confidence)", "动态主题稳定性和映射置信度。"),
    MetricNameSpec("动态主题层", "dynamic_topic_source_pool", "动态主题来源池(dynamic_topic_source_pool)", "动态主题样本来源池。"),
    MetricNameSpec("动态主题层", "dynamic_to_fixed_topic_candidate", "动态主题固定映射候选(dynamic_to_fixed_topic_candidate)", "动态主题建议映射到的固定主题。"),
    MetricNameSpec("动态主题层", "dynamic_mapping_status", "动态主题映射状态(dynamic_mapping_status)", "动态主题与固定 taxonomy 的映射状态。", True),
    MetricNameSpec("动态二分类修复", "dynamic_binary_candidate_label", "动态二分类候选标签(dynamic_binary_candidate_label)", "动态主题建议的二分类标签。"),
    MetricNameSpec("动态二分类修复", "dynamic_binary_candidate_confidence", "动态二分类候选置信度(dynamic_binary_candidate_confidence)", "动态二分类候选置信度。"),
    MetricNameSpec("动态二分类修复", "dynamic_binary_candidate_action", "动态二分类候选动作(dynamic_binary_candidate_action)", "动态二分类候选动作。"),
    MetricNameSpec("动态二分类修复", "dynamic_binary_candidate_reason", "动态二分类候选原因(dynamic_binary_candidate_reason)", "动态二分类候选依据。"),
    MetricNameSpec("动态二分类修复", "dynamic_binary_review_priority", "动态二分类复核优先级(dynamic_binary_review_priority)", "动态二分类复核优先级。"),
    MetricNameSpec("动态二分类修复", "dynamic_binary_override_applied", "动态二分类修复是否应用(dynamic_binary_override_applied)", "动态二分类 refine 是否实际覆盖字段。", True),
    MetricNameSpec("动态二分类修复", "dynamic_binary_override_label", "动态二分类修复标签(dynamic_binary_override_label)", "动态二分类 refine 覆盖后的标签。", True),
    MetricNameSpec("动态二分类修复", "dynamic_binary_override_topic", "动态二分类修复主题(dynamic_binary_override_topic)", "动态二分类 refine 覆盖后的主题。"),
    MetricNameSpec("动态二分类修复", "dynamic_binary_override_reason", "动态二分类修复原因(dynamic_binary_override_reason)", "动态二分类 refine 覆盖原因。"),
    MetricNameSpec("动态二分类修复", "dynamic_binary_override_source", "动态二分类修复来源(dynamic_binary_override_source)", "动态二分类 refine 来源。"),
    MetricNameSpec("策略与LLM裁决", "binary_policy_action", "二分类策略动作(binary_policy_action)", "UrbanBinaryPolicyV2 的最终策略动作。", True),
    MetricNameSpec("策略与LLM裁决", "binary_policy_reason", "二分类策略原因(binary_policy_reason)", "UrbanBinaryPolicyV2 的策略解释。"),
    MetricNameSpec("策略与LLM裁决", "binary_policy_conflict_type", "二分类冲突类型(binary_policy_conflict_type)", "冲突样本的冲突类型。", True),
    MetricNameSpec("策略与LLM裁决", "llm_adjudication_required", "LLM裁决需求(llm_adjudication_required)", "是否需要 LLM 困难样本裁决。"),
    MetricNameSpec("策略与LLM裁决", "llm_adjudication_label", "LLM裁决标签(llm_adjudication_label)", "LLM 输出的二分类裁决标签。"),
    MetricNameSpec("策略与LLM裁决", "llm_adjudication_confidence", "LLM裁决置信度(llm_adjudication_confidence)", "LLM 裁决置信度。"),
    MetricNameSpec("策略与LLM裁决", "llm_adjudication_reason", "LLM裁决原因(llm_adjudication_reason)", "LLM 裁决解释。"),
    MetricNameSpec("策略与LLM裁决", "llm_used", "LLM裁决实际使用(llm_used)", "LLM 是否实际覆盖最终结果。"),
    MetricNameSpec("策略与LLM裁决", "llm_attempted", "LLM裁决尝试计数(llm_attempted)", "LLM 是否被尝试调用。"),
    MetricNameSpec("评估指标", "File", "结果文件(File)", "被评估的预测文件。"),
    MetricNameSpec("评估指标", "Metric", "评估任务(Metric)", "被评估的任务或指标名称。"),
    MetricNameSpec("评估指标", "Total", "样本总数(Total)", "参与统计的样本数。"),
    MetricNameSpec("评估指标", "Correct", "正确数(Correct)", "预测正确的样本数。"),
    MetricNameSpec("评估指标", "Accuracy", "准确率(Accuracy)", "正确预测占总样本比例。"),
    MetricNameSpec("评估指标", "Precision", "精确率(Precision)", "预测正例中的真实正例比例。"),
    MetricNameSpec("评估指标", "Recall", "召回率(Recall)", "真实正例中被预测为正例的比例。"),
    MetricNameSpec("评估指标", "F1", "F1值(F1)", "Precision 和 Recall 的调和平均。"),
    MetricNameSpec("评估指标", "TP", "真正例(TP)", "真实正例且预测正例的样本数。"),
    MetricNameSpec("评估指标", "TN", "真负例(TN)", "真实负例且预测负例的样本数。"),
    MetricNameSpec("评估指标", "FP", "假正例(FP)", "真实负例但预测正例的样本数。"),
    MetricNameSpec("评估指标", "FN", "假负例(FN)", "真实正例但预测负例的样本数。"),
    MetricNameSpec("评估指标", "Count", "数量(Count)", "分组样本数。"),
    MetricNameSpec("评估指标", "Share", "占比(Share)", "分组样本占比。"),
)


DISPLAY_NAME_BY_SOURCE = {spec.source_field: spec.display_name for spec in METRIC_NAME_SPECS}


def display_name_for_field(source_field: str) -> str:
    return DISPLAY_NAME_BY_SOURCE.get(source_field, source_field)


def rename_columns_for_display(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    return frame.rename(
        columns={column: display_name_for_field(column) for column in frame.columns}
    )


def metric_dictionary_frame(source_fields: Iterable[str] | None = None) -> pd.DataFrame:
    selected = set(source_fields or [])
    specs = [
        spec
        for spec in METRIC_NAME_SPECS
        if not selected or spec.source_field in selected or spec.display_name in selected
    ]
    return pd.DataFrame(
        [
            {
                "stage": spec.stage,
                "display_name": spec.display_name,
                "source_field": spec.source_field,
                "definition": spec.definition,
                "used_for_final_binary": spec.used_for_final_binary,
            }
            for spec in specs
        ],
        columns=[
            "stage",
            "display_name",
            "source_field",
            "definition",
            "used_for_final_binary",
        ],
    )
