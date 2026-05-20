from __future__ import annotations

import pandas as pd
import time
import re
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, Union
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from .merged_output import build_metric_dictionary_frame, build_review_ready_merged_frame, load_task_input_frame
from ..prompting.generator import PromptGenerator
from ..runtime.config import Config, Schema
from ..runtime.llm_client import DeepSeekClient, LLMQuotaExceededError
from ..runtime.memory import ConversationMemory
from ..runtime.project_paths import (
    build_merged_prediction_stem,
    build_spatial_prediction_stem,
    build_urban_prediction_stem,
    ensure_run_layout,
    run_paths,
)
from ..runtime.resume import ResumeCheckpoint, default_checkpoint_path, input_fingerprint
from ..spatial.extraction import SpatialExtractionStrategy
from ..strategies.stepwise_long import StepwiseLongContextStrategy
from ..urban.hybrid.classifier import UrbanHybridClassifier
from ..urban.core.metadata import UrbanMetadataRecord
from ..urban.topic_model.local_classifier import UrbanTopicClassifier
from ..urban.taxonomy.core import legacy_topic_for_label, urban_flag_for_topic_label
from ..urban.pipeline.io import (
    build_urban_output_row,
    build_urban_prediction_frame,
    write_urban_prediction_checkpoint,
)
from ..urban.pipeline.postprocess import postprocess_urban_predictions


class TaskType(Enum):
    URBAN_RENEWAL = "urban_renewal"
    SPATIAL = "spatial"
    BOTH = "both"


class UrbanMethod(Enum):
    PURE_LLM_API = "pure_llm_api"
    LOCAL_TOPIC_CLASSIFIER = "local_topic_classifier"
    THREE_STAGE_HYBRID = "three_stage_hybrid"


def _prediction_task_key(task_type: TaskType | str) -> str:
    raw_value = task_type.value if isinstance(task_type, TaskType) else str(task_type)
    task_key = raw_value.strip().lower()
    if task_key in {"urban_renewal", "spatial", "merged"}:
        return task_key
    raise ValueError(f"Unknown prediction task type: {task_type!r}")


class TaskRouter:
    def __init__(
        self,
        client: DeepSeekClient = None,
        urban_client: DeepSeekClient = None,
        spatial_client: DeepSeekClient = None,
        prompt_gen: PromptGenerator = None,
        shot_mode: str = "zero",
        urban_shot_mode: str = None,
        spatial_shot_mode: str = None,
        urban_method: Union[UrbanMethod, str] = UrbanMethod.THREE_STAGE_HYBRID,
        hybrid_llm_assist_enabled: Optional[bool] = None,
    ):
        self.config = Config()
        self.urban_client = urban_client or client or DeepSeekClient()
        self.spatial_client = spatial_client or DeepSeekClient()
        self.urban_shot_mode = urban_shot_mode or shot_mode
        self.spatial_shot_mode = spatial_shot_mode or shot_mode
        self.urban_method = (
            urban_method
            if isinstance(urban_method, UrbanMethod)
            else UrbanMethod(str(urban_method))
        )
        if hybrid_llm_assist_enabled is None:
            hybrid_llm_assist_enabled = Config.URBAN_HYBRID_LLM_ASSIST_ENABLED
        self.hybrid_llm_assist_enabled = bool(hybrid_llm_assist_enabled)
        if prompt_gen is not None:
            self.urban_prompt_gen = prompt_gen
            self.spatial_prompt_gen = prompt_gen
        else:
            self.urban_prompt_gen = PromptGenerator(
                shot_mode=self.urban_shot_mode,
                default_theme="urban_renewal",
            )
            self.spatial_prompt_gen = PromptGenerator(
                shot_mode=self.spatial_shot_mode,
                default_theme="spatial",
            )

        self.urban_renewal_strategy = StepwiseLongContextStrategy(
            self.urban_client, self.urban_prompt_gen
        )
        self.urban_topic_classifier = UrbanTopicClassifier()
        self.urban_hybrid_classifier = UrbanHybridClassifier(
            self.urban_renewal_strategy,
            llm_assist_enabled=self.hybrid_llm_assist_enabled,
        )
        self.spatial_strategy = SpatialExtractionStrategy(
            self.spatial_client, self.spatial_prompt_gen
        )
        self._validate_prompt_routes()

    def _validate_prompt_routes(self):
        self.urban_prompt_gen.get_step_system_prompt()
        self.spatial_prompt_gen.get_spatial_system_prompt()
        urban_template = self.urban_prompt_gen.registry.get_template_file(
            "urban_renewal",
            self.urban_prompt_gen.shot_mode,
        )
        spatial_template = self.spatial_prompt_gen.registry.get_template_file(
            "spatial",
            self.spatial_prompt_gen.shot_mode,
        )
        print(
            f"[INFO] Prompt route validated: urban shot={self.urban_prompt_gen.shot_mode}, template={urban_template}; spatial shot={self.spatial_prompt_gen.shot_mode}, template={spatial_template}"
        )

    def _normalize_audit_path(self, path: Optional[Union[str, Path]]) -> str:
        if not path:
            return ""
        try:
            return str(Path(path).resolve())
        except Exception:
            return str(path)

    def _build_session_audit_metadata(
        self,
        *,
        task_type: TaskType,
        input_path: Path,
        output_path: Optional[Path],
        strategy_name: str,
        run_id: str,
        sample_index: int,
        run_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        metadata = {
            "task_type": task_type.value,
            "input_file": self._normalize_audit_path(input_path),
            "output_file": self._normalize_audit_path(output_path),
            "strategy_name": strategy_name,
            "run_id": run_id,
            "sample_index": int(sample_index),
            "task_name": input_path.stem,
        }
        if run_context:
            for key in (
                "experiment_track",
                "dataset_id",
                "truth_file",
                "task_mode",
                "urban_method",
                "session_policy",
                "order_id",
                "order_seed",
                "max_samples_per_window",
                "urban_flow_audit_enabled",
            ):
                metadata[key] = run_context.get(key)
        return metadata

    def _raise_fatal_task_error(self, task_name: str, error: Exception):
        raise RuntimeError(
            f"任务执行失败并已终止: task={task_name}, error={type(error).__name__}: {error}"
        ) from error

    def run(
        self,
        input_file: str = None,
        output_file: str = None,
        limit: int = None,
        task_type: TaskType = TaskType.BOTH,
        run_context: Optional[Dict[str, Any]] = None,
    ):
        if task_type == TaskType.URBAN_RENEWAL:
            return self.run_urban_renewal(input_file, output_file, limit, run_context=run_context)
        elif task_type == TaskType.SPATIAL:
            return self.run_spatial(input_file, output_file, limit, run_context=run_context)
        else:
            return self.run_both(input_file, output_file, limit, run_context=run_context)

    def _run_context_value(
        self,
        run_context: Optional[Dict[str, Any]],
        key: str,
        default: Any = None,
    ) -> Any:
        if not run_context:
            return default
        return run_context.get(key, default)

    def _prepare_frame_for_run(
        self,
        df: pd.DataFrame,
        run_context: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        prepared = df.copy()
        order_seed = self._run_context_value(run_context, "order_seed")
        order_id = str(self._run_context_value(run_context, "order_id", "canonical_title_order") or "canonical_title_order")
        if order_seed is not None:
            return prepared.sample(frac=1, random_state=int(order_seed)).reset_index(drop=True)
        if order_id == "canonical_title_order" and Schema.TITLE in prepared.columns:
            sort_key = prepared[Schema.TITLE].fillna("").astype(str).str.strip().str.lower()
            return prepared.assign(_sort_key=sort_key).sort_values("_sort_key", kind="stable").drop(
                columns=["_sort_key"]
            ).reset_index(drop=True)
        return prepared.reset_index(drop=True)

    def _urban_checkpoint_interval(
        self,
        row_count: int,
        run_context: Optional[Dict[str, Any]] = None,
    ) -> int:
        raw_value = self._run_context_value(run_context, "urban_checkpoint_interval")
        if raw_value not in (None, ""):
            try:
                return max(1, int(raw_value))
            except (TypeError, ValueError):
                pass
        if row_count >= 5000:
            return 1000
        if row_count >= 1000:
            return 100
        return 10

    def _resume_checkpoint_for_output(
        self,
        output_path: Path,
        run_context: Optional[Dict[str, Any]] = None,
    ) -> ResumeCheckpoint:
        raw_path = self._run_context_value(run_context, "resume_checkpoint")
        path = Path(raw_path) if raw_path else default_checkpoint_path(output_path)
        return ResumeCheckpoint(path)

    def _ordered_result_rows(self, rows: list[Optional[Dict[str, Any]]]) -> list[Dict[str, Any]]:
        return [item for item in rows if item]

    def _write_partial_prediction_rows(
        self,
        rows: list[Optional[Dict[str, Any]]],
        output_path: Path,
    ) -> None:
        ordered_rows = self._ordered_result_rows(rows)
        if ordered_rows:
            write_urban_prediction_checkpoint(ordered_rows, output_path)

    def _record_resume_interruption(
        self,
        run_context: Optional[Dict[str, Any]],
        checkpoint: ResumeCheckpoint,
        *,
        completed_rows: int,
    ) -> None:
        if run_context is None:
            return
        if checkpoint.path is not None:
            run_context["_last_resume_checkpoint"] = str(checkpoint.path)
        run_context["_last_resume_completed_rows"] = int(completed_rows)

    def _run_id_for_explicit_output(self, output_path: Path, run_id: str | None) -> str:
        return run_id or output_path.stem

    def _urban_flow_audit_enabled(
        self,
        run_context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Return whether urban rows should carry per-row audit metadata."""

        raw_value = self._run_context_value(run_context, "urban_flow_audit_enabled", True)
        if isinstance(raw_value, bool):
            return bool(raw_value)
        return str(raw_value).strip().lower() not in {"0", "false", "off", "no"}

    def run_urban_renewal(
        self,
        input_file: str = None,
        output_file: str = None,
        limit: int = None,
        run_id: str = None,
        run_context: Optional[Dict[str, Any]] = None,
    ) -> Path:
        print("\n[INFO] Running Urban Renewal Classification only...")

        input_path = Path(input_file) if input_file else self.config.require_default_train_input_file()
        task_name = input_path.stem

        if output_file:
            output_path = Path(output_file)
            timestamp = self._run_id_for_explicit_output(output_path, run_id)
        else:
            timestamp = run_id or time.strftime("%Y%m%d_%H%M%S")
            output_path = self._default_prediction_file(
                task_name,
                timestamp,
                run_context,
                task_type=TaskType.URBAN_RENEWAL,
            )
        self._ensure_output_parent(output_path)

        print(f"Reading from {input_path}")
        df = self._read_input(input_path)
        df = self._prepare_frame_for_run(df, run_context=run_context)
        max_samples_per_window = self._run_context_value(run_context, "max_samples_per_window")
        if max_samples_per_window is not None:
            self.urban_renewal_strategy.max_samples_per_window = max(1, int(max_samples_per_window))

        if limit:
            df = df.head(limit)

        rows = list(df.iterrows())
        fingerprint = input_fingerprint(input_path, rows=len(rows))
        resume_checkpoint = self._resume_checkpoint_for_output(output_path, run_context=run_context)
        results_list: list[Optional[Dict[str, Any]]] = [None] * len(rows)
        checkpoint_interval = self._urban_checkpoint_interval(len(df), run_context=run_context)

        for position, (index, row) in enumerate(tqdm(rows, total=len(rows))):
            resume_key = resume_checkpoint.key(
                row_index=int(index),
                task_type=TaskType.URBAN_RENEWAL.value,
                run_id=timestamp,
                input_fingerprint=fingerprint,
            )
            existing_row = resume_checkpoint.completed_row(resume_key)
            if existing_row:
                results_list[position] = existing_row
                continue

            title = str(row.get(Schema.TITLE, "") or "")
            abstract = str(row.get(Schema.ABSTRACT, "") or "")

            if not title and not abstract:
                continue

            session_path = self._get_urban_session_path(task_name, index, timestamp)
            metadata = self._extract_metadata(row)
            audit_metadata = None
            if self._urban_flow_audit_enabled(run_context):
                audit_metadata = self._build_session_audit_metadata(
                    task_type=TaskType.URBAN_RENEWAL,
                    input_path=input_path,
                    output_path=output_path,
                    strategy_name=self.urban_method.value,
                    run_id=timestamp,
                    sample_index=index,
                    run_context=run_context,
                )
            try:
                result = self._run_urban_method(
                    title,
                    abstract,
                    metadata,
                    session_path,
                    audit_metadata=audit_metadata,
                    run_context=run_context,
                )
            except LLMQuotaExceededError as exc:
                completed_rows = len(self._ordered_result_rows(results_list))
                resume_checkpoint.append_record(
                    resume_key,
                    status="quota_exhausted",
                    row={Schema.TITLE: title, Schema.ABSTRACT: abstract},
                    error=str(exc),
                    metadata={"completed_rows": completed_rows},
                )
                self._record_resume_interruption(run_context, resume_checkpoint, completed_rows=completed_rows)
                self._write_partial_prediction_rows(results_list, output_path)
                print(
                    "[WARN] LLM quota exhausted; saved resumable checkpoint. "
                    f"completed_rows={completed_rows} "
                    f"checkpoint={resume_checkpoint.path}"
                )
                raise

            output_row = self._build_urban_output_row(
                title,
                abstract,
                result,
            )
            results_list[position] = output_row
            resume_checkpoint.append_completed(resume_key, row=output_row)

            if (index + 1) % checkpoint_interval == 0 or (index + 1) == len(df):
                self._write_partial_prediction_rows(results_list, output_path)

        ordered_results = self._ordered_result_rows(results_list)
        if ordered_results:
            final_df = build_urban_prediction_frame(ordered_results)
            postprocess_context = dict(run_context or {})
            if resume_checkpoint.path is not None:
                postprocess_context.setdefault("resume_checkpoint", str(resume_checkpoint.path))
            postprocess_context.setdefault("resume_run_id", timestamp)
            postprocess_context.setdefault("resume_input_fingerprint", fingerprint)
            postprocess_context.setdefault("resume_task_type", TaskType.URBAN_RENEWAL.value)
            try:
                final_df = self._postprocess_urban_prediction_frame(final_df, run_context=postprocess_context)
            except LLMQuotaExceededError as exc:
                final_df.to_excel(output_path, index=False, engine="openpyxl")
                self._record_resume_interruption(
                    run_context,
                    resume_checkpoint,
                    completed_rows=len(ordered_results),
                )
                resume_checkpoint.write_summary(
                    extra={
                        "completed_rows": len(ordered_results),
                        "output": str(output_path),
                        "run_id": timestamp,
                        "task_type": TaskType.URBAN_RENEWAL.value,
                        "postprocess_status": "quota_exhausted",
                        "error": str(exc),
                    }
                )
                print(
                    "[WARN] LLM quota exhausted during urban postprocess; saved resumable checkpoint. "
                    f"completed_rows={len(ordered_results)} checkpoint={resume_checkpoint.path}"
                )
                raise
            if run_context is not None and "urban_postprocess_layers" in postprocess_context:
                run_context["urban_postprocess_layers"] = postprocess_context["urban_postprocess_layers"]
            final_df.to_excel(output_path, index=False, engine="openpyxl")
            resume_checkpoint.write_summary(
                extra={
                    "completed_rows": len(ordered_results),
                    "output": str(output_path),
                    "run_id": timestamp,
                    "task_type": TaskType.URBAN_RENEWAL.value,
                }
            )

        print(f"[INFO] Urban Renewal results saved to: {output_path}")
        return output_path

    def _postprocess_urban_prediction_frame(
        self,
        frame: pd.DataFrame,
        *,
        run_context: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        context = run_context if run_context is not None else {}
        if self.urban_method == UrbanMethod.THREE_STAGE_HYBRID and self.hybrid_llm_assist_enabled:
            context.setdefault("urban_stable_strategy_llm_strategy", self.urban_renewal_strategy)
            context.setdefault("urban_stable_strategy_llm_enabled", True)
        return postprocess_urban_predictions(
            frame,
            run_context=context,
            llm_client=self.urban_client,
            hybrid_llm_assist_enabled=self.hybrid_llm_assist_enabled,
            urban_method=self.urban_method,
        )

    def _run_urban_method(
        self,
        title: str,
        abstract: str,
        metadata: Dict[str, Any],
        session_path: Path,
        audit_metadata: Optional[Dict[str, Any]] = None,
        run_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        record = UrbanMetadataRecord.from_row(
            {
                Schema.TITLE: title,
                Schema.ABSTRACT: abstract,
                **(metadata or {}),
            }
        )
        if self.urban_method == UrbanMethod.PURE_LLM_API:
            effective_session_path = (
                None
                if self._run_context_value(run_context, "session_policy") == "cross_paper_long_context"
                else session_path
            )
            if audit_metadata is None:
                return self._run_urban_pure_llm(title, abstract, record, effective_session_path)
            return self._run_urban_pure_llm(
                title,
                abstract,
                record,
                effective_session_path,
                audit_metadata=audit_metadata,
            )
        if self.urban_method == UrbanMethod.LOCAL_TOPIC_CLASSIFIER:
            return self._run_urban_local_classifier(record)
        if audit_metadata is None:
            return self.urban_hybrid_classifier.classify(
                title,
                abstract,
                metadata=metadata,
                session_path=session_path,
                finalize_strategy=False,
            )
        return self.urban_hybrid_classifier.classify(
            title,
            abstract,
            metadata=metadata,
            session_path=session_path,
            audit_metadata=audit_metadata,
            finalize_strategy=False,
        )

    def _run_urban_pure_llm(
        self,
        title: str,
        abstract: str,
        record: UrbanMetadataRecord,
        session_path: Path,
        audit_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        process_kwargs = {
            "session_path": session_path,
            "metadata": record.to_output_dict(),
            "auxiliary_context": None,
        }
        if audit_metadata is not None:
            process_kwargs["audit_metadata"] = audit_metadata
        result = self.urban_renewal_strategy.process(
            title,
            abstract,
            **process_kwargs,
        )
        label = str(result.get(Schema.IS_URBAN_RENEWAL, "0") or "0")
        if label not in {"0", "1"}:
            label = "0"
        output = dict(record.to_output_dict())
        output.update(result)
        output.update(
            {
                Schema.IS_URBAN_RENEWAL: label,
                "final_label": label,
                "decision_source": UrbanMethod.PURE_LLM_API.value,
                "decision_reason": result.get("urban_parse_reason", "pure_llm_api"),
                "llm_used": 1,
                "llm_attempted": 1,
                "llm_failure_reason": "",
            }
        )
        return output

    def _run_urban_local_classifier(
        self,
        record: UrbanMetadataRecord,
    ) -> Dict[str, Any]:
        prediction = self.urban_topic_classifier.predict(record)
        label = urban_flag_for_topic_label(prediction.topic_label)
        legacy_label, legacy_group, legacy_name = legacy_topic_for_label(prediction.topic_label)
        return {
            **record.to_output_dict(),
            Schema.IS_URBAN_RENEWAL: label,
            "urban_flag": label,
            "urban_parse_reason": "title_abstract_classifier",
            "final_label": label,
            "confidence": prediction.confidence,
            "llm_used": 0,
            "llm_attempted": 0,
            "llm_failure_reason": "",
            "llm_family_hint": "",
            "llm_family_hint_reason": "",
            "review_flag": int(prediction.topic_label == "Unknown"),
            "review_reason": "local_classifier_unknown" if prediction.topic_label == "Unknown" else "",
            "decision_source": UrbanMethod.LOCAL_TOPIC_CLASSIFIER.value,
            "decision_reason": f"{prediction.topic_label}:{prediction.topic_name}",
            "legacy_topic_label": legacy_label,
            "legacy_topic_group": legacy_group,
            "legacy_topic_name": legacy_name,
            "topic_rule": "",
            "topic_rule_group": "",
            "topic_rule_name": "",
            "topic_rule_score": 0.0,
            "topic_rule_margin": 0.0,
            "topic_rule_top3": "",
            "topic_rule_matches": "",
            "review_flag_rule": 0,
            "review_reason_rule": "",
            "topic_local_label": prediction.topic_label,
            "topic_local_group": prediction.topic_group,
            "topic_local_name": prediction.topic_name,
            "topic_local_confidence": prediction.confidence,
            "topic_local_margin": prediction.margin,
            "topic_local_top3": "; ".join(prediction.top_candidates),
            "topic_final": prediction.topic_label,
            "topic_final_group": prediction.topic_group,
            "topic_final_name": prediction.topic_name,
            "topic_label": prediction.topic_label,
            "topic_group": prediction.topic_group,
            "topic_name": prediction.topic_name,
            "topic_confidence": prediction.confidence,
            "topic_margin": prediction.margin,
            "topic_matches": "; ".join(prediction.matched_terms),
            "topic_binary_score": prediction.binary_score,
            "topic_binary_probability": prediction.binary_probability,
            "bertopic_hint_label": "",
            "bertopic_hint_group": "",
            "bertopic_hint_name": "",
            "bertopic_hint_conflict_flag": 0,
            "bertopic_cluster_quality": "",
            "bertopic_dynamic_topic_id": -1,
            "bertopic_dynamic_topic_words": "",
        }

    def run_spatial(
        self,
        input_file: str = None,
        output_file: str = None,
        limit: int = None,
        run_id: str = None,
        run_context: Optional[Dict[str, Any]] = None,
    ) -> Path:
        print("\n[INFO] Running Spatial Attribute Extraction only...")

        input_path = Path(input_file) if input_file else self.config.require_default_train_input_file()
        task_name = input_path.stem

        if output_file:
            output_path = Path(output_file)
            timestamp = self._run_id_for_explicit_output(output_path, run_id)
        else:
            timestamp = run_id or time.strftime("%Y%m%d_%H%M%S")
            output_path = self._default_prediction_file(
                task_name,
                timestamp,
                run_context,
                task_type=TaskType.SPATIAL,
            )
        self._ensure_output_parent(output_path)

        print(f"Reading from {input_path}")
        df = self._read_input(input_path)
        df = self._prepare_frame_for_run(df, run_context=run_context)

        if limit:
            df = df.head(limit)

        rows = list(df.iterrows())
        fingerprint = input_fingerprint(input_path, rows=len(rows))
        resume_checkpoint = self._resume_checkpoint_for_output(output_path, run_context=run_context)
        results_list: list[Optional[Dict[str, Any]]] = [None] * len(rows)

        def process_one(position: int, index: int, row: pd.Series) -> tuple[int, Dict[str, Any]]:
            title = str(row.get(Schema.TITLE, "") or "")
            abstract = str(row.get(Schema.ABSTRACT, "") or "")

            if not title and not abstract:
                return position, {}

            session_path = self._get_spatial_session_path(task_name, index, timestamp)
            audit_metadata = self._build_session_audit_metadata(
                task_type=TaskType.SPATIAL,
                input_path=input_path,
                output_path=output_path,
                strategy_name=f"spatial:{self.spatial_shot_mode}",
                run_id=timestamp,
                sample_index=index,
                run_context=run_context,
            )
            result = self.spatial_strategy.process(
                title,
                abstract,
                session_path,
                audit_metadata=audit_metadata,
            )
            return position, (
                self._build_spatial_output_row(
                    title,
                    abstract,
                    result,
                )
            )

        max_workers = max(1, int(self.config.MAX_WORKERS))
        if max_workers == 1:
            for completed, (position, (index, row)) in enumerate(tqdm(list(enumerate(rows)), total=len(rows)), start=1):
                resume_key = resume_checkpoint.key(
                    row_index=int(index),
                    task_type=TaskType.SPATIAL.value,
                    run_id=timestamp,
                    input_fingerprint=fingerprint,
                )
                existing_row = resume_checkpoint.completed_row(resume_key)
                if existing_row:
                    results_list[position] = existing_row
                else:
                    try:
                        _, result_row = process_one(position, index, row)
                    except LLMQuotaExceededError as exc:
                        completed_rows_count = len(self._ordered_result_rows(results_list))
                        resume_checkpoint.append_record(
                            resume_key,
                            status="quota_exhausted",
                            row={
                                Schema.TITLE: str(row.get(Schema.TITLE, "") or ""),
                                Schema.ABSTRACT: str(row.get(Schema.ABSTRACT, "") or ""),
                            },
                            error=str(exc),
                            metadata={"completed_rows": completed_rows_count},
                        )
                        self._record_resume_interruption(
                            run_context,
                            resume_checkpoint,
                            completed_rows=completed_rows_count,
                        )
                        self._write_partial_prediction_rows(results_list, output_path)
                        print(
                            "[WARN] LLM quota exhausted; saved resumable checkpoint. "
                            f"completed_rows={completed_rows_count} "
                            f"checkpoint={resume_checkpoint.path}"
                        )
                        raise
                    results_list[position] = result_row
                    resume_checkpoint.append_completed(resume_key, row=result_row)
                if completed % 10 == 0 or completed == len(rows):
                    self._write_partial_prediction_rows(results_list, output_path)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for position, (index, row) in enumerate(rows):
                    resume_key = resume_checkpoint.key(
                        row_index=int(index),
                        task_type=TaskType.SPATIAL.value,
                        run_id=timestamp,
                        input_fingerprint=fingerprint,
                    )
                    existing_row = resume_checkpoint.completed_row(resume_key)
                    if existing_row:
                        results_list[position] = existing_row
                        continue
                    future = executor.submit(process_one, position, index, row)
                    futures[future] = (position, index, row, resume_key)

                for completed, future in enumerate(tqdm(as_completed(futures), total=len(futures)), start=1):
                    position, index, row, resume_key = futures[future]
                    try:
                        _, result_row = future.result()
                    except LLMQuotaExceededError as exc:
                        completed_rows_count = len(self._ordered_result_rows(results_list))
                        resume_checkpoint.append_record(
                            resume_key,
                            status="quota_exhausted",
                            row={
                                Schema.TITLE: str(row.get(Schema.TITLE, "") or ""),
                                Schema.ABSTRACT: str(row.get(Schema.ABSTRACT, "") or ""),
                            },
                            error=str(exc),
                            metadata={"completed_rows": completed_rows_count},
                        )
                        self._record_resume_interruption(
                            run_context,
                            resume_checkpoint,
                            completed_rows=completed_rows_count,
                        )
                        for pending in futures:
                            if pending is not future:
                                pending.cancel()
                        self._write_partial_prediction_rows(results_list, output_path)
                        print(
                            "[WARN] LLM quota exhausted; saved resumable checkpoint. "
                            f"completed_rows={completed_rows_count} "
                            f"checkpoint={resume_checkpoint.path}"
                        )
                        raise
                    results_list[position] = result_row
                    resume_checkpoint.append_completed(resume_key, row=result_row)
                    if completed % 10 == 0 or completed == len(futures):
                        self._write_partial_prediction_rows(results_list, output_path)

        ordered_rows = self._ordered_result_rows(results_list)
        if ordered_rows:
            temp_df = pd.DataFrame(ordered_rows)
            temp_df.to_excel(output_path, index=False, engine="openpyxl")
            resume_checkpoint.write_summary(
                extra={
                    "completed_rows": len(ordered_rows),
                    "output": str(output_path),
                    "run_id": timestamp,
                    "task_type": TaskType.SPATIAL.value,
                }
            )

        print(f"[INFO] Spatial results saved to: {output_path}")
        return output_path

    def run_both(
        self,
        input_file: str = None,
        output_file: str = None,
        limit: int = None,
        run_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Path]:
        print("\n[INFO] Running both tasks with strict serial isolation...")
        run_id = time.strftime("%Y%m%d_%H%M%S")

        print("[INFO] Phase A: Urban Renewal task (isolated)")
        urban_path = self.run_urban_renewal(
            input_file=input_file,
            output_file=None,
            limit=limit,
            run_id=run_id,
            run_context=run_context,
        )

        print("[INFO] Phase B: Spatial task (isolated)")
        spatial_path = self.run_spatial(
            input_file=input_file,
            output_file=None,
            limit=limit,
            run_id=run_id,
            run_context=run_context,
        )

        merged_path = self._merge_results(urban_path, spatial_path, run_id, output_file=output_file)

        return {
            "urban_renewal": urban_path,
            "spatial": spatial_path,
            "merged": merged_path
        }

    def _process_urban_renewal(
        self,
        title: str,
        abstract: str,
        memory: ConversationMemory
    ) -> Dict[str, Any]:
        prompt1 = self.urban_prompt_gen.get_step_prompt(1, title, abstract, include_context=True)
        memory.add_user_message(prompt1)

        resp1 = self.urban_client.chat_completion(memory.get_messages())
        if not resp1:
            self._safe_save_memory(memory, "urban_empty_response")
            return {
                "是否属于城市更新研究": "0",
                "urban_parse_reason": "empty_response",
            }

        memory.add_assistant_message(resp1)
        parsed_label, parse_reason = self._parse_single_output(resp1)
        self._safe_save_memory(memory, "urban_sample_completed")

        result = {
            "是否属于城市更新研究": parsed_label,
            "urban_parse_reason": parse_reason,
        }
        return result

    def _parse_single_output(self, text: str) -> tuple[str, str]:
        raw_text = text.strip()
        if not raw_text:
            return "0", "empty_text"

        explicit_patterns = [
            r'(?:最终答案|最终结论|答案|结论)\s*[:：是为]?\s*([01])\b',
            r'"?(?:是否属于城市更新研究|is_urban_renewal)"?\s*[:=]\s*"?([01])"?',
        ]
        for pattern in explicit_patterns:
            match = re.search(pattern, raw_text, re.IGNORECASE)
            if match:
                return match.group(1), "explicit_answer_pattern"

        clean_text = re.sub(r'(Step|Field|Phase)\s*\d+', '', raw_text, flags=re.IGNORECASE)
        line_match = re.search(r'(?m)^\s*([01])\s*$', clean_text)
        if line_match:
            return line_match.group(1), "single_digit_line"

        match = re.search(r'(?<!\d)(1|0)(?!\d)', clean_text)
        if match:
            return match.group(1), "fallback_first_digit"

        if re.search(r'\b(yes|true)\b', raw_text, re.IGNORECASE):
            return "1", "fallback_boolean_yes"

        return "0", "no_label_detected"

    def _safe_save_memory(self, memory: ConversationMemory, scene: str):
        try:
            memory.save()
        except Exception as error:
            print(f"[WARN] Failed to persist session in {scene}: {error}")

    def _create_memory(self, system_prompt: str) -> ConversationMemory:
        return ConversationMemory(system_prompt=system_prompt, skip_index=True)

    def _build_urban_output_row(
        self,
        title: str,
        abstract: str,
        result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return build_urban_output_row(title, abstract, result)

    def _build_spatial_output_row(
        self,
        title: str,
        abstract: str,
        result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        result = result or {}
        return {
            Schema.TITLE: title,
            Schema.ABSTRACT: abstract,
            Schema.IS_SPATIAL: str(result.get(Schema.IS_SPATIAL, "0") or "0"),
            Schema.SPATIAL_LEVEL: result.get(Schema.SPATIAL_LEVEL, "Not mentioned") or "Not mentioned",
            Schema.SPATIAL_DESC: result.get(Schema.SPATIAL_DESC, "Not mentioned") or "Not mentioned",
            "Reasoning": result.get("Reasoning", ""),
            "Confidence": result.get("Confidence", "Low"),
            Schema.SPATIAL_VALIDATION_STATUS: result.get(Schema.SPATIAL_VALIDATION_STATUS, ""),
            Schema.SPATIAL_VALIDATION_REASON: result.get(Schema.SPATIAL_VALIDATION_REASON, ""),
            Schema.SPATIAL_AREA_EVIDENCE: result.get(Schema.SPATIAL_AREA_EVIDENCE, ""),
            Schema.LLM_SPATIAL_SCALE_LEVEL_RAW: result.get(Schema.LLM_SPATIAL_SCALE_LEVEL_RAW, ""),
            Schema.RESOLVED_STUDY_AREA: result.get(Schema.RESOLVED_STUDY_AREA, ""),
            Schema.RESOLVED_GEO_ID: result.get(Schema.RESOLVED_GEO_ID, ""),
            Schema.AREA_HIERARCHY_PATH: result.get(Schema.AREA_HIERARCHY_PATH, ""),
            Schema.MAPPED_SPATIAL_SCALE_LEVEL: result.get(Schema.MAPPED_SPATIAL_SCALE_LEVEL, ""),
            Schema.SCALE_DECISION_SOURCE: result.get(Schema.SCALE_DECISION_SOURCE, ""),
            Schema.GEO_RESOLUTION_STATUS: result.get(Schema.GEO_RESOLUTION_STATUS, ""),
            Schema.GEO_RESOLUTION_REASON: result.get(Schema.GEO_RESOLUTION_REASON, ""),
            Schema.GEO_RESOLUTION_CONFIDENCE: result.get(Schema.GEO_RESOLUTION_CONFIDENCE, ""),
            Schema.GEO_SOURCE: result.get(Schema.GEO_SOURCE, ""),
        }

    def _get_urban_session_path(self, task_name: str, index: int, run_id: str) -> Path:
        return self.config.SESSIONS_DIR / task_name / run_id / f"urban_{index}" / "session.json"

    def _ensure_output_parent(self, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    def _default_prediction_dir(
        self,
        task_name: str,
        run_id: str,
        run_context: Optional[Dict[str, Any]] = None,
        task_type: TaskType | str = TaskType.URBAN_RENEWAL,
    ) -> Path:
        context = run_context or {}
        dataset_id = str(context.get("dataset_id") or task_name)
        experiment_track = str(context.get("experiment_track") or "research_matrix")
        layout = run_paths(dataset_id, experiment_track, run_id, project_root=self.config.DATA_DIR.parent)
        ensure_run_layout(layout)
        return layout.prediction_dir_for_task(_prediction_task_key(task_type))

    def _default_prediction_file(
        self,
        task_name: str,
        run_id: str,
        run_context: Optional[Dict[str, Any]] = None,
        task_type: TaskType | str = TaskType.URBAN_RENEWAL,
    ) -> Path:
        context = run_context or {}
        dataset_id = str(context.get("dataset_id") or task_name)
        task_key = _prediction_task_key(task_type)
        output_dir = self._default_prediction_dir(task_name, run_id, run_context, task_type=task_key)
        if task_key == "urban_renewal":
            stem = build_urban_prediction_stem(
                dataset_id=dataset_id,
                urban_method=self.urban_method.value,
                shot_mode=self.urban_shot_mode if self.urban_method != UrbanMethod.LOCAL_TOPIC_CLASSIFIER else None,
                llm_assist_enabled=(
                    bool(context.get("hybrid_llm_assist_enabled", getattr(self, "hybrid_llm_assist_enabled", False)))
                    if self.urban_method == UrbanMethod.THREE_STAGE_HYBRID
                    else None
                ),
                run_tag=run_id,
            )
        elif task_key == "spatial":
            stem = build_spatial_prediction_stem(
                dataset_id=dataset_id,
                spatial_shot=self.spatial_shot_mode,
                run_tag=run_id,
            )
        else:
            stem = build_merged_prediction_stem(dataset_id=dataset_id, run_tag=run_id)
        return output_dir / f"{stem}.xlsx"

    def _build_urban_output_filename(self, timestamp: str) -> str:
        if self.urban_method == UrbanMethod.LOCAL_TOPIC_CLASSIFIER:
            return f"urban_renewal_{self.urban_method.value}_{timestamp}.xlsx"
        return f"urban_renewal_{self.urban_method.value}_{self.urban_shot_mode}_{timestamp}.xlsx"

    def _get_spatial_session_path(self, task_name: str, index: int, run_id: str) -> Path:
        return self.config.SESSIONS_DIR / task_name / run_id / f"spatial_{index}" / "session.json"

    def _merge_results(
        self,
        urban_path: Path,
        spatial_path: Path,
        timestamp: str,
        output_file: str = None,
    ) -> Optional[Path]:
        print("\n[INFO] Merging results...")

        try:
            df_urban = pd.read_excel(urban_path, engine="openpyxl")
            df_spatial = pd.read_excel(spatial_path, engine="openpyxl")

            if "Article Title" not in df_urban.columns or "Article Title" not in df_spatial.columns:
                print("[WARN] Missing 'Article Title' column for merge.")
                return None

            if len(df_urban) != len(df_spatial):
                raise ValueError(
                    "Cannot merge urban and spatial predictions by row order: "
                    f"urban_rows={len(df_urban)}, spatial_rows={len(df_spatial)}"
                )

            merged = df_urban.reset_index(drop=True).copy()
            spatial_fields = [
                Schema.IS_SPATIAL,
                Schema.SPATIAL_LEVEL,
                Schema.SPATIAL_DESC,
                "Reasoning",
                "Confidence",
                Schema.SPATIAL_VALIDATION_STATUS,
                Schema.SPATIAL_VALIDATION_REASON,
                Schema.SPATIAL_AREA_EVIDENCE,
            ]
            spatial_values = df_spatial.reset_index(drop=True)
            for column in spatial_fields:
                if column not in spatial_values.columns:
                    continue
                target_column = f"{column}_spatial" if column in merged.columns else column
                merged[target_column] = spatial_values[column]

            input_df = None
            for parent in urban_path.parents:
                input_df = load_task_input_frame(parent)
                if input_df is not None:
                    break
            merged_raw = merged.copy()
            merged = build_review_ready_merged_frame(merged, input_df=input_df)

            if output_file:
                merge_output = Path(output_file)
            elif urban_path.parent.name == "urban_renewal" and urban_path.parent.parent.name == "predictions":
                run_dir = urban_path.parent.parent.parent
                dataset_id = run_dir.parent.parent.parent.name if run_dir.parent.parent.name == "runs" else urban_path.stem
                merge_output = urban_path.parent.parent / "merged" / f"{build_merged_prediction_stem(dataset_id=dataset_id, run_tag=timestamp)}.xlsx"
            else:
                merge_output = urban_path.parent / f"merged_{timestamp}.xlsx"
            self._ensure_output_parent(merge_output)
            with pd.ExcelWriter(merge_output, engine="openpyxl") as writer:
                merged.to_excel(writer, sheet_name="Review View", index=False)
                build_metric_dictionary_frame().to_excel(writer, sheet_name="Metric Dictionary", index=False)
                merged_raw.to_excel(writer, sheet_name="Raw Predictions", index=False)
            print(f"[INFO] Merged results saved to: {merge_output}")
            return merge_output

        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise RuntimeError(f"Failed to merge results: {e}") from e

    def _read_input(self, input_path: Path) -> pd.DataFrame:
        df = pd.read_excel(input_path, engine="openpyxl")
        if Schema.TITLE not in df.columns or Schema.ABSTRACT not in df.columns:
            df = pd.read_excel(input_path, engine="openpyxl", header=None)
            df = df.dropna(axis=1, how="all")
            col_names = []
            if df.shape[1] >= 1:
                col_names.append(Schema.TITLE)
            if df.shape[1] >= 2:
                col_names.append(Schema.ABSTRACT)
            label_cols = [Schema.IS_URBAN_RENEWAL, Schema.IS_SPATIAL, Schema.SPATIAL_LEVEL, Schema.SPATIAL_DESC]
            remaining = df.shape[1] - len(col_names)
            col_names.extend(label_cols[:max(0, min(remaining, len(label_cols)))])
            if remaining > len(label_cols):
                col_names.extend([f"extra_{i+1}" for i in range(remaining - len(label_cols))])
            df.columns = col_names
        if Schema.TITLE not in df.columns:
            df[Schema.TITLE] = ""
        if Schema.ABSTRACT not in df.columns:
            df[Schema.ABSTRACT] = ""
        optional_columns = [
            Schema.AUTHOR_KEYWORDS,
            Schema.KEYWORDS_PLUS,
            Schema.KEYWORDS,
            Schema.WOS_CATEGORIES,
            Schema.RESEARCH_AREAS,
        ]
        for column in optional_columns:
            if column not in df.columns:
                df[column] = ""
        return df[[Schema.TITLE, Schema.ABSTRACT] + optional_columns].copy()

    def _extract_metadata(self, row: pd.Series) -> Dict[str, Any]:
        return {
            Schema.AUTHOR_KEYWORDS: row.get(Schema.AUTHOR_KEYWORDS, ""),
            Schema.KEYWORDS_PLUS: row.get(Schema.KEYWORDS_PLUS, ""),
            Schema.KEYWORDS: row.get(Schema.KEYWORDS, ""),
            Schema.WOS_CATEGORIES: row.get(Schema.WOS_CATEGORIES, ""),
            Schema.RESEARCH_AREAS: row.get(Schema.RESEARCH_AREAS, ""),
        }
