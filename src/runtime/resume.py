"""JSONL checkpoint helpers for resumable API-backed runs."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


@dataclass(frozen=True)
class ResumeKey:
    row_index: int
    task_type: str
    run_id: str
    input_fingerprint: str

    def as_id(self) -> str:
        return "|".join(
            [
                str(int(self.row_index)),
                str(self.task_type or ""),
                str(self.run_id or ""),
                str(self.input_fingerprint or ""),
            ]
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "row_index": int(self.row_index),
            "task_type": str(self.task_type or ""),
            "run_id": str(self.run_id or ""),
            "input_fingerprint": str(self.input_fingerprint or ""),
            "key": self.as_id(),
        }


class ResumeCheckpoint:
    """Append-only checkpoint that treats only completed rows as resumable."""

    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self.records: dict[str, dict[str, Any]] = {}
        if self.path and self.path.exists():
            self._load()

    @classmethod
    def default_for_output(cls, output_path: str | Path) -> "ResumeCheckpoint":
        return cls(default_checkpoint_path(output_path))

    def key(
        self,
        *,
        row_index: int,
        task_type: str,
        run_id: str,
        input_fingerprint: str,
    ) -> ResumeKey:
        return ResumeKey(
            row_index=int(row_index),
            task_type=str(task_type or ""),
            run_id=str(run_id or ""),
            input_fingerprint=str(input_fingerprint or ""),
        )

    def completed_row(self, key: ResumeKey) -> dict[str, Any] | None:
        record = self.records.get(key.as_id())
        if not record or record.get("status") != "completed":
            return None
        row = record.get("row")
        return dict(row) if isinstance(row, dict) else None

    def append_completed(self, key: ResumeKey, *, row: Mapping[str, Any]) -> None:
        self.append_record(key, status="completed", row=dict(row))

    def append_record(
        self,
        key: ResumeKey,
        *,
        status: str,
        row: Mapping[str, Any] | None = None,
        error: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        record = {
            **key.as_dict(),
            "status": str(status or ""),
            "updated_at": time.time(),
            "row": dict(row or {}),
        }
        if error:
            record["error"] = str(error)
        if metadata:
            record["metadata"] = dict(metadata)
        self.records[key.as_id()] = record
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self.write_summary()

    def completed_rows_for(
        self,
        *,
        task_type: str,
        run_id: str,
        input_fingerprint: str,
    ) -> dict[int, dict[str, Any]]:
        rows: dict[int, dict[str, Any]] = {}
        for record in self.records.values():
            if record.get("status") != "completed":
                continue
            if str(record.get("task_type") or "") != str(task_type or ""):
                continue
            if str(record.get("run_id") or "") != str(run_id or ""):
                continue
            if str(record.get("input_fingerprint") or "") != str(input_fingerprint or ""):
                continue
            row = record.get("row")
            if isinstance(row, dict):
                rows[int(record.get("row_index", -1))] = dict(row)
        return rows

    def write_summary(self, *, extra: Mapping[str, Any] | None = None) -> None:
        if self.path is None:
            return
        counts: dict[str, int] = {}
        for record in self.records.values():
            status = str(record.get("status") or "")
            counts[status] = counts.get(status, 0) + 1
        summary = {
            "checkpoint": str(self.path),
            "updated_at": time.time(),
            **counts,
            "total_records": len(self.records),
        }
        if extra:
            summary.update(dict(extra))
        summary_path_for_checkpoint(self.path).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    def _load(self) -> None:
        if self.path is None:
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = str(record.get("key") or "")
                if not key:
                    key = ResumeKey(
                        row_index=int(record.get("row_index", -1)),
                        task_type=str(record.get("task_type") or ""),
                        run_id=str(record.get("run_id") or ""),
                        input_fingerprint=str(record.get("input_fingerprint") or ""),
                    ).as_id()
                    record["key"] = key
                self.records[key] = record


def default_checkpoint_path(output_path: str | Path) -> Path:
    return Path(f"{Path(output_path)}.checkpoint.jsonl")


def summary_path_for_checkpoint(checkpoint_path: str | Path) -> Path:
    path = Path(checkpoint_path)
    suffix = ".checkpoint.jsonl"
    if path.name.endswith(suffix):
        return path.with_name(f"{path.name[: -len(suffix)]}.resume.json")
    return path.with_suffix(".resume.json")


def input_fingerprint(path: str | Path, *, rows: int | None = None) -> str:
    item = Path(path)
    parts = [str(item.resolve() if item.exists() else item)]
    if item.exists():
        stat = item.stat()
        parts.extend([str(int(stat.st_mtime_ns)), str(int(stat.st_size))])
    if rows is not None:
        parts.append(str(int(rows)))
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def ordered_completed_rows(
    checkpoint: ResumeCheckpoint,
    *,
    task_type: str,
    run_id: str,
    input_fingerprint: str,
    row_indices: Iterable[int],
) -> list[dict[str, Any] | None]:
    completed = checkpoint.completed_rows_for(
        task_type=task_type,
        run_id=run_id,
        input_fingerprint=input_fingerprint,
    )
    return [completed.get(int(index)) for index in row_indices]
