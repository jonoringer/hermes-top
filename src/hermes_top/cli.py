from __future__ import annotations

import argparse
import collections
import json
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
from importlib.metadata import PackageNotFoundError, version as package_version
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ANSI_CLEAR = "\033[2J\033[H"
ANSI_HOME_AND_CLEAR = "\033[H\033[J"
ANSI_CURSOR_HIDE = "\033[?25l"
ANSI_CURSOR_SHOW = "\033[?25h"
GPU_PATTERN = re.compile(
    r"\b(cuda|cudnn|nvidia|vllm|llama\.cpp|ollama|torch|transformers|triton|mlx)\b",
    re.IGNORECASE,
)
CPU_PATTERN = re.compile(
    r"\b(python|node|bash|sh|make|ffmpeg|convert|build|compile|train|infer|render)\b",
    re.IGNORECASE,
)
WEB_PATTERN = re.compile(r"\b(http|https|url|fetch|request|response|api)\b", re.IGNORECASE)
SPARK_CHARS = " .:-=+*#%@"


@dataclass
class SessionInfo:
    session_id: str
    title: str
    source: str
    started_at: str | None
    ended_at: str | None
    updated_at: str | None
    message_count: int = 0
    latest_message_at: str | None = None
    latest_message_role: str | None = None
    latest_message_preview: str = ""


@dataclass
class Operation:
    session_id: str
    session_title: str
    source: str
    started_at: str | None
    duration_seconds: float
    status: str
    kind: str
    tool_name: str
    label: str
    detail: str
    resource_hint: str
    call_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "session_title": self.session_title,
            "source": self.source,
            "started_at": self.started_at,
            "duration_seconds": round(self.duration_seconds, 3),
            "status": self.status,
            "kind": self.kind,
            "tool_name": self.tool_name,
            "label": self.label,
            "detail": self.detail,
            "resource_hint": self.resource_hint,
            "call_id": self.call_id,
        }


@dataclass
class GpuStat:
    index: int
    name: str
    utilization_gpu: float
    memory_used_mb: float | None = None
    memory_total_mb: float | None = None
    temperature_c: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "utilization_gpu": self.utilization_gpu,
            "memory_used_mb": self.memory_used_mb,
            "memory_total_mb": self.memory_total_mb,
            "temperature_c": self.temperature_c,
        }


@dataclass
class SystemSnapshot:
    cpu_count: int | None
    load_1m: float | None
    load_5m: float | None
    load_15m: float | None
    load_percent: float | None
    gpu_stats: list[GpuStat]
    gpu_query_error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "cpu_count": self.cpu_count,
            "load_1m": self.load_1m,
            "load_5m": self.load_5m,
            "load_15m": self.load_15m,
            "load_percent": self.load_percent,
            "gpu_stats": [gpu.as_dict() for gpu in self.gpu_stats],
            "gpu_query_error": self.gpu_query_error,
        }


def default_db_path() -> Path:
    hermes_home = os.environ.get("HERMES_HOME")
    if hermes_home:
        return Path(hermes_home).expanduser() / "state.db"
    return Path.home() / ".hermes" / "state.db"


def installed_version() -> str:
    try:
        return package_version("hermes-top")
    except PackageNotFoundError:
        return "0.0.0+local"


def safe_float(value: str) -> float | None:
    text = (value or "").strip()
    if not text or text in {"[N/A]", "N/A", "na"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def collect_system_snapshot() -> SystemSnapshot:
    cpu_count = os.cpu_count()
    load_1m: float | None = None
    load_5m: float | None = None
    load_15m: float | None = None
    load_percent: float | None = None
    try:
        load_1m, load_5m, load_15m = os.getloadavg()
        if cpu_count and cpu_count > 0:
            load_percent = min(max((load_1m / cpu_count) * 100.0, 0.0), 999.0)
    except (AttributeError, OSError):
        pass

    gpu_stats: list[GpuStat] = []
    gpu_query_error: str | None = None
    if shutil.which("nvidia-smi"):
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=1.5,
                check=False,
            )
            if proc.returncode == 0:
                for line in proc.stdout.splitlines():
                    parts = [part.strip() for part in line.split(",")]
                    if len(parts) < 6:
                        continue
                    index = int(parts[0]) if parts[0].isdigit() else len(gpu_stats)
                    gpu_stats.append(
                        GpuStat(
                            index=index,
                            name=parts[1],
                            utilization_gpu=safe_float(parts[2]) or 0.0,
                            memory_used_mb=safe_float(parts[3]),
                            memory_total_mb=safe_float(parts[4]),
                            temperature_c=safe_float(parts[5]),
                        )
                    )
            else:
                gpu_query_error = (proc.stderr or proc.stdout or "nvidia-smi failed").strip()
        except (OSError, subprocess.SubprocessError) as exc:
            gpu_query_error = str(exc)

    return SystemSnapshot(
        cpu_count=cpu_count,
        load_1m=load_1m,
        load_5m=load_5m,
        load_15m=load_15m,
        load_percent=load_percent,
        gpu_stats=gpu_stats,
        gpu_query_error=gpu_query_error,
    )


def build_sparkline(values: Iterable[float | None], width: int = 20, max_value: float = 100.0) -> str:
    points = [value for value in values if value is not None]
    if not points:
        return " " * width
    clipped = list(points)[-width:]
    spark = []
    for value in clipped:
        ratio = 0.0 if max_value <= 0 else max(min(value / max_value, 1.0), 0.0)
        index = int(round(ratio * (len(SPARK_CHARS) - 1)))
        spark.append(SPARK_CHARS[index])
    return "".join(spark).rjust(width)


def format_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.0f}%"


def format_load(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="hermes-top",
        description="Top-style monitor for Hermes Agent sessions and in-flight tool activity.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {installed_version()}")
    parser.add_argument("--db-path", default=str(default_db_path()), help="Path to Hermes state.db")
    parser.add_argument("--limit", type=int, default=25, help="Maximum rows to display")
    parser.add_argument("--refresh", type=float, default=2.0, help="Refresh interval in seconds")
    parser.add_argument(
        "--max-idle-age",
        type=float,
        default=1800.0,
        help="Hide idle/recent sessions older than this many seconds in the default view (default: 1800)",
    )
    parser.add_argument("--json", action="store_true", help="Emit a JSON snapshot instead of a live table")
    parser.add_argument("--once", action="store_true", help="Render one snapshot and exit")
    parser.set_defaults(include_idle=True)
    parser.add_argument(
        "--active-only",
        dest="include_idle",
        action="store_false",
        help="Hide idle sessions and only show active Hermes operations",
    )
    parser.add_argument(
        "--all-sessions",
        action="store_true",
        help="Show all idle sessions regardless of age",
    )
    parser.add_argument(
        "--include-idle",
        dest="include_idle",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def connect_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}


def choose_column(columns: set[str], *candidates: str) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def read_sessions(conn: sqlite3.Connection) -> dict[str, SessionInfo]:
    columns = table_columns(conn, "sessions")
    if not columns:
        return {}

    session_id_col = choose_column(columns, "id", "session_id")
    title_col = choose_column(columns, "title", "session_title", "name")
    source_col = choose_column(columns, "source", "platform")
    started_col = choose_column(columns, "started_at", "created_at")
    ended_col = choose_column(columns, "ended_at", "closed_at")
    updated_col = choose_column(columns, "updated_at", "last_active_at", "modified_at")

    select_parts = [f"{session_id_col} AS session_id"]
    select_parts.append(f"{title_col} AS title" if title_col else "NULL AS title")
    select_parts.append(f"{source_col} AS source" if source_col else "'unknown' AS source")
    select_parts.append(f"{started_col} AS started_at" if started_col else "NULL AS started_at")
    select_parts.append(f"{ended_col} AS ended_at" if ended_col else "NULL AS ended_at")
    select_parts.append(f"{updated_col} AS updated_at" if updated_col else "NULL AS updated_at")

    query = f"SELECT {', '.join(select_parts)} FROM sessions"
    sessions: dict[str, SessionInfo] = {}
    for row in conn.execute(query):
        session_id = str(row["session_id"])
        title = (row["title"] or "").strip() if isinstance(row["title"], str) else (row["title"] or "")
        if not title:
            title = session_id[:12]
        sessions[session_id] = SessionInfo(
            session_id=session_id,
            title=title,
            source=(row["source"] or "unknown"),
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            updated_at=row["updated_at"],
        )
    return sessions


def read_messages(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    columns = table_columns(conn, "messages")
    if not columns:
        return []

    message_id_col = choose_column(columns, "id", "message_id")
    session_col = choose_column(columns, "session_id")
    role_col = choose_column(columns, "role")
    content_col = choose_column(columns, "content", "text")
    created_col = choose_column(columns, "created_at", "timestamp")
    tool_call_id_col = choose_column(columns, "tool_call_id")
    tool_name_col = choose_column(columns, "tool_name", "name")
    tool_calls_col = choose_column(columns, "tool_calls")

    if not message_id_col or not session_col or not role_col:
        return []

    select_parts = [
        f"{message_id_col} AS message_id",
        f"{session_col} AS session_id",
        f"{role_col} AS role",
        f"{content_col} AS content" if content_col else "NULL AS content",
        f"{created_col} AS created_at" if created_col else "NULL AS created_at",
        f"{tool_call_id_col} AS tool_call_id" if tool_call_id_col else "NULL AS tool_call_id",
        f"{tool_name_col} AS tool_name" if tool_name_col else "NULL AS tool_name",
        f"{tool_calls_col} AS tool_calls" if tool_calls_col else "NULL AS tool_calls",
    ]
    query = f"SELECT {', '.join(select_parts)} FROM messages ORDER BY {message_id_col}"
    return list(conn.execute(query))


def safe_json_loads(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def parse_time(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        if value > 1_000_000_000_000:
            value = value / 1000.0
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.isdigit():
            return parse_time(int(text))
        text = text.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(text)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            pass
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
            try:
                return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
    return None


def summarize_text(value: Any, limit: int = 52) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = " ".join(value.split())
    else:
        text = " ".join(json.dumps(value, ensure_ascii=True).split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def summarize_message_content(value: Any, limit: int = 80) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        stripped = value.strip()
        parsed = safe_json_loads(stripped)
        if parsed is not None:
            return summarize_message_content(parsed, limit=limit)
        return summarize_text(stripped, limit=limit)
    if isinstance(value, dict):
        for key in ("text", "content", "message", "detail", "summary", "result"):
            if key in value and value[key] not in (None, ""):
                return summarize_message_content(value[key], limit=limit)
        return summarize_text(value, limit=limit)
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                if item.get("type") == "text" and item.get("text"):
                    parts.append(str(item["text"]))
                    continue
                if item.get("type") in {"input_text", "output_text"} and item.get("text"):
                    parts.append(str(item["text"]))
                    continue
                if item.get("type") == "tool_result" and item.get("content"):
                    parts.append(summarize_message_content(item["content"], limit=limit))
                    continue
                if item.get("tool_calls"):
                    names = []
                    for call in item["tool_calls"]:
                        if not isinstance(call, dict):
                            continue
                        fn = call.get("function") if isinstance(call.get("function"), dict) else {}
                        names.append(str(fn.get("name") or call.get("name") or "tool"))
                    if names:
                        parts.append("tool calls: " + ", ".join(names))
                        continue
            elif item not in (None, ""):
                parts.append(str(item))
            if " ".join(parts):
                break
        if parts:
            return summarize_text(" ".join(part for part in parts if part), limit=limit)
    return summarize_text(value, limit=limit)


def stringify_args(arguments: Any) -> str:
    if arguments is None:
        return ""
    if isinstance(arguments, str):
        parsed = safe_json_loads(arguments)
        if parsed is not None:
            arguments = parsed
        else:
            return summarize_text(arguments)
    if isinstance(arguments, dict):
        preferred_keys = ("action", "command", "query", "url", "path", "prompt", "location")
        parts = []
        for key in preferred_keys:
            if key in arguments and arguments[key] not in (None, ""):
                parts.append(f"{key}={summarize_text(arguments[key], 28)}")
        if parts:
            return ", ".join(parts)
    return summarize_text(arguments)


def extract_tool_calls(row: sqlite3.Row) -> list[dict[str, Any]]:
    explicit = safe_json_loads(row["tool_calls"])
    if isinstance(explicit, list):
        return [call for call in explicit if isinstance(call, dict)]

    content = safe_json_loads(row["content"])
    if isinstance(content, dict) and isinstance(content.get("tool_calls"), list):
        return [call for call in content["tool_calls"] if isinstance(call, dict)]
    if isinstance(content, list):
        calls: list[dict[str, Any]] = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("tool_calls"), list):
                calls.extend(call for call in part["tool_calls"] if isinstance(call, dict))
        if calls:
            return calls
    return []


def describe_message(row: sqlite3.Row) -> str:
    role = str(row["role"] or "message")
    if role == "assistant":
        tool_calls = extract_tool_calls(row)
        if tool_calls:
            names = []
            for call in tool_calls[:3]:
                fn = call.get("function") if isinstance(call.get("function"), dict) else {}
                names.append(str(fn.get("name") or call.get("name") or "tool"))
            suffix = " ..." if len(tool_calls) > 3 else ""
            return f"assistant called {', '.join(names)}{suffix}"
    if role == "tool":
        tool_name = str(row["tool_name"] or "tool")
        detail = summarize_message_content(row["content"], limit=72)
        return f"{tool_name}: {detail}" if detail else f"{tool_name} result"
    detail = summarize_message_content(row["content"], limit=72)
    return f"{role}: {detail}" if detail else role


def classify_tool(tool_name: str, detail: str) -> tuple[str, str]:
    name = (tool_name or "").lower()
    combined = f"{tool_name} {detail}"
    if name.startswith("web") or WEB_PATTERN.search(combined):
        return "web", "network"
    if name in {"terminal", "process", "bash", "shell", "python"}:
        if GPU_PATTERN.search(combined):
            return "process", "gpu-likely"
        if CPU_PATTERN.search(combined):
            return "process", "cpu-likely"
        return "process", "compute"
    if GPU_PATTERN.search(combined):
        return "tool", "gpu-likely"
    if CPU_PATTERN.search(combined):
        return "tool", "cpu-likely"
    return "tool", "unknown"


def extract_background_result(content: Any) -> dict[str, Any] | None:
    if isinstance(content, str):
        parsed = safe_json_loads(content)
        if parsed is not None:
            return extract_background_result(parsed)
        return None
    if isinstance(content, dict):
        background = content.get("background")
        has_proc_key = any(key in content for key in ("pid", "process_id", "job_id", "command", "session_id"))
        if background is True or has_proc_key:
            return content
        for value in content.values():
            found = extract_background_result(value)
            if found:
                return found
    if isinstance(content, list):
        for item in content:
            found = extract_background_result(item)
            if found:
                return found
    return None


def build_operations(sessions: dict[str, SessionInfo], messages: Iterable[sqlite3.Row]) -> list[Operation]:
    now = datetime.now(timezone.utc)
    outstanding: dict[str, Operation] = {}
    background_ops: dict[str, Operation] = {}
    active_sessions: set[str] = set()

    for row in messages:
        session = sessions.get(str(row["session_id"]))
        if not session:
            continue
        active_sessions.add(session.session_id)
        created_at = parse_time(row["created_at"]) or parse_time(session.started_at) or now
        session.message_count += 1
        session.latest_message_at = created_at.isoformat()
        session.latest_message_role = str(row["role"] or "")
        session.latest_message_preview = describe_message(row)

        if row["role"] == "assistant":
            for call in extract_tool_calls(row):
                fn = call.get("function") if isinstance(call.get("function"), dict) else {}
                tool_name = str(fn.get("name") or call.get("name") or "tool")
                arguments = fn.get("arguments") if fn else call.get("arguments")
                detail = stringify_args(arguments)
                kind, resource_hint = classify_tool(tool_name, detail)
                call_id = str(call.get("id") or "")
                if not call_id:
                    continue
                outstanding[call_id] = Operation(
                    session_id=session.session_id,
                    session_title=session.title,
                    source=session.source,
                    started_at=created_at.isoformat(),
                    duration_seconds=max((now - created_at).total_seconds(), 0.0),
                    status="running",
                    kind=kind,
                    tool_name=tool_name,
                    label=tool_name,
                    detail=detail,
                    resource_hint=resource_hint,
                    call_id=call_id,
                )
            continue

        if row["role"] == "tool":
            tool_call_id = row["tool_call_id"]
            if tool_call_id:
                outstanding.pop(str(tool_call_id), None)

            tool_name = str(row["tool_name"] or "tool")
            result = safe_json_loads(row["content"])
            background = extract_background_result(result)
            if tool_name in {"terminal", "process"} and background:
                detail = stringify_args(background)
                kind, resource_hint = classify_tool(tool_name, detail)
                key = str(
                    background.get("process_id")
                    or background.get("job_id")
                    or background.get("session_id")
                    or row["message_id"]
                )
                status = str(background.get("status") or "running").lower()
                background_ops[key] = Operation(
                    session_id=session.session_id,
                    session_title=session.title,
                    source=session.source,
                    started_at=created_at.isoformat(),
                    duration_seconds=max((now - created_at).total_seconds(), 0.0),
                    status=status,
                    kind=kind,
                    tool_name=tool_name,
                    label=str(background.get("command") or background.get("action") or tool_name),
                    detail=detail,
                    resource_hint=resource_hint,
                    call_id=str(row["tool_call_id"] or ""),
                )

    operations = list(outstanding.values())
    operations.extend(op for op in background_ops.values() if op.status not in {"done", "finished", "completed", "exited"})

    for session_id in active_sessions:
        session = sessions[session_id]
        if session.ended_at:
            continue
        if any(op.session_id == session_id for op in operations):
            continue
        latest_activity = (
            parse_time(session.latest_message_at)
            or parse_time(session.updated_at)
            or parse_time(session.started_at)
            or now
        )
        idle_age = max((now - latest_activity).total_seconds(), 0.0)
        status = "recent" if idle_age <= 30 else "idle"
        operations.append(
            Operation(
                session_id=session.session_id,
                session_title=session.title,
                source=session.source,
                started_at=latest_activity.isoformat(),
                duration_seconds=idle_age,
                status=status,
                kind="session",
                tool_name="",
                label=session.latest_message_role or "session",
                detail=(
                    f"{session.latest_message_preview} [msgs={session.message_count}]"
                    if session.latest_message_preview
                    else f"No recent Hermes message detected [msgs={session.message_count}]"
                ),
                resource_hint="n/a",
                call_id=None,
            )
        )

    def sort_key(op: Operation) -> tuple[int, float, str, str]:
        if op.status == "running":
            return (0, -op.duration_seconds, op.session_id, op.tool_name)
        if op.status == "recent":
            return (1, op.duration_seconds, op.session_id, op.tool_name)
        if op.status == "idle":
            return (2, op.duration_seconds, op.session_id, op.tool_name)
        return (0, -op.duration_seconds, op.session_id, op.tool_name)

    operations.sort(key=sort_key)
    return operations


def human_duration(seconds: float) -> str:
    whole = max(int(seconds), 0)
    hours, rem = divmod(whole, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def clip(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    if width == 1:
        return text[:1]
    return text[: width - 1] + "…"


def render_table(
    db_path: Path,
    operations: list[Operation],
    include_idle: bool,
    limit: int,
    hidden_idle_count: int = 0,
    system: SystemSnapshot | None = None,
    load_history: collections.deque[float | None] | None = None,
    gpu_history: collections.deque[float | None] | None = None,
    active_count: int | None = None,
    idle_count: int | None = None,
) -> str:
    if active_count is None:
        active_count = sum(1 for op in operations if op.status not in {"idle", "recent"})
    if idle_count is None:
        idle_count = sum(1 for op in operations if op.status in {"idle", "recent"})
    rows = [op for op in operations if include_idle or op.status not in {"idle", "recent"}][:limit]
    terminal_width = shutil.get_terminal_size((140, 40)).columns
    cols = [
        ("STATUS", 9, lambda op: op.status),
        ("KIND", 8, lambda op: op.kind),
        ("DURATION", 10, lambda op: human_duration(op.duration_seconds)),
        ("SOURCE", 8, lambda op: op.source),
        ("TOOL", 16, lambda op: op.label or op.tool_name or "—"),
        ("RESOURCE", 10, lambda op: op.resource_hint),
        ("SESSION", 26, lambda op: op.session_title),
    ]
    used = sum(width for _, width, _ in cols) + len(cols) - 1
    detail_width = max(24, terminal_width - used - 1)
    header = " ".join(name.ljust(width) for name, width, _ in cols) + " " + "DETAIL"
    lines = [
        f"hermes-top  db={db_path}",
        (
            f"rows={len(rows)}  showing={'active+idle' if include_idle else 'active-only'}"
            f"  active={active_count}  idle={idle_count}"
        ),
        "",
    ]
    if system is not None:
        load_line = (
            "host load  "
            f"1m={format_load(system.load_1m)}  "
            f"5m={format_load(system.load_5m)}  "
            f"15m={format_load(system.load_15m)}  "
            f"cpu-load={format_percent(system.load_percent)}"
        )
        if system.cpu_count:
            load_line += f"  cores={system.cpu_count}"
        lines.append(load_line)
        if load_history is not None:
            lines.append(f"load hist  {build_sparkline(load_history, width=24)}")
        if system.gpu_stats:
            avg_gpu = sum(gpu.utilization_gpu for gpu in system.gpu_stats) / len(system.gpu_stats)
            lines.append(
                f"gpu load   avg={format_percent(avg_gpu)}  count={len(system.gpu_stats)}"
            )
            if gpu_history is not None:
                lines.append(f"gpu hist   {build_sparkline(gpu_history, width=24)}")
            for gpu in system.gpu_stats:
                mem = ""
                if gpu.memory_used_mb is not None and gpu.memory_total_mb is not None:
                    mem = f" mem={gpu.memory_used_mb:.0f}/{gpu.memory_total_mb:.0f}MiB"
                temp = f" temp={gpu.temperature_c:.0f}C" if gpu.temperature_c is not None else ""
                lines.append(
                    f"gpu{gpu.index:<2} {clip(gpu.name, 24):<24} util={format_percent(gpu.utilization_gpu):>4}{mem}{temp}"
                )
        elif system.gpu_query_error:
            lines.append(f"gpu load   unavailable ({summarize_text(system.gpu_query_error, limit=60)})")
        else:
            lines.append("gpu load   no NVIDIA GPUs detected")
        lines.append("")
    lines.extend(
        [
            header,
            "-" * min(len(header) + detail_width + 1, terminal_width),
        ]
    )
    if not rows:
        if idle_count or hidden_idle_count:
            noun = "session" if idle_count == 1 else "sessions"
            verb = "is" if idle_count == 1 else "are"
            lines.append(
                "No active Hermes-owned operations found in the session database."
            )
            if hidden_idle_count:
                hidden_noun = "session" if hidden_idle_count == 1 else "sessions"
                hidden_verb = "is" if hidden_idle_count == 1 else "are"
                lines.append(
                    f"{hidden_idle_count} older idle {hidden_noun} {hidden_verb} hidden. Re-run with --all-sessions to show them."
                )
            elif idle_count and not include_idle:
                lines.append(
                    f"{idle_count} idle {noun} {verb} hidden. Re-run without --active-only to show them."
                )
        else:
            lines.append("No active Hermes-owned operations found in the session database.")
        return "\n".join(lines)

    for op in rows:
        prefix = " ".join(clip(getter(op), width).ljust(width) for _, width, getter in cols)
        lines.append(prefix + " " + clip(op.detail or "—", detail_width))
    if hidden_idle_count:
        hidden_noun = "session" if hidden_idle_count == 1 else "sessions"
        lines.append("")
        lines.append(
            f"{hidden_idle_count} older idle {hidden_noun} hidden. Re-run with --all-sessions to show them."
        )
    return "\n".join(lines)


def snapshot(db_path: Path, include_idle: bool, all_sessions: bool, max_idle_age: float) -> dict[str, Any]:
    if not db_path.exists():
        return {
            "db_path": str(db_path),
            "exists": False,
            "operations": [],
            "warning": "Hermes state.db was not found. Start Hermes first or pass --db-path.",
        }

    with connect_db(db_path) as conn:
        sessions = read_sessions(conn)
        operations = build_operations(sessions, read_messages(conn))

    active_count = sum(1 for op in operations if op.status not in {"idle", "recent"})
    idle_count = sum(1 for op in operations if op.status in {"idle", "recent"})

    hidden_idle_count = 0
    if include_idle and not all_sessions:
        filtered_operations = []
        for op in operations:
            if op.status in {"idle", "recent"} and op.duration_seconds > max(max_idle_age, 0.0):
                hidden_idle_count += 1
                continue
            filtered_operations.append(op)
        operations = filtered_operations

    if not include_idle:
        operations = [op for op in operations if op.status not in {"idle", "recent"}]

    return {
        "db_path": str(db_path),
        "exists": True,
        "operation_count": len(operations),
        "active_count": active_count,
        "idle_count": idle_count,
        "hidden_idle_count": hidden_idle_count,
        "system": collect_system_snapshot().as_dict(),
        "operations": [op.as_dict() for op in operations],
    }


def run_once(args: argparse.Namespace) -> int:
    db_path = Path(args.db_path).expanduser()
    data = snapshot(
        db_path,
        include_idle=args.include_idle,
        all_sessions=args.all_sessions,
        max_idle_age=args.max_idle_age,
    )
    if args.json:
        print(json.dumps(data, indent=2))
        return 0

    if not data["exists"]:
        print(f"hermes-top  db={db_path}")
        print("")
        print(data["warning"])
        return 1

    operations = [
        Operation(
            session_id=item["session_id"],
            session_title=item["session_title"],
            source=item["source"],
            started_at=item["started_at"],
            duration_seconds=item["duration_seconds"],
            status=item["status"],
            kind=item["kind"],
            tool_name=item["tool_name"],
            label=item["label"],
            detail=item["detail"],
            resource_hint=item["resource_hint"],
            call_id=item["call_id"],
        )
        for item in data["operations"]
    ]
    system = None
    if isinstance(data.get("system"), dict):
        system_dict = data["system"]
        system = SystemSnapshot(
            cpu_count=system_dict.get("cpu_count"),
            load_1m=system_dict.get("load_1m"),
            load_5m=system_dict.get("load_5m"),
            load_15m=system_dict.get("load_15m"),
            load_percent=system_dict.get("load_percent"),
            gpu_stats=[
                GpuStat(
                    index=gpu.get("index", idx),
                    name=gpu.get("name", f"GPU {idx}"),
                    utilization_gpu=gpu.get("utilization_gpu", 0.0),
                    memory_used_mb=gpu.get("memory_used_mb"),
                    memory_total_mb=gpu.get("memory_total_mb"),
                    temperature_c=gpu.get("temperature_c"),
                )
                for idx, gpu in enumerate(system_dict.get("gpu_stats", []))
                if isinstance(gpu, dict)
            ],
            gpu_query_error=system_dict.get("gpu_query_error"),
        )
    print(
        render_table(
            db_path,
            operations,
            args.include_idle,
            args.limit,
            hidden_idle_count=data.get("hidden_idle_count", 0),
            system=system,
            load_history=collections.deque([system.load_percent] if system else [], maxlen=24),
            gpu_history=collections.deque(
                [
                    (
                        sum(gpu.utilization_gpu for gpu in system.gpu_stats) / len(system.gpu_stats)
                        if system and system.gpu_stats
                        else None
                    )
                ],
                maxlen=24,
            ),
            active_count=data.get("active_count"),
            idle_count=data.get("idle_count"),
        )
    )
    return 0


def run_live(args: argparse.Namespace) -> int:
    stop_event = threading.Event()
    load_history: collections.deque[float | None] = collections.deque(maxlen=24)
    gpu_history: collections.deque[float | None] = collections.deque(maxlen=24)

    def handle_signal(_signum: int, _frame: Any) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    db_path = Path(args.db_path).expanduser()
    use_tty_refresh = sys.stdout.isatty()
    if use_tty_refresh:
        sys.stdout.write(ANSI_CURSOR_HIDE)
        sys.stdout.flush()
    try:
        while not stop_event.is_set():
            data = snapshot(
                db_path,
                include_idle=args.include_idle,
                all_sessions=args.all_sessions,
                max_idle_age=args.max_idle_age,
            )
            sys.stdout.write(ANSI_HOME_AND_CLEAR if use_tty_refresh else ANSI_CLEAR)
            if not data["exists"]:
                sys.stdout.write(f"hermes-top  db={db_path}\n\n{data['warning']}\n")
            else:
                system = None
                if isinstance(data.get("system"), dict):
                    system_dict = data["system"]
                    system = SystemSnapshot(
                        cpu_count=system_dict.get("cpu_count"),
                        load_1m=system_dict.get("load_1m"),
                        load_5m=system_dict.get("load_5m"),
                        load_15m=system_dict.get("load_15m"),
                        load_percent=system_dict.get("load_percent"),
                        gpu_stats=[
                            GpuStat(
                                index=gpu.get("index", idx),
                                name=gpu.get("name", f"GPU {idx}"),
                                utilization_gpu=gpu.get("utilization_gpu", 0.0),
                                memory_used_mb=gpu.get("memory_used_mb"),
                                memory_total_mb=gpu.get("memory_total_mb"),
                                temperature_c=gpu.get("temperature_c"),
                            )
                            for idx, gpu in enumerate(system_dict.get("gpu_stats", []))
                            if isinstance(gpu, dict)
                        ],
                        gpu_query_error=system_dict.get("gpu_query_error"),
                    )
                if system is not None:
                    load_history.append(system.load_percent)
                    gpu_history.append(
                        (
                            sum(gpu.utilization_gpu for gpu in system.gpu_stats) / len(system.gpu_stats)
                            if system.gpu_stats
                            else None
                        )
                    )
                operations = [
                    Operation(
                        session_id=item["session_id"],
                        session_title=item["session_title"],
                        source=item["source"],
                        started_at=item["started_at"],
                        duration_seconds=item["duration_seconds"],
                        status=item["status"],
                        kind=item["kind"],
                        tool_name=item["tool_name"],
                        label=item["label"],
                        detail=item["detail"],
                        resource_hint=item["resource_hint"],
                        call_id=item["call_id"],
                    )
                    for item in data["operations"]
                ]
                sys.stdout.write(
                    render_table(
                        db_path,
                        operations,
                        args.include_idle,
                        args.limit,
                        hidden_idle_count=data.get("hidden_idle_count", 0),
                        system=system,
                        load_history=load_history,
                        gpu_history=gpu_history,
                        active_count=data.get("active_count"),
                        idle_count=data.get("idle_count"),
                    )
                )
            sys.stdout.write("\n\nPress Ctrl+C to exit.\n")
            sys.stdout.flush()
            stop_event.wait(max(args.refresh, 0.2))
    finally:
        if use_tty_refresh:
            sys.stdout.write("\n")
            sys.stdout.write(ANSI_CURSOR_SHOW)
            sys.stdout.flush()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.json or args.once:
        return run_once(args)
    return run_live(args)


if __name__ == "__main__":
    raise SystemExit(main())
