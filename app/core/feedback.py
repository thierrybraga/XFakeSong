"""Central feedback hub for terminal logs, API snapshots, and UI notifications."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Deque, Dict, List, Literal, Optional

FeedbackLevel = Literal["success", "info", "warning", "error", "critical"]

_LEVEL_ICONS = {
    "success": "[OK]",
    "info": "[INFO]",
    "warning": "[WARN]",
    "error": "[ERROR]",
    "critical": "[CRITICAL]",
}
_LEVEL_COLORS = {
    "success": "#10b981",
    "info": "#06b6d4",
    "warning": "#f59e0b",
    "error": "#ef4444",
    "critical": "#dc2626",
}
_LOG_TO_FEEDBACK = {
    logging.DEBUG: "info",
    logging.INFO: "info",
    logging.WARNING: "warning",
    logging.ERROR: "error",
    logging.CRITICAL: "critical",
}
_FEEDBACK_TO_LOG = {
    "success": logging.INFO,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}
_STANDARD_LOG_ATTRS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}


@dataclass
class FeedbackEvent:
    """Structured feedback event shared by terminal, API, and frontend."""

    level: FeedbackLevel
    title: str
    message: str = ""
    source: str = "system"
    category: str = "event"
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    error_code: Optional[str] = None
    hint: Optional[str] = None
    link: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def icon(self) -> str:
        return _LEVEL_ICONS.get(self.level, "[INFO]")

    @property
    def color(self) -> str:
        return _LEVEL_COLORS.get(self.level, "#94a3b8")

    @property
    def ago(self) -> str:
        delta = time.time() - self.timestamp
        if delta < 60:
            return f"{int(delta)}s atras"
        if delta < 3600:
            return f"{int(delta / 60)}min atras"
        if delta < 86400:
            return f"{int(delta / 3600)}h atras"
        return f"{int(delta / 86400)}d atras"

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["icon"] = self.icon
        data["color"] = self.color
        data["ago"] = self.ago
        return data


class FeedbackHub:
    """Thread-safe ring buffer for system feedback events."""

    def __init__(self, maxlen: int = 200):
        self._events: Deque[FeedbackEvent] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._unread_count = 0

    def publish(
        self,
        level: FeedbackLevel,
        title: str,
        message: str = "",
        *,
        source: str = "system",
        category: str = "event",
        error_code: Optional[str] = None,
        hint: Optional[str] = None,
        link: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        emit_log: bool = True,
    ) -> FeedbackEvent:
        event = FeedbackEvent(
            level=level,
            title=title,
            message=message,
            source=source,
            category=category,
            error_code=error_code,
            hint=hint,
            link=link,
            metadata=metadata or {},
        )
        with self._lock:
            self._events.append(event)
            self._unread_count += 1

        if emit_log:
            log_message = title if not message else f"{title} - {message}"
            logging.getLogger(source).log(
                _FEEDBACK_TO_LOG.get(level, logging.INFO),
                log_message,
                extra={"xf_feedback_event": True},
            )
        return event

    def publish_from_log(self, record: logging.LogRecord) -> FeedbackEvent:
        level = _LOG_TO_FEEDBACK.get(record.levelno, "info")
        title = record.getMessage()
        extra = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _STANDARD_LOG_ATTRS and not key.startswith("_")
        }
        message = ""
        if (
            title == "request_completed"
            and {"method", "path", "status"} <= extra.keys()
        ):
            message = (
                f"{extra.get('method')} {extra.get('path')} -> "
                f"{extra.get('status')} ({extra.get('elapsed_ms')}ms)"
            )
        return self.publish(
            level,
            title,
            message,
            source=record.name,
            category="log",
            metadata={
                "logger": record.name,
                "levelname": record.levelname,
                "pathname": record.pathname,
                "lineno": record.lineno,
                **extra,
            },
            emit_log=False,
        )

    def get_events(self, limit: int = 50) -> List[FeedbackEvent]:
        with self._lock:
            events = list(self._events)
        events.reverse()
        return events[: max(0, limit)]

    def clear(self) -> int:
        with self._lock:
            count = len(self._events)
            self._events.clear()
            self._unread_count = 0
        return count

    def mark_all_read(self) -> None:
        with self._lock:
            self._unread_count = 0

    def unread_count(self) -> int:
        with self._lock:
            return self._unread_count

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            events = list(self._events)
            unread = self._unread_count
        counts = {"success": 0, "info": 0, "warning": 0, "error": 0, "critical": 0}
        for event in events:
            counts[event.level] = counts.get(event.level, 0) + 1
        latest = events[-1].as_dict() if events else None
        return {
            "total": len(events),
            "unread": unread,
            "counts": counts,
            "latest": latest,
        }


class FeedbackLogHandler(logging.Handler):
    """Logging handler that mirrors terminal logs into the feedback hub."""

    def emit(self, record: logging.LogRecord) -> None:
        if getattr(record, "xf_feedback_event", False):
            return
        if record.name.startswith("uvicorn.access"):
            return
        try:
            feedback_hub.publish_from_log(record)
        except Exception:
            self.handleError(record)


feedback_hub = FeedbackHub()
_handler_lock = threading.Lock()


def install_feedback_logging(level: int = logging.INFO) -> None:
    """Install the feedback logging handler once on the root logger."""

    root = logging.getLogger()
    with _handler_lock:
        for handler in root.handlers:
            if isinstance(handler, FeedbackLogHandler):
                handler.setLevel(level)
                return
        handler = FeedbackLogHandler(level=level)
        root.addHandler(handler)


def configure_logging(
    *,
    level: int = logging.INFO,
    log_file: str = "system.log",
    force: bool = False,
) -> None:
    """Configure terminal/file logging and attach the feedback mirror."""

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        handlers.insert(0, logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=force,
    )
    install_feedback_logging(level=level)


def publish_feedback(
    level: FeedbackLevel,
    title: str,
    message: str = "",
    **kwargs: Any,
) -> FeedbackEvent:
    return feedback_hub.publish(level, title, message, **kwargs)


def get_feedback_events(limit: int = 50) -> List[FeedbackEvent]:
    return feedback_hub.get_events(limit=limit)


def clear_feedback_events() -> int:
    return feedback_hub.clear()


def mark_feedback_read() -> None:
    feedback_hub.mark_all_read()


def get_feedback_unread_count() -> int:
    return feedback_hub.unread_count()


def get_feedback_summary() -> Dict[str, Any]:
    return feedback_hub.summary()


__all__ = [
    "FeedbackEvent",
    "FeedbackHub",
    "FeedbackLevel",
    "FeedbackLogHandler",
    "clear_feedback_events",
    "configure_logging",
    "feedback_hub",
    "get_feedback_events",
    "get_feedback_summary",
    "get_feedback_unread_count",
    "install_feedback_logging",
    "mark_feedback_read",
    "publish_feedback",
]
