"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        # request_id -> {user_id, text, start_time}
        self._open: dict[str, dict] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Open a correlation record: request id + input text + start time.

        Every request gets a stable ``request_id`` so the matching output,
        block decision and reviewer action can be replayed later.
        """
        request_id = request_id or f"REQ-{uuid4().hex[:12].upper()}"
        self._open[request_id] = {
            "user_id": user_id,
            "text": text,
            "start": time.time(),
        }
        return request_id

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Close the correlation record and append one audit entry."""
        request_id = request_id or f"REQ-{uuid4().hex[:12].upper()}"
        opened = self._open.pop(request_id, None)
        start = opened["start"] if opened else time.time()
        latency_ms = round((time.time() - start) * 1000, 2)
        self.logs.append(
            {
                "request_id": request_id,
                "user_id": user_id,
                "timestamp": utc_now_iso(),
                "input_text": (opened or {}).get("text", ""),
                "output_text": text,
                "blocked": bool(blocked),
                "layer": layer,
                "latency_ms": latency_ms,
            }
        )
        return request_id

    def export_json(self, filepath: str = "outputs/audit_log.json"):
        """Write the audit trail to disk (JSON array)."""
        out = Path(filepath)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(self.logs, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return out


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
