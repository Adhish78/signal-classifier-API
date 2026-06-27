import json
import logging
import sys
import time
from datetime import UTC, datetime

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

logger = logging.getLogger("api.middleware")


class JSONFormatter(logging.Formatter):
    """
    Custom log formatter that outputs log entries as structured JSON strings.
    
    Rationale:
    In modern cloud environments, unstructured plain text logs are hard to parse.
    Formatting logs as JSON allows centralized log aggregators (e.g., Datadog,
    ELK stack, Splunk) to automatically index key fields (like log levels,
    timestamps, and custom request details) without requiring complex regex rules.
    """
    def format(self, record: logging.LogRecord) -> str:
        try:
            log_data = json.loads(record.getMessage())
            if not isinstance(log_data, dict):
                log_data = {"message": record.getMessage()}
        except (json.JSONDecodeError, TypeError):
            log_data = {"message": record.getMessage()}

        if "timestamp" not in log_data:
            log_data["timestamp"] = datetime.fromtimestamp(
                record.created, tz=UTC
            ).isoformat()
        if "level" not in log_data:
            log_data["level"] = record.levelname
        if "logger" not in log_data:
            log_data["logger"] = record.name

        return json.dumps(log_data)


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to intercept all HTTP requests, measure their total execution
    latency, and log a structured JSON summary.
    """
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        start_time = time.perf_counter()
        timestamp = datetime.now(UTC).isoformat()

        response = await call_next(request)

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        log_payload = {
            "timestamp": timestamp,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
        }

        logger.info(json.dumps(log_payload))
        return response


def setup_logging() -> None:
    root_logger = logging.getLogger()
    # Set to INFO by default
    root_logger.setLevel(logging.INFO)

    # Avoid duplicate handlers
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(console_handler)
