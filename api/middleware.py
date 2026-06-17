import json
import logging
import time
import sys
from datetime import datetime, timezone
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

logger = logging.getLogger("api.middleware")


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        try:
            log_data = json.loads(record.getMessage())
            if not isinstance(log_data, dict):
                log_data = {"message": record.getMessage()}
        except (json.JSONDecodeError, TypeError):
            log_data = {"message": record.getMessage()}

        if "timestamp" not in log_data:
            log_data["timestamp"] = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        if "level" not in log_data:
            log_data["level"] = record.levelname
        if "logger" not in log_data:
            log_data["logger"] = record.name

        return json.dumps(log_data)


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start_time = time.perf_counter()
        timestamp = datetime.now(timezone.utc).isoformat()

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
