from __future__ import annotations

import json
import logging
import re
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("fraud_intelligence.api")

_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """Add request correlation and structured HTTP request logs."""

    async def dispatch(
        self,
        request: Request,
        call_next,
    ) -> Response:
        start_time = time.perf_counter()

        incoming_request_id = request.headers.get("X-Request-ID", "")
        if _REQUEST_ID_PATTERN.fullmatch(incoming_request_id):
            request_id = incoming_request_id
        else:
            request_id = str(uuid.uuid4())

        request.state.request_id = request_id
        status_code = 500

        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            duration_ms = round(
                (time.perf_counter() - start_time) * 1000,
                2,
            )
            log_record = {
                "event": "http_request",
                "request_id": request_id,
                "method": request.method,
                "route": getattr(request.scope.get("route"), "path", "<unmatched>"),
                "status_code": status_code,
                "duration_ms": duration_ms,
            }
            logger.info(json.dumps(log_record, separators=(",", ":")))
