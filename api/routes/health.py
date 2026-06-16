import time

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
def health_check(request: Request) -> dict[str, str | float]:
    start_time = getattr(request.app.state, "start_time", time.time())
    return {"status": "healthy", "uptime": time.time() - start_time}
