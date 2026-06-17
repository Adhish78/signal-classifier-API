import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.config import Settings
from api.routes.health import router as health_router
from api.routes.metadata import router as metadata_router
from api.routes.predict import router as predict_router
from api.routes.metrics import router as metrics_router
from api.middleware import StructuredLoggingMiddleware, setup_logging
from src.inference import InferenceEngine

# Configure JSON structured logging globally
setup_logging()

settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Load the ONNX model inference engine on startup
    app.state.inference_engine = InferenceEngine(settings.model_path)
    yield


app = FastAPI(
    title=settings.app_name, docs_url="/docs", redoc_url="/redoc", lifespan=lifespan
)

app.add_middleware(StructuredLoggingMiddleware)

# Initialize application start time for uptime tracking
app.state.start_time = time.time()
app.state.total_predictions = 0
app.state.failed_predictions = 0
app.state.average_inference_time_ms = 0.0
app.state.min_inference_time_ms = 0.0
app.state.max_inference_time_ms = 0.0

# Include routes
app.include_router(health_router)
app.include_router(metadata_router)
app.include_router(predict_router)
app.include_router(metrics_router)
