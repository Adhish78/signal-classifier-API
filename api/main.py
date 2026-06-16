import time

from fastapi import FastAPI

from api.config import Settings
from api.routes.health import router as health_router
from api.routes.metadata import router as metadata_router
from api.routes.predict import router as predict_router

settings = Settings()

app = FastAPI(title=settings.app_name, docs_url="/docs", redoc_url="/redoc")

# Initialize application start time for uptime tracking
app.state.start_time = time.time()

# Include routes
app.include_router(health_router)
app.include_router(metadata_router)
app.include_router(predict_router)
