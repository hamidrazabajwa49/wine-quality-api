import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from .. import __version__
from ..config import settings
from ..logger import get_logger
from ..predictor import Predictor
from .routes import router

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = Predictor(settings.model_path)
    log.info("Model %s loaded", app.state.predictor.version)
    yield


app = FastAPI(
    title="Wine Quality API",
    version=__version__,
    description="Predicts the quality score (0-10) of a red wine from 11 lab measurements.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = uuid.uuid4().hex[:12]
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    log.info("%s %s %s %.1fms id=%s", request.method, request.url.path, response.status_code, elapsed_ms, request_id)
    return response


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
    log.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/docs")


app.include_router(router)
