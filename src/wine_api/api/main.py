from contextlib import asynccontextmanager
from fastapi import FastAPI

from .. import __version__
from ..config import settings
from ..logger import get_logger
from .routes import router

log = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = Predictor(settings.model_path)
    log.info("Model %s loaded", app.state.predictor.version)
    yield

app = FastAPI(title="Wine Quality API", version=__version__,lifespan=lifespan)
app.include_router(router)
