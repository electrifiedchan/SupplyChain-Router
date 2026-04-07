import uvicorn
from fastapi.responses import JSONResponse
from openenv.core.env_server import create_fastapi_app
from server.environment import SupplyChainEnv
from models import LogisticsAction, LogisticsObservation

# Pass the CLASS 'SupplyChainEnv', the framework will handle instantiation
app = create_fastapi_app(
    SupplyChainEnv,
    LogisticsAction,
    LogisticsObservation,
)


@app.get("/")
async def health_check():
    """Root health check — HuggingFace Spaces pings this to verify the app is live."""
    return JSONResponse({"status": "ok", "env": "disaster-relief-logistics"})


def main() -> None:
    """Entry point for openenv validate and pyproject.toml [project.scripts]."""
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()