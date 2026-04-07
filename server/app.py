<<<<<<< HEAD
import uvicorn
from openenv.core.env_server import create_fastapi_app
from server.environment import SupplyChainEnv
from models import LogisticsAction, LogisticsObservation

# Pass the CLASS 'SupplyChainEnv', the framework will handle instantiation
app = create_fastapi_app(
    SupplyChainEnv,
    LogisticsAction,
    LogisticsObservation,
)


def main() -> None:
    """Entry point for openenv validate and pyproject.toml [project.scripts]."""
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
=======
import uvicorn
from openenv.core.env_server import create_fastapi_app
from server.environment import SupplyChainEnv
from models import LogisticsAction, LogisticsObservation

# Pass the CLASS 'SupplyChainEnv', the framework will handle instantiation
app = create_fastapi_app(
    SupplyChainEnv,
    LogisticsAction,
    LogisticsObservation,
)


def main() -> None:
    """Entry point for openenv validate and pyproject.toml [project.scripts]."""
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
>>>>>>> 530d28dcc2acbad075ba47a00e9e840d702cb383
    main()