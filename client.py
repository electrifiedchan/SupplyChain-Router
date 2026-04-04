"""
SupplyChain-Router Environment Client

Typed EnvClient wrapper for the Disaster Relief Logistics environment.
Use this to connect an agent to the environment server.

Usage:
    async with SupplyChainEnvClient.from_env() as env:
        obs = await env.reset()
        while not obs.done:
            action = LogisticsAction(helicopter_id="Heli_A", pallet_id="Pallet_1")
            result = await env.step(action)
            obs = result.observation
"""

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from models import LogisticsAction, LogisticsObservation, LogisticsState


class SupplyChainEnvClient(EnvClient[LogisticsAction, LogisticsObservation, LogisticsState]):
    """
    Typed client for the Disaster Relief Logistics environment.

    Handles WebSocket communication, serialization, and deserialization
    following the OpenEnv EnvClient pattern.
    """

    def _step_payload(self, action: LogisticsAction) -> dict:
        """Serialize action to the JSON payload the server expects."""
        return {
            "helicopter_id": action.helicopter_id,
            "pallet_id": action.pallet_id,
        }

    def _parse_result(self, payload: dict) -> StepResult[LogisticsObservation]:
        """Deserialize the server's step response into a typed StepResult."""
        obs_data = payload.get("observation", {})
        done = payload.get("done", obs_data.get("done", False))
        reward = payload.get("reward", obs_data.get("reward", 0.0))

        observation = LogisticsObservation(
            done=done,
            reward=reward,
            step_count=obs_data.get("step_count", 0),
            task_difficulty=obs_data.get("task_difficulty", "easy"),
            remaining_pallets=obs_data.get("remaining_pallets", {}),
            helicopters=obs_data.get("helicopters", {}),
            info=obs_data.get("info", {}),
        )

        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
        )

    def _parse_state(self, payload: dict) -> LogisticsState:
        """Deserialize the server's state response into a typed LogisticsState."""
        return LogisticsState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            difficulty=payload.get("difficulty", "easy"),
            pallets_remaining=payload.get("pallets_remaining", 0),
            done=payload.get("done", False),
        )
