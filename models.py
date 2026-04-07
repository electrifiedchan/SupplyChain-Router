from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Dict, Literal, List, Any, Optional
from openenv.core.env_server import Action, Observation, State

# ─── 1. ENTITY MODELS (The Game Pieces) ──────────────────────────────────────

class ReliefPallet(BaseModel):
    """
    Represents one pallet of disaster relief supplies.

    The hazard_class determines containment rules:
      - 'safe'     : no restrictions, load anywhere
      - 'chemical' : cannot share a helicopter with 'medical' pallets
      - 'medical'  : cannot share a helicopter with 'chemical' pallets
                     (contamination risk triggers heavy penalty)

    The priority determines score impact:
      - 'standard' : normal scoring
      - 'critical'  : undelivered critical pallets apply a large score penalty
    """

    weight: int = Field(
        ...,
        gt=0,
        description="Base weight of the pallet in lbs. Must be greater than zero.",
        json_schema_extra={"example": 30},
    )
    priority: Literal["standard", "critical"] = Field(
        ...,
        description=(
            "Delivery urgency. 'critical' pallets (e.g., blood, medicine) "
            "apply a heavy score penalty if left unrouted."
        ),
        json_schema_extra={"example": "critical"},
    )
    hazard_class: Literal["safe", "chemical", "medical"] = Field(
        ...,
        description=(
            "Hazard type of the pallet contents. "
            "Loading 'chemical' and 'medical' pallets into the same helicopter "
            "triggers the Dynamic Weight Trap — a containment penalty is added "
            "to that helicopter's load."
        ),
        json_schema_extra={"example": "medical"},
    )


class Helicopter(BaseModel):
    """
    Represents one helicopter available for disaster relief routing.

    current_load tracks the REAL weight including any containment penalties
    applied by the Dynamic Weight Trap. The AI must account for this when
    deciding what to load next.
    """

    max_capacity: int = Field(
        ...,
        gt=0,
        description="Maximum weight the helicopter can carry before it is unsafe to fly.",
        json_schema_extra={"example": 80},
    )
    current_load: int = Field(
        0,
        ge=0,
        description=(
            "Current total weight in lbs, including any dynamic containment "
            "penalties. Updated after every move."
        ),
        json_schema_extra={"example": 30},
    )
    loaded_pallets: List[str] = Field(
        default_factory=list,
        description=(
            "Ordered list of pallet IDs currently inside this helicopter. "
            "No pallet ID should appear more than once — the environment "
            "firewall enforces this, but treat duplicates as a critical bug."
        ),
        json_schema_extra={"example": ["Pallet_1", "Pallet_3"]},
    )
    containment_penalty_active: bool = Field(
        False,
        description=(
            "True if this helicopter is currently carrying both chemical and "
            "medical pallets. The Dynamic Weight Trap has added extra weight "
            "to current_load to simulate containment equipment."
        ),
    )

    @property
    def remaining_capacity(self) -> int:
        """How many more lbs this helicopter can safely accept."""
        return max(0, self.max_capacity - self.current_load)

    @field_validator("loaded_pallets")
    @classmethod
    def no_duplicate_pallets(cls, v: List[str]) -> List[str]:
        """Catch duplicate pallet IDs at the model layer before they reach the firewall."""
        if len(v) != len(set(v)):
            duplicates = [p for p in v if v.count(p) > 1]
            raise ValueError(
                f"loaded_pallets contains duplicate pallet IDs: {list(set(duplicates))}. "
                f"Each pallet can only be loaded once."
            )
        return v

    @model_validator(mode="after")
    def load_cannot_exceed_capacity(self) -> "Helicopter":
        """Second line of defense: model rejects overloaded state."""
        if self.current_load > self.max_capacity:
            raise ValueError(
                f"current_load ({self.current_load} lb) exceeds "
                f"max_capacity ({self.max_capacity} lb). "
                f"This helicopter would crash."
            )
        return self


# ─── 2. COMMUNICATION PROTOCOL MODELS ────────────────────────────────────────

class LogisticsInfo(BaseModel):
    """
    Evaluator metadata returned with every step.
    Separate from LogisticsObservation so the AI sees the board
    and the evaluator sees the scoring data — never mixed.
    """

    reason: str = Field(
        ...,
        description="Human-readable explanation of what happened this step.",
        json_schema_extra={"example": "Valid move: Pallet_1 loaded onto Heli_A."},
    )
    penalty_applied: float = Field(
        0.0,
        ge=0.0,
        description=(
            "Penalty amount this step. 0.0 means a clean legal move. "
            "1.0 means a firewall rule was broken and the episode ended."
        ),
        json_schema_extra={"example": 0.0},
    )
    dynamic_weight_trap_triggered: bool = Field(
        False,
        description=(
            "True if this move triggered the Dynamic Weight Trap — "
            "chemical and medical pallets now share a helicopter, "
            "adding containment weight to that helicopter's load."
        ),
    )
    containment_weight_added: int = Field(
        0,
        ge=0,
        description=(
            "How many lbs of containment equipment were added to a helicopter "
            "this step due to the Dynamic Weight Trap. 0 if trap not triggered."
        ),
    )
    oracle_comparison: Optional[str] = Field(
        None,
        description=(
            "On episode completion, a summary comparing the AI's solution "
            "to the Oracle's mathematical baseline. None during active play."
        ),
    )


class LogisticsAction(Action):
    """
    The exact JSON the AI must send to make one move.
    Load one pallet onto one helicopter per action.
    """

    helicopter_id: str = Field(
        ...,
        description="The ID of the target helicopter (e.g., 'Heli_A').",
        json_schema_extra={"example": "Heli_A"},
    )
    pallet_id: str = Field(
        ...,
        description="The ID of the pallet to load (e.g., 'Pallet_1').",
        json_schema_extra={"example": "Pallet_1"},
    )


class LogisticsObservation(Observation):
    """
    The complete board state sent to the AI after every action.
    Inherits done: bool and reward: Optional[float] from Observation base.

    The AI uses this to decide its next move.
    Scoring metadata (penalties, oracle data) is kept separate in LogisticsInfo
    so the AI cannot read its own penalty score and game the reward signal.
    """

    step_count: int = Field(
        0,
        ge=0,
        description="How many steps have been taken in this episode.",
        json_schema_extra={"example": 3},
    )
    task_difficulty: Literal["easy", "medium", "hard"] = Field(
        "easy",
        description=(
            "Current scenario difficulty. Controls how many pallets exist, "
            "their weights, and whether the Dynamic Weight Trap is active."
        ),
        json_schema_extra={"example": "hard"},
    )
    remaining_pallets: Dict[str, ReliefPallet] = Field(
        default_factory=dict,
        description=(
            "All pallets not yet loaded onto any helicopter. "
            "Key is pallet ID, value is full pallet details including "
            "weight, priority, and hazard class."
        ),
    )
    helicopters: Dict[str, Helicopter] = Field(
        default_factory=dict,
        description=(
            "All helicopters and their current state. "
            "Key is helicopter ID, value includes current load, "
            "max capacity, loaded pallets, and containment status."
        ),
    )
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Evaluator metadata from the last step. "
            "Contains reason string and penalty data. "
            "Use LogisticsInfo to parse this field."
        ),
    )


class LogisticsState(State):
    """
    Internal episode state — returned by the /state endpoint.
    Inherits episode_id: Optional[str] and step_count: int from State base.
    """
    difficulty: str = Field(
        "easy",
        description="Current difficulty level: easy, medium, or hard.",
    )
    pallets_remaining: int = Field(
        0,
        ge=0,
        description="Number of pallets not yet routed.",
    )
    done: bool = Field(
        False,
        description="Whether the current episode has ended.",
    )