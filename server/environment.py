import sys
import os
import copy
import logging
from typing import Dict, Any, Optional, List, Tuple

from openenv.core.env_server import Environment

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import (
    ReliefPallet,
    Helicopter,
    LogisticsObservation,
    LogisticsInfo,
    LogisticsState,
)

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
CONTAINMENT_PENALTY_WEIGHT: int = 50
MAX_STEPS: int = 15
DIFFICULTY_CYCLE: List[str] = ["easy", "medium", "hard"]
# Dense reward: intermediate reward is now computed dynamically per step
FAILURE_REWARD: float = 0.001        # Keep this clamped!
REPETITION_PENALTY: float = 0.0      # Must be 0.0 to not drag the sum below zero
REPETITION_WINDOW: int = 3           # track last N actions for repeat detection
PENALTY_AMOUNT: float = 1.0
UTILIZATION_WEIGHT: float = 0.60     # blended win score: 60% fleet utilization
PRIORITY_WEIGHT: float = 0.40        # blended win score: 40% critical delivery

# ─── Scenario Definitions ─────────────────────────────────────────────────────
SCENARIOS: Dict[str, Dict[str, Any]] = {
    "easy": {
        "pallets": {
            "Pallet_1": ReliefPallet(weight=20, priority="standard", hazard_class="safe"),
            "Pallet_2": ReliefPallet(weight=30, priority="standard", hazard_class="safe"),
            "Pallet_3": ReliefPallet(weight=40, priority="standard", hazard_class="safe"),
            "Pallet_4": ReliefPallet(weight=10, priority="standard", hazard_class="safe"),
        },
        "helicopters": {
            "Heli_A": Helicopter(max_capacity=60),
            "Heli_B": Helicopter(max_capacity=60),
        },
    },
    "medium": {
        "pallets": {
            "Pallet_1": ReliefPallet(weight=20, priority="critical", hazard_class="safe"),
            "Pallet_2": ReliefPallet(weight=30, priority="standard", hazard_class="safe"),
            "Pallet_3": ReliefPallet(weight=40, priority="critical", hazard_class="safe"),
            "Pallet_4": ReliefPallet(weight=30, priority="standard", hazard_class="safe"),
            "Pallet_5": ReliefPallet(weight=50, priority="standard", hazard_class="safe"),
        },
        "helicopters": {
            "Heli_A": Helicopter(max_capacity=100),
            "Heli_B": Helicopter(max_capacity=100),
        },
    },
    "hard": {
        "pallets": {
            "Pallet_1": ReliefPallet(weight=30, priority="critical",  hazard_class="medical"),
            "Pallet_2": ReliefPallet(weight=40, priority="standard",  hazard_class="chemical"),
            "Pallet_3": ReliefPallet(weight=20, priority="critical",  hazard_class="safe"),
            "Pallet_4": ReliefPallet(weight=50, priority="standard",  hazard_class="safe"),
            "Pallet_5": ReliefPallet(weight=30, priority="standard",  hazard_class="medical"),
            "Pallet_6": ReliefPallet(weight=20, priority="critical",  hazard_class="chemical"),
        },
        "helicopters": {
            "Heli_A": Helicopter(max_capacity=100),
            "Heli_B": Helicopter(max_capacity=100),
            "Heli_C": Helicopter(max_capacity=80),
        },
    },
}

def _validate_scenarios() -> None:
    """Crash at startup if any scenario is mathematically impossible."""
    for difficulty, scenario in SCENARIOS.items():
        total_weight = sum(p.weight for p in scenario["pallets"].values())
        total_capacity = sum(h.max_capacity for h in scenario["helicopters"].values())
        if total_weight > total_capacity:
            raise ValueError(
                f"Scenario '{difficulty}' is impossible: "
                f"{total_weight} lb pallets cannot fit in {total_capacity} lb capacity."
            )
        logger.info(
            "✅ Scenario '%s': %d lb pallets / %d lb capacity",
            difficulty, total_weight, total_capacity,
        )

_validate_scenarios()

# ─── Environment ──────────────────────────────────────────────────────────────
class SupplyChainEnv(Environment):
    """
    Disaster Relief Logistics Environment.

    Returns only LogisticsObservation from step() and reset().
    reward and info are embedded inside the observation so the
    openenv framework can call .model_dump() on the returned object.
    """

    def __init__(self) -> None:
        super().__init__()
        # episode_count=0 so the first reset() call produces episode 1 → "easy"
        self._episode_count: int = 0
        self._step: int = 0
        self._difficulty: str = "easy"
        self._done: bool = False
        self._failure_reason: Optional[str] = None
        self._action_history: List[Tuple[str, str]] = []
        # Pre-load the EASY scenario as the initial stand-by state
        # (reset() will properly initialise episode 1 = EASY on first call)
        _init_scenario = copy.deepcopy(SCENARIOS["easy"])
        self._pallets: Dict[str, ReliefPallet] = _init_scenario["pallets"]
        self._helicopters: Dict[str, Helicopter] = _init_scenario["helicopters"]
        self._pallets_reference: Dict[str, ReliefPallet] = copy.deepcopy(_init_scenario["pallets"])
        self._useful_load: Dict[str, int] = {h_id: 0 for h_id in self._helicopters}

    # ── Public API ─────────────────────────────────────────────────────────────

    def reset(self) -> LogisticsObservation:
        """Reset to next difficulty and return clean starting observation."""
        self._initialise_episode()
        return self._build_observation(
            reward=0.0,
            info=LogisticsInfo(reason="Episode started. Good luck."),
        )

    def step(self, action: Any) -> LogisticsObservation:
        """
        Execute one action.
        Returns LogisticsObservation with reward and info embedded.
        The openenv framework calls .model_dump() on the returned object.
        """
        try:
            if self._done:
                return self._trigger_failure(
                    "Logic Error: step() called after episode already finished."
                )

            if self._step >= MAX_STEPS:
                return self._trigger_failure(
                    f"Timeout: {MAX_STEPS} steps reached."
                )

            self._step += 1

            # ── Parse action ──────────────────────────────────────────────────
            h_id, p_id, parse_error = self._parse_action(action)
            if parse_error:
                return self._trigger_failure(parse_error)

            # ── Repetition Check (soft 0.0 penalty, no episode end) ──────────
            if (h_id, p_id) in self._action_history[-REPETITION_WINDOW:]:
                logger.warning(
                    "🔁 Repeat action detected: %s → %s. Applying 0.0 penalty.", p_id, h_id
                )
                self._action_history.append((h_id, p_id))
                return self._build_observation(
                    reward=REPETITION_PENALTY,
                    info=LogisticsInfo(
                        reason=(
                            f"Repeat action: '{p_id}' → '{h_id}' was already attempted. "
                            f"Try a different pallet or helicopter."
                        ),
                        penalty_applied=abs(REPETITION_PENALTY),
                    ),
                )

            # ── Layer 1: Existence Firewall ───────────────────────────────────
            if h_id not in self._helicopters:
                return self._trigger_failure(
                    f"Hallucination: Helicopter '{h_id}' does not exist. "
                    f"Valid: {list(self._helicopters.keys())}"
                )

            if p_id not in self._pallets:
                return self._trigger_failure(
                    f"Hallucination: Pallet '{p_id}' does not exist "
                    f"or already routed. "
                    f"Remaining: {list(self._pallets.keys())}"
                )

            target_heli = self._helicopters[h_id]
            target_pallet = self._pallets[p_id]

            # ── Layer 2: Dynamic Weight Trap (hard only) ──────────────────────
            trap_triggered = False
            if (
                self._difficulty == "hard"
                and not target_heli.containment_penalty_active
            ):
                existing_hazards = {
                    self._pallets_reference[pid].hazard_class
                    for pid in target_heli.loaded_pallets
                }
                incoming = target_pallet.hazard_class
                mixing = (
                    incoming == "chemical" and "medical" in existing_hazards
                ) or (
                    incoming == "medical" and "chemical" in existing_hazards
                )
                if mixing:
                    target_heli.containment_penalty_active = True
                    target_heli.current_load += CONTAINMENT_PENALTY_WEIGHT
                    trap_triggered = True
                    logger.warning(
                        "🚨 TRAP SPRUNG on %s: +%d lb containment penalty.",
                        h_id, CONTAINMENT_PENALTY_WEIGHT,
                    )

            # ── Layer 3: Capacity Check (after trap applied) ──────────────────
            if target_heli.current_load + target_pallet.weight > target_heli.max_capacity:
                overflow = (
                    target_heli.current_load + target_pallet.weight
                ) - target_heli.max_capacity
                reason = (
                    f"Physics Violation: '{p_id}' ({target_pallet.weight} lb) "
                    f"onto '{h_id}' ({target_heli.current_load}/"
                    f"{target_heli.max_capacity} lb) overflows by {overflow} lb."
                )
                if trap_triggered:
                    reason += (
                        f" Crash caused by Dynamic Weight Trap "
                        f"(+{CONTAINMENT_PENALTY_WEIGHT} lb containment penalty)."
                    )
                return self._trigger_failure(reason)

            # ── Apply legal move ──────────────────────────────────────────────
            self._action_history.append((h_id, p_id))  # type: ignore[arg-type]
            self._pallets.pop(p_id)
            target_heli.loaded_pallets.append(p_id)
            target_heli.current_load += target_pallet.weight
            self._useful_load[h_id] += target_pallet.weight

            # ── Dense Reward: multi-signal progress fraction ─────────────
            total_pallets = len(self._pallets_reference)
            delivered_pallets = total_pallets - len(self._pallets)

            # Signal 1: count-based delivery progress
            count_progress = (
                delivered_pallets / total_pallets if total_pallets > 0 else 0.0
            )

            # Signal 2: weight-based delivery progress
            total_weight = sum(p.weight for p in self._pallets_reference.values())
            delivered_weight = total_weight - sum(
                p.weight for p in self._pallets.values()
            )
            weight_progress = (
                delivered_weight / total_weight if total_weight > 0 else 0.0
            )

            # Signal 3: critical-pallet delivery progress
            total_criticals = sum(
                1 for p in self._pallets_reference.values()
                if p.priority == "critical"
            )
            if total_criticals > 0:
                remaining_criticals = sum(
                    1 for p in self._pallets.values()
                    if p.priority == "critical"
                )
                critical_progress = (
                    (total_criticals - remaining_criticals) / total_criticals
                )
            else:
                # No criticals in scenario → mirror count progress
                critical_progress = count_progress

            # Blend: 40% count + 35% weight + 25% priority
            raw_progress = (
                0.40 * count_progress
                + 0.35 * weight_progress
                + 0.25 * critical_progress
            )

            # Smooth concave scaling — front-loads reward so early correct
            # moves feel meaningful (trajectory ≈ 0.45 → 0.62 → 0.78)
            step_reward = max(0.01, min(0.99, raw_progress ** 0.55))

            logger.info(
                "📈 Dense reward: count=%.2f weight=%.2f crit=%.2f "
                "→ raw=%.3f → reward=%.3f",
                count_progress, weight_progress, critical_progress,
                raw_progress, step_reward,
            )

            logger.info(
                "✅ %s (%d lb, %s, %s) → %s | Load: %d/%d lb",
                p_id, target_pallet.weight,
                target_pallet.priority, target_pallet.hazard_class,
                h_id,
                target_heli.current_load, target_heli.max_capacity,
            )

            # ── Win condition ─────────────────────────────────────────────────
            if len(self._pallets) == 0:
                self._done = True
                blended_score, oracle_report = self._evaluate_final_solution()
                return self._build_observation(
                    reward=blended_score,
                    info=LogisticsInfo(
                        reason=f"Mission Complete! {oracle_report}",
                        penalty_applied=0.0,
                        dynamic_weight_trap_triggered=trap_triggered,
                        containment_weight_added=(
                            CONTAINMENT_PENALTY_WEIGHT if trap_triggered else 0
                        ),
                        oracle_comparison=oracle_report,
                    ),
                )

            # ── Timeout condition ─────────────────────────────────────────────
            if self._step >= MAX_STEPS:
                self._done = True
                return self._build_observation(
                    reward=FAILURE_REWARD,
                    info=LogisticsInfo(
                        reason=(
                            f"Timeout: {MAX_STEPS} steps reached. "
                            f"{len(self._pallets)} pallet(s) unrouted."
                        ),
                        penalty_applied=0.0,
                    ),
                )

            # ── Normal step ───────────────────────────────────────────────────
            return self._build_observation(
                reward=step_reward,
                info=LogisticsInfo(
                    reason=(
                        f"Valid: '{p_id}' ({target_pallet.weight} lb, "
                        f"{target_pallet.priority}, {target_pallet.hazard_class}) "
                        f"→ '{h_id}'. "
                        f"Load: {target_heli.current_load}/{target_heli.max_capacity} lb. "
                        f"{len(self._pallets)} pallet(s) remaining."
                    ),
                    penalty_applied=0.0,
                    dynamic_weight_trap_triggered=trap_triggered,
                    containment_weight_added=(
                        CONTAINMENT_PENALTY_WEIGHT if trap_triggered else 0
                    ),
                ),
            )

        except Exception as e:
            logger.error("CRITICAL ERROR in step(): %s", e, exc_info=True)
            return self._trigger_failure(f"Internal Server Error: {str(e)}")

    # ── State property ─────────────────────────────────────────────────────────

    SUPPORTS_CONCURRENT_SESSIONS = True

    @property
    def state(self) -> LogisticsState:
        return LogisticsState(
            step_count=self._step,
            difficulty=self._difficulty,
            pallets_remaining=len(self._pallets),
            done=self._done,
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _initialise_episode(self) -> None:
        self._episode_count += 1
        self._difficulty = DIFFICULTY_CYCLE[
            (self._episode_count - 1) % len(DIFFICULTY_CYCLE)
        ]
        scenario = copy.deepcopy(SCENARIOS[self._difficulty])
        self._step = 0
        self._done = False
        self._failure_reason = None
        self._pallets = scenario["pallets"]
        self._helicopters = scenario["helicopters"]
        self._pallets_reference = copy.deepcopy(scenario["pallets"])
        self._useful_load = {h_id: 0 for h_id in self._helicopters}
        self._action_history = []
        logger.info(
            "🚁 Episode %d → Difficulty: [%s] | %d pallets | %d helicopters",
            self._episode_count,
            self._difficulty.upper(),
            len(self._pallets),
            len(self._helicopters),
        )

    def _parse_action(
        self, action: Any
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Safely extract helicopter_id and pallet_id.
        Returns (h_id, p_id, error_string).
        error_string is None on success.
        """
        if isinstance(action, dict):
            h_id = action.get("helicopter_id")
            p_id = action.get("pallet_id")
        else:
            h_id = getattr(action, "helicopter_id", None)
            p_id = getattr(action, "pallet_id", None)

        if not h_id and not p_id:
            return None, None, f"Malformed action: missing both IDs. Got: {action!r}"
        if not h_id:
            return None, None, f"Malformed action: missing helicopter_id. Got: {action!r}"
        if not p_id:
            return None, None, f"Malformed action: missing pallet_id. Got: {action!r}"
        if not isinstance(h_id, str):
            return None, None, f"helicopter_id must be a string, got {type(h_id).__name__}"
        if not isinstance(p_id, str):
            return None, None, f"pallet_id must be a string, got {type(p_id).__name__}"

        return h_id, p_id, None

    def _evaluate_final_solution(self) -> Tuple[float, str]:
        """
        Blended score formula (Phase 2):
          60% Fleet Utilization  — useful weight routed / total fleet capacity
          40% Critical Priority  — critical pallets delivered / total critical pallets

        Returns (blended_score, human_readable_report).
        """
        helis = list(self._helicopters.values())
        all_pallets = self._pallets_reference

        # Metric 1: useful weight (excludes containment penalty padding)
        total_capacity = sum(h.max_capacity for h in helis)
        total_useful = sum(self._useful_load.values())
        utilization = total_useful / total_capacity if total_capacity > 0 else 0.0

        # Metric 2: critical pallets routed
        total_criticals = sum(1 for p in all_pallets.values() if p.priority == "critical")
        routed_criticals = sum(
            1
            for h in helis
            for pid in h.loaded_pallets
            if all_pallets[pid].priority == "critical"
        )
        priority = routed_criticals / total_criticals if total_criticals > 0 else 1.0

        blended = round(
            UTILIZATION_WEIGHT * utilization + PRIORITY_WEIGHT * priority, 3
        )
        # Clamp the score to strictly adhere to OpenEnv rules (0 < score < 1)
        blended = max(0.001, min(0.999, float(blended)))

        report = (
            f"Score: {blended:.3f} | "
            f"Utilization: {utilization:.0%} ({total_useful}/{total_capacity} lb) | "
            f"Criticals: {routed_criticals}/{total_criticals} ({priority:.0%})"
        )
        logger.info("📊 %s", report)
        return blended, report

    def _trigger_failure(self, reason: str) -> LogisticsObservation:
        """End the episode immediately with clamped minimum reward and penalty flag."""
        if self._failure_reason is None:
            self._failure_reason = reason
        self._done = True
        logger.warning("🚨 Episode terminated: %s", reason)
        return self._build_observation(
            reward=FAILURE_REWARD,
            info=LogisticsInfo(reason=reason, penalty_applied=PENALTY_AMOUNT),
        )

    def _build_observation(
        self,
        reward: float,
        info: LogisticsInfo,
    ) -> LogisticsObservation:
        """
        Build what the AI sees plus embedded reward and info.
        The openenv framework calls .model_dump() on this object.
        """
        return LogisticsObservation(
            step_count=self._step,
            task_difficulty=self._difficulty,
            remaining_pallets=copy.deepcopy(self._pallets),
            helicopters=copy.deepcopy(self._helicopters),
            done=self._done,
            reward=reward,
            info=info.model_dump(),
        )