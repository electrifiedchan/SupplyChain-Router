import sys
import os
import copy
import logging
import random
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

print("--- LOGISTICS SERVER INITIALIZED: HELI_B CAPACITY = 75 ---", flush=True)

# ─── Constants ────────────────────────────────────────────────────────────────
CONTAINMENT_PENALTY_WEIGHT: int = 50
MAX_STEPS: int = 15
DIFFICULTY_CYCLE: List[str] = ["easy", "medium", "hard"]
FAILURE_REWARD: float = 0.001
REPETITION_PENALTY: float = 0.0
REPETITION_WINDOW: int = 3
PENALTY_AMOUNT: float = 1.0
UTILIZATION_WEIGHT: float = 0.60
PRIORITY_WEIGHT: float = 0.40

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
            "Pallet_1": ReliefPallet(weight=45, priority="critical", hazard_class="medical"),
            "Pallet_2": ReliefPallet(weight=40, priority="standard", hazard_class="safe"),
            "Pallet_3": ReliefPallet(weight=35, priority="critical", hazard_class="medical"),
            "Pallet_4": ReliefPallet(weight=30, priority="standard", hazard_class="safe"),
            "Pallet_5": ReliefPallet(weight=20, priority="standard", hazard_class="safe"),
            "Pallet_6": ReliefPallet(weight=10, priority="standard", hazard_class="safe"),
        },
        "helicopters": {
            "Heli_A": Helicopter(max_capacity=80),
            "Heli_B": Helicopter(max_capacity=80), 
            "Heli_C": Helicopter(max_capacity=60),
        },
    },
    "hard": {
        "pallets": {
            "Pallet_1": ReliefPallet(weight=40, priority="critical",  hazard_class="medical"),
            "Pallet_2": ReliefPallet(weight=40, priority="critical",  hazard_class="chemical"),
            "Pallet_3": ReliefPallet(weight=20, priority="critical",  hazard_class="safe"),
            "Pallet_4": ReliefPallet(weight=40, priority="standard",  hazard_class="safe"),
            "Pallet_5": ReliefPallet(weight=20, priority="standard",  hazard_class="medical"),
            "Pallet_6": ReliefPallet(weight=20, priority="standard",  hazard_class="chemical"),
        },
        "helicopters": {
            "Heli_A": Helicopter(max_capacity=110),
            "Heli_B": Helicopter(max_capacity=110),
            "Heli_C": Helicopter(max_capacity=40),
        },
    },
}

# ─── Baseline capacities (for scoring — never mutated after init) ─────────────
SCENARIO_BASELINE_CAPACITY: Dict[str, int] = {
    name: sum(h.max_capacity for h in s["helicopters"].values())
    for name, s in SCENARIOS.items()
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

    SUPPORTS_CONCURRENT_SESSIONS = False  # Architecture fix #7

    def __init__(self) -> None:
        super().__init__()
        self._episode_count: int = 0
        self._step: int = 0
        self._difficulty: str = "easy"
        self._done: bool = False
        self._failure_reason: Optional[str] = None
        self._anomaly_triggered: bool = False
        self._active_alert: str = ""
        self._action_history: List[Tuple[str, str]] = []
        _init_scenario = copy.deepcopy(SCENARIOS["easy"])
        self._pallets: Dict[str, ReliefPallet] = _init_scenario["pallets"]
        self._helicopters: Dict[str, Helicopter] = _init_scenario["helicopters"]
        self._pallets_reference: Dict[str, ReliefPallet] = copy.deepcopy(_init_scenario["pallets"])
        self._useful_load: Dict[str, int] = {h_id: 0 for h_id in self._helicopters}

    def reset(self) -> LogisticsObservation:
        self._initialise_episode()
        return self._build_observation(
            reward=0.0,
            info=LogisticsInfo(reason="Episode started. Good luck."),
        )

    def step(self, action: Any) -> LogisticsObservation:
        try:
            if self._done:
                return self._trigger_failure("Logic Error: step() called after episode already finished.")
            if self._step >= MAX_STEPS:
                return self._trigger_failure(f"Timeout: {MAX_STEPS} steps reached.")

            self._step += 1

            try:
                h_id, p_id, parse_error = self._parse_action(action)
                if parse_error or h_id is None or p_id is None:
                    return self._build_observation(
                        reward=0.01,
                        info=LogisticsInfo(reason="NO_VALID_ACTION_AVAILABLE", penalty_applied=0.0)
                    )
            except Exception:
                return self._build_observation(
                    reward=0.01,
                    info=LogisticsInfo(reason="INVALID_OR_EMPTY_ACTION", penalty_applied=0.0)
                )

            if (h_id, p_id) in self._action_history[-REPETITION_WINDOW:]:
                self._action_history.append((h_id, p_id))
                return self._build_observation(
                    reward=REPETITION_PENALTY,
                    info=LogisticsInfo(
                        reason=f"Repeat action: '{p_id}' → '{h_id}' was already attempted.",
                        penalty_applied=abs(REPETITION_PENALTY),
                    ),
                )

            if h_id not in self._helicopters:
                return self._trigger_failure(f"Hallucination: Helicopter '{h_id}' does not exist.")
            if p_id not in self._pallets:
                return self._trigger_failure(f"Hallucination: Pallet '{p_id}' does not exist or already routed.")

            # 🛡️ THE HALLUCINATION FIREWALL 🛡️
            try:
                target_heli = self._helicopters[h_id]
                target_pallet = self._pallets[p_id]

                trap_triggered = False
                if self._difficulty == "hard" and not target_heli.containment_penalty_active:
                    existing_hazards = {
                        self._pallets_reference[pid].hazard_class
                        for pid in target_heli.loaded_pallets
                    }
                    incoming = target_pallet.hazard_class
                    mixing = (incoming == "chemical" and "medical" in existing_hazards) or \
                             (incoming == "medical" and "chemical" in existing_hazards)
                    if mixing:
                        target_heli.containment_penalty_active = True
                        target_heli.current_load += CONTAINMENT_PENALTY_WEIGHT
                        trap_triggered = True

                if target_heli.current_load + target_pallet.weight > target_heli.max_capacity:
                    overflow = (target_heli.current_load + target_pallet.weight) - target_heli.max_capacity
                    reason = f"Physics Violation: '{p_id}' onto '{h_id}' overflows by {overflow} lb."
                    if trap_triggered:
                        reason += f" Crash caused by Dynamic Weight Trap (+{CONTAINMENT_PENALTY_WEIGHT} lb)."
                    return self._trigger_failure(reason)

                self._action_history.append((h_id, p_id))
                self._pallets.pop(p_id)
                target_heli.loaded_pallets.append(p_id)
                target_heli.current_load += target_pallet.weight
                self._useful_load[h_id] += target_pallet.weight

            except KeyError as e:
                logger.warning("FIREWALL INTERCEPT: LLM Hallucinated invalid ID: %s", e)
                return self._build_observation(
                    reward=0.0,
                    info=LogisticsInfo(
                        reason=f"ACTION REJECTED: Invalid ID {e}. Must use exact IDs provided.",
                        penalty_applied=0.0,
                    ),
                )

            total_pallets = len(self._pallets_reference)
            delivered_pallets = total_pallets - len(self._pallets)
            count_progress = delivered_pallets / total_pallets if total_pallets > 0 else 0.0

            total_weight = sum(p.weight for p in self._pallets_reference.values())
            delivered_weight = total_weight - sum(p.weight for p in self._pallets.values())
            weight_progress = delivered_weight / total_weight if total_weight > 0 else 0.0

            total_criticals = sum(1 for p in self._pallets_reference.values() if p.priority == "critical")
            if total_criticals > 0:
                remaining_criticals = sum(1 for p in self._pallets.values() if p.priority == "critical")
                critical_progress = (total_criticals - remaining_criticals) / total_criticals
            else:
                critical_progress = count_progress

            raw_progress = (0.40 * count_progress + 0.35 * weight_progress + 0.25 * critical_progress)
            step_reward = max(0.01, min(0.99, raw_progress ** 0.55))

            if len(self._pallets) == 0:
                self._done = True
                blended_score, oracle_report = self._evaluate_final_solution()
                task_id = f"{self._difficulty}_ep{self._episode_count}"
                print(f"[END] task={task_id} score={blended_score:.3f} steps={self._step}", flush=True)
                return self._build_observation(
                    reward=blended_score,
                    info=LogisticsInfo(
                        reason=f"Mission Complete! {oracle_report}",
                        penalty_applied=0.0,
                        dynamic_weight_trap_triggered=trap_triggered,
                        containment_weight_added=CONTAINMENT_PENALTY_WEIGHT if trap_triggered else 0,
                        oracle_comparison=oracle_report,
                    ),
                )

            if self._step >= MAX_STEPS:
                self._done = True
                task_id = f"{self._difficulty}_ep{self._episode_count}"
                print(f"[END] task={task_id} score={FAILURE_REWARD:.3f} steps={self._step}", flush=True)
                return self._build_observation(
                    reward=FAILURE_REWARD,
                    info=LogisticsInfo(reason=f"Timeout: {MAX_STEPS} steps reached.", penalty_applied=0.0),
                )

            # 🚨 TWO-STEP STOCHASTIC ANOMALY (HARD MODE) 🚨
            if self._difficulty == "hard":
                if self._step == 3 and not self._anomaly_triggered:
                    self._anomaly_triggered = True
                    self._active_alert = (
                        "CRITICAL TELEMETRY WARNING: "
                        "SPECI KTBW 141347Z 18032G58KT 1/4SM +TSRA FG BKN005 OVC010CB. "
                        "!KTBW NOTAM: SEVERE WEATHER APPROACHING Heli_C. "
                        "EVACUATE CARGO IMMEDIATELY. GROUNDING IMMINENT IN 1 STEP."
                    )
                elif self._step >= 4 and self._anomaly_triggered:
                    if "Heli_C" in self._helicopters:
                        # Reordered for Pydantic 'validate_assignment' safety
                        self._helicopters["Heli_C"].current_load = 0   # 1. Reset load
                        self._helicopters["Heli_C"].loaded_pallets = [] # 2. Clear inventory
                        self._helicopters["Heli_C"].max_capacity = 0    # 3. Finally ground
                        
                    self._active_alert = (
                        "CRITICAL TELEMETRY OVERRIDE: "
                        "TORNADO IMPACT CONFIRMED. Heli_C IS GROUNDED. "
                        "CARGO EVACUATED."
                    )

            return self._build_observation(
                reward=step_reward,
                info=LogisticsInfo(
                    reason=self._active_alert if self._active_alert else "Action accepted.",
                    penalty_applied=0.0,
                    dynamic_weight_trap_triggered=trap_triggered,
                    containment_weight_added=CONTAINMENT_PENALTY_WEIGHT if trap_triggered else 0,
                ),
            )

        except Exception as e:
            logger.error("CRITICAL ERROR in step(): %s", e, exc_info=True)
            return self._trigger_failure(f"Internal Server Error: {str(e)}")

    @property
    def state(self) -> LogisticsState:
        return LogisticsState(
            step_count=self._step,
            difficulty=self._difficulty,
            pallets_remaining=len(self._pallets),
            done=self._done,
        )

    def _initialise_episode(self) -> None:
        self.current_seed = random.randint(1000, 9999)
        random.seed(self.current_seed)
        print(f"[START] initialization seed={self.current_seed}", flush=True)

        self._episode_count += 1
        self._difficulty = DIFFICULTY_CYCLE[(self._episode_count - 1) % len(DIFFICULTY_CYCLE)]
        scenario = copy.deepcopy(SCENARIOS[self._difficulty])
        self._step = 0
        self._done = False
        self._failure_reason = None
        self._anomaly_triggered = False
        self._active_alert = ""
        self._helicopters = scenario["helicopters"]
        
        pallets_base = scenario["pallets"]
        total_fleet_capacity = sum(h.max_capacity for h in self._helicopters.values())
        
        base_weights = {pid: p.weight for pid, p in pallets_base.items()}
        
        for _ in range(10):
            total_pallet_weight = 0
            for pid, p in pallets_base.items():
                base_weight = base_weights[pid]
                p.weight = max(1, base_weight + random.randint(-5, 5))
                total_pallet_weight += p.weight
                
            if total_pallet_weight <= total_fleet_capacity:
                break
        else:
            pallets_base = copy.deepcopy(SCENARIOS[self._difficulty])["pallets"]
            total_pallet_weight = sum(p.weight for p in pallets_base.values())
            
        self._pallets = pallets_base
        self._pallets_reference = copy.deepcopy(self._pallets)
        self._useful_load = {h_id: 0 for h_id in self._helicopters}
        self._action_history = []
        
        self.baseline_payload_weight = total_pallet_weight
        SCENARIO_BASELINE_CAPACITY[self._difficulty] = total_pallet_weight

    def _parse_action(self, action: Any) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        if isinstance(action, dict):
            h_id = action.get("helicopter_id")
            p_id = action.get("pallet_id")
        else:
            h_id = getattr(action, "helicopter_id", None)
            p_id = getattr(action, "pallet_id", None)

        h_id = str(h_id) if h_id is not None else ""
        p_id = str(p_id) if p_id is not None else ""

        if not h_id and not p_id: return None, None, "Malformed action: missing both IDs."
        if not h_id: return None, None, "Malformed action: missing helicopter_id."
        if not p_id: return None, None, "Malformed action: missing pallet_id."

        return h_id, p_id, None

    def _evaluate_final_solution(self) -> Tuple[float, str]:
        helis = list(self._helicopters.values())
        all_pallets = self._pallets_reference
        
        active_max_capacity = sum(h.max_capacity for h in helis if h.max_capacity > 0)
        active_useful_load = sum(
            self._useful_load[h_id] 
            for h_id, h in self._helicopters.items() 
            if h.max_capacity > 0
        )
        utilization = active_useful_load / active_max_capacity if active_max_capacity > 0 else 0.0

        total_criticals = sum(1 for p in all_pallets.values() if p.priority == "critical")
        routed_criticals = sum(
            1 for h in helis for pid in h.loaded_pallets if all_pallets[pid].priority == "critical"
        )
        priority = routed_criticals / total_criticals if total_criticals > 0 else 1.0

        blended = round(UTILIZATION_WEIGHT * utilization + PRIORITY_WEIGHT * priority, 3)
        blended = max(0.001, min(0.999, float(blended)))

        report = f"Score: {blended:.3f} | Utilization: {utilization:.0%} | Criticals: {routed_criticals}/{total_criticals}"
        return blended, report

    def _trigger_failure(self, reason: str) -> LogisticsObservation:
        if self._failure_reason is None:
            self._failure_reason = reason
        self._done = True
        task_id = f"{self._difficulty}_ep{self._episode_count}"
        print(f"[END] task={task_id} score={FAILURE_REWARD:.3f} steps={self._step}", flush=True)
        return self._build_observation(
            reward=FAILURE_REWARD,
            info=LogisticsInfo(reason=reason, penalty_applied=PENALTY_AMOUNT),
        )

    def _build_observation(self, reward: float, info: LogisticsInfo) -> LogisticsObservation:
        return LogisticsObservation(
            step_count=self._step,
            task_difficulty=self._difficulty,
            remaining_pallets=copy.deepcopy(self._pallets),
            helicopters=copy.deepcopy(self._helicopters),
            done=self._done,
            reward=reward,
            info=info.model_dump(),
        )