import os
import sys
import json
import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple

from openai import OpenAI
from client import SupplyChainEnvClient
from models import LogisticsAction

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── 1. MANDATORY LOGGING FORMATTERS ─────────────────────────────────────────

def safe_score(raw_score: float) -> float:
    """Guarantees score is strictly between 0 and 1"""
    return max(0.01, min(0.99, float(raw_score)))

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(
    step: int,
    action: dict,
    reward: float,
    done: bool,
    error: str = None,
) -> None:
    done_str = "true" if done else "false"
    error_str = error if error else "null"
    import json
    action_str = json.dumps(action) if action else "{}"
    
    safe_reward = safe_score(reward)
    
    print(
        f"[STEP] step={step} action={action_str} reward={safe_reward:.2f} done={done_str} error={error_str}",
        flush=True,
    )

def log_end(task_id: str, score_val: float, steps: int) -> None:
    final_score = safe_score(score_val)
    print(f"[END] task={task_id} score={final_score:.2f} steps={steps}", flush=True)

# ─── 2. SETTINGS & INTERACTIVE AUTH ───────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
ENV_BASE_URL = os.environ.get("ENV_URL", "https://electrifiedchan-disaster-relief-logistics.hf.space")
MAX_STEPS = 15
SUCCESS_THRESHOLD = 0.80  # blended score must hit 0.80 to count as success
MAX_RETRIES = 2
RETRY_DELAY = 2.0

# Read key from HF_TOKEN (hackathon spec)
API_KEY = (
    os.environ.get("HF_TOKEN")
    or ""
)

if not API_KEY or API_KEY == "NO_KEY_SET":
    logger.warning("⚠️ No key provided. AI brain disabled. Greedy fallback will run.")
    API_KEY = "NO_KEY_SET"

# ─── 3. PROMPT BUILDER ────────────────────────────────────────────────────────

def _build_prompt(obs: Dict[str, Any]) -> str:
    """Full detailed prompt including hazmat trap rules."""
    pallets = obs.get("remaining_pallets", {})
    helicopters = obs.get("helicopters", {})
    difficulty = obs.get("task_difficulty", "unknown")
    step = obs.get("step_count", "?")

    heli_lines = []
    for h_id, h in helicopters.items():
        free = h["max_capacity"] - h["current_load"]
        trap = h.get("containment_penalty_active", False)
        loaded = h.get("loaded_pallets", [])
        trap_warning = " ⚠️ HAZMAT MIXED - DO NOT ADD chemical/medical" if trap else ""
        heli_lines.append(
            f"  {h_id}: {h['current_load']}/{h['max_capacity']} lb used | "
            f"{free} lb free | loaded={loaded} | "
            f"containment_penalty_active={trap}{trap_warning}"
        )

    hazmat_section = ""
    if difficulty == "hard":
        hazmat_section = """
⚠️ HAZMAT TRAP (HARD MODE) — THIS IS THE #1 CAUSE OF MISSION FAILURE:
- If you load a "chemical" pallet into a helicopter that has ANY "medical" pallet,
  OR a "medical" pallet into a helicopter that has ANY "chemical" pallet,
  then +50 lb of containment equipment is INSTANTLY added to that helicopter.
- This almost always causes an overload crash and score = 0.0.
- STRATEGY: Keep ALL chemical pallets on one helicopter, ALL medical on another.
  Use a third helicopter for "safe" pallets. NEVER mix chemical and medical.
"""

    return f"""You are an emergency response AI routing disaster relief pallets to helicopters.
Difficulty: {difficulty.upper()} | Step: {step}

RULES (violating ANY rule = mission failure, score 0.0):
1. NEVER exceed a helicopter's max_capacity (check free space).
2. Load "critical" priority pallets before "standard" ones.
{hazmat_section}
HELICOPTERS:
{chr(10).join(heli_lines)}

REMAINING PALLETS:
{json.dumps(pallets, indent=2)}

Think about which helicopter can safely hold this pallet WITHOUT exceeding capacity
and WITHOUT mixing chemical+medical hazard classes.

Output exactly ONE JSON object. No markdown. No explanation.
Example: {{"helicopter_id": "Heli_A", "pallet_id": "Pallet_1"}}
"""

# ─── 4. ACTION VALIDATOR ──────────────────────────────────────────────────────

def _validate_action(
    action: Dict[str, Any], obs: Dict[str, Any]
) -> Tuple[bool, str]:
    """Check action IDs exist and weight fits before sending to environment."""
    h_id = action.get("helicopter_id")
    p_id = action.get("pallet_id")
    helicopters = obs.get("helicopters", {})
    pallets = obs.get("remaining_pallets", {})

    if not h_id or not isinstance(h_id, str):
        return False, f"Invalid helicopter_id: {h_id!r}"
    if not p_id or not isinstance(p_id, str):
        return False, f"Invalid pallet_id: {p_id!r}"
    if h_id not in helicopters:
        return False, f"Helicopter '{h_id}' not in {list(helicopters.keys())}"
    if p_id not in pallets:
        return False, f"Pallet '{p_id}' not in {list(pallets.keys())}"

    heli = helicopters[h_id]
    pallet = pallets[p_id]
    free = heli["max_capacity"] - heli["current_load"]

    # Check hazmat trap (would add +50 lb containment penalty)
    difficulty = obs.get("task_difficulty", "easy")
    hazard = pallet.get("hazard_class", "safe")
    if difficulty == "hard" and _would_trigger_trap(h_id, hazard):
        penalty_weight = 50
        effective_free = free - penalty_weight
        if pallet["weight"] > effective_free:
            return False, (
                f"HAZMAT TRAP: '{p_id}' ({hazard}) on '{h_id}' would trigger "
                f"+{penalty_weight} lb containment → only {effective_free} lb free."
            )

    if pallet["weight"] > free:
        return False, (
            f"Pallet '{p_id}' ({pallet['weight']} lb) exceeds "
            f"'{h_id}' free capacity ({free} lb)."
        )
    return True, "Valid"

# ─── 5. HAZARD TRACKING & GREEDY FALLBACK ────────────────────────────────────

# Track hazard classes loaded onto each helicopter across steps
# (since loaded pallets are removed from remaining_pallets, we can't look them up later)
_heli_hazard_map: Dict[str, set] = {}


def _update_hazard_tracking(obs: Dict[str, Any], action: Optional[Dict[str, str]] = None) -> None:
    """Track which hazard classes are on each helicopter."""
    global _heli_hazard_map
    helicopters = obs.get("helicopters", {})
    pallets = obs.get("remaining_pallets", {})

    # Initialize helis we haven't seen
    for h_id in helicopters:
        if h_id not in _heli_hazard_map:
            _heli_hazard_map[h_id] = set()

    # If we just performed an action, record the hazard class
    if action:
        p_id = action.get("pallet_id", "")
        h_id = action.get("helicopter_id", "")
        if p_id in pallets and h_id in _heli_hazard_map:
            _heli_hazard_map[h_id].add(pallets[p_id].get("hazard_class", "safe"))


def _reset_hazard_tracking() -> None:
    """Reset tracking for a new episode."""
    global _heli_hazard_map
    _heli_hazard_map = {}


def _would_trigger_trap(h_id: str, pallet_hazard: str) -> bool:
    """Check if adding this hazard class to helicopter would trigger the trap."""
    if pallet_hazard == "safe":
        return False
    existing = _heli_hazard_map.get(h_id, set())
    if pallet_hazard == "chemical" and "medical" in existing:
        return True
    if pallet_hazard == "medical" and "chemical" in existing:
        return True
    return False


def _greedy_fallback(obs: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """
    Rule-based fallback when AI model fails.
    Respects hazmat rules and prefers critical pallets.
    """
    pallets = obs.get("remaining_pallets", {})
    helicopters = obs.get("helicopters", {})
    difficulty = obs.get("task_difficulty", "easy")

    if not pallets or not helicopters:
        return None

    # Sort: critical first, then heaviest first
    sorted_pallets = sorted(
        pallets.items(),
        key=lambda x: (
            0 if x[1].get("priority") == "critical" else 1,
            -x[1].get("weight", 0),
        ),
    )

    for p_id, p_data in sorted_pallets:
        weight = p_data.get("weight", 0)
        hazard = p_data.get("hazard_class", "safe")

        for h_id, h_data in helicopters.items():
            free = h_data["max_capacity"] - h_data["current_load"]
            if weight > free:
                continue

            # On hard: check hazmat compatibility using our tracking
            if difficulty == "hard" and _would_trigger_trap(h_id, hazard):
                continue

            return {"helicopter_id": h_id, "pallet_id": p_id}

    return None

# ─── 6. AI BRAIN ──────────────────────────────────────────────────────────────

async def get_model_action(
    client: OpenAI, obs: Dict[str, Any]
) -> Tuple[Optional[Dict[str, str]], bool]:
    """
    Returns (action, used_fallback).
    used_fallback=True means greedy answered, not the AI model.
    """
    if API_KEY == "NO_KEY_SET":
        return _greedy_fallback(obs), True

    prompt = _build_prompt(obs)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Use stream=True as NVIDIA NIM requires for devstral
            stream = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.15,
                top_p=0.95,
                max_tokens=512,
                seed=42,
                stream=True,
            )

            # Collect streamed chunks into full response
            content_parts = []
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    content_parts.append(chunk.choices[0].delta.content)
            content = "".join(content_parts).strip()

            # Strip markdown fences
            if content.startswith("```json"):
                content = content[7:].rstrip("`").strip()
            elif content.startswith("```"):
                content = content[3:].rstrip("`").strip()

            action = json.loads(content)
            is_valid, reason = _validate_action(action, obs)

            if not is_valid:
                logger.warning(
                    "Attempt %d: invalid action %s — %s", attempt, action, reason
                )
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY)
                continue

            logger.info("✅ Model action (attempt %d): %s", attempt, action)
            return action, False

        except json.JSONDecodeError as e:
            logger.warning("Attempt %d: JSON parse error: %s", attempt, e)
        except Exception as e:
            logger.warning("Attempt %d: API Error - %s", attempt, e)

        if attempt < MAX_RETRIES:
            await asyncio.sleep(RETRY_DELAY)

    logger.warning("All model attempts failed. Using greedy fallback.")
    return _greedy_fallback(obs), True

# ─── 7. MAIN LOOP ─────────────────────────────────────────────────────────────

async def _run_episode(env: "SupplyChainEnvClient", ai_client: OpenAI,
                       episode_num: int) -> tuple:
    """Run one full episode. Returns (success, score, difficulty, rewards)."""
    rewards: List[float] = []
    steps_taken: int = 0
    score: float = 0.001

    reset_result = await env.reset()
    obs = reset_result.observation.model_dump()

    _reset_hazard_tracking()
    _update_hazard_tracking(obs)

    difficulty = obs.get("task_difficulty", "unknown")
    log_start(
        task=f"disaster-relief-{difficulty} (episode {episode_num}/3)",
        env="OpenEnv-SupplyChain",
        model=MODEL_NAME,
    )

    for step in range(1, MAX_STEPS + 1):
        if obs.get("done", False):
            break

        action_dict, used_fallback = await get_model_action(ai_client, obs)

        if action_dict is None:
            rewards.append(0.001)
            steps_taken = step
            log_step(step=step, action={}, reward=0.001, done=True,
                     error="No valid action from AI or fallback")
            break

        if used_fallback:
            logger.info("Step %d: greedy fallback → %s", step, action_dict)

        _update_hazard_tracking(obs, action_dict)

        try:
            result = await env.step(LogisticsAction(
                helicopter_id=action_dict["helicopter_id"],
                pallet_id=action_dict["pallet_id"],
            ))
        except RuntimeError as e:
            rewards.append(0.001)
            steps_taken = step
            log_step(step=step, action=action_dict, reward=0.001, done=True, error=str(e))
            break

        obs = result.observation.model_dump()
        reward = float(result.reward or 0.001)
        done = result.done
        info = obs.get("info", {})

        penalty = float(info.get("penalty_applied", 0.0))
        reason = info.get("reason", "")
        error_str = reason if penalty > 0.0 else None

        rewards.append(reward)
        steps_taken = step
        log_step(step=step, action=action_dict, reward=reward, done=done, error=error_str)

        if done:
            score = reward
            break

    success = score >= SUCCESS_THRESHOLD
    task_name = f"disaster-relief-{difficulty} (episode {episode_num}/3)"
    total_score = sum(rewards)
    step_count = steps_taken
    # Ensure the variable names match the ones in your local loop scope
    log_end(task_id=task_name, score_val=total_score, steps=step_count)
    return success, score, difficulty, rewards


async def main() -> None:
    ai_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    all_results = []
    any_episode_started = False

    try:
        # One WebSocket connection → 3 episodes: EASY → MEDIUM → HARD
        async with SupplyChainEnvClient(base_url=ENV_BASE_URL) as env:
            for ep in range(1, 4):
                any_episode_started = True
                success, score, diff, rewards = await _run_episode(env, ai_client, ep)
                all_results.append((diff, score, success))

        # ── Final summary ──────────────────────────────────────────────────
        print("\n" + "=" * 55)
        print("3-MODE EVALUATION COMPLETE")
        print("=" * 55)
        for diff, score, ok in all_results:
            label = "PASS" if ok else "FAIL"
            print(f"  {diff.upper():8s}  score={score:.3f}  [{label}]")
        passed = sum(1 for _, _, ok in all_results if ok)
        print(f"\n  Overall: {passed}/3 modes passed (threshold >= {SUCCESS_THRESHOLD})")

    except Exception as e:
        logger.error("Fatal error: %s", e)
        # Spec: [END] must always be emitted, even on exception
        if not any_episode_started:
            log_start(task="disaster-relief-easy", env="OpenEnv-SupplyChain", model=MODEL_NAME)
            log_end(task_id="disaster-relief-easy", score_val=0.001, steps=0)

if __name__ == "__main__":
    asyncio.run(main())